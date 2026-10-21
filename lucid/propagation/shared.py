"""The surface photon propagator: one factory for any Detector subclass.

Geometry enters only through the Detector methods called below (configure_grid,
assign_sensor_to_cells, grid_cell_centers, build_inverted_sensor_map, bounds_check,
intersect_ray, point_to_grid_cell, compute_normal); everything else here is shape-agnostic.

Not the only propagator: `lucid/propagation/string/` traverses a per-DOM volume rather than a
surface grid, and carries its own deposit implementation. The two have diverged: that one clamps
the closest-approach parameter where this one gates on it, and has no first-hit survival product.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from lucid.propagation.base import (
    compute_sensor_intersections_base,
    compute_flat_sensor_intersections,
    flat_footprint_distance,
    _surface_frame,
    process_intersection_normals,
    find_closest_sensors,
)
from lucid.overlap import create_overlap_prob


def first_hit_survival(weights, times):
    """Cap the deposit at one photon: P(hit s_i) = p_i * prod_{j before i} (1 - p_j).

    `overlap_prob` is applied INDEPENDENTLY per candidate, so on its own it leaves the total uncapped:
    at grazing incidence a ray skimming the wall passes within r of a whole row of sensors and each
    takes full weight. Physically the photon
    deposits on the FIRST sensor it reaches, so with candidates ordered by arrival time and p_i the
    conditional hit probability given the photon gets there, sum(P) = 1 - prod(1 - p_j) <= 1.
    It splits weight smoothly between adjacent sensors (p1, p2*(1-p1)) and reduces to p_1 when
    only one candidate is in range. Done in log space; gated-out candidates have p = 0 and
    contribute exactly 1 to the product.

    TIES ARE BROKEN BY SLOT. Two live candidates can arrive at EXACTLY the same time -- a ray
    equidistant from two sensors -- and with a strict `t_j < t_i` neither counts as earlier, both
    keep their full p, and the sum can exceed 1. The slot index orders them, so the cap holds by construction rather than almost always.

    THE ORDERING IS A HARD COMPARISON. The product is smooth in the p's, but which p's multiply
    which is decided by a threshold on arrival time, so when two candidates swap order the
    per-sensor split jumps while the total stays put. Aggregate checks cannot see it; a
    per-sensor gradient can.

    Parameters
    ----------
    weights : (C, N) per-candidate overlap probabilities
    times : (C, N) per-candidate arrival times

    Returns
    -------
    (C, N) capped weights, summing to at most 1 over C for every photon.
    """
    # Clipped below 1 so log1p(-p) stays finite -- STRAIGHT-THROUGH, so the clip shapes only the
    # forward value (bit-identical to a plain clip). Must not be a plain jnp.clip for the
    # gradient: in step mode (temperature=None) the forward overlap is EXACTLY 1 inside a
    # sphere and 0 outside, where a plain clip has zero derivative (above its maximum) or half
    # (at a tie with its minimum), which would remove the straight-through surrogate gradient
    # the hard step exists to keep.
    p = weights + jax.lax.stop_gradient(jnp.clip(weights, 0.0, 1.0 - 1e-6) - weights)
    slot = jnp.arange(p.shape[0])
    # before[i, j, n]: candidate j reaches photon n's path before candidate i does
    before = ((times[None, :, :] < times[:, None, :])
              | ((times[None, :, :] == times[:, None, :])
                 & (slot[None, :, None] < slot[:, None, None])))
    log_survive = jnp.sum(before * jnp.log1p(-p)[None, :, :], axis=1)
    return p * jnp.exp(log_survive)


def validate_sensor_map(assignments_geometric, inverted_sensor_map, num_sensors,
                        detector, max_candidates_per_ray):
    """Check consistency between forward (sensor→cells) and inverse (cell→sensors) maps.

    Runs at propagator build time (numpy, not JIT). Raises warnings for
    any issues that could silently degrade simulation quality.

    Checks:
    1. Index bounds — no out-of-range sensor IDs
    2. Cell coverage — fraction of cells with at least one sensor
    3. Sensor visibility — are all sensors reachable through the map?
    4. Overcrowding — cells where geometric assignments exceed max_candidates_per_ray
    5. Forward-inverse consistency — geometric assignments present in inverse map
    """
    inv = np.asarray(inverted_sensor_map)
    fwd = np.asarray(assignments_geometric)
    total_cells, slots = inv.shape
    valid_mask = inv != -1

    # --- 1. Index bounds ---
    if valid_mask.any():
        min_idx, max_idx = int(inv[valid_mask].min()), int(inv[valid_mask].max())
        if min_idx < 0 or max_idx >= num_sensors:
            warnings.warn(
                f"Sensor map: out-of-range indices [{min_idx}, {max_idx}] "
                f"for {num_sensors} sensors")

    # --- 2. Cell coverage ---
    cells_with_sensors = int(np.any(valid_mask, axis=1).sum())
    coverage_pct = 100.0 * cells_with_sensors / total_cells if total_cells > 0 else 0
    if coverage_pct < 90.0:
        warnings.warn(
            f"Sensor map: low cell coverage — {cells_with_sensors}/{total_cells} "
            f"({coverage_pct:.1f}%) cells have sensors. Photons hitting empty "
            f"cells will produce zero weights.")

    # --- 3. Sensor visibility ---
    sensors_in_map = set(int(x) for x in inv[valid_mask])
    missing_sensors = set(range(num_sensors)) - sensors_in_map
    if missing_sensors:
        warnings.warn(
            f"Sensor map: {len(missing_sensors)}/{num_sensors} sensors do not "
            f"appear in any cell's inverse map. These sensors can never be hit.")

    # --- 4. Overcrowding (geometric assignments exceed max_candidates_per_ray) ---
    # Count how many geometric assignments each cell receives
    cell_geo_count = np.zeros(total_cells, dtype=int)
    for sensor_id in range(fwd.shape[0]):
        for slot in range(fwd.shape[1]):
            coords = fwd[sensor_id, slot]
            if np.all(coords == -1):
                continue
            linear_idx = int(detector.point_to_grid_cell_from_coords(coords))
            if 0 <= linear_idx < total_cells:
                cell_geo_count[linear_idx] += 1

    max_geo = int(cell_geo_count.max()) if total_cells > 0 else 0
    if max_geo > max_candidates_per_ray:
        warnings.warn(
            f"Sensor map: max geometric sensor assignments per cell ({max_geo}) "
            f"exceeds max_candidates_per_ray={max_candidates_per_ray}. "
            f"Auto-adjusting to {max_geo}.")

    # --- 5. Forward-inverse consistency ---
    n_missing = 0
    n_checked = 0
    for sensor_id in range(fwd.shape[0]):
        for slot in range(fwd.shape[1]):
            coords = fwd[sensor_id, slot]
            if np.all(coords == -1):
                continue
            linear_idx = int(detector.point_to_grid_cell_from_coords(coords))
            if linear_idx < 0 or linear_idx >= total_cells:
                continue
            n_checked += 1
            if sensor_id not in inv[linear_idx]:
                n_missing += 1

    if n_missing > 0:
        warnings.warn(
            f"Sensor map: {n_missing}/{n_checked} geometric assignments are "
            f"missing from the inverse map — likely dropped due to "
            f"max_candidates_per_ray={max_candidates_per_ray} overflow.")


def flat_sensor_normals(detector, sensor_positions):
    """Outward surface normal of each flat sensor.

    A flat sensor faces into the detector, so its outward normal is the opposite of its viewing
    direction when the detector carries those (``Cylinder.from_pmt_file`` keeps the npz
    ``directions`` as ``pmt_directions``). Otherwise it comes from a ray cast from the detector
    centre to the sensor centre (the surface detectors are convex, so the ray reaches every
    surface point), which needs the centre to lie on the surface: a disc whose centre is past a
    cap's rim would get the barrel's normal.
    """
    dirs = getattr(detector, 'pmt_directions', None)
    if dirs is not None:
        d = jnp.asarray(dirs, dtype=jnp.float32)
        return -d / jnp.linalg.norm(d, axis=1, keepdims=True)
    pos = jnp.asarray(sensor_positions, dtype=jnp.float32)
    dirs = pos / (jnp.linalg.norm(pos, axis=1, keepdims=True) + 1e-12)
    ip, _, _, surf = detector.intersect_ray(jnp.zeros_like(pos), dirs)
    return detector.compute_normal(ip, surf)


def validate_flat_sensor_coverage(detector, sensor_positions, sensor_model,
                                  inverted_sensor_map, n_grid=9):
    """Check that the cell lookup offers every flat sensor wherever a photon can land on it.

    Runs at propagator build time. A sensor larger than a cell (a PMT window on a grid sized by
    SiPMs) is assigned geometrically to a few cells only and relies on the closest-sensor
    fallback for the rest; this checks that it is there. Samples an ``n_grid`` x ``n_grid``
    lattice over every footprint, casts a ray from the detector centre through each sample (the
    surface detectors are convex, so every surface point is reached), and wherever the landing
    point falls inside the sensor requires the sensor among the candidates of its cell. Raises
    ValueError otherwise: the sensor would silently lose that part of its area.
    """
    pos = jnp.asarray(sensor_positions, dtype=jnp.float32)
    n = pos.shape[0]
    u, v = _surface_frame(sensor_model.normal)
    g = (jnp.arange(n_grid) + 0.5) / n_grid * 2.0 - 1.0
    a, b = [x.ravel() for x in jnp.meshgrid(g, g, indexing='ij')]
    half = sensor_model.half_size[:, None, None]
    samples = pos[:, None, :] + half * (a[None, :, None] * u[:, None, :]
                                        + b[None, :, None] * v[:, None, :])
    samples = samples.reshape(-1, 3)
    owner = jnp.repeat(jnp.arange(n), a.shape[0])
    dirs = samples / (jnp.linalg.norm(samples, axis=1, keepdims=True) + 1e-12)
    land, _, grid_info, surf = detector.intersect_ray(jnp.zeros_like(samples), dirs)
    x, coplanar = flat_footprint_distance(
        land, detector.compute_normal(land, surf), pos[owner],
        sensor_model.is_disc[owner], sensor_model.half_size[owner], sensor_model.normal[owner])
    on_sensor = np.asarray(coplanar & (x < 1.0))
    cells = np.asarray(detector.point_to_grid_cell(grid_info))
    offered = np.any(np.asarray(inverted_sensor_map)[cells] == np.asarray(owner)[:, None], axis=1)
    missed = on_sensor & ~offered
    if missed.any():
        bad = np.unique(np.asarray(owner)[missed])
        raise ValueError(
            f"Flat sensors: {int(missed.sum())} of {int(on_sensor.sum())} sampled footprint "
            f"points, on {len(bad)} sensors (e.g. {bad[:5].tolist()}), land in cells whose "
            "candidates do not include the sensor. Raise max_candidates_per_ray or the npz "
            "sensor_radius.")


def create_propagator(detector, sensor_positions, sensor_radius,
                      temperature=0.2, max_candidates_per_ray=4,
                      overlap_st_width_frac=0.35, overlap_renorm=1.0,
                      overlap_mode='interp', deposit_leg_bound=False,
                      sensor_model=None,
                      **grid_params):
    """Build a JIT-compiled photon propagator using detector methods.

    Parameters
    ----------
    detector : Detector
        Detector instance with Phase 9 methods implemented.
    sensor_positions : jnp.ndarray, shape (n_sensors, 3)
    sensor_radius : float
    temperature : float
        Soft-assignment temperature for overlap probability.
    overlap_st_width_frac : float
        Straight-through surrogate width (fraction of r) for the hard-step
        overlap. Backward-gradient only; default 0.35.
    overlap_renorm : float
        Soft-overlap renormalization constant C (default 1.0 = OFF).
    overlap_mode : str
        Soft-overlap lookup interpolation: 'interp' (default) or 'cubic'.
    deposit_leg_bound : bool
        Bound the deposit to the leg the photon actually travels, [0, t_geometry], instead of
        weighting by distance from the unbounded ray LINE. Default False; a Python bool
        resolved at trace time, so when off the leg-bound arithmetic is absent from the graph
        rather than present and unused, and the result is bit-identical to the behaviour without it.

        The line does not stop at the wall, so for a ray at incidence theta it passes within a
        sensor radius of sensors displaced along the wall from the landing point, over-counting
        hits by (1 - cos theta)/2 per ray. Normal incidence is unaffected. Switching it on
        changes the forward model, and with it any energy scale calibrated without it.
    sensor_model : FlatSensorModel or None
        None (default) keeps the spherical sensor of ``sensor_radius`` centred on the wall.
        A :class:`lucid.propagation.base.FlatSensorModel` makes every sensor a flat pad in the
        surface (a square or a disc, each its own size), hit iff the photon's landing point
        falls inside it, so it collects ``area * cos(theta)``. Its deposit is always the hard
        step: ``temperature`` and the soft-overlap options apply to the spherical model only.
        ``sensor_radius`` still sizes the cell assignment; sensors larger than that are found
        through the closest-sensor fallback, which :func:`validate_flat_sensor_coverage`
        checks at build time. A Python-level choice made at trace time, so the default graph
        is unchanged.
    max_candidates_per_ray : int
    **grid_params
        Geometry-specific grid parameters passed to ``detector.configure_grid()``.
        Cylinder: n_cap, n_angular, n_height.
        Sphere: n_divisions.
        Box: n_x, n_y, n_z.

    Returns
    -------
    callable
        JIT-compiled ``propagate_photons(origins, directions) -> dict``
    """
    sensor_positions = jnp.array(sensor_positions)
    num_sensors = len(sensor_positions)

    # Configure grid on detector — caller passes geometry-specific params.
    # max_candidates_per_ray is always forwarded so auto-derivation can
    # ensure no cell exceeds this limit.
    grid_params.setdefault('max_candidates_per_ray', max_candidates_per_ray)
    detector.configure_grid(**grid_params)

    # 1. Geometric sensor-to-cell assignments
    assignments_geometric = detector.assign_sensor_to_cells(sensor_positions, sensor_radius)

    # 2. Grid cell centers
    grid_centers = detector.grid_cell_centers()

    # 3. Distance-based fallback assignments (shared)
    assignments_distance = find_closest_sensors(
        grid_centers, sensor_positions, max_candidates_per_ray)

    # 4. Build inverted sensor map (geometry-specific decoder)
    inverted_sensor_map = detector.build_inverted_sensor_map(
        assignments_geometric, assignments_distance,
        max_candidates_per_ray, num_sensors)

    # 4b. Validate the sensor map
    validate_sensor_map(assignments_geometric, inverted_sensor_map,
                        num_sensors, detector, max_candidates_per_ray)

    # 5. Overlap probability (shared)
    # temperature=None → step function (hard assignment, non-differentiable)
    # temperature=float → Gaussian kernel with sigma = temperature * sensor_radius
    if temperature is None:
        overlap_prob = create_overlap_prob(
            None, sensor_radius,
            st_width_frac=overlap_st_width_frac, renorm=overlap_renorm, mode=overlap_mode)
    else:
        overlap_prob = create_overlap_prob(
            temperature * sensor_radius, sensor_radius,
            st_width_frac=overlap_st_width_frac, renorm=overlap_renorm, mode=overlap_mode)

    # 5b. Flat sensors: every footprint reachable through the lookup; a hard step at the
    # pad edge, on the distance normalised by each pad's half size.
    if sensor_model is not None:
        sensor_model = sensor_model._replace(
            normal=flat_sensor_normals(detector, sensor_positions))
        validate_flat_sensor_coverage(detector, sensor_positions, sensor_model,
                                      inverted_sensor_map)
        flat_overlap_prob = create_overlap_prob(None, 1.0, st_width_frac=overlap_st_width_frac)

    # 6. Bounds check closure
    def bounds_check(positions):
        return detector.bounds_check(positions)

    # 7. JIT-compiled propagation function
    @jax.jit
    def propagate_photons(photon_origins, photon_directions):
        """Trace photon rays through detector geometry.

        Parameters
        ----------
        photon_origins : jnp.ndarray, shape (n_rays, 3) or (3,)
        photon_directions : jnp.ndarray, shape (n_rays, 3) or (3,)

        Returns
        -------
        dict with keys: sensor_weights, sensor_indices, times, positions,
             normals, inside_sensor, per_sensor_positions, sensor_normals
        """
        single_ray = photon_origins.ndim == 1
        if single_ray:
            photon_origins = photon_origins[None, :]
            photon_directions = photon_directions[None, :]

        # a. Ray-geometry intersection
        intersection_point, t_geometry, grid_info, surface_info = \
            detector.intersect_ray(photon_origins, photon_directions)

        # b. Map to grid cell indices
        idx = detector.point_to_grid_cell(grid_info)

        # c. Look up candidate sensors (stop_gradient: geometry is static)
        potential_sensors = jax.lax.stop_gradient(inverted_sensor_map[idx])

        # d. Compute sensor intersections (shared, vmapped over sensor slots)
        if sensor_model is None:
            def compute_for_slot(slot_sensors):
                return compute_sensor_intersections_base(
                    slot_sensors, sensor_positions, sensor_radius,
                    photon_origins, photon_directions,
                    bounds_check, overlap_prob,
                    # None when off drops the leg-bound code from the trace; see `deposit_leg_bound`.
                    t_geometry=t_geometry if deposit_leg_bound else None)
        else:
            landing_normals = detector.compute_normal(intersection_point, surface_info)

            def compute_for_slot(slot_sensors):
                return compute_flat_sensor_intersections(
                    slot_sensors, sensor_positions, photon_origins, photon_directions,
                    intersection_point, landing_normals, sensor_model, flat_overlap_prob)

        (weights, sensor_times, sensor_indices,
         sensor_normals_all, inside_sensor,
         sensor_hit_positions) = jax.vmap(
            compute_for_slot, in_axes=1, out_axes=0)(potential_sensors)

        # First-hit semantics: the photon deposits on the first sensor it reaches. See
        # `first_hit_survival` for the cap, its tie-break, and what its hard ordering costs.
        weights = first_hit_survival(
            weights, jnp.squeeze(sensor_times, -1) if sensor_times.ndim == 3 else sensor_times)

        # e. Compute geometry surface normals
        geometry_normals = detector.compute_normal(intersection_point, surface_info)

        # f. Process intersection normals (shared)
        final_results = process_intersection_normals(
            photon_origins, photon_directions, intersection_point,
            t_geometry, sensor_normals_all, sensor_hit_positions,
            inside_sensor, geometry_normals)

        hit_positions = final_results['positions']
        final_normals = final_results['normals']

        # g. Assemble result dict
        result = {
            'times': sensor_times,
            'sensor_weights': weights,
            'sensor_indices': sensor_indices,
            'per_sensor_positions': sensor_hit_positions,
            'positions': hit_positions,
            'normals': final_normals,
            'sensor_normals': sensor_normals_all,
            'inside_sensor': inside_sensor,
        }

        if single_ray:
            result = jax.tree.map(lambda x: x[0] if x.ndim > 0 else x, result)

        return result

    return propagate_photons
