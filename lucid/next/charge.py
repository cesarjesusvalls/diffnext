"""Ionization charge of a point deposit in the NEXT-like detector, transported with JAXTPC.

The ionization drifts along the detector axis to the gate. JAXTPC computes the drift
(``tools.drift.compute_drift_to_plane``); its ``SimParams`` carry the transport parameters (drift
velocity, electron lifetime, transverse and longitudinal diffusion coefficients), and the
survival fraction exp(-t / lifetime) and diffusion widths sqrt(2 D t) follow JAXTPC's own
conventions (``tools.physics``, ``tools.wires``).

Electrons reaching the electroluminescence gap are represented by macro-electrons placed with
fixed standard normals, so their arrival positions are differentiable in the deposit position
and the diffusion.

Needs JAXTPC (https://github.com/OmarAlterkait/JAXTPC) on the python path: its ``tools`` package.
"""
from typing import NamedTuple

import jax.numpy as jnp

try:
    from tools.config import SimParams
    from tools.drift import compute_drift_to_plane
except ImportError as err:          # pragma: no cover - depends on the environment
    raise ImportError(
        "lucid.next.charge needs JAXTPC's 'tools' package on the python path "
        "(e.g. PYTHONPATH=$JAXTPC_PATH; https://github.com/OmarAlterkait/JAXTPC)") from err


def xenon_drift_params(velocity_mm_us=1.0, lifetime_ms=1000.0,
                       diffusion_trans_mm_sqrtcm=1.0, diffusion_long_mm_sqrtcm=0.3):
    """JAXTPC ``SimParams`` for the NEXT gas from NEXT-style constants.

    Defaults are the nexus NEXT-100 values: drift velocity 1 mm/us and diffusion 1.0 (transverse)
    and 0.3 (longitudinal) mm/sqrt(cm) (``Next100FieldCage``), electron lifetime 1000 ms
    (``Next100Vessel``). A width k*sqrt(L) after drifting L equals sqrt(2 D t) with t = L / v, so
    D = k^2 v / 2. Recombination, response models and space charge are not used.
    """
    v = velocity_mm_us / 10.0                            # cm/us
    k_t, k_l = diffusion_trans_mm_sqrtcm / 10.0, diffusion_long_mm_sqrtcm / 10.0   # cm/sqrt(cm)
    return SimParams(velocity_cm_us=jnp.asarray(v, jnp.float32),
                     lifetime_us=jnp.asarray(lifetime_ms * 1000.0, jnp.float32),
                     diffusion_trans_cm2_us=jnp.asarray(k_t ** 2 * v / 2.0, jnp.float32),
                     diffusion_long_cm2_us=jnp.asarray(k_l ** 2 * v / 2.0, jnp.float32),
                     recomb_params=None, response_models=None, sce_models=None)


class DriftResult(NamedTuple):
    """Charge transport from a point deposit to the gate."""
    distance_m: jnp.ndarray        # drift length
    time_us: jnp.ndarray           # drift time
    survival: jnp.ndarray          # fraction of electrons not attached, exp(-t / lifetime)
    sigma_trans_m: jnp.ndarray     # transverse spread of the cloud at the gate
    sigma_time_us: jnp.ndarray     # longitudinal spread, in arrival time


def drift_to_gate(deposit_m, gate_z_m, params):
    """Drift the ionization of a deposit at ``deposit_m`` (x, y, z; m, LUCiD frame, z towards the
    gate) to the gate plane at ``gate_z_m`` with JAXTPC."""
    deposit_m = jnp.asarray(deposit_m, jnp.float32)
    # JAXTPC drifts along x: put the drift axis first, in cm; electrons move towards +z.
    pos_cm = 100.0 * jnp.stack([deposit_m[2], deposit_m[0], deposit_m[1]])[None, :]
    dist_cm, t_us, _ = compute_drift_to_plane(pos_cm, 100.0 * gate_z_m, 1,
                                              params.velocity_cm_us, 0.0)
    t = t_us[0]
    survival = jnp.exp(-t / params.lifetime_us)
    # eps inside the sqrt keeps the gradient finite for a deposit at the gate
    sigma_t_cm = jnp.sqrt(2.0 * params.diffusion_trans_cm2_us * t + 1e-12)
    sigma_l_us = jnp.sqrt(2.0 * params.diffusion_long_cm2_us / params.velocity_cm_us ** 2 * t + 1e-12)
    return DriftResult(dist_cm[0] / 100.0, t, survival, sigma_t_cm / 100.0, sigma_l_us)


def electron_arrivals(deposit_m, drift, normals):
    """Arrival (x, y) at the gate of macro-electrons: the deposit's (x, y) plus the transverse
    spread times fixed standard normals ``normals`` (M, 2)."""
    return jnp.asarray(deposit_m, jnp.float32)[:2] + drift.sigma_trans_m * normals
