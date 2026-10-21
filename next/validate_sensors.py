"""Validate the NEXT-100-like geometry with SiPMs and PMTs: build-time coverage check, hand-aimed
rays, and LUCiD's direct light (K=1, 1M lattice rays) against exact solid angles."""
import os
import time
import numpy as np
import jax, jax.numpy as jnp
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.simulation import setup_event_simulator
from lucid.sources.calibration_sources import isotropic_source

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config') + '/'
GEOM, PHYS = CFG + 'NEXT100_like_geom_config.json', CFG + 'NEXT100_like_physics_config.json'
t0 = time.time()
g = DetectorGeometry.from_config(GEOM, temperature=None)       # runs the coverage check
det = g.detector; R, H = det.r, det.H
pts = np.asarray(g.sensor_points)
half = np.asarray(det.sensor_half_size); disc = np.asarray(det.sensor_shape) == 'disc'
print(f"built in {time.time()-t0:.1f} s: {g.num_sensors} sensors ({(~disc).sum()} SiPMs top, {disc.sum()} PMTs bottom), "
      f"R={R} H={H}, grid n_cap={det._n_cap}")

def fire(target, off, tan, sign, label):
    land = target + np.array([off[0], off[1], 0.0])
    d = np.array([tan[0], tan[1], sign], dtype=float); d /= np.linalg.norm(d)
    o = land - d * (0.1 / abs(d[2]))
    r = g.propagator(jnp.asarray(o[None], jnp.float32), jnp.asarray(d[None], jnp.float32))
    w = np.asarray(r['sensor_weights'])[:, 0]; idx = np.asarray(r['sensor_indices'])[:, 0]
    print(f"  {label:34s} hit {[int(i) for i, x in zip(idx, w) if x > 0]}")

s0 = int(np.argmin(np.where(disc, 1e9, np.hypot(pts[:, 0] - 0.0115, pts[:, 1] - 0.0141))))
p0 = int(np.argmin(np.where(disc, np.hypot(pts[:, 0], pts[:, 1]), 1e9)))
print(f"SiPM {s0} at {pts[s0]}, PMT {p0} at {pts[p0]}")
fire(pts[s0], (0.0006, 0), (0.8, 0.3), 1, 'SiPM, 0.6 mm off, oblique')
fire(pts[s0], (0.0007, 0), (0.8, 0.3), 1, 'SiPM, 0.7 mm off, oblique')
fire(pts[p0], (0, 0), (0, 0), -1, 'PMT, centre')
fire(pts[p0], (0.039, 0), (0.5, -0.4), -1, 'PMT, 39 mm off, oblique')
fire(pts[p0], (0.0, -0.041), (0.5, -0.4), -1, 'PMT, 41 mm off, oblique')
fire(pts[p0], (0.028, 0.028), (0, 0), -1, 'PMT, (28, 28) mm: inside disc')
fire(pts[p0], (0.0285, 0.0285), (0, 0), -1, 'PMT, (28.5, 28.5) mm: outside disc')

def omega_rect(x1, x2, y1, y2, h):
    F = lambda x, y: np.arctan(x * y / (h * np.sqrt(x * x + y * y + h * h)))
    return F(x2, y2) - F(x1, y2) - F(x2, y1) + F(x1, y1)

def omega_discs(src, h, step=2e-4):
    """Solid angle of each PMT disc, clipped to the cylinder, by midpoint integration."""
    out = np.zeros(disc.sum())
    g1 = np.arange(-0.04 + step / 2, 0.04, step)
    gx, gy = np.meshgrid(g1, g1, indexing='ij'); gx, gy = gx.ravel(), gy.ravel()
    inside_disc = gx ** 2 + gy ** 2 < 0.04 ** 2
    for k, c in enumerate(pts[disc]):
        x, y = c[0] + gx, c[1] + gy
        m = inside_disc & (x ** 2 + y ** 2 < R ** 2)
        rho2 = (x[m] - src[0]) ** 2 + (y[m] - src[1]) ** 2
        out[k] = np.sum(h / (h * h + rho2) ** 1.5) * step * step
    return out

N = 1_000_000
sim = setup_event_simulator(GEOM, n_photons=N, temperature=None, K=1, is_calibration=True,
                            physics_config=PHYS, default_detector_params=True,
                            wavelength_mode=False, reflection_model='scalar', hit_mode='aggregated')
s = 0.65e-3
for name, src in [('EL gap, on axis', [0.0037, 0.0064, H / 2 - 0.019]),
                  ('detector centre', [0.0, 0.0, 0.0]),
                  ('drift, x=0.3 m', [0.30, 0.0, H / 2 - 0.40])]:
    q, _ = sim(isotropic_source(src, intensity=N), jax.random.PRNGKey(0))
    q = np.asarray(q, dtype=np.float64)
    h_top = H / 2 - src[2]; rel = pts[~disc, :2] - np.array(src[:2])
    ex_sipm = N / (4 * np.pi) * omega_rect(rel[:, 0] - s, rel[:, 0] + s, rel[:, 1] - s, rel[:, 1] + s, h_top)
    ex_pmt = N / (4 * np.pi) * omega_discs(src, src[2] + H / 2)
    qs, qp = q[~disc], q[disc]
    print(f"{name:16s}: SiPMs LUCiD {qs.sum():8.1f} exact {ex_sipm.sum():8.1f} ({(qs.sum()-ex_sipm.sum())/np.sqrt(ex_sipm.sum()):+.2f} sigma) | "
          f"PMTs LUCiD {qp.sum():8.1f} exact {ex_pmt.sum():8.1f} ({(qp.sum()-ex_pmt.sum())/np.sqrt(ex_pmt.sum()):+.2f} sigma); "
          f"worst PMT |diff|/sqrt(exact) {np.max(np.abs(qp-ex_pmt)/np.sqrt(np.maximum(ex_pmt,1))):.2f}")
