"""Gradient test: recover a point-source position from its PMT light with LUCiD's autodiff.

"Observed" light is LUCiD's own prediction for a source at a known position (noise-free, same
1M-ray lattice), so a working gradient must lead back to that position. Each iteration computes
the expected rays on the 60 PMTs and their Jacobian with respect to the source position by
forward-mode autodiff (three tangents per batch), and takes an Adam step on
chi2 = mean((pred - obs)^2 / obs).

  --check : compare the autodiff gradient of chi2 with central finite differences at the starts.
  --fit   : run the fits and save every iteration (position, chi2, gradient, PMT light).
"""
import argparse
import json
import os
import time
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from lucid.simulation import setup_event_simulator
from lucid.sources.calibration_sources import get_isotropic_rays

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')
GEOM, PHYS = f'{CFG}/NEXT100_like_geom_config.json', f'{CFG}/NEXT100_like_physics_config.json'
H2, GATE = 0.7414, 0.02413                       # half height; gate 24.13 mm in front of the SiPMs
z_at = lambda drift: H2 - GATE - drift           # drift distance from the gate -> LUCiD z (m)
CASES = [
    dict(name='A', true=[0.15, -0.20, z_at(0.6)], start=[-0.20, 0.15, z_at(0.3)]),
    dict(name='B', true=[-0.30, 0.25, z_at(1.0)], start=[0.10, -0.10, z_at(0.5)]),
]


class RaySlice(NamedTuple):
    directions: jnp.ndarray
    origins: jnp.ndarray
    weights: jnp.ndarray

    def __call__(self, n_photons, key, n_water=1.33):
        return self.directions, self.origins, self.weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['check', 'fit'], required=True)
    ap.add_argument('--K', type=int, default=200)
    ap.add_argument('--n-rays', type=int, default=1_000_000)
    ap.add_argument('--batch', type=int, default=50_000)
    ap.add_argument('--pos-grad-threshold', type=int, default=None,
                    help='LUCiD: steps whose landing positions keep their gradient (None = all K)')
    ap.add_argument('--phys', default=PHYS, help='physics config')
    ap.add_argument('--exclude-rim', action='store_true',
                    help='leave the PMTs that reach past the cylinder radius out of chi2')
    ap.add_argument('--n-grad-iters', type=int, default=None,
                    help='LUCiD: steps whose directions keep their gradient (None = all K)')
    ap.add_argument('--iters', type=int, default=80)
    ap.add_argument('--lr', type=float, nargs=2, default=[0.03, 0.002], help='Adam step, start and end (m)')
    ap.add_argument('--out', default='fit_point_source.npz')
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)

    sim = setup_event_simulator(GEOM, n_photons=args.batch, temperature=None, K=args.K,
                                is_calibration=True, physics_config=args.phys,
                                default_detector_params=True, wavelength_mode=False,
                                reflection_model='scalar', hit_mode='aggregated',
                                pos_grad_threshold=args.pos_grad_threshold,
                                n_grad_iters=args.n_grad_iters)
    log(f"K={args.K} pos_grad_threshold={args.pos_grad_threshold} n_grad_iters={args.n_grad_iters}")
    det = sim.det_geom.detector
    pmt = np.flatnonzero(np.asarray(det.sensor_shape) == 'disc')
    if args.exclude_rim:
        xy = np.asarray(sim.det_geom.sensor_points)[pmt, :2]
        pmt = pmt[np.hypot(xy[:, 0], xy[:, 1]) + np.asarray(det.sensor_half_size)[pmt] <= det.r]
    log(f"physics {args.phys}; {len(pmt)} PMTs in chi2")
    pmt_j = jnp.asarray(pmt)
    nb = args.n_rays // args.batch
    dirs, _, w = get_isotropic_rays(jnp.zeros(3), float(args.n_rays), args.n_rays, jax.random.PRNGKey(0))
    dirs = dirs.reshape(nb, args.batch, 3); w = w.reshape(nb, args.batch)
    keys = jax.random.split(jax.random.PRNGKey(7), nb)

    def light_b(pos, d, wb, key):
        q, _ = sim(RaySlice(d, jnp.broadcast_to(pos, d.shape), wb), key)
        return q[pmt_j]

    light_b_jit = jax.jit(light_b)

    @jax.jit
    def light_and_jac_b(pos, d, wb, key):
        f = lambda p: light_b(p, d, wb, key)
        q, jt = jax.vmap(lambda t: jax.jvp(f, (pos,), (t,)), out_axes=(None, 0))(jnp.eye(3))
        return q, jt.T

    def light(pos):
        pos = jnp.asarray(pos, jnp.float32)
        return sum(np.asarray(light_b_jit(pos, dirs[b], w[b], keys[b]), np.float64) for b in range(nb))

    def light_and_jac(pos):
        pos = jnp.asarray(pos, jnp.float32)
        q, J = 0.0, 0.0
        for b in range(nb):
            qb, Jb = light_and_jac_b(pos, dirs[b], w[b], keys[b])
            q = q + np.asarray(qb, np.float64); J = J + np.asarray(Jb, np.float64)
        return q, J

    def chi2(q, obs):
        return float(np.mean((q - obs) ** 2 / np.maximum(obs, 1.0)))

    def chi2_grad(q, J, obs):
        return 2.0 * np.mean(((q - obs) / np.maximum(obs, 1.0))[:, None] * J, axis=0)

    t0 = time.time(); _ = light(CASES[0]['true']); log(f"forward, {args.n_rays} rays, K={args.K}: {time.time()-t0:.1f} s (with compile)")
    t0 = time.time(); _ = light(CASES[0]['true']); log(f"forward: {time.time()-t0:.1f} s")
    t0 = time.time(); _ = light_and_jac(CASES[0]['true']); log(f"forward + Jacobian: {time.time()-t0:.1f} s (with compile)")
    t0 = time.time(); _ = light_and_jac(CASES[0]['true']); log(f"forward + Jacobian: {time.time()-t0:.1f} s")

    if args.mode == 'check':
        for c in CASES:
            obs = light(c['true'])
            for label, p in (('start', c['start']), ('5 cm from truth', np.array(c['true']) + [0.03, -0.03, 0.02])):
                p = np.asarray(p, np.float64)
                q, J = light_and_jac(p)
                g_ad = chi2_grad(q, J, obs)
                line = f"case {c['name']} at {label}: chi2 {chi2(q, obs):10.2f}  AD grad {np.round(g_ad, 1)}"
                for h in (0.002, 0.01):
                    g_fd = np.array([(chi2(light(p + h * e), obs) - chi2(light(p - h * e), obs)) / (2 * h)
                                     for e in np.eye(3)])
                    cos = g_ad @ g_fd / (np.linalg.norm(g_ad) * np.linalg.norm(g_fd))
                    line += f" | FD h={h*1000:.0f} mm {np.round(g_fd, 1)} cos {cos:.4f}"
                log(line)
        return

    out = dict(pmt_xy=np.asarray(sim.det_geom.sensor_points)[pmt, :2], radius=det.r, height=det.H,
               K=args.K, n_rays=args.n_rays, lr=np.array(args.lr), iters=args.iters)
    for c in CASES:
        obs = light(c['true'])
        pos = np.array(c['start'], np.float64)
        m = np.zeros(3); v = np.zeros(3); b1, b2, eps = 0.9, 0.999, 1e-12
        hist = dict(pos=[], chi2=[], grad=[], light=[])
        for it in range(args.iters + 1):
            q, J = light_and_jac(pos)
            g = chi2_grad(q, J, obs)
            hist['pos'].append(pos.copy()); hist['chi2'].append(chi2(q, obs)); hist['grad'].append(g); hist['light'].append(q)
            dist = np.linalg.norm(pos - c['true']) * 1000
            log(f"case {c['name']} it {it:3d}  pos ({pos[0]*1000:7.1f}, {pos[1]*1000:7.1f}, {pos[2]*1000:7.1f}) mm  "
                f"chi2/ndf {hist['chi2'][-1]:12.3f}  |pos - true| {dist:7.1f} mm")
            if it == args.iters:
                break
            lr = args.lr[0] * (args.lr[1] / args.lr[0]) ** (it / max(args.iters - 1, 1))
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** (it + 1)); vh = v / (1 - b2 ** (it + 1))
            pos = pos - lr * mh / (np.sqrt(vh) + eps)
        for k, val in hist.items():
            out[f"{c['name']}_{k}"] = np.array(val)
        out[f"{c['name']}_true"] = np.array(c['true']); out[f"{c['name']}_obs"] = obs
        np.savez(args.out, **out)
    log(f"wrote {args.out}")


if __name__ == '__main__':
    main()
