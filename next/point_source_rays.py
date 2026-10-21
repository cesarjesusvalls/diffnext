"""Expected rays per SiPM from an isotropic point source in the NEXT-100-like LUCiD detector.

For every source position, emits ``--n-rays`` rays on LUCiD's Fibonacci lattice
(``get_isotropic_rays``, weight 1 per ray) and runs LUCiD's calibration-mode simulator twice:

* direct : K = 1, the first leg only (no reflection; no scattering when scattering is on);
* total  : K = ``--k-total`` legs, every reflection order up to K.

QE = 1 and SiPM reflectivity = 0 in the physics config, so the per-sensor "charge" is the
expected number of rays that reach each SiPM. The 1M rays are fed in batches of ``--batch``
(same lattice, split), to bound the (K, candidates, rays) scan buffers on the GPU.
"""
import argparse
import json
import os
import sys
import time
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from lucid.simulation import setup_event_simulator
from lucid.sources.calibration_sources import get_isotropic_rays

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')


class RaySlice(NamedTuple):
    """A fixed set of rays, callable like LUCiD's calibration sources."""
    directions: jnp.ndarray
    origins: jnp.ndarray
    weights: jnp.ndarray

    def __call__(self, n_photons, key, n_water=1.33):
        return self.directions, self.origins, self.weights


def build_sim(geom, phys, batch, K):
    return setup_event_simulator(
        geom, n_photons=batch, temperature=None, K=K, is_calibration=True,
        physics_config=phys, default_detector_params=True, wavelength_mode=False,
        reflection_model='scalar', hit_mode='aggregated')


def expected_rays(sim, pos, n_rays, batch, key):
    dirs, origins, w = get_isotropic_rays(jnp.asarray(pos, jnp.float32), float(n_rays), n_rays, key)
    total = None
    for b in range(n_rays // batch):
        sl = slice(b * batch, (b + 1) * batch)
        q, _ = sim(RaySlice(dirs[sl], origins[sl], w[sl]), jax.random.fold_in(key, b))
        q = np.asarray(q, dtype=np.float64)
        total = q if total is None else total + q
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geom', default='NEXT100_like')
    ap.add_argument('--phys', default='NEXT100_like')
    ap.add_argument('--positions', required=True, help='JSON file: list of {"name", "xyz_m"}')
    ap.add_argument('--n-rays', type=int, default=1_000_000)
    ap.add_argument('--batch', type=int, default=100_000)
    ap.add_argument('--k-total', type=int, nargs='+', default=[200])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-direct', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    assert args.n_rays % args.batch == 0

    geom = os.path.join(CFG, f'{args.geom}_geom_config.json')
    phys = os.path.join(CFG, f'{args.phys}_physics_config.json')
    positions = json.load(open(args.positions))
    xyz = np.array([p['xyz_m'] for p in positions], dtype=np.float64)
    key = jax.random.PRNGKey(args.seed)
    log = lambda *a: print(*a, flush=True)
    log(f"jax devices {jax.devices()}; geom {geom}; phys {phys}; {len(xyz)} positions; "
        f"n_rays {args.n_rays} batch {args.batch} K {args.k_total}")

    out = dict(xyz=xyz, names=np.array([p['name'] for p in positions]),
               n_rays=args.n_rays, geom=args.geom, phys=args.phys, seed=args.seed)

    runs = ([] if args.no_direct else [('direct', 1)]) + [(f'total_K{k}', k) for k in args.k_total]
    sim0 = None
    for label, K in runs:
        t0 = time.time()
        sim = build_sim(geom, phys, args.batch, K)
        sim0 = sim0 or sim
        log(f"[{label}] simulator built in {time.time() - t0:.1f} s")
        q_all = []
        for i, p in enumerate(xyz):
            t1 = time.time()
            q = expected_rays(sim, p, args.n_rays, args.batch, jax.random.fold_in(key, 1000 * i + K))
            q_all.append(q)
            log(f"[{label}] {positions[i]['name']:>16s}  sum={q.sum():12.3f}  max={q.max():10.3f}  "
                f"nonzero={int((q > 0).sum()):5d}  {time.time() - t1:6.1f} s")
        out[label] = np.array(q_all)
        np.savez(args.out, **out)

    det = sim0.det_geom.detector
    out['sensor_xyz'] = np.asarray(sim0.det_geom.sensor_points, dtype=np.float64)
    out['sensor_id'] = np.asarray(det.pmt_id)
    if hasattr(det, 'sensor_shape'):
        out['sensor_shape'] = np.asarray(det.sensor_shape).astype(str)
        out['sensor_half_size'] = np.asarray(det.sensor_half_size, dtype=np.float64)
    out['radius'] = det.r
    out['height'] = det.H
    np.savez(args.out, **out)
    log(f"wrote {args.out}")


if __name__ == '__main__':
    main()
