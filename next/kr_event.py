"""S1 and S2 of a point deposit (41.5 keV, the Kr-83m line) at three drift distances in the
NEXT-like detector, with the integrated JAXTPC-charge + LUCiD-light simulation, and checks that
gradients flow from the sensors back to the deposit and the physics parameters.

  HOLD_JOBID=<job> JAXTPC_PATH=<JAXTPC checkout> next/run_gpu_s3df.sh next/kr_event.py --out kr.npz
"""
import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

from lucid.next.event import NextEventSimulator, default_params

ap = argparse.ArgumentParser()
ap.add_argument('--energy-MeV', type=float, default=0.0415)
ap.add_argument('--xy-m', type=float, nargs=2, default=[0.10, -0.05])
ap.add_argument('--drift-m', type=float, nargs='+', default=[0.1, 0.6, 1.1])
ap.add_argument('--n-rays', type=int, default=1_000_000)
ap.add_argument('--K', type=int, default=200)
ap.add_argument('--out', required=True)
args = ap.parse_args()
log = lambda *a: print(*a, flush=True)

t0 = time.time()
ev = NextEventSimulator(n_rays=args.n_rays, K=args.K)
p = default_params()
det = ev.detector
shape = np.asarray(det.sensor_shape).astype(str)
sipm, pmt = shape == 'square', shape == 'disc'
log(f"simulator built in {time.time() - t0:.1f} s: {sipm.sum()} SiPMs, {pmt.sum()} PMTs; gate z {ev.gate_z:.4f} m, "
    f"EL gap z {ev.el_gap_z}; {args.n_rays} rays per signal, K={args.K}")

out = dict(energy_MeV=args.energy_MeV, sensor_xyz=np.asarray(ev.sim.det_geom.sensor_points),
           sensor_shape=shape, sensor_half_size=np.asarray(det.sensor_half_size), sensor_id=np.asarray(det.pmt_id),
           radius=det.r, height=det.H, gate_z=ev.gate_z, el_gap_z=np.array(ev.el_gap_z),
           cathode_z=float(det.cathode_z), drift_m=np.array(args.drift_m),
           w_ion_eV=float(p.w_ion_eV), s1_yield=float(p.s1_yield_per_MeV), el_gain=float(p.el_gain),
           velocity_mm_us=float(p.drift.velocity_cm_us) * 10, lifetime_us=float(p.drift.lifetime_us),
           D_T=float(p.drift.diffusion_trans_cm2_us), D_L=float(p.drift.diffusion_long_cm2_us))
rows = {k: [] for k in ('deposit', 's1', 's2', 'xy_e', 'drift_time_us', 'survival', 'sigma_trans_m',
                        'sigma_time_us', 'n_electrons', 'n_s2_photons')}
for d in args.drift_m:
    dep = np.array([args.xy_m[0], args.xy_m[1], ev.gate_z - d])
    t1 = time.time()
    s1 = np.asarray(ev.s1(dep, args.energy_MeV, p), np.float64)
    s2, dr, xy, ne, nph = ev.s2(dep, args.energy_MeV, p)
    s2 = np.asarray(s2, np.float64)
    for k, v in (('deposit', dep), ('s1', s1), ('s2', s2), ('xy_e', np.asarray(xy)),
                 ('drift_time_us', float(dr.time_us)), ('survival', float(dr.survival)),
                 ('sigma_trans_m', float(dr.sigma_trans_m)), ('sigma_time_us', float(dr.sigma_time_us)),
                 ('n_electrons', float(ne)), ('n_s2_photons', float(nph))):
        rows[k].append(v)
    log(f"drift {d:.2f} m: t {float(dr.time_us):7.1f} us, sigma_T {float(dr.sigma_trans_m)*1e3:5.2f} mm, "
        f"sigma_t {float(dr.sigma_time_us):5.2f} us, {float(ne):7.1f} e-, {float(nph):10.0f} S2 photons | "
        f"S1 on SiPMs {s1[sipm].sum():7.2f} PMTs {s1[pmt].sum():8.2f} | S2 on SiPMs {s2[sipm].sum():9.0f} "
        f"PMTs {s2[pmt].sum():10.0f} | {time.time() - t1:.1f} s")
for k, v in rows.items():
    out[k] = np.array(v)
np.savez(args.out, **out)
log(f"wrote {args.out}")

# ---- gradients, at the middle drift distance: exact where the answer is known ----
dep = jnp.asarray([args.xy_m[0], args.xy_m[1], ev.gate_z - args.drift_m[len(args.drift_m) // 2]], jnp.float32)
E = jnp.asarray(args.energy_MeV, jnp.float32)
sipm_j, pmt_j = jnp.asarray(sipm), jnp.asarray(pmt)


def tangent(field, sub=None):
    t = jax.tree_util.tree_map(jnp.zeros_like, p)
    if sub is None:
        return t._replace(**{field: jnp.ones_like(getattr(p, field))})
    return t._replace(drift=t.drift._replace(**{sub: jnp.ones_like(getattr(p.drift, sub))}))


s2_sum = lambda q: jnp.sum(jnp.where(sipm_j, q, 0.0))
f_s2 = lambda pp: s2_sum(ev.s2(dep, E, pp)[0])
f_s1 = lambda e: jnp.sum(ev.s1(dep, e, p))
S2, dS2_gain = jax.jvp(f_s2, (p,), (tangent('el_gain'),))
log(f"dS2/d(el_gain) = {float(dS2_gain):.6g}; S2/el_gain = {float(S2 / p.el_gain):.6g}")
S1, dS1_E = jax.jvp(f_s1, (E,), (jnp.ones_like(E),))
log(f"dS1/dE = {float(dS1_E):.6g}; S1/E = {float(S1 / E):.6g}")
_, dS2_tau = jax.jvp(f_s2, (p,), (tangent('drift', 'lifetime_us'),))
dr = ev.s2(dep, E, p)[1]
log(f"dS2/d(lifetime) = {float(dS2_tau):.6g}; S2 t/tau^2 = {float(S2 * dr.time_us / p.drift.lifetime_us ** 2):.6g}")
_, dS2_dx = jax.jvp(lambda d_: s2_sum(ev.s2(d_, E, p)[0]), (dep,), (jnp.array([1.0, 0.0, 0.0]),))
_, dS2_DT = jax.jvp(f_s2, (p,), (tangent('drift', 'diffusion_trans_cm2_us'),))
log(f"dS2/dx_deposit = {float(dS2_dx):.6g} (finite: {bool(jnp.isfinite(dS2_dx))}); "
    f"dS2/dD_T = {float(dS2_DT):.6g} (finite: {bool(jnp.isfinite(dS2_DT))})")
