"""The fixed-point condition at the truth, for LUCiD's charge term and for LeaveOneOutProblem.

The fit's step is A^-1 g with g = J^T r. At the truth (pseudo-data from the truth parameters and
gains), an estimator whose expected g is not zero settles away from the truth, by about
-H^-1 E[g]. This measures E[g] at the truth over independent replicas of the forward (the fit's
own draws: step j's keys), as tests/test_fitting_estimator_unbiased.py does on a toy, for

  profiled      LUCiD's CalibrationProblem: gains profiled on the replica's 4 draws (log gauge)
  profiled 16   the same, 4 replicas averaged into one 16-draw forward
  leave-one-out LeaveOneOutProblem on the same 4 draws

with one 8-draw Jacobian at the truth shared by all, and the shift -H^-1 E[g] it implies, with H
the charge metric plus w_time times the waveform's (w_time: the weight the fit used, saved as its
`w_time`). Configuration: the joint fit's second round (v, lifetime, D_L, EL gain, R free; D_T / v
held; no absorption; 10% gains).

  python fixed_point_check.py next_letters.npz out.npz --w-time 0.00707 [--replicas 16]
"""
import argparse
import time

import numpy as np
import jax.numpy as jnp

from lucid.detector_params import _flatten_detector_params, _nest_flat_kwargs
from lucid.fitting import (CalibrationForward, CalibrationJacobian, closure_data, neyman_residual,
                           profile_gains)
from lucid.fitting.calibrate import _seeded_keys
from lucid.next.calibration import (LeaveOneOutProblem, S2FitParams, S2WaveformTerm, deposit_groups,
                                    s2_calibration_sim)
from lucid.next.charge import xenon_drift_params
from lucid.next.event import NextEventSimulator, default_params
from lucid.next.waveform import EL_DRIFT_VELOCITY_MM_US

ap = argparse.ArgumentParser()
ap.add_argument('sample')
ap.add_argument('out')
ap.add_argument('--w-time', type=float, required=True, help='the waveform weight, a fit\'s w_time')
ap.add_argument('--data-key', type=int, default=5000, help='the pseudo-data\'s key base')
ap.add_argument('--replicas', type=int, default=16)
ap.add_argument('--draws', type=int, default=4)
ap.add_argument('--jacobian-draws', type=int, default=8)
args = ap.parse_args()
log = lambda *a: print(*a, flush=True)

sample = np.load(args.sample)
ev = NextEventSimulator(n_rays=1_000_000, batch=100_000, K=200, pos_grad_threshold=0, optics_input=True)
NS = len(np.asarray(ev.detector.sensor_shape))
sipm = np.asarray(ev.detector.sensor_shape).astype(str) == 'square'
truth = default_params()._replace(drift=xenon_drift_params(lifetime_ms=10.0))
optics = _nest_flat_kwargs(_flatten_detector_params(ev.default_detector_params))
free = ['drift velocity', 'electron lifetime', 'longitudinal diffusion', 'EL gain', 'wall reflectivity']
par = S2FitParams(truth, free, optics, hold_dt_over_v=True)
th = jnp.asarray(par.theta_from_physical(), jnp.float32)
groups = deposit_groups(sample['deposits'], float(sample['energy_MeV']), sample['letter_of'])
sim = s2_calibration_sim(ev)
log_k = 0.1 * np.random.default_rng(6000).standard_normal(NS)
k_true = np.exp(log_k - log_k.mean())
fkey, jkey = _seeded_keys(0)
fwd = CalibrationForward(sim, groups, par, NS)
t0 = time.time()
data = closure_data(fwd, th, k_true, n_draws=8, key_base=args.data_key)
G = data.shape[0]
q_floor = 0.01 * float(jnp.mean(data))
w = jnp.sqrt(jnp.clip(data, q_floor, None))
log(f"pseudo-data {time.time() - t0:.0f} s")

loo = LeaveOneOutProblem(fwd, CalibrationJacobian(sim, groups, par, NS, key0=jkey), par, data, q_floor,
                         n_forward_draws=args.draws, jacobian_draws=args.jacobian_draws, forward_key0=fkey)
t0 = time.time()
J = loo.jacobian(th, jnp.zeros(NS), 0, data, q_floor, draws=range(args.jacobian_draws))   # dM/dtheta / sqrt(Q)
log(f"Jacobian, {args.jacobian_draws} draws: {time.time() - t0:.0f} s")

# The waveform term at the truth: zero gradient, curvature on v and D_L.
edges = np.arange(0.0, 1600.5, 1.0)
gap_us = (ev.el_gap_z[1] - ev.el_gap_z[0]) * 1000.0 / EL_DRIFT_VELOCITY_MM_US
timing = S2WaveformTerm(par, groups, data, ev.gate_z, edges, gap_us)
timing.observed = timing.predict(th)
_, Ht, _ = timing.grad_metric_loss(th)
Ht = np.asarray(Ht, np.float64)

rows = {'profiled': [], 'profiled, lk from the previous replica': [], 'leave-one-out': []}
M16, g16, k_prev = [], [], None
t0 = time.time()
for j in range(args.replicas):
    M = loo._draws(th, j)                                       # the fit's draws at step j
    mu = M.mean(0)
    k = profile_gains(mu.sum(0) + 1e-12, loo.data_sum, gauge='log')
    r = neyman_residual(k[None] * mu, data, q_floor)
    rows['profiled'].append(np.asarray(jnp.einsum('n,gnp,gn->p', k, J, r) / G))
    if k_prev is not None:
        rows['profiled, lk from the previous replica'].append(np.asarray(jnp.einsum('n,gnp,gn->p', k_prev, J, r) / G))
    k_prev = k
    Mn, m, kl = loo._gains(M)
    rl = (kl[:, None] * M - data[None] * (Mn / m)[:, None]) / w[None]
    rows['leave-one-out'].append(np.asarray(jnp.einsum('bn,gnp,bgn->p', kl, J, rl) / (G * M.shape[0])))
    M16.append(mu)
    if len(M16) == 4:
        mu16 = sum(M16) / 4
        k16 = profile_gains(mu16.sum(0) + 1e-12, loo.data_sum, gauge='log')
        r16 = neyman_residual(k16[None] * mu16, data, q_floor)
        g16.append(np.asarray(jnp.einsum('n,gnp,gn->p', k16, J, r16) / G))
        M16 = []
    log(f"replica {j + 1}/{args.replicas}: {time.time() - t0:.0f} s")
rows['profiled 16'] = g16

w_time = args.w_time
Hc =np.asarray(jnp.einsum('gnp,gnq,n->pq', J, J, jnp.asarray(k_true) ** 2) / G, np.float64)
H = Hc + w_time * Ht
Hinv = np.linalg.inv(H)
res = {}
log(f"\nE[g] at the truth over replicas (z = mean / s.e.m.), and the shift -H^-1 E[g] it implies "
    f"(H: charge + {w_time:g} x waveform), as fitted / truth - 1:")
for label, gs in rows.items():
    gs = np.asarray(gs, np.float64)
    n = len(gs)
    mean, sem = gs.mean(0), gs.std(0, ddof=1) / np.sqrt(n)
    shift = -Hinv @ mean
    cov = Hinv @ (np.cov(gs.T, ddof=1) / n) @ Hinv
    err = np.sqrt(np.diag(cov))
    res[label] = dict(g=gs, mean=mean, sem=sem, shift=shift, shift_err=err)
    log(f"  {label} ({n} samples)")
    for i, name in enumerate(par.free):
        f = (lambda t: 1.0 - np.exp(t)) if name == 'wall reflectivity' else np.exp
        t = float(th[i])
        frac = f(t + shift[i]) / f(t) - 1
        frac_err = abs(f(t + shift[i] + err[i]) / f(t) - 1 - frac)
        log(f"    {name:23s} z {mean[i] / sem[i]:+6.2f}   shift {frac:+8.3%} +- {frac_err:.3%}")
np.savez(args.out, names=np.array(par.free), theta=np.asarray(th), Hc=Hc, Ht=Ht, w_time=w_time,
         **{f"{k.replace(' ', '_').replace(',', '')}_{q}": v for k, r in res.items() for q, v in r.items()})
log(f"wrote {args.out}")
