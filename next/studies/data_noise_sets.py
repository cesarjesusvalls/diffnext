"""How far does the pseudo-data's own Monte-Carlo noise move the fit, and did it move it there?

To first order only: fit C (see predict_shift.py) does not end where this map puts it.

The pseudo-data are 8 forward draws at the truth, so they differ from the noise-free model by dQ.
To first order the fit's fixed point (Gauss-Newton with LUCiD's Jacobian, gains profiled, plus the
noise-free waveform term) moves by d theta = L dQ, with

  r = (k_n m[g, n] - Q[g, n]) / w,   k_n = c Q_n / m_n,   w = sqrt(clip(Q, q_floor))
  L = -(J_L^T J_full / G + w_time H_t)^-1 J_L^T (dr/dQ) / G

J_L: LUCiD's Jacobian (gains held), J_full: the residual's full derivative (the profiled gains and
the gauge move with theta). Both gauges: LUCiD's mean(log k) = 0 and LeaveOneOutProblem's
light-weighted one (c = sum m / sum Q); on a noise-free model the two problems differ only there.

1. The statistical error of the closure: Cov(d theta) = L diag(Var Q) L^T, Var Q from single draws.
2. The shift this pseudo-data realised: L dQ_hat, dQ_hat = Q - k_true m_hat, m_hat from n_model
   draws (its own noise adds L diag(k^2 var / n_model) L^T to the spread of that estimate).

3. Each disjoint set of 8 model draws as another pseudo-data set: its shift
   L (Q_j - k m_hat(other draws)) shows the spread a fresh pseudo-data draw would have, without the
   diagonal-variance approximation.

Saves the map (`<gauge>_Lcol`) and the model draws (`M`) for key_check.py, key_families.py,
key_split_check.py and predict_shift.py.

  python data_noise_sets.py next_letters.npz out.npz --w-time 0.00707 --model-draws 64
"""
import argparse
import time

import numpy as np
import jax.numpy as jnp

from lucid.detector_params import _flatten_detector_params, _nest_flat_kwargs
from lucid.fitting import CalibrationForward, CalibrationJacobian, closure_data
from lucid.fitting.calibrate import _seeded_keys
from lucid.next.calibration import S2FitParams, S2WaveformTerm, deposit_groups, s2_calibration_sim
from lucid.next.charge import xenon_drift_params
from lucid.next.event import NextEventSimulator, default_params
from lucid.next.waveform import EL_DRIFT_VELOCITY_MM_US

ap = argparse.ArgumentParser()
ap.add_argument('sample')
ap.add_argument('out')
ap.add_argument('--w-time', type=float, required=True, help='the waveform weight, a fit\'s w_time')
ap.add_argument('--data-key', type=int, default=5000, help='the pseudo-data\'s key base')
ap.add_argument('--model-draws', type=int, default=48)
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
ones = jnp.ones(NS)
t0 = time.time()
data = closure_data(fwd, th, k_true, n_draws=8, key_base=args.data_key)          # the fits' pseudo-data
G = data.shape[0]
q_floor = 0.01 * float(jnp.mean(data))
log(f"pseudo-data {time.time() - t0:.0f} s")
t0 = time.time()
J_unit = np.asarray(CalibrationJacobian(sim, groups, par, NS, key0=jkey)(
    th, jnp.zeros(NS), 0, data, q_floor, draws=range(args.jacobian_draws)), np.float64)
log(f"Jacobian, {args.jacobian_draws} draws: {time.time() - t0:.0f} s")
t0 = time.time()
M = np.stack([np.asarray(fwd(th, 60_000 + 131 * b, ones), np.float64) for b in range(args.model_draws)])
log(f"{args.model_draws} model draws: {time.time() - t0:.0f} s")

edges = np.arange(0.0, 1600.5, 1.0)
gap_us = (ev.el_gap_z[1] - ev.el_gap_z[0]) * 1000.0 / EL_DRIFT_VELOCITY_MM_US
timing = S2WaveformTerm(par, groups, data, ev.gate_z, edges, gap_us)
timing.observed = timing.predict(th)
Ht = np.asarray(timing.grad_metric_loss(th)[1], np.float64)

Q = np.asarray(data, np.float64)
m_hat = M.mean(0)                                         # unit-gain model at the truth
var1 = M.var(0, ddof=1)                                   # one draw's variance, unit gain
var_Q = k_true[None] ** 2 * var1 / 8                      # the pseudo-data's
dQ_hat = Q - k_true[None] * m_hat
w = np.sqrt(np.clip(Q, q_floor, None))
P = len(free)

m_n, Q_n = m_hat.sum(0), Q.sum(0)
f = m_hat / m_n
k = Q_n / m_n                                             # c = 1 at the truth
D = J_unit * w[..., None]                                 # dm / dtheta, unit gain
D_n = D.sum(0)
J_L = k[None, :, None] * D / w[..., None]
res = {}
for gauge in ('log', 'light-weighted'):
    # gauge: d log c / d theta, and d log c / dQ_n
    if gauge == 'log':
        dlogc_dth = (D_n / m_n[:, None]).mean(0)
        a = 1.0 / (NS * Q_n)
    else:
        dlogc_dth = D_n.sum(0) / m_n.sum()
        a = np.full(NS, 1.0 / Q_n.sum())
    J_full = k[None, :, None] * (D - f[..., None] * D_n[None] + m_hat[..., None] * dlogc_dth) / w[..., None]
    H = J_L.reshape(-1, P).T @ J_full.reshape(-1, P) / G + args.w_time * Ht
    Hinv = np.linalg.inv(H)
    # dr[g, n] = (f[g, n] dQ_n - dQ[g, n]) / w[g, n] - (Q_n f[g, n] / w[g, n]) sum_m a_m dQ_m
    # L dQ = -Hinv J_L^T dr / G, column by column in closed form:
    u = np.einsum('gnp,gn->np', J_L, f / w)                # sum_g J_L f / w, per sensor
    v = np.einsum('gnp,gn->p', J_L, Q_n[None] * f / w)     # gauge term
    Lcol = (-J_L / w[..., None] + u[None] - a[None, :, None] * v[None, None]) / G   # (G, NS, P): d(J_L^T dr / G) / dQ
    Lcol = -np.einsum('pq,gnq->gnp', Hinv, Lcol)
    cov = np.einsum('gnp,gn,gnq->pq', Lcol, var_Q, Lcol)
    cov_hat = np.einsum('gnp,gn,gnq->pq', Lcol, k_true[None] ** 2 * var1 / args.model_draws, Lcol)
    shift = np.einsum('gnp,gn->p', Lcol, dQ_hat)
    n_sets = args.model_draws // 8
    set_shifts = []
    for j in range(n_sets):
        rest = np.r_[M[:8 * j], M[8 * (j + 1):]].mean(0)
        set_shifts.append(np.einsum('gnp,gn->p', Lcol, k_true[None] * (M[8 * j:8 * (j + 1)].mean(0) - rest)))
    set_shifts = np.array(set_shifts)
    res[gauge] = dict(cov=cov, shift=shift, cov_hat=cov_hat, H=H, Lcol=Lcol, set_shifts=set_shifts)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    log(f"\n{gauge} gauge: statistical error from the 8-draw pseudo-data, and this pseudo-data's shift "
        f"(+- what {args.model_draws} model draws leave)")
    for i, n in enumerate(free):
        fn = (lambda t: 1.0 - np.exp(t)) if n == 'wall reflectivity' else np.exp
        t = float(th[i])
        pct = lambda d: 100 * (fn(t + d) / fn(t) - 1)
        log(f"  {n:23s} sigma {abs(pct(sd[i])):7.3f}%   shift {pct(shift[i]):+7.3f}% +- "
            f"{abs(pct(np.sqrt(cov_hat[i, i]))):.3f}%")
    for i, n in enumerate(free):
        if n in ('electron lifetime', 'EL gain', 'wall reflectivity'):
            fn = (lambda t: 1.0 - np.exp(t)) if n == 'wall reflectivity' else np.exp
            t = float(th[i])
            pct = 100 * (fn(t + set_shifts[:, i]) / fn(t) - 1)
            log(f"  {n:23s} {n_sets} fresh pseudo-data sets: shifts " + ' '.join(f'{x:+.3f}' for x in pct)
                + f"%  (sd {pct.std(ddof=1):.3f}%)")
    iG, iR = free.index('EL gain'), free.index('wall reflectivity')
    log(f"  correlation EL gain - log(1 - R): {corr[iG, iR]:+.3f}; lifetime - log(1 - R): "
        f"{corr[free.index('electron lifetime'), iR]:+.3f}")
np.savez(args.out, names=np.array(free), theta=np.asarray(th), var1=var1, m_hat=m_hat, dQ_hat=dQ_hat, Ht=Ht, M=M,
         **{f"{g.replace('-', '_')}_{q}": x for g, r in res.items() for q, x in r.items()})
log(f"wrote {args.out}")
