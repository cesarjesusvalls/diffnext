"""Where the linearised fixed point puts a closure fit, given its pseudo-data.

  shift = L (Q - k_true m)

L: data_noise_sets.py's map (both gauges: LUCiD's log gauge, and LeaveOneOutProblem's
light-weighted one), m: its 64-draw unit-gain model at the truth. +- what those 64 draws leave.
Pseudo-data from saved fits (their `data`), or generated here with closure_data's keys.

  python predict_shift.py data_noise_sets.npz [fit.npz ...] [--data-key 20000 --sample next_letters.npz]
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('noise_sets')
ap.add_argument('fits', nargs='*')
ap.add_argument('--data-key', type=int, nargs='*', default=[])
ap.add_argument('--sample')
args = ap.parse_args()
log = lambda *a: print(*a, flush=True)

ns = np.load(args.noise_sets)
names = [str(n) for n in ns['names']]
M = ns['M'].astype(np.float64)
m, var1 = M.mean(0), M.var(0, ddof=1)
th = ns['theta'].astype(np.float64)


def pct(i, d):
    f = (lambda t: 1.0 - np.exp(t)) if names[i] == 'wall reflectivity' else np.exp
    return 100 * (f(th[i] + d) / f(th[i]) - 1)


def report(label, Q, k, fit=None):
    dQ = Q - k[None] * m
    log(f"\n{label}")
    for gauge in ('log', 'light_weighted'):
        L = ns[f'{gauge}_Lcol']
        s = np.einsum('gnp,gn->p', L, dQ)
        sd = np.sqrt(np.einsum('gnp,gn,gnp->p', L, k[None] ** 2 * var1 / len(M), L))
        log(f"  {gauge.replace('_', '-')} gauge, predicted: "
            + ', '.join(f"{n} {pct(i, s[i]):+.3f}% +- {abs(pct(i, sd[i])):.3f}%" for i, n in enumerate(names)))
    if fit is not None:
        target = fit['real_target'] if 'real_target' in fit else fit['real_true']
        log("  fit (against its target):  "
            + ', '.join(f"{n} {100 * (fit['real_fit'][i] / target[i] - 1):+.3f}%"
                        for i, n in enumerate(names)))


for path in args.fits:
    d = np.load(path)
    assert [str(n) for n in d['names']] == names, path
    key = int(d['data_key']) if 'data_key' in d else 5000
    est = str(d['estimator']) if 'estimator' in d else 'profiled'
    report(f"{path} (estimator {est}, pseudo-data keys {key} + 131 b)",
           d['data'].astype(np.float64), d['k_true'].astype(np.float64), d)

if args.data_key:
    import jax.numpy as jnp
    from lucid.detector_params import _flatten_detector_params, _nest_flat_kwargs
    from lucid.fitting import CalibrationForward, closure_data
    from lucid.next.calibration import S2FitParams, deposit_groups, s2_calibration_sim
    from lucid.next.charge import xenon_drift_params
    from lucid.next.event import NextEventSimulator, default_params

    sample = np.load(args.sample)
    ev = NextEventSimulator(n_rays=1_000_000, batch=100_000, K=200, pos_grad_threshold=0, optics_input=True)
    NS = len(np.asarray(ev.detector.sensor_shape))
    truth = default_params()._replace(drift=xenon_drift_params(lifetime_ms=10.0))
    optics = _nest_flat_kwargs(_flatten_detector_params(ev.default_detector_params))
    par = S2FitParams(truth, names, optics, hold_dt_over_v=True)
    theta = jnp.asarray(par.theta_from_physical(), jnp.float32)
    assert np.allclose(np.asarray(theta), th, atol=1e-6)
    groups = deposit_groups(sample['deposits'], float(sample['energy_MeV']), sample['letter_of'])
    fwd = CalibrationForward(s2_calibration_sim(ev), groups, par, NS)
    log_k = 0.1 * np.random.default_rng(6000).standard_normal(NS)
    k_true = np.exp(log_k - log_k.mean())
    for key in args.data_key:
        Q = np.asarray(closure_data(fwd, theta, k_true, n_draws=8, key_base=key), np.float64)
        report(f"pseudo-data generated here, keys {key} + 131 b", Q, k_true)
