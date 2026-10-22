"""Is the joint fit's end point a minimum of its own (noisy) charge loss, lower than the truth's?

The fit's estimator (profiled gains, log gauge, Neyman residual, 4-draw forward average) at a few
points, on common random numbers: 4 replicas of 4 draws each. Also the same with all 16 draws
averaged, where a bias from the model's Monte-Carlo variance would shrink.

  python loss_compare.py joint_fit.npz next_letters.npz out.npz
"""
import sys
import numpy as np
import jax.numpy as jnp

from lucid.detector_params import _flatten_detector_params, _nest_flat_kwargs
from lucid.fitting import CalibrationForward, neyman_residual, profile_gains
from lucid.next.calibration import S2FitParams, deposit_groups, s2_calibration_sim
from lucid.next.charge import xenon_drift_params
from lucid.next.event import NextEventSimulator, default_params

fit, sample, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
names = [str(n) for n in fit['names']]
ev = NextEventSimulator(n_rays=1_000_000, batch=100_000, K=200, pos_grad_threshold=0, optics_input=True)
NS = len(np.asarray(ev.detector.sensor_shape))
truth = default_params()._replace(drift=xenon_drift_params(lifetime_ms=float(fit['lifetime_ms'])))
optics = _nest_flat_kwargs({**_flatten_detector_params(ev.default_detector_params),
                            'absorption_length': jnp.asarray(float(fit['absorption_length_m']), jnp.float32)})
par = S2FitParams(truth, names, optics)
groups = deposit_groups(sample['deposits'], float(sample['energy_MeV']), sample['letter_of'])
fwd = CalibrationForward(s2_calibration_sim(ev), groups, par, NS)
Q = jnp.asarray(fit['data'])
q_floor = 0.01 * float(jnp.mean(Q))

t0, t1 = fit['theta_true'], fit['fit']
iG, iR, iL = (names.index(n) for n in ('EL gain', 'wall reflectivity', 'absorption length'))
points = {'truth': t0, 'fit': t1, 'midway': 0.5 * (t0 + t1)}
p = t1.copy(); p[iG] = t0[iG]; points['fit, EL gain at truth'] = p
p = t1.copy(); p[[iR, iL]] = t0[[iR, iL]]; points['fit, R and abs at truth'] = p


def loss(mu):
    k = profile_gains(mu.sum(0) + 1e-12, Q.sum(0), gauge='log')
    r = neyman_residual(k[None] * mu, Q, q_floor)
    return float(jnp.sum(r ** 2) / Q.shape[0])


rows = {}
for label, th in points.items():
    th = jnp.asarray(th, jnp.float32)
    reps = [fwd.average(th, 20_000 + 100 * j, jnp.ones(NS), 4) for j in range(4)]   # common keys across points
    l4 = [loss(m) for m in reps]
    l16 = loss(sum(reps) / 4)
    rows[label] = (np.mean(l4), np.std(l4) / 2, l16, l4)
    print(f"{label:26s} 4-draw loss {np.mean(l4):10.1f} +- {np.std(l4) / 2:6.1f} (replicas "
          + ' '.join(f'{x:.0f}' for x in l4) + f");  16-draw loss {l16:10.1f}", flush=True)
base = rows['truth']
for label, r in rows.items():
    print(f"  {label:26s} minus truth: 4-draw {r[0] - base[0]:+9.1f}, 16-draw {r[2] - base[2]:+9.1f}; "
          f"per replica " + ' '.join(f'{a - b:+.0f}' for a, b in zip(r[3], base[3])))
np.savez(out, labels=np.array(list(rows)), l4=np.array([r[3] for r in rows.values()]),
         l16=np.array([r[2] for r in rows.values()]), thetas=np.array([points[k] for k in rows]))
