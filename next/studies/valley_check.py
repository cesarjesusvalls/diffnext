"""Two follow-ups on the joint fit.

1. How degenerate are the wall reflectivity, absorption length and EL gain on this sample? The
   charge term's Gauss-Newton metric (LUCiD's CalibrationJacobian, 2 draws, at the truth and at
   the fit): the correlations it implies and its weakest direction.
2. How well do the SiPM gains come back as a function of how much light the SiPM sees?

  python valley_check.py joint_fit.npz next_letters.npz out.npz
"""
import sys
import numpy as np
import jax.numpy as jnp

from lucid.detector_params import _flatten_detector_params, _nest_flat_kwargs
from lucid.fitting import CalibrationForward, CalibrationJacobian, profile_gains
from lucid.fitting.calibrate import _seeded_keys
from lucid.next.calibration import S2FitParams, deposit_groups, s2_calibration_sim
from lucid.next.charge import xenon_drift_params
from lucid.next.event import NextEventSimulator, default_params

fit, sample, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
names = [str(n) for n in fit['names']]

# 2 first: numbers only
sipm, k_true, k_fit, k_tr = fit['sipm'], fit['k_true'], fit['gains_fit'], fit['gains_at_truth']
light = fit['data'].sum(0) / k_true                      # photons per sensor over the 4 letters, unit gain
edges = np.quantile(light[sipm], [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0])
print('SiPM gains by light (quantile bins of the total S2 photons over the 4 letters):')
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = sipm & (light >= lo) & (light <= hi)
    rows.append((lo, hi, m.sum(), np.std(k_fit[m] / k_true[m] - 1), np.std(k_tr[m] / k_true[m] - 1)))
    print(f"  {lo:10.4g} - {hi:10.4g} photons  {m.sum():5d} SiPMs  fitted / truth - 1 rms {rows[-1][3]:.4f}  "
          f"(at the truth parameters {rows[-1][4]:.4f})", flush=True)

# 1
ev = NextEventSimulator(n_rays=1_000_000, batch=100_000, K=200, pos_grad_threshold=0, optics_input=True)
NS = len(np.asarray(ev.detector.sensor_shape))
truth = default_params()._replace(drift=xenon_drift_params(lifetime_ms=float(fit['lifetime_ms'])))
optics = _nest_flat_kwargs({**_flatten_detector_params(ev.default_detector_params),
                            'absorption_length': jnp.asarray(float(fit['absorption_length_m']), jnp.float32)})
par = S2FitParams(truth, names, optics)
groups = deposit_groups(sample['deposits'], float(sample['energy_MeV']), sample['letter_of'])
sim = s2_calibration_sim(ev)
fwd = CalibrationForward(sim, groups, par, NS)
fkey, jkey = _seeded_keys(0)
jac = CalibrationJacobian(sim, groups, par, NS, key0=jkey)
data = jnp.asarray(fit['data'])
q_floor = 0.01 * float(jnp.mean(data))
res = {}
for label, th in (('truth', fit['theta_true']), ('fit', fit['fit'])):
    th = jnp.asarray(th, jnp.float32)
    k = profile_gains(fwd.average(th, 9100, jnp.ones(NS), 2).sum(0) + 1e-12, data.sum(0), gauge='log')
    J = jac(th, jnp.log(k), 0, data, q_floor, draws=range(2))
    Hc = np.asarray(jnp.einsum('gnp,gnq->pq', J, J) / data.shape[0], np.float64)
    charge_only = [i for i, n in enumerate(names) if n != 'longitudinal diffusion']   # D_L: waveform only
    sub = Hc[np.ix_(charge_only, charge_only)]
    cov = np.linalg.inv(sub)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    d = np.sqrt(np.diag(sub))
    w, V = np.linalg.eigh(sub / np.outer(d, d))
    sub_names = [names[i] for i in charge_only]
    print(f"\nat the {label}: correlations implied by the charge term's metric (D_L left out: the charge has none)")
    print('  ' + ' '.join(f'{n[:12]:>13s}' for n in sub_names))
    for n, row in zip(sub_names, corr):
        print(f'  {n[:12]:12s} ' + ' '.join(f'{c:+13.3f}' for c in row))
    print(f"  weakest direction (scaled metric eigenvalue {w[0]:.2e}, next {w[1]:.2e}): "
          + ', '.join(f'{n} {c:+.2f}' for n, c in zip(sub_names, V[:, 0])), flush=True)
    res[label] = dict(Hc=Hc, corr=corr, eig=w, vec=V)
np.savez(out, names=np.array(names), gain_bins=np.array(rows), **{f'{k}_{l}': v for l, r in res.items() for k, v in r.items()})
