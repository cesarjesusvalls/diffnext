"""Why does D_L drift up in the joint fit? Three numbers.

1. Drift-velocity jitter: the step-to-step scatter of log v over the last 40 Gauss-Newton steps, in
   this fit (D_T held) and in the charge-only fit (D_T free).
2. The waveform term alone, everything at the truth but log v jittered by N(0, s) and log D_L
   offset: where the jitter-averaged Sobolev loss has its minimum. If a jittered v favours wide
   pulses, the minimum sits above D_L's truth.
3. The damping D_L sees: its curvature (waveform only, times w_time) against LUCiD's Levenberg
   term mu * median(diag H), with the charge term's diag H from 2 Jacobian draws at the fit.

  python dl_drift_check.py joint_fit.npz charge_fit_fixed.npz next_letters.npz out.npz
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

from lucid.detector_params import _flatten_detector_params, _nest_flat_kwargs
from lucid.fitting import CalibrationJacobian, profile_gains
from lucid.fitting.calibrate import _seeded_keys
from lucid.next.calibration import S2FitParams, S2WaveformTerm, deposit_groups, s2_calibration_sim
from lucid.next.charge import xenon_drift_params
from lucid.next.event import NextEventSimulator, default_params
from lucid.next.waveform import EL_DRIFT_VELOCITY_MM_US
from tools.losses import sobolev_loss_single

fit, ref, sample, out = np.load(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3]), sys.argv[4]
names = [str(n) for n in fit['names']]
iv, idl = names.index('drift velocity'), names.index('longitudinal diffusion')

# 1. jitter of log v
h, hr = fit['history'] - fit['theta_true'], ref['history'] - ref['theta_true']
s_joint, s_ref = np.std(np.diff(h[-40:, iv])), np.std(np.diff(hr[-40:, 0]))
rms_joint, rms_ref = np.sqrt(np.mean(h[-40:, iv] ** 2)), np.sqrt(np.mean(hr[-40:, 0] ** 2))
print(f"log v over the last 40 steps: rms offset {rms_joint:.2e} (joint) vs {rms_ref:.2e} (charge-only); "
      f"step-to-step std {s_joint:.2e} vs {s_ref:.2e}", flush=True)

# 2. jitter-averaged waveform loss vs D_L
ev = NextEventSimulator(n_rays=1_000_000, batch=100_000, K=200, pos_grad_threshold=0, optics_input=True)
NS = len(np.asarray(ev.detector.sensor_shape))
truth = default_params()._replace(drift=xenon_drift_params(lifetime_ms=float(fit['lifetime_ms'])))
optics = _nest_flat_kwargs({**_flatten_detector_params(ev.default_detector_params),
                            'absorption_length': jnp.asarray(float(fit['absorption_length_m']), jnp.float32)})
par = S2FitParams(truth, names, optics)
th0 = jnp.asarray(par.theta_from_physical(), jnp.float32)
groups = deposit_groups(sample['deposits'], float(sample['energy_MeV']), sample['letter_of'])
edges = fit['edges_us']
gap_us = (ev.el_gap_z[1] - ev.el_gap_z[0]) * 1000.0 / EL_DRIFT_VELOCITY_MM_US
term = S2WaveformTerm(par, groups, fit['data'], ev.gate_z, edges, gap_us)
term.observed = term.predict(th0)
# One parameter set at a time (lax.map): vmapped, the Sobolev padding needs ~30 GB. The average
# over the v jitter is a 32-node Gauss-Hermite quadrature of N(0, s^2).
loss = jax.jit(lambda ths: jax.lax.map(
    lambda th: sobolev_loss_single(term.predict(th)[None], term.observed[None], term.weight), ths))
nodes, weights = np.polynomial.hermite_e.hermegauss(32)
weights = weights / weights.sum()
dls = np.linspace(-0.2, 0.8, 21)
rows = []
for s in (0.0, rms_ref, rms_joint, 2 * rms_joint):
    vals = []
    for dl in dls:
        th = jnp.tile(th0, (len(nodes), 1)).at[:, idl].add(dl).at[:, iv].add(s * jnp.asarray(nodes, jnp.float32))
        vals.append(float(np.asarray(loss(th)) @ weights))
    vals = np.array(vals)
    rows.append(vals)
    print(f"v jitter {s:.1e}: jitter-averaged waveform loss is lowest at log D_L offset {dls[np.argmin(vals)]:+.2f} "
          f"(loss there {vals.min():.3g}; at the truth {vals[np.argmin(np.abs(dls))]:.3g})", flush=True)

# 3. D_L's curvature against the Levenberg term at the fit
gt, Ht, _ = term.grad_metric_loss(jnp.asarray(fit['fit'], jnp.float32))
w = float(fit['w_time'])
fkey, jkey = _seeded_keys(0)
sim = s2_calibration_sim(ev)
data = jnp.asarray(fit['data'])
q_floor = 0.01 * float(jnp.mean(data))
jac = CalibrationJacobian(sim, groups, par, NS, key0=jkey)
from lucid.fitting import CalibrationForward
fwd = CalibrationForward(sim, groups, par, NS)
th_fit = jnp.asarray(fit['fit'], jnp.float32)
mu_model = fwd.average(th_fit, 9100, jnp.ones(NS), 2)
k = profile_gains(mu_model.sum(0) + 1e-12, data.sum(0), gauge='log')
J = jac(th_fit, jnp.log(k), 0, data, q_floor, draws=range(2))
Hc = np.asarray(jnp.einsum('gnp,gnq->pq', J, J) / data.shape[0])
H = Hc + w * np.asarray(Ht)
diag = np.diag(H)
mu = float(fit['mu']) if 'mu' in fit else 1e-6
print('diag H at the fit (charge + w x waveform): ' + ', '.join(f'{n} {d:.3g}' for n, d in zip(names, diag)))
print(f"D_L: curvature {diag[idl]:.3g} vs Levenberg mu x median(diag H) = {mu * np.median(diag):.3g}; "
      f"waveform v-D_L correlation {Ht[iv, idl] / np.sqrt(Ht[iv, iv] * Ht[idl, idl]):+.3f}")
np.savez(out, dls=dls, losses=np.array(rows), jitters=np.array([0.0, rms_ref, rms_joint, 2 * rms_joint]),
         diag=diag, Hc=Hc, Ht=np.asarray(Ht), w=w, names=np.array(names), hist_joint=h, hist_ref=hr)
