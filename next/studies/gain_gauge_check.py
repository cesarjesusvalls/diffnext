"""At the TRUE parameters: how far does the profiled-gain gauge mean(log k) = 0 sit from zero when the
forward is one noisy draw, as in the fit? The fitted EL gain must absorb exactly that offset:
gain_fit / gain_true = exp(m), m = mean_j log(sum_s Q_sj / sum_s M_sj).

  python gain_gauge_check.py charge_fit.npz next_letters.npz out.npz
"""
import sys
import numpy as np
import jax.numpy as jnp
from lucid.fitting import CalibrationForward
from lucid.next.calibration import S2FitParams, deposit_groups, s2_calibration_sim
from lucid.next.charge import xenon_drift_params
from lucid.next.event import NextEventSimulator, default_params

fit, sample, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
ev = NextEventSimulator(n_rays=1_000_000, batch=100_000, K=200, pos_grad_threshold=0)
NS = len(np.asarray(ev.detector.sensor_shape))
truth = default_params()._replace(drift=xenon_drift_params(lifetime_ms=10.0))
par = S2FitParams(truth)
th = jnp.asarray(par.theta_from_physical(), jnp.float32)
fwd = CalibrationForward(s2_calibration_sim(ev), deposit_groups(sample['deposits'], float(sample['energy_MeV']),
                                                                sample['letter_of']), par, NS)
Q = fit['data'].sum(0)                      # pseudo-data per sensor, summed over the 4 letters
sipm = fit['sipm']
draws = np.stack([np.asarray(fwd(th, 1000 + 13 * s, jnp.ones(NS))).sum(0) for s in range(8)])  # the fit's key stream
logk = np.log(Q[None] / draws)              # (draws, sensors), raw profiled log-gain at truth
rel = draws.std(0) / draws.mean(0)          # forward Monte-Carlo noise per sensor
m1 = logk.mean(1)
m4 = np.log(Q / draws[:4].mean(0)).mean()
print(f"m = mean log k at truth, one draw:  {m1.mean():+.4f} +- {m1.std():.4f}  -> predicted gain bias {np.exp(m1.mean()) - 1:+.2%}")
print(f"   SiPMs only {logk[:, sipm].mean():+.4f}   PMTs only {logk[:, ~sipm].mean():+.4f}")
print(f"forward noise per sensor (relative): SiPMs median {np.median(rel[sipm]):.3f}, PMTs median {np.median(rel[~sipm]):.4f}")
print(f"Jensen estimate mean(rel^2)/2 (all sensors) {np.mean(rel ** 2) / 2:+.4f}")
print(f"m with the forward averaged over 4 draws: {m4:+.4f} -> {np.exp(m4) - 1:+.2%}")
np.savez(out, logk=logk, rel=rel, sipm=sipm, Q=Q, m1=m1, m4=m4, sensor_xyz=np.asarray(ev.sim.det_geom.sensor_points))
