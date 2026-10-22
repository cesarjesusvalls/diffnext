"""Where does the closure's own charge chi^2 put D_T? Scan log D_T with the other parameters at the
truth, the fit's estimator (profiled gains, log gauge, Neyman residual), its pseudo-data, and 8
forward draws per point on common random numbers.

  python dt_scan.py charge_fit_fixed.npz next_letters.npz out.npz
"""
import sys
import numpy as np
import jax.numpy as jnp
from lucid.fitting import CalibrationForward
from lucid.fitting.calib import profile_gains, neyman_residual
from lucid.next.calibration import S2FitParams, deposit_groups, s2_calibration_sim
from lucid.next.charge import xenon_drift_params
from lucid.next.event import NextEventSimulator, default_params

fit, sample, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
ev = NextEventSimulator(n_rays=1_000_000, batch=100_000, K=200, pos_grad_threshold=0)
NS = len(np.asarray(ev.detector.sensor_shape))
truth = default_params()._replace(drift=xenon_drift_params(lifetime_ms=10.0))
par = S2FitParams(truth)                                 # v, lifetime, D_T, D_L, EL gain
th0 = par.theta_from_physical()
fwd = CalibrationForward(s2_calibration_sim(ev), deposit_groups(sample['deposits'], float(sample['energy_MeV']),
                                                                sample['letter_of']), par, NS)
Q = jnp.asarray(fit['data'])
q_floor = 0.01 * float(jnp.mean(Q))
sipm = fit['sipm']
offsets = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
rows = []
for dlog in offsets:
    th = th0.copy(); th[2] += dlog
    mu = fwd.average(jnp.asarray(th, jnp.float32), 900_000, jnp.ones(NS), 8)
    k = profile_gains(mu.sum(0) + 1e-12, Q.sum(0), gauge='log')
    r = np.asarray(neyman_residual(k[None] * mu, Q, q_floor))
    rows.append((float((r ** 2).sum() / r.shape[0]), float((r[:, sipm] ** 2).sum() / r.shape[0]),
                 float((r[:, ~sipm] ** 2).sum() / r.shape[0])))
    print(f"log D_T offset {dlog:+.2f} (D_T {np.exp(dlog) - 1:+6.1%}):  chi2 {rows[-1][0]:12.1f}  SiPMs {rows[-1][1]:12.1f}"
          f"  PMTs {rows[-1][2]:10.1f}", flush=True)
rows = np.array(rows)
c = np.polyfit(offsets, rows[:, 0], 2)
print(f"parabola minimum at log offset {-c[1] / (2 * c[0]):+.3f}  (D_T {np.exp(-c[1] / (2 * c[0])) - 1:+.1%})")
np.savez(out, offsets=offsets, chi2=rows)
