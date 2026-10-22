"""Closure test: recover the charge transport and EL gain from the S2 of the "NEXT" sample.

Pseudo-data at known parameters, a start drawn as ``lucid.fitting.closure`` draws it, then LUCiD's
damped Gauss-Newton (``lucid.fitting.minimize``) on the S2 charge per sensor (SiPMs and PMTs;
Neyman residual, per-sensor gains profiled) plus the S2 waveform compared with JAXTPC's Sobolev
loss (``lucid.next.calibration``). Each letter of the sample is one calibration source.

  --mode check   AD Jacobian of one source against central finite differences (common random
                 numbers), per parameter, for SiPMs and PMTs: LUCiD's SourceModel cross-check
  --mode crb     Cramer-Rao bound of the S2 charge alone (lucid.fitting.crb)
  --mode fit     the closure fit

--free picks the fitted parameters, the rest staying at the truth: v, lifetime, dt, dl and gain
(charge transport and EL gain), reflectivity and absorption (wall reflectivity, absorption length).

  HOLD_JOBID=<job> JAXTPC_PATH=<JAXTPC> next/run_gpu_s3df.sh next/charge_calibration.py \
      --sample next_letters.npz --mode fit --out charge_fit.npz
"""
import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

from lucid.detector_params import _flatten_detector_params, _nest_flat_kwargs
from lucid.fitting import (CalibrationForward, CalibrationJacobian, CalibrationProblem, SourceModel,
                           closure_data, crb, damped_gauss_newton, minimize, profile_gains)
from lucid.fitting.calibrate import _seeded_keys
from lucid.next.calibration import (CHARGE_PARAMS, PARAM_NAMES, ChargeTimingProblem, LeaveOneOutProblem,
                                    S2FitParams, S2WaveformTerm, deposit_groups, s2_calibration_sim)
from lucid.next.charge import xenon_drift_params
from lucid.next.event import NextEventSimulator, default_params
from lucid.next.waveform import EL_DRIFT_VELOCITY_MM_US

KEYS = dict(zip(('v', 'lifetime', 'dt', 'dl', 'gain', 'reflectivity', 'absorption'), PARAM_NAMES))

ap = argparse.ArgumentParser()
ap.add_argument('--sample', required=True, help='a next_letters.py output')
ap.add_argument('--mode', choices=('check', 'crb', 'fit'), required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--free', nargs='+', choices=list(KEYS), default=['v', 'lifetime', 'dt', 'dl', 'gain'],
                help='fitted parameters; the rest stay at the truth')
ap.add_argument('--lifetime-ms', type=float, default=10.0, help='truth electron lifetime')
ap.add_argument('--wall-reflectivity', type=float, help='truth (default: the physics config\'s)')
ap.add_argument('--absorption-length-m', type=float, help='truth (default: the physics config\'s)')
ap.add_argument('--gain-spread', type=float, default=0.0,
                help='truth per-sensor gains: log-normal with this sigma, geometric mean 1')
ap.add_argument('--hold-dt-over-v', action='store_true',
                help='hold the transverse diffusion at its truth D_T / v rather than D_T, so that it moves with v')
ap.add_argument('--n-rays', type=int, default=1_000_000, help='rays per letter')
ap.add_argument('--batch', type=int, default=100_000)
ap.add_argument('--K', type=int, default=200)
ap.add_argument('--pos-grad-threshold', type=int, default=1,
                help='propagation steps whose ray positions carry gradients (LUCiD knob)')
ap.add_argument('--source', type=int, default=3, help='check: which letter')
ap.add_argument('--fd-step', type=float, default=0.2, help='check: log-space step')
ap.add_argument('--draws', type=int, default=16, help='check: Monte-Carlo draws averaged')
ap.add_argument('--fd-params', type=int, nargs='+',
                help='check: parameters given finite differences (default all; the rest are left at zero)')
ap.add_argument('--steps', type=int, default=30)
ap.add_argument('--refresh', type=int, default=2)
ap.add_argument('--lam', type=float, default=0.01, help='Marquardt damping (lucid.fitting default)')
ap.add_argument('--mu', type=float, default=0.1,
                help='Levenberg damping, times the median curvature (lucid.fitting default)')
ap.add_argument('--jacobian-draws', type=int, default=2)
ap.add_argument('--forward-draws', type=int, default=1,
                help='forward draws averaged into the residual each step (lucid.fitting n_forward_draws)')
ap.add_argument('--estimator', choices=('profiled', 'leave-one-out'), default='profiled',
                help='the charge term: LUCiD\'s CalibrationProblem (gains profiled on the residual\'s draws), '
                     'or LeaveOneOutProblem (each draw\'s gains from the other draws)')
ap.add_argument('--data-key', type=int, default=5000,
                help='fit: the pseudo-data\'s key base (lucid.fitting.closure_data: draw b of letter s uses '
                     'base + 131 b + 1000 s)')
ap.add_argument('--perturbation', type=float, default=0.4)
ap.add_argument('--polyak', type=int, default=10)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--bin-us', type=float, default=1.0)
ap.add_argument('--window-us', type=float, default=1600.0)
args = ap.parse_args()
log = lambda *a: print(*a, flush=True)

sample = np.load(args.sample)
free = [n for n in PARAM_NAMES if n in {KEYS[k] for k in args.free}]
optics_truth = {f: v for f, v in (('wall_reflection_rate', args.wall_reflectivity),
                                  ('absorption_length', args.absorption_length_m)) if v is not None}
use_optics = bool(optics_truth) or any(n not in CHARGE_PARAMS for n in free)
ev = NextEventSimulator(n_rays=args.n_rays, batch=args.batch, K=args.K,
                        pos_grad_threshold=args.pos_grad_threshold, optics_input=use_optics)
NS = len(np.asarray(ev.detector.sensor_shape))
sipm = np.asarray(ev.detector.sensor_shape).astype(str) == 'square'
truth = default_params()._replace(drift=xenon_drift_params(lifetime_ms=args.lifetime_ms))
optics = None
if use_optics:
    optics = _nest_flat_kwargs({**_flatten_detector_params(ev.default_detector_params),
                                **{f: jnp.asarray(v, jnp.float32) for f, v in optics_truth.items()}})
par = S2FitParams(truth, free, optics, hold_dt_over_v=args.hold_dt_over_v)
theta_true = par.theta_from_physical()
groups = deposit_groups(sample['deposits'], float(sample['energy_MeV']), sample['letter_of'])
sim = s2_calibration_sim(ev)
log_k = args.gain_spread * np.random.default_rng(6000 + args.seed).standard_normal(NS)
k_true = np.exp(log_k - log_k.mean())                     # the fit's gauge: mean log gain 0
log(f"{len(groups)} sources ({[int((g.energy_MeV > 0).sum()) for g in groups]} deposits), {NS} sensors, "
    f"{args.n_rays} rays per source, K={args.K}, pos_grad_threshold={args.pos_grad_threshold}")
log('truth: ' + ', '.join(f'{n} {v:.4g}' for n, v in zip(par.free, par.to_real(theta_true))))
truth_optics = _flatten_detector_params(optics if optics is not None else ev.default_detector_params)
absorption = float(truth_optics['absorption_length'])
log(f"truth optics: wall reflectivity {float(truth_optics['wall_reflection_rate']):.4g}, absorption length "
    + ('none' if absorption >= 1e6 else f'{absorption:.4g} m') + f"; per-sensor gain spread {args.gain_spread:g}")
held = ['transverse diffusion / drift velocity' if n == 'transverse diffusion' and args.hold_dt_over_v else n
        for n in PARAM_NAMES if n not in free]
log('held at the truth: ' + (', '.join(held) or 'nothing'))


def source_model(g):
    return SourceModel(lambda th, ek, pk: sim(g, par.to_dp(th, 0, jnp.ones(NS)), ek)[0])


if args.mode == 'check':
    # Expectations, not one draw: a ray carries hundreds of photons and the sensors are hard-edged,
    # so a single draw's derivative in a position-type parameter is mostly hits flipping on and off.
    sm = source_model(groups[args.source])
    th = jnp.asarray(theta_true, jnp.float32)
    lk = jnp.zeros(NS)
    keys = [jax.random.PRNGKey(11 + i) for i in range(args.draws)]
    t0 = time.time()
    J_draws = np.stack([sm.ad_jacobian(th, lk, k, k) for k in keys])
    J_ad = J_draws.mean(axis=0)
    t_ad = time.time() - t0
    m_avg = lambda t: sum(np.asarray(sm.m(t, lk, k, k)) for k in keys) / args.draws
    m0 = m_avg(th)
    fd_params = range(par.P) if args.fd_params is None else args.fd_params
    J_fd = np.stack([(m_avg(th.at[d].add(args.fd_step / 2)) - m_avg(th.at[d].add(-args.fd_step / 2)))
                     / args.fd_step if d in fd_params else np.zeros(NS) for d in range(par.P)], axis=1)
    log(f"{args.draws} draws: AD Jacobians {t_ad:.0f} s, central differences with step {args.fd_step} "
        f"{time.time() - t0 - t_ad:.0f} s")
    for name, m in (('SiPMs', sipm), ('PMTs', ~sipm)):
        for d, pn in enumerate(par.free):
            a, f = J_ad[m, d], J_fd[m, d]
            nf = np.linalg.norm(f)
            cos = float(a @ f / (np.linalg.norm(a) * nf + 1e-30))
            proj = float(a @ f / (nf ** 2 + 1e-30))
            log(f"  {name:5s} {pn:23s} |FD| {nf:10.4g}  |AD|/|FD| {np.linalg.norm(a) / (nf + 1e-30):7.3f}  "
                f"cos {cos:6.3f}  AD.FD/|FD|^2 {proj:7.3f}")
    np.savez(args.out, J_ad=J_ad, J_draws=J_draws, J_fd=J_fd, m0=m0, sipm=sipm, theta=theta_true, fd_step=args.fd_step,
             names=np.array(par.free), source=args.source, pos_grad_threshold=args.pos_grad_threshold)

elif args.mode == 'crb':
    t0 = time.time()
    res = crb([source_model(g) for g in groups], theta_true, NS)
    raw = np.sqrt(np.clip(np.diag(res['cov_raw']), 0, None))
    log(f"charge-only Cramer-Rao bound ({time.time() - t0:.0f} s); fractional sigma, raw and x sqrt(12):")
    for n, s, h in zip(par.free, raw, res['sigma']):
        log(f"  {n:23s} {s:10.3g}  {h:10.3g}")
    w, V = np.linalg.eigh(res['fisher'])
    log('Fisher eigenvalues (log-params): ' + ' '.join(f'{x:.3g}' for x in w))
    log('weakest direction: ' + ', '.join(f'{n} {c:+.2f}' for n, c in zip(par.free, V[:, 0])))
    np.savez(args.out, theta=theta_true, sigma_raw=raw, sigma=res['sigma'], fisher=res['fisher'],
             cov_raw=res['cov_raw'], names=np.array(par.free))

else:
    fkey, jkey = _seeded_keys(args.seed)
    fwd = CalibrationForward(sim, groups, par, NS)
    t0 = time.time()
    data = closure_data(fwd, theta_true, k_true, n_draws=8, key_base=args.data_key)
    log(f"pseudo-data: 8 draws per source, keys {args.data_key} + 131 b, {time.time() - t0:.0f} s")
    rng = np.random.default_rng(4000 + args.seed)          # lucid.fitting.closure's start
    start = theta_true + rng.uniform(-args.perturbation, args.perturbation, par.P)
    jac = CalibrationJacobian(sim, groups, par, NS, key0=jkey)
    kw = dict(n_forward_draws=args.forward_draws, jacobian_draws=args.jacobian_draws, forward_key0=fkey)
    q_floor = 0.01 * float(jnp.mean(data))
    # What the fit should return: the truth, but for the EL gain in LeaveOneOutProblem's gauge,
    # which normalises the gains by the light they see.
    theta_target = theta_true.copy()
    if args.estimator == 'leave-one-out':
        charge = LeaveOneOutProblem(fwd, jac, par, data, q_floor, **kw)
        if 'EL gain' in par.free:
            light_mean_gain = float(data.sum() / (data.sum(0) / k_true).sum())
            theta_target[par.free.index('EL gain')] += np.log(light_mean_gain)
            log(f"leave-one-out gains; in their gauge the EL gain to recover is the truth's times "
                f"{light_mean_gain:.5f} (the light-weighted mean truth gain)")
    else:
        charge = CalibrationProblem(fwd, jac, par, data, q_floor, gauge='log', **kw)
    edges = np.arange(0.0, args.window_us + args.bin_us / 2, args.bin_us)
    gap_us = (ev.el_gap_z[1] - ev.el_gap_z[0]) * 1000.0 / EL_DRIFT_VELOCITY_MM_US
    timing = S2WaveformTerm(par, groups, data, ev.gate_z, edges, gap_us)
    timing.observed = timing.predict(jnp.asarray(theta_true, jnp.float32))
    prob = ChargeTimingProblem(charge, timing)

    t_start = time.time()

    def on_step(step, theta, g, H, loss):
        lc, lt = prob.terms[-1]
        log(f"  step {step + 1:3d}  {time.time() - t_start:6.0f} s  loss {lc:11.5g} + {prob.w_time:.3g} x "
            f"{lt:9.4g}  " + ' '.join(f'{x:+.3f}' for x in np.asarray(theta) - theta_target))

    log('start - truth (log): ' + ' '.join(f'{x:+.3f}' for x in start - theta_target))
    res = minimize(prob, jnp.asarray(start, jnp.float32), args.steps,
                   damped_gauss_newton(lam=args.lam, mu=args.mu), max_step=0.5, refresh=args.refresh,
                   readout='polyak', polyak=args.polyak, on_step=on_step)
    hist = np.asarray(res['history'], np.float64)
    fit = np.asarray(res['theta'], np.float64)
    log(f"done in {time.time() - t_start:.0f} s; fitted / truth - 1 (Polyak mean of the last {args.polyak}):")
    for n, f_, t_, s_ in zip(par.free, par.to_real(fit), par.to_real(theta_target), par.to_real(start)):
        log(f"  {n:23s} truth {t_:10.4g}  start {s_ / t_ - 1:+7.1%}  fit {f_ / t_ - 1:+8.3%}")
    # The gain map: profiled on 16 draws at the fit, and at the truth, where what is left of the
    # spread is the Monte-Carlo noise of the pseudo-data and the model.
    gains = lambda th: np.asarray(profile_gains(fwd.average(jnp.asarray(th, jnp.float32), 9000, jnp.ones(NS), 16)
                                                .sum(0) + 1e-12, data.sum(0), gauge='log'))
    gains_fit, gains_at_truth = gains(fit), gains(theta_true)
    for name, m in (('SiPMs', sipm), ('PMTs', ~sipm)):
        log(f"  {name:5s} gains: truth spread {np.std(np.log(k_true[m])):.3f}, fitted / truth - 1 rms "
            f"{np.std(gains_fit[m] / k_true[m] - 1):.4f} (at the truth parameters "
            f"{np.std(gains_at_truth[m] / k_true[m] - 1):.4f})")
    model_start = np.asarray(fwd(jnp.asarray(start, jnp.float32), 7000, jnp.ones(NS)))
    model_fit = np.asarray(fwd(jnp.asarray(fit, jnp.float32), 7000, jnp.ones(NS)))
    np.savez(args.out, names=np.array(par.free), theta_true=theta_true, start=start, fit=fit, history=hist,
             real_true=par.to_real(theta_true), real_start=par.to_real(start), real_fit=par.to_real(fit),
             estimator=args.estimator, theta_target=theta_target, real_target=par.to_real(theta_target),
             data_key=args.data_key,
             real_history=np.stack([par.to_real(h) for h in hist]),
             gain_spread=args.gain_spread, k_true=k_true, gains_fit=gains_fit, gains_at_truth=gains_at_truth,
             wall_reflectivity=float(truth_optics['wall_reflection_rate']),
             absorption_length_m=absorption,
             terms=np.array(prob.terms), w_time=prob.w_time, gnorm=res['gnorm'], loss=res['loss'],
             data=np.asarray(data), model_start=model_start, model_fit=model_fit, sipm=sipm,
             sensor_xyz=np.asarray(ev.sim.det_geom.sensor_points), edges_us=edges,
             wave_obs=np.asarray(timing.observed),
             wave_start=np.asarray(timing.predict(jnp.asarray(start, jnp.float32))),
             wave_fit=np.asarray(timing.predict(jnp.asarray(fit, jnp.float32))),
             wave_hist=np.stack([np.asarray(timing.predict(jnp.asarray(h, jnp.float32))) for h in hist]),
             letter_drift_m=sample['drift_m'], word=sample['word'], lifetime_ms=args.lifetime_ms,
             hold_dt_over_v=args.hold_dt_over_v, n_rays=args.n_rays, K=args.K,
             pos_grad_threshold=args.pos_grad_threshold, lam=args.lam, mu=args.mu, forward_draws=args.forward_draws,
             jacobian_draws=args.jacobian_draws)
    log(f"wrote {args.out}")
