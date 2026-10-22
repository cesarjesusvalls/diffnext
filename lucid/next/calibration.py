"""Calibrating the NEXT-like detector on its S2, with LUCiD's fitting.

The free parameters are any of the JAXTPC transport (drift velocity, electron lifetime, transverse
and longitudinal diffusion), the EL gain and the optics (wall reflectivity, absorption length);
the rest are held. W stays fixed: the S2 depends on the gain and W only through gain / W. A global
QE is not offered either: it scales every sensor as the EL gain does.

Two observables, each through an existing tool:

* **S2 charge per sensor** (SiPMs and PMTs), fitted the way ``lucid.fitting`` fits optical
  constants: the Neyman residual with per-sensor gains profiled, its Jacobian by ``jacfwd``
  (``CalibrationForward``, ``CalibrationJacobian``, ``CalibrationProblem``). The event simulator
  enters through LUCiD's parameterisation interface (``W``, ``P``, ``to_dp``, ``to_real``):
  :class:`S2FitParams` maps the fit vector to its parameters, and each calibration source is a
  :class:`DepositGroup`.
* **S2 waveform** summed over sensors, compared with JAXTPC's Sobolev loss
  (``tools.losses.sobolev_loss_single``), whose long reach gives the drift velocity a gradient even
  when the predicted and observed pulses do not overlap.

The charge alone sees the transport only through gain / W, v * lifetime and D_T / v, and does not
see D_L at all; the pulse times fix v and their widths D_L. :class:`ChargeTimingProblem` adds the
two terms in the problem interface ``lucid.fitting.minimize`` drives, so the step is LUCiD's damped
Gauss-Newton.

The charge term is LUCiD's ``CalibrationProblem``, or :class:`LeaveOneOutProblem`, which takes each
forward draw's gains from the other draws: its expected step at the truth is zero for any number of
draws, where ``CalibrationProblem``'s settles away from the truth by an amount that falls as
1 / draws.
"""
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from lucid.detector_params import _flatten_detector_params, _nest_flat_kwargs
from lucid.fitting import CalibrationProblem
from lucid.next.charge import drift_to_gate
from lucid.next.event import el_yield
from lucid.next.waveform import s2_time_profile

# The free parameters a fit can take: JAXTPC SimParams fields (cm, us), the EL gain, and
# DetectorParams leaves (m).
_DRIFT_FIELDS = {'drift velocity': 'velocity_cm_us', 'electron lifetime': 'lifetime_us',
                 'transverse diffusion': 'diffusion_trans_cm2_us',
                 'longitudinal diffusion': 'diffusion_long_cm2_us'}
_OPTICS_FIELDS = {'wall reflectivity': 'wall_reflection_rate',
                  'absorption length': 'absorption_length'}
CHARGE_PARAMS = tuple(_DRIFT_FIELDS) + ('EL gain',)
PARAM_NAMES = CHARGE_PARAMS + tuple(_OPTICS_FIELDS)


class DepositGroup(NamedTuple):
    """A calibration source: point deposits (N, 3; m) and their energies (N,; MeV, 0 = padding)."""
    positions_m: jnp.ndarray
    energy_MeV: jnp.ndarray


def deposit_groups(deposits_m, energy_MeV, group_of):
    """One :class:`DepositGroup` per label in ``group_of``, padded to a common length so that
    LUCiD can stack them and compile one program for all."""
    deposits_m = np.asarray(deposits_m, np.float32)
    energy = np.broadcast_to(np.asarray(energy_MeV, np.float32), (len(deposits_m),))
    labels = np.unique(group_of)
    n = max(int(np.sum(group_of == g)) for g in labels)
    groups = []
    for g in labels:
        m = group_of == g
        pos = np.zeros((n, 3), np.float32)
        e = np.zeros(n, np.float32)
        pos[:m.sum()], e[:m.sum()] = deposits_m[m], energy[m]
        pos[m.sum():] = deposits_m[m][0]           # padding sits on a real deposit, with no energy
        groups.append(DepositGroup(jnp.asarray(pos), jnp.asarray(e)))
    return groups


class S2FitParams:
    """``theta``: the log of each parameter named in ``free`` (units as in ``PARAM_NAMES``' fields),
    except the wall reflectivity R, fitted as ``log(1 - R)``.

    In log R, as ``lucid.fitting.CalibrationParams`` fits it, a start 0.4 above R = 0.94 is
    R = 1.4, a wall that makes light at every bounce. ``1 - R`` is the loss per bounce, which the
    light collection follows, and stays a loss in any step.

    LUCiD's parameterisation interface: ``to_dp`` hands the simulator the event parameters, the
    per-sensor gains and the optics (``None`` when ``optics`` is: the physics config's, baked in);
    everything not in ``theta`` is held at ``base`` and ``optics``.

    ``hold_dt_over_v`` holds the transverse diffusion at ``base``'s D_T / v instead of at D_T, so
    that it moves with a free drift velocity. The transverse spread, sigma_T^2 = 2 (D_T / v) d,
    then does not depend on v, and v reaches the charge through the electron survival alone: held
    at D_T, the spread's derivative, mostly Monte-Carlo noise on the SiPMs, lands on v.
    """
    W = 1

    def __init__(self, base, free=CHARGE_PARAMS, optics=None, hold_dt_over_v=False):
        unknown = [n for n in free if n not in PARAM_NAMES]
        if unknown:
            raise KeyError(f'not fit parameters: {unknown}; choose from {PARAM_NAMES}')
        if optics is None and any(n in _OPTICS_FIELDS for n in free):
            raise ValueError('freeing an optics parameter needs the optics DetectorParams')
        if hold_dt_over_v and 'transverse diffusion' in free:
            raise ValueError('hold_dt_over_v holds the transverse diffusion, which is also free')
        self.base, self.optics, self.free = base, optics, tuple(free)
        self.hold_dt_over_v = bool(hold_dt_over_v)
        self.P = len(self.free)
        self._optics_flat = (None if optics is None else
                             {k: jnp.asarray(v) for k, v in _flatten_detector_params(optics).items()})

    def _values(self, theta):
        theta = jnp.asarray(theta)
        v = {n: 1.0 - jnp.exp(theta[i]) if n == 'wall reflectivity' else jnp.exp(theta[i])
             for i, n in enumerate(self.free)}
        if self.hold_dt_over_v and 'drift velocity' in v:
            d = self.base.drift
            v['transverse diffusion'] = d.diffusion_trans_cm2_us / d.velocity_cm_us * v['drift velocity']
        return v

    def charge_params(self, theta):
        v = self._values(theta)
        drift = self.base.drift._replace(**{f: v[n] for n, f in _DRIFT_FIELDS.items() if n in v})
        return self.base._replace(drift=drift, el_gain=v.get('EL gain', self.base.el_gain))

    def detector_params(self, theta):
        if self.optics is None:
            return None
        v = self._values(theta)
        return _nest_flat_kwargs({**self._optics_flat,
                                  **{f: v[n] for n, f in _OPTICS_FIELDS.items() if n in v}})

    def to_dp(self, theta, wl, gains):
        return self.charge_params(theta), gains, self.detector_params(theta)

    def to_real(self, theta):
        return np.array([1.0 - np.exp(t) if n == 'wall reflectivity' else np.exp(t)
                         for n, t in zip(self.free, np.asarray(theta, np.float64))])

    def theta_from_physical(self, params=None, optics=None):
        params = self.base if params is None else params
        optics = self.optics if optics is None else optics
        d = params.drift
        value = {n: getattr(d, f) for n, f in _DRIFT_FIELDS.items()}
        value['EL gain'] = params.el_gain
        if optics is not None:
            flat = _flatten_detector_params(optics)
            value.update({n: flat[f] for n, f in _OPTICS_FIELDS.items()})
        return np.array([np.log(1.0 - float(value[n])) if n == 'wall reflectivity'
                         else np.log(float(value[n])) for n in self.free])


def s2_calibration_sim(ev):
    """``sim(source, (params, gains, optics), key) -> (light per sensor, None)``: the S2 of a
    :class:`DepositGroup` from ``ev`` (a ``NextEventSimulator``), in the form LUCiD's
    ``CalibrationForward`` and ``CalibrationJacobian`` call."""
    def sim(source, dp, key):
        params, gains, optics = dp
        light, _, _ = ev.s2_group(source.positions_m, source.energy_MeV, params, key,
                                  detector_params=optics)
        return gains * light, None
    return sim


class S2WaveformTerm:
    """The S2 waveform summed over sensors, compared with JAXTPC's Sobolev loss.

    Each group's pulse is its observed S2 (``observed_light`` summed over sensors) times its time
    profile at ``theta``, so the term carries the timing and leaves the amplitudes to the charge
    term. ``grad_metric_loss`` follows LUCiD's least-squares convention (``g = J^T r``,
    ``H = J^T J`` for a loss ``|r|^2``): half the gradient, and half the Hessian of the loss with
    the waveform linearised, which for this quadratic loss is its Gauss-Newton metric.
    """

    def __init__(self, params, groups, observed_light, gate_z_m, edges_us, gap_time_us,
                 max_pad=1024, s=2.0):
        from tools.losses import make_sobolev_weight, sobolev_loss_single

        self.params = params
        self.edges_us = jnp.asarray(edges_us, jnp.float32)
        self.weight = make_sobolev_weight(1, len(edges_us) - 1, max_pad=max_pad, s=s)
        positions = jnp.stack([g.positions_m for g in groups])
        energies = jnp.stack([g.energy_MeV for g in groups])
        amplitude = jnp.asarray(np.asarray(observed_light).sum(axis=1), jnp.float32)
        self.observed = None

        def predict(theta):
            p = params.charge_params(theta)

            def pulse(pos, e, a):
                drift = jax.vmap(lambda d: drift_to_gate(d, gate_z_m, p.drift))(pos)
                n_ph = el_yield(e * 1e6 / p.w_ion_eV * drift.survival, p)
                prof = jax.vmap(lambda dr: s2_time_profile(dr, self.edges_us, gap_time_us))(drift)
                return a * (n_ph @ prof) / n_ph.sum()

            return jax.vmap(pulse)(positions, energies, amplitude).sum(axis=0)

        def loss(theta, observed):
            return sobolev_loss_single(predict(theta)[None], observed[None], self.weight)

        def grad_metric_loss(theta, observed):
            value, grad = jax.value_and_grad(loss)(theta, observed)
            w0, J = predict(theta), jax.jacfwd(predict)(theta)
            lin = lambda d: sobolev_loss_single((w0 + J @ d)[None], observed[None], self.weight)
            return 0.5 * grad, 0.5 * jax.hessian(lin)(jnp.zeros_like(theta)), value

        self.predict = jax.jit(predict)
        self._gml = jax.jit(grad_metric_loss)

    def grad_metric_loss(self, theta):
        theta = jnp.asarray(theta, jnp.float32)
        return self._gml(theta, self.observed)


class LeaveOneOutProblem(CalibrationProblem):
    """LUCiD's ``CalibrationProblem`` with each forward draw normalised by the others, so that the
    expected step at the truth is zero for any number of forward draws (two or more).

    ``CalibrationProblem`` profiles each gain on the residual's own draws, ``k_n = c Q_n / M_n``
    (``M_n``: the model's light on sensor ``n``, summed over sources), in the gauge
    ``mean(log k) = 0``. Both are nonlinear in the Monte-Carlo forward, so the residual's
    expectation at the truth is not zero and the fit settles away from it, by an amount that falls
    as 1 / draws (``tests/test_fitting_estimator_unbiased.py``). Here draw ``b`` gives

        r_b[g, n] = (k_n M_b[g, n] - Q[g, n] M_b[n] / m_n) / sqrt(Q[g, n]),
        k_n = c Q_n / m_n,   c = sum_n m_n / sum_n Q_n,

    with ``m_n`` the sensor's light averaged over the OTHER draws: LUCiD's residual with the gain
    taken from the other draws, and the data scaled by this draw's own fluctuation ``M_b[n] / m_n``,
    which keeps LUCiD's per-sensor sum rule (``sum_g r_b[g, n] sqrt(Q[g, n])`` is ``(c - 1) Q_n``,
    here times ``M_b[n] / m_n``: zero but for the gauge). Given the other draws, ``r_b`` is linear
    in draw ``b``, so its expectation is the noise-free residual, zero at the truth. Nothing else in
    the step depends on draw ``b``: the Jacobian's gain is the same leave-one-out ``k``, the
    metric's comes from the previous evaluation, and the Jacobian has its own stream, as in LUCiD.

    The gauge, ``sum_n Q_n / k_n = sum_n Q_n`` (at unit gain the model's total light is the
    observed), is linear in the model where ``mean(log k) = 0`` is not. The EL gain it returns goes
    with gains normalised that way: for pseudo-data with truth gains ``k_true``, the truth's EL
    gain times ``sum(Q) / sum(Q / k_true)``.

    ``k_cap`` bounds a gain where the other draws saw almost no light on a sensor.
    """

    def __init__(self, forward, jacobian, params, data, q_floor, *, n_forward_draws=2,
                 jacobian_draws=2, forward_key0=1000, forward_key_stride=13, k_cap=10.0):
        if n_forward_draws < 2:
            raise ValueError('leave-one-out normalisation needs at least 2 forward draws')
        super().__init__(forward, jacobian, params, data, q_floor, n_forward_draws=n_forward_draws,
                         jacobian_draws=jacobian_draws, forward_key0=forward_key0,
                         forward_key_stride=forward_key_stride)
        self.k_cap = float(k_cap)
        self._k_metric = None
        self._H = None

    def _draws(self, theta, step):
        """``(draws, S, NS)``: ``CalibrationForward.average``'s draws, kept apart."""
        ones = jnp.ones(self.data.shape[1])
        key = self.forward_key(step)
        return jnp.stack([self.forward(theta, key + 131 * b, ones) for b in range(self.n_forward_draws)])

    def _gains(self, M):
        """Per draw: its light per sensor, the other draws' mean of it, and the gains from that."""
        Mn = M.sum(1)
        m = (Mn.sum(0)[None] - Mn) / (M.shape[0] - 1)
        c = m.sum(1, keepdims=True) / self.data_sum.sum()
        m = jnp.maximum(m, c * self.data_sum[None] / self.k_cap)
        return Mn, m, c * self.data_sum[None] / m

    def grad_metric_loss(self, theta, step, refresh=True):
        M = self._draws(theta, step)
        Mn, m, k = self._gains(M)
        w = jnp.sqrt(jnp.clip(self.data, self.q_floor, None))
        r = (k[:, None] * M - self.data[None] * (Mn / m)[:, None]) / w[None]          # (draws, S, NS)

        n = self.n_configs
        if refresh or self._J is None:
            # dM/dtheta / sqrt(Q) at unit gain; the gains are applied below
            self._J = self.jacobian(theta, jnp.zeros(self.data.shape[1]), step, self.data, self.q_floor,
                                    draws=range(self.jacobian_draws))
            k_metric = k.mean(0) if self._k_metric is None else self._k_metric
            self._H = jnp.einsum('gnp,gnq,n->pq', self._J, self._J, k_metric ** 2) / n
        self._k_metric = k.mean(0)
        g = jnp.einsum('bn,gnp,bgn->p', k, self._J, r) / (n * M.shape[0])
        loss = float(jnp.sum(r.mean(0) ** 2) / n)
        return g, self._H, loss

    def gains(self, theta, step=0):
        Mn = self._draws(theta, step).sum(1).mean(0)
        return Mn.sum() / self.data_sum.sum() * self.data_sum / Mn


class ChargeTimingProblem:
    """LUCiD's ``CalibrationProblem`` on the S2 charge plus the :class:`S2WaveformTerm`, in the
    problem interface ``lucid.fitting.minimize`` drives.

    The waveform term is weighted by ``w_time``. With None it is set at the first evaluation (the
    starting point) so that both terms put the same curvature on the drift velocity, the one
    parameter both see. ``terms`` records the (charge, waveform) losses of every evaluation.
    """

    def __init__(self, charge, timing, w_time=None):
        self.charge, self.timing, self.w_time = charge, timing, w_time
        self.terms = []

    def accumulate(self, theta, dtheta):
        return self.charge.accumulate(theta, dtheta)

    def grad_metric_loss(self, theta, step, refresh=True):
        gc, Hc, lc = self.charge.grad_metric_loss(theta, step, refresh)
        gt, Ht, lt = (np.asarray(x, np.float64) for x in self.timing.grad_metric_loss(theta))
        if self.w_time is None:
            i = self.charge.params.free.index('drift velocity')
            self.w_time = float(np.asarray(Hc)[i, i] / Ht[i, i])
        self.terms.append((float(lc), float(lt)))
        w = self.w_time
        return (np.asarray(gc, np.float64) + w * gt, np.asarray(Hc, np.float64) + w * Ht,
                float(lc) + w * float(lt))
