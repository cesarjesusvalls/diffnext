"""S2 time profile of a drifted deposit: the fraction of its EL light emitted in each time bin.

The electrons of a deposit reach the gate spread by longitudinal diffusion, at N(t_drift, sigma_t)
from JAXTPC (``lucid.next.charge.drift_to_gate``), and each emits uniformly while it crosses the EL
gap. The emission time is the sum of the two; its CDF has a closed form, so the binned profile is
exact and differentiable in the drift velocity and the longitudinal diffusion.
"""
import jax.numpy as jnp
from jax.scipy.special import ndtr

EL_DRIFT_VELOCITY_MM_US = 2.5      # nexus NEXT-100 drift velocity in the EL gap


def _int_ndtr(z):
    """Antiderivative of the standard normal CDF, z Phi(z) + phi(z)."""
    return z * ndtr(z) + jnp.exp(-0.5 * z * z) / jnp.sqrt(2.0 * jnp.pi)


def emission_cdf(t_us, drift, gap_time_us):
    """P(emission time <= t_us): arrival N(drift.time_us, drift.sigma_time_us), then uniform
    emission over [0, gap_time_us]."""
    s = drift.sigma_time_us
    a = (t_us - drift.time_us) / s
    b = (t_us - drift.time_us - gap_time_us) / s
    before = s / gap_time_us * (_int_ndtr(a) - _int_ndtr(b))
    after = 1.0 - s / gap_time_us * (_int_ndtr(-b) - _int_ndtr(-a))
    # the two forms are equal; in float32 each is accurate only on its own side of the pulse
    return jnp.where(t_us < drift.time_us + 0.5 * gap_time_us, before, after)


def s2_time_profile(drift, edges_us, gap_time_us):
    """Fraction of a deposit's S2 emitted between consecutive ``edges_us`` (B + 1,) -> (B,)."""
    c = emission_cdf(edges_us, drift, gap_time_us)
    return c[1:] - c[:-1]
