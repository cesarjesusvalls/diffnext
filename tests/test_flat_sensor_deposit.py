"""Flat sensors deposit at the sensor's reflectance (``deposit_as_sensor``).

A flat sensor's deposit is a hard step, so a photon off every sensor deposits nothing whatever
reflectance multiplies it: the forward is the same with the sensor's reflectance as with the
wall's. What changes is the straight-through gradient, whose outer half (photons landing just
outside a sensor, on the wall) was scaled by ``1 - R_wall`` before.
"""
import os
from typing import NamedTuple

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.simulation import photon_iteration_update_factors, setup_event_simulator
from lucid.simulation.reflection import ScalarReflection, scalar_reflection
from lucid.sources.calibration_sources import get_isotropic_rays

pytestmark = pytest.mark.slow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOM = os.path.join(ROOT, 'config', 'NEXT100_like_geom_config.json')
PHYS = os.path.join(ROOT, 'config', 'NEXT100_like_physics_config.json')
R_WALL, R_SENSOR = 0.94, 0.0


def _step(hit_sensor, deposit_as_sensor):
    return photon_iteration_update_factors(
        position=jnp.array([0.0, 0.0, 0.5]), direction=jnp.array([0.0, 0.0, 1.0]), time=0.0,
        surface_distance=0.2, normal=jnp.array([0.0, 0.0, -1.0]), scatter_length=50.0,
        mie_scatter_length=1.0e9, g=0.0,
        refl_params=ScalarReflection(wall_rate=jnp.asarray(R_WALL), sensor_rate=jnp.asarray(R_SENSOR)),
        absorption_length=100.0, hit_sensor=hit_sensor, lam=jnp.asarray(0.0),
        rng_key=jax.random.PRNGKey(3), speed_of_light=0.3, reflection_fn=scalar_reflection,
        deposit_as_sensor=deposit_as_sensor)


def test_wall_landing_deposits_at_the_sensor_reflectance():
    off, on = _step(False, False), _step(False, True)
    npt.assert_allclose(on[3] * (1.0 - R_WALL), off[3] * (1.0 - R_SENSOR), rtol=1e-6)
    for i in (0, 1, 2, 4, 5, 6):           # everything but the deposit is untouched
        npt.assert_array_equal(on[i], off[i])


def test_sensor_landing_is_unchanged():
    for a, b in zip(_step(True, False), _step(True, True)):
        npt.assert_array_equal(a, b)


class _RaySource(NamedTuple):
    """Fixed rays as a pytree, the form the calibration simulator traces."""
    directions: jnp.ndarray
    origins: jnp.ndarray
    weights: jnp.ndarray

    def __call__(self, n_photons, key, n_water=1.33):
        return self.directions, self.origins, self.weights


def _source(n, key):
    dirs, _, _ = get_isotropic_rays(jnp.zeros(3), 1.0, n, key)
    origin = jnp.array([0.01, -0.02, 0.7414 - 0.019])      # 19 mm below the SiPM plane, like EL light
    return _RaySource(dirs, jnp.broadcast_to(origin, dirs.shape), jnp.ones(n))


def _charges(monkeypatch, flat_sensors):
    original = DetectorGeometry.from_config
    if not flat_sensors:
        monkeypatch.setattr(DetectorGeometry, 'from_config',
                            staticmethod(lambda *a, **k: original(*a, **k)._replace(flat_sensors=False)))
    sim = setup_event_simulator(GEOM, n_photons=20_000, temperature=None, K=3, is_calibration=True,
                                physics_config=PHYS, default_detector_params=True,
                                wavelength_mode=False, reflection_model='scalar',
                                hit_mode='aggregated')
    monkeypatch.setattr(DetectorGeometry, 'from_config', original)
    return np.asarray(sim(_source(20_000, jax.random.PRNGKey(5)), jax.random.PRNGKey(6))[0])


def test_flat_sensor_forward_is_unchanged(monkeypatch):
    with_sensor_deposit = _charges(monkeypatch, True)
    with_landing_deposit = _charges(monkeypatch, False)
    assert with_sensor_deposit.sum() > 0
    npt.assert_array_equal(with_sensor_deposit, with_landing_deposit)
