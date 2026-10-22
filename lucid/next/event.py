"""Integrated charge and light simulation of a point energy deposit in the NEXT-like detector.

S1: E * s1_yield photons, emitted isotropically at the deposit.
S2: N_e = E / W ionization electrons drift to the gate (``lucid.next.charge``, JAXTPC); the
fraction exp(-t / lifetime) that survives crosses the electroluminescence gap, and each electron
makes ``el_yield`` photons, emitted isotropically and uniformly across the gap at its diffused
arrival point.

Both are propagated through the same LUCiD detector (flat SiPMs and PMTs, reflecting walls), so
the light per sensor is the expected number of photons reaching it. Every step is a JAX function
of the deposit and of ``ChargeLightParams``: gradients run from the sensors back to the energy,
the position, the transport parameters and the EL gain, and to the optics when they are passed
in as ``DetectorParams``.
"""
import os
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from lucid.simulation import setup_event_simulator
from lucid.sources.calibration_sources import get_isotropic_rays
from lucid.next.charge import xenon_drift_params, drift_to_gate, electron_arrivals

_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config')
GEOM = os.path.join(_CONFIG, 'NEXT100_like_geom_config.json')
PHYS = os.path.join(_CONFIG, 'NEXT100_like_physics_config.json')


class ChargeLightParams(NamedTuple):
    """Physics of a deposit: quanta, charge transport and charge-to-light."""
    w_ion_eV: jnp.ndarray          # energy per ionization electron
    s1_yield_per_MeV: jnp.ndarray  # primary scintillation photons per MeV
    el_gain: jnp.ndarray           # EL photons per electron crossing the gap (placeholder)
    drift: Any                     # JAXTPC SimParams: velocity, lifetime, diffusion


def default_params():
    """nexus values: W = 22.4 eV (IonizationClustering) and 25510 photons/MeV (Next100Vessel).
    The EL gain, 3264 photons per electron, is nexus XenonELLightYield, (140 E/p - 116) p
    photons/cm, at 34.5 kV/cm and 13.5 bar over the 1 cm gap. Transport: xenon_drift_params."""
    return ChargeLightParams(jnp.asarray(22.4, jnp.float32), jnp.asarray(25510.0, jnp.float32),
                             jnp.asarray(3264.0, jnp.float32), xenon_drift_params())


def el_yield(n_electrons, params):
    """Placeholder charge-to-light conversion: photons = el_gain * electrons."""
    return params.el_gain * n_electrons


class RaySource(NamedTuple):
    """Fixed rays, in the form LUCiD's calibration simulator takes a source."""
    directions: jnp.ndarray
    origins: jnp.ndarray
    weights: jnp.ndarray

    def __call__(self, n_photons, key, n_water=1.33):
        return self.directions, self.origins, self.weights


class NextEventSimulator:
    """S1 and S2 light per sensor for point deposits in the NEXT-like LUCiD detector.

    Parameters
    ----------
    n_rays : rays per light signal, fed to LUCiD in batches of ``batch``.
    K : propagation steps (reflections) per ray.
    n_electrons_mc : macro-electrons representing the drifting cloud.
    seed : fixes the ray directions, macro-electron normals and emission depths, so repeated
        calls are deterministic functions of the deposit and the parameters.
    pos_grad_threshold, n_grad_iters : LUCiD's cut on how many propagation steps carry position
        and direction gradients (None: all K). Parameters entering through the photon numbers
        (energy, yields, EL gain, lifetime) differentiate exactly either way; position-type
        gradients carried through every diffuse reflection grow without bound.
    optics_input : also build the LUCiD simulator that takes ``DetectorParams`` as an argument,
        so that ``s2_group`` can be given optics other than the physics config's, and fitted.
        ``sim`` keeps the physics-config values baked in.
    """

    def __init__(self, geom=GEOM, phys=PHYS, n_rays=1_000_000, batch=100_000, K=200,
                 n_electrons_mc=4096, seed=0, pos_grad_threshold=None, n_grad_iters=None,
                 optics_input=False):
        if n_rays % batch:
            raise ValueError('n_rays must be a multiple of batch')
        kw = dict(n_photons=batch, temperature=None, K=K, is_calibration=True, physics_config=phys,
                  wavelength_mode=False, reflection_model='scalar', hit_mode='aggregated',
                  pos_grad_threshold=pos_grad_threshold, n_grad_iters=n_grad_iters)
        self.sim = setup_event_simulator(geom, default_detector_params=True, **kw)
        self.sim_optics = (setup_event_simulator(geom, default_detector_params=False, **kw)
                           if optics_input else None)
        self.default_detector_params = self.sim.default_detector_params
        self.detector = self.sim.det_geom.detector
        self.gate_z = float(self.detector.gate_z)
        self.el_gap_z = tuple(float(v) for v in self.detector.el_gap_z)
        self.n_rays, self.batch, self.nb = n_rays, batch, n_rays // batch
        k_dir, k_e, k_z, k_prop = jax.random.split(jax.random.PRNGKey(seed), 4)
        dirs, _, _ = get_isotropic_rays(jnp.zeros(3), 1.0, n_rays, k_dir)
        self.dirs = dirs.reshape(self.nb, batch, 3)
        self.normals = jax.random.normal(k_e, (n_electrons_mc, 2))
        self.el_u = jax.random.uniform(k_z, (self.nb, batch))
        self.electron_of_ray = (jnp.arange(n_rays) % n_electrons_mc).reshape(self.nb, batch)
        self.keys = jax.random.split(k_prop, self.nb)

    def _propagate(self, origins, n_photons):
        """Expected photons per sensor from n_photons spread evenly over the rays;
        origins(b) gives the (batch, 3) ray origins of batch b."""
        w = jnp.full((self.batch,), 1.0) * (n_photons / self.n_rays)
        total = 0.0
        for b in range(self.nb):
            q, _ = self.sim(RaySource(self.dirs[b], origins(b), w), self.keys[b])
            total = total + q
        return total

    def s1(self, deposit_m, energy_MeV, params):
        """Expected S1 photons per sensor for a deposit at ``deposit_m`` (x, y, z; m)."""
        deposit_m = jnp.asarray(deposit_m, jnp.float32)
        origins = lambda b: jnp.broadcast_to(deposit_m, (self.batch, 3))
        return self._propagate(origins, energy_MeV * params.s1_yield_per_MeV)

    def s2(self, deposit_m, energy_MeV, params):
        """Expected S2 photons per sensor, with the drift result, the macro-electron arrival
        (x, y), and the numbers of electrons crossing the gap and of photons they make."""
        deposit_m = jnp.asarray(deposit_m, jnp.float32)
        drift = drift_to_gate(deposit_m, self.gate_z, params.drift)
        n_electrons = energy_MeV * 1e6 / params.w_ion_eV * drift.survival
        n_photons = el_yield(n_electrons, params)
        xy = electron_arrivals(deposit_m, drift, self.normals)
        z0, z1 = self.el_gap_z

        def origins(b):
            z = z0 + (z1 - z0) * self.el_u[b]
            return jnp.concatenate([xy[self.electron_of_ray[b]], z[:, None]], axis=1)

        return self._propagate(origins, n_photons), drift, xy, n_electrons, n_photons

    def s2_group(self, deposits_m, energy_MeV, params, key, detector_params=None):
        """Expected S2 photons per sensor from a group of point deposits, with the drift result
        and photon number of each.

        ``deposits_m`` (N, 3) and ``energy_MeV`` (N,); a deposit with zero energy is padding. Every
        random number (ray directions, each ray's macro-electron spread, emission depth, the
        propagation) is drawn from ``key``, so a new key is a new Monte-Carlo draw. The rays are
        shared out evenly among the real deposits, so a group costs one ``n_rays`` budget, and
        the batches run under ``lax.map``, so the call can sit inside ``jit`` or ``jacfwd``.
        ``detector_params`` (a LUCiD ``DetectorParams``; needs ``optics_input``) replaces the
        physics config's optics.
        """
        if detector_params is not None and self.sim_optics is None:
            raise ValueError('detector_params needs NextEventSimulator(optics_input=True)')
        deposits_m = jnp.asarray(deposits_m, jnp.float32)
        energy_MeV = jnp.asarray(energy_MeV, jnp.float32)
        n_dep = deposits_m.shape[0]
        drift = jax.vmap(lambda d: drift_to_gate(d, self.gate_z, params.drift))(deposits_m)
        n_photons = el_yield(energy_MeV * 1e6 / params.w_ion_eV * drift.survival, params)

        k_dir, k_e, k_z, k_prop = jax.random.split(key, 4)
        dirs, _, _ = get_isotropic_rays(jnp.zeros(3), 1.0, self.n_rays, k_dir)
        live = jnp.flatnonzero(energy_MeV > 0, size=n_dep, fill_value=0)
        dep_of_ray = live[jnp.arange(self.n_rays) % jnp.sum(energy_MeV > 0)]
        rays_per_dep = jnp.zeros(n_dep).at[dep_of_ray].add(1.0)
        xy = (deposits_m[dep_of_ray, :2] + drift.sigma_trans_m[dep_of_ray, None]
              * jax.random.normal(k_e, (self.n_rays, 2)))
        z0, z1 = self.el_gap_z
        z = z0 + (z1 - z0) * jax.random.uniform(k_z, (self.n_rays,))
        origins = jnp.concatenate([xy, z[:, None]], axis=1)
        weights = n_photons[dep_of_ray] / rays_per_dep[dep_of_ray]

        def batch(args):
            o, d, w, k = args
            if detector_params is None:
                return self.sim(RaySource(d, o, w), k)[0]
            return self.sim_optics(RaySource(d, o, w), detector_params, k)[0]

        shape = lambda a: a.reshape((self.nb, self.batch) + a.shape[1:])
        light = jax.lax.map(batch, (shape(origins), shape(dirs), shape(weights),
                                    jax.random.split(k_prop, self.nb)))
        return light.sum(axis=0), drift, n_photons
