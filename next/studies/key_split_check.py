"""Which random stream makes the key-5000 draws different: the electrons' spread and emission depth,
or the propagation?

NextEventSimulator.s2_group splits each source's key into (directions, electron spread, depth,
propagation); the directions are a Fibonacci lattice that ignores its key. This rebuilds a draw
with the electron and depth streams from one key and the propagation stream from another, and
projects each draw on the reflectivity shift (data_noise_sets.npz's log-gauge map), as
key_check.py does. Also prints the raw streams' first moments for the suspect keys.

  python key_split_check.py next_letters.npz data_noise_sets.npz
"""
import sys

import numpy as np
import jax
import jax.numpy as jnp

from lucid.detector_params import _flatten_detector_params, _nest_flat_kwargs
from lucid.next.calibration import S2FitParams, deposit_groups
from lucid.next.charge import drift_to_gate, xenon_drift_params
from lucid.next.event import NextEventSimulator, RaySource, default_params, el_yield
from lucid.sources.calibration_sources import get_isotropic_rays

log = lambda *a: print(*a, flush=True)
sample, ref = np.load(sys.argv[1]), np.load(sys.argv[2])
ev = NextEventSimulator(n_rays=1_000_000, batch=100_000, K=200, pos_grad_threshold=0, optics_input=True)
NS = len(np.asarray(ev.detector.sensor_shape))
truth = default_params()._replace(drift=xenon_drift_params(lifetime_ms=10.0))
optics = _nest_flat_kwargs(_flatten_detector_params(ev.default_detector_params))
free = [str(n) for n in ref['names']]
par = S2FitParams(truth, free, optics, hold_dt_over_v=True)
th = jnp.asarray(par.theta_from_physical(), jnp.float32)
p = par.charge_params(th)
groups = deposit_groups(sample['deposits'], float(sample['energy_MeV']), sample['letter_of'])
log_k = 0.1 * np.random.default_rng(6000).standard_normal(NS)
k = np.exp(log_k - log_k.mean())
L = ref['log_Lcol'][..., free.index('wall reflectivity')]
m = ref['M'].mean(0)
pct = lambda d: 100 * 0.06 * (1 - np.exp(d)) / 0.94


@jax.jit
def light(pos, e, key_ez, key_prop):
    """s2_group's body, with the electron/depth streams and the propagation stream from two keys."""
    n_dep = pos.shape[0]
    drift = jax.vmap(lambda d: drift_to_gate(d, ev.gate_z, p.drift))(pos)
    n_photons = el_yield(e * 1e6 / p.w_ion_eV * drift.survival, p)
    k_dir, k_e, k_z, _ = jax.random.split(key_ez, 4)
    _, _, _, k_prop = jax.random.split(key_prop, 4)
    dirs, _, _ = get_isotropic_rays(jnp.zeros(3), 1.0, ev.n_rays, k_dir)
    live = jnp.flatnonzero(e > 0, size=n_dep, fill_value=0)
    dep_of_ray = live[jnp.arange(ev.n_rays) % jnp.sum(e > 0)]
    rays_per_dep = jnp.zeros(n_dep).at[dep_of_ray].add(1.0)
    xy = pos[dep_of_ray, :2] + drift.sigma_trans_m[dep_of_ray, None] * jax.random.normal(k_e, (ev.n_rays, 2))
    z0, z1 = ev.el_gap_z
    z = z0 + (z1 - z0) * jax.random.uniform(k_z, (ev.n_rays,))
    origins = jnp.concatenate([xy, z[:, None]], axis=1)
    weights = n_photons[dep_of_ray] / rays_per_dep[dep_of_ray]
    shape = lambda a: a.reshape((ev.nb, ev.batch) + a.shape[1:])
    out = jax.lax.map(lambda a: ev.sim_optics(RaySource(a[1], a[0], a[2]), optics, a[3])[0],
                      (shape(origins), shape(dirs), shape(weights), jax.random.split(k_prop, ev.nb)))
    return out.sum(axis=0)


def shift(base_ez, base_prop):
    M = np.stack([np.asarray(light(g.positions_m, g.energy_MeV, jax.random.PRNGKey(base_ez + 1000 * s),
                                   jax.random.PRNGKey(base_prop + 1000 * s)), np.float64)
                  for s, g in enumerate(groups)])
    return pct(float(np.einsum('gn,gn->', L, k[None] * (M - m))))


for label, ez, prop in (('both from 5000 + 131 b (as the pseudo-data)', 5000, 5000),
                        ('electrons/depth 5000 + 131 b, propagation 5007 + 131 b', 5000, 5007),
                        ('electrons/depth 5007 + 131 b, propagation 5000 + 131 b', 5007, 5000),
                        ('both from 5007 + 131 b', 5007, 5007)):
    s = [shift(ez + 131 * b, prop + 131 * b) for b in range(4)]
    log(f"{label:56s} R shift " + ' '.join(f'{x:+.3f}' for x in s) + f"%  mean {np.mean(s):+.3f}%")

for base in (5000, 5007, 60000):
    for b in range(3):
        _, k_e, k_z, k_prop = jax.random.split(jax.random.PRNGKey(base + 131 * b), 4)
        n = jax.random.normal(k_e, (ev.n_rays, 2))
        u = jax.random.uniform(k_z, (ev.n_rays,))
        log(f"key {base + 131 * b}: k_e {np.asarray(k_e)}, normal mean {float(n.mean()):+.5f} sd {float(n.std()):.5f}; "
            f"uniform mean {float(u.mean()):.5f}; k_prop {np.asarray(k_prop)}")
