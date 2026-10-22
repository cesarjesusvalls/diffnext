"""Are the pseudo-data's draws (CalibrationForward keys 5000 + 131 b) different from other draws?

Single forward draws at the truth for several key families, each projected on the reflectivity
shift that pseudo-data made of it would give (data_noise_sets.npz's log-gauge map, against its
64-draw model mean). Draws from one distribution scatter by ~0.135% in R whatever their keys.

  python key_check.py next_letters.npz data_noise_sets.npz out.npz
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from lucid.detector_params import _flatten_detector_params, _nest_flat_kwargs
from lucid.fitting import CalibrationForward
from lucid.next.calibration import S2FitParams, deposit_groups, s2_calibration_sim
from lucid.next.charge import xenon_drift_params
from lucid.next.event import NextEventSimulator, default_params

log = lambda *a: print(*a, flush=True)
sample, ref = np.load(sys.argv[1]), np.load(sys.argv[2])
log('jax', jax.__version__, 'PRNG implementation:', getattr(jax.config, 'jax_default_prng_impl', '?'),
    ', threefry partitionable:', getattr(jax.config, 'jax_threefry_partitionable', '?'))
ev = NextEventSimulator(n_rays=1_000_000, batch=100_000, K=200, pos_grad_threshold=0, optics_input=True)
NS = len(np.asarray(ev.detector.sensor_shape))
truth = default_params()._replace(drift=xenon_drift_params(lifetime_ms=10.0))
optics = _nest_flat_kwargs(_flatten_detector_params(ev.default_detector_params))
free = [str(n) for n in ref['names']]
par = S2FitParams(truth, free, optics, hold_dt_over_v=True)
th = jnp.asarray(par.theta_from_physical(), jnp.float32)
groups = deposit_groups(sample['deposits'], float(sample['energy_MeV']), sample['letter_of'])
fwd = CalibrationForward(s2_calibration_sim(ev), groups, par, NS)
ones = jnp.ones(NS)
log_k = 0.1 * np.random.default_rng(6000).standard_normal(NS)
k = np.exp(log_k - log_k.mean())
L = ref['log_Lcol'][..., free.index('wall reflectivity')]
m = ref['M'].mean(0)
pct = lambda d: 100 * 0.06 * (1 - np.exp(d)) / 0.94                   # d log(1 - R) -> R / 0.94 - 1

families = {'pseudo-data keys 5000 + 131 b': [5000 + 131 * b for b in range(8)],
            'the same keys again': [5000 + 131 * b for b in range(4)],
            '5007 + 131 b': [5007 + 131 * b for b in range(8)],
            '20000 + 131 b': [20000 + 131 * b for b in range(8)],
            '60000 + 131 b (in the 64-draw mean)': [60000 + 131 * b for b in range(4)]}
out = {}
t0 = time.time()
for label, keys in families.items():
    s = np.array([pct(float(np.einsum('gn,gn->', L, k[None] * (np.asarray(fwd(th, kb, ones), np.float64) - m))))
                  for kb in keys])
    out[label] = s
    log(f"{label:38s} R shift per draw " + ' '.join(f'{x:+.3f}' for x in s) + f"%   mean {s.mean():+.3f}%  "
        f"({time.time() - t0:.0f} s)")
np.savez(sys.argv[3], **{k_.replace(' ', '_'): v for k_, v in out.items()})
