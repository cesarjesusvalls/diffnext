"""The optics-input simulator at the physics-config optics must give the baked-in simulator's S2.

  python optics_path_check.py next_letters.npz
"""
import sys
import time
import numpy as np
import jax
from lucid.next.calibration import deposit_groups
from lucid.next.event import NextEventSimulator, default_params

t0 = time.time()
ev = NextEventSimulator(n_rays=1_000_000, batch=100_000, K=200, pos_grad_threshold=0, optics_input=True)
print(f"setup with both simulators: {time.time() - t0:.1f} s", flush=True)
s = np.load(sys.argv[1])
g = deposit_groups(s['deposits'], float(s['energy_MeV']), s['letter_of'])[3]
p = default_params()
key = jax.random.PRNGKey(1)
for label, dp in (('baked', None), ('input', ev.default_detector_params)):
    t0 = time.time()
    q = np.asarray(ev.s2_group(g.positions_m, g.energy_MeV, p, key, detector_params=dp)[0])
    print(f"{label}: {time.time() - t0:.1f} s (with compile), total {q.sum():.6g}", flush=True)
    if dp is None:
        a = q
rel = np.abs(q - a) / np.maximum(np.abs(a), 1e-30)
print(f"max |input - baked| / baked over sensors with light: {rel[a > 0].max():.3g}; "
      f"identical: {np.array_equal(q, a)}")
