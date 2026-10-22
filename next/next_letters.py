"""The word "NEXT" written in point deposits, drifted and imaged: S2 per sensor for every point.

Each letter is drawn with point deposits every ~12 mm along its strokes (13 cm wide, 18 cm tall,
centres 20 cm apart), and each letter sits at its own distance from the gate, so the letters reach
the EL gap one after another.
Every point is a 41.5 keV deposit simulated with lucid.next (JAXTPC drift + LUCiD light).

  HOLD_JOBID=<job> JAXTPC_PATH=<JAXTPC> next/run_gpu_s3df.sh next/next_letters.py --out letters.npz
"""
import argparse
import time

import numpy as np

from lucid.next.event import NextEventSimulator, default_params

# Strokes in letter units, x and y in [-1, 1], scaled to HALF_W and HALF_H (mm).
STROKES = {
    'N': [[(-1, -1), (-1, 1)], [(-1, 1), (1, -1)], [(1, -1), (1, 1)]],
    'E': [[(-1, -1), (-1, 1)], [(-1, 1), (1, 1)], [(-1, 0), (0.7, 0)], [(-1, -1), (1, -1)]],
    'X': [[(-1, 1), (1, -1)], [(-1, -1), (1, 1)]],
    'T': [[(-1, 1), (1, 1)], [(0, 1), (0, -1)]],
}
HALF_W, HALF_H = 65.0, 90.0


def letter_points(letter, centre_x, step=12.0):
    """Points every ~step mm along a letter's strokes (mm), shared vertices kept once."""
    pts = []
    for (x0, y0), (x1, y1) in STROKES[letter]:
        x0, x1, y0, y1 = x0 * HALF_W, x1 * HALF_W, y0 * HALF_H, y1 * HALF_H
        n = max(int(round(np.hypot(x1 - x0, y1 - y0) / step)), 1)
        for t in np.linspace(0.0, 1.0, n + 1):
            pts.append((centre_x + x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    pts = np.array(pts)
    _, keep = np.unique(np.round(pts, 3), axis=0, return_index=True)
    return pts[np.sort(keep)]


ap = argparse.ArgumentParser()
ap.add_argument('--word', default='NEXT')
ap.add_argument('--drift-m', type=float, nargs='+', default=[0.3, 0.5, 0.7, 0.9],
                help='distance from the gate of each letter')
ap.add_argument('--pitch-mm', type=float, default=200.0, help='distance between letter centres')
ap.add_argument('--energy-MeV', type=float, default=0.0415)
ap.add_argument('--n-rays', type=int, default=1_000_000)
ap.add_argument('--K', type=int, default=200)
ap.add_argument('--out', required=True)
args = ap.parse_args()
log = lambda *a: print(*a, flush=True)

ev = NextEventSimulator(n_rays=args.n_rays, K=args.K)
p = default_params()
det = ev.detector
shape = np.asarray(det.sensor_shape).astype(str)
centres = (np.arange(len(args.word)) - (len(args.word) - 1) / 2) * args.pitch_mm
deposits, letter_of = [], []
for k, (ch, cx) in enumerate(zip(args.word, centres)):
    xy = letter_points(ch, cx)
    z = ev.gate_z - args.drift_m[k]
    deposits += [[x / 1000.0, y / 1000.0, z] for x, y in xy]
    letter_of += [k] * len(xy)
deposits = np.array(deposits)
log(f"{len(deposits)} point deposits: " + ', '.join(f"{ch} {letter_of.count(k)} at {args.drift_m[k]:.1f} m"
                                                   for k, ch in enumerate(args.word)))

s2, tdrift, sig_t, sig_l = [], [], [], []
t0 = time.time()
for j, dep in enumerate(deposits):
    light, dr, _, _, _ = ev.s2(dep, args.energy_MeV, p)
    s2.append(np.asarray(light, np.float32))
    tdrift.append(float(dr.time_us)); sig_t.append(float(dr.sigma_trans_m)); sig_l.append(float(dr.sigma_time_us))
    if (j + 1) % 20 == 0 or j + 1 == len(deposits):
        log(f"  {j + 1}/{len(deposits)} points, {time.time() - t0:.0f} s")

np.savez(args.out, word=args.word, deposits=deposits, letter_of=np.array(letter_of), drift_m=np.array(args.drift_m),
         s2=np.array(s2), drift_time_us=np.array(tdrift), sigma_trans_m=np.array(sig_t), sigma_time_us=np.array(sig_l),
         sensor_xyz=np.asarray(ev.sim.det_geom.sensor_points), sensor_shape=shape,
         sensor_half_size=np.asarray(det.sensor_half_size), radius=det.r, height=det.H, gate_z=ev.gate_z,
         el_gap_z=np.array(ev.el_gap_z), cathode_z=float(det.cathode_z), energy_MeV=args.energy_MeV,
         velocity_mm_us=float(p.drift.velocity_cm_us) * 10, D_T=float(p.drift.diffusion_trans_cm2_us))
log(f"wrote {args.out}")
