"""Compact JSON for the PMT-light viewer: geometry of the NEXT-like LUCiD cylinder and the
expected rays per PMT (direct, indirect, total) for every source position of a run."""
import json, sys
import numpy as np

RUN, POSITIONS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
d = np.load(RUN)
pos = json.load(open(POSITIONS))
K = max(int(k.split('K')[1]) for k in d.files if k.startswith('total_K'))
direct, total = d['direct'], d[f'total_K{K}']
indirect = total - direct
pmt = d['sensor_shape'] == 'disc'
H = float(d['height']) * 1000.0
gate = H / 2 - 24.13
data = dict(
    radius_mm=float(d['radius']) * 1000.0, height_mm=H,
    gate_z_mm=round(gate, 2), cathode_z_mm=round(gate - 1187.37, 2),
    sipm_xy=np.round(d['sensor_xyz'][~pmt, :2] * 1000.0, 3).tolist(),
    sipm_half_mm=round(float(d['sensor_half_size'][~pmt][0]) * 1000.0, 3),
    pmt_xy=np.round(d['sensor_xyz'][pmt, :2] * 1000.0, 2).tolist(),
    pmt_radius_mm=round(float(d['sensor_half_size'][pmt][0]) * 1000.0, 2),
    pmt_id=d['sensor_id'][pmt].astype(int).tolist(),
    n_rays=int(d['n_rays']), K=K,
    positions=[dict(name=p['name'], drift_m=p['drift_m'],
                    xyz_mm=[round(v * 1000.0, 2) for v in p['xyz_m']],
                    direct=np.round(direct[i, pmt], 1).tolist(),
                    indirect=np.round(indirect[i, pmt], 1).tolist(),
                    total=np.round(total[i, pmt], 1).tolist(),
                    sipm=dict(direct=round(float(direct[i, ~pmt].sum()), 1),
                              indirect=round(float(indirect[i, ~pmt].sum()), 1),
                              total=round(float(total[i, ~pmt].sum()), 1)))
               for i, p in enumerate(pos)])
s = json.dumps(data, separators=(',', ':'))
open(OUT, 'w').write(s)
print(f"wrote {OUT}: {len(s) / 1e3:.0f} kB; {len(data['sipm_xy'])} SiPMs, {len(data['pmt_xy'])} PMTs, "
      f"{len(data['positions'])} positions, gate z {data['gate_z_mm']} mm, cathode z {data['cathode_z_mm']} mm")
