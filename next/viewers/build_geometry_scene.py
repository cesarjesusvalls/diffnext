"""Pack the NEXT-100 geometry (nexus / Geant4 VRML export) and the NEXT-like LUCiD geometry
into one JSON of mesh groups for the 3D comparison page.

Frame: the nexus Next100OpticalGeometry export frame, in mm. The gate (end of the gate grid)
is at z = 0 and z grows towards the cathode; the light-tube axis is the z axis; the tracking
and energy planes are centred at (3.7, 6.4) mm. The LUCiD cylinder is placed with its axis on
the light-tube axis and its readout cap on the SiPM sensitive surface (z = -24.13 mm), so
z_nexus = (H/2 - z_lucid) - 24.13 mm.
"""
import base64
import json
import re
import sys

import numpy as np

import os
WRL, OUT = sys.argv[1], sys.argv[2]      # nexus VRML export (nexus_export/), output scene.json
LUCID_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config') + '/'

# ---- nexus groups -------------------------------------------------------------------------
NEXUS_GROUPS = [
    # key, label, role, volumes, kind
    ('n_sipm', 'SiPM sensitive areas', 'sensor', ['SIPM_S13372_SENSAREA'], 'mesh'),
    ('n_mask', 'SiPM-board masks', 'reflector', ['SIPM_BOARD_MASK'], 'mesh'),
    ('n_hole', 'Mask holes (6 x 6 mm)', 'reflector', ['SIPM_BOARD_MASK_HOLE'], 'lines'),
    ('n_tube', 'Light tube', 'reflector', ['LIGHT_TUBE_DRIFT', 'LIGHT_TUBE_BUFFER'], 'mesh'),
    ('n_el', 'EL region: gate, anode, meshes, gap', 'electrode',
     ['GATE_RING', 'ANODE_RING', 'EL_GRID_GATE', 'EL_GRID_ANODE', 'EL_GAP'], 'mesh'),
    ('n_cath', 'Cathode ring and mesh', 'electrode', ['CATHODE_RING', 'CATHODE_GRID'], 'mesh'),
    ('n_pmt', 'Energy-plane PMTs and windows', 'sensor',
     ['PMT', 'PMT_PHOTOCATHODE', 'SAPPHIRE_WINDOW', 'SAPPHIRE_WDW_TPB', 'OPTICAL_PAD'], 'mesh'),
    ('n_cu', 'Copper plates', 'structure', ['TP_COPPER_PLATE', 'EP_COPPER_PLATE'], 'mesh'),
    ('n_fc', 'Field rings, staves, HDPE tube', 'structure', ['FIELD_RING', 'STAVE', 'HDPE_TUBE'], 'mesh'),
]

FLOAT = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


def parse_vrml(path):
    txt = open(path).read()
    solids = {}
    for blk in txt.split('#---------- SOLID: ')[1:]:
        name = blk.split('\n', 1)[0].strip().rsplit(':', 1)[0]
        pts = re.search(r'point\s*\[(.*?)\]', blk, re.S).group(1)
        idx = re.search(r'coordIndex\s*\[(.*?)\]', blk, re.S).group(1)
        p = np.array(FLOAT.findall(pts), dtype=np.float64).reshape(-1, 3)
        faces, cur = [], []
        for tok in idx.replace(',', ' ').split():
            i = int(tok)
            if i < 0:
                if len(cur) >= 3:
                    faces.append(cur)
                cur = []
            else:
                cur.append(i)
        solids.setdefault(name, []).append((p, faces))
    return solids


def triangles(p, faces):
    tri = [(f[0], f[k], f[k + 1]) for f in faces for k in range(1, len(f) - 1)]
    return np.asarray(tri, dtype=np.int64)


def edges(p, faces):
    e = set()
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            e.add((min(a, b), max(a, b)))
    return np.asarray(sorted(e), dtype=np.int64)


def pack(parts, kind):
    """Merge (points, faces) parts into one buffer: positions + triangle or line indices."""
    pos, ind, off = [], [], 0
    for p, faces in parts:
        ix = triangles(p, faces) if kind == 'mesh' else edges(p, faces)
        pos.append(p)
        ind.append(ix + off)
        off += len(p)
    pos = np.concatenate(pos).astype(np.float32)
    ind = np.concatenate(ind).astype(np.uint32)
    return pos, ind


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode('ascii')


def group_record(key, label, detector, role, kind, pos, ind, count):
    lo, hi = pos.min(0), pos.max(0)
    return dict(key=key, label=label, detector=detector, role=role, kind=kind, count=int(count),
                positions=b64(pos), indices=b64(ind),
                bbox=[[round(float(v), 2) for v in lo], [round(float(v), 2) for v in hi]])


solids = parse_vrml(WRL)
groups = []
for key, label, role, vols, kind in NEXUS_GROUPS:
    parts = [part for v in vols for part in solids[v]]
    pos, ind = pack(parts, kind)
    groups.append(group_record(key, label, 'nexus', role, kind, pos, ind, len(parts)))
    print(f"{key:8s} {len(parts):5d} solids  {len(pos):7d} verts  bbox z {pos[:, 2].min():8.2f} .. "
          f"{pos[:, 2].max():8.2f}  r_max {np.hypot(pos[:, 0], pos[:, 1]).max():7.2f}")
for v in ('EL_GAP', 'EL_GRID_GATE', 'EL_GRID_ANODE', 'GATE_RING', 'ANODE_RING', 'CATHODE_GRID',
          'LIGHT_TUBE_DRIFT', 'LIGHT_TUBE_BUFFER', 'SAPPHIRE_WINDOW', 'SIPM_BOARD_MASK'):
    allp = np.concatenate([p for p, _ in solids[v]])
    print(f"   {v:20s} z {allp[:, 2].min():9.3f} .. {allp[:, 2].max():9.3f}   "
          f"r {np.hypot(allp[:, 0], allp[:, 1]).min():7.2f} .. {np.hypot(allp[:, 0], allp[:, 1]).max():7.2f}")

# ---- LUCiD groups -------------------------------------------------------------------------
cfg = json.load(open(LUCID_CFG + 'NEXT100_like_geom_config.json'))
npz = np.load(LUCID_CFG + cfg['geometry_definitions']['npz_file_path'])
R = float(npz['radius']) * 1000.0
H = float(npz['height']) * 1000.0
Z_TOP = -24.13                   # readout cap (SiPM sensitive surface) in the nexus frame
Z_BOT = Z_TOP + H                # PMT cap (sapphire-window face)
seg = 180
th = np.linspace(0.0, 2 * np.pi, seg, endpoint=False)
ring = np.column_stack([R * np.cos(th), R * np.sin(th)])
i = np.arange(seg); j = (i + 1) % seg

bar_pos = np.concatenate([np.column_stack([ring, np.full(seg, Z_TOP)]),
                          np.column_stack([ring, np.full(seg, Z_BOT)])]).astype(np.float32)
bar_ind = np.concatenate([np.column_stack([i, j, i + seg]),
                          np.column_stack([j, j + seg, i + seg])]).astype(np.uint32)
groups.append(group_record('l_barrel', 'Barrel wall', 'lucid', 'reflector', 'mesh',
                           bar_pos, bar_ind.ravel(), 1))


def disc(z):
    pos = np.concatenate([[[0.0, 0.0, z]], np.column_stack([ring, np.full(seg, z)])]).astype(np.float32)
    ind = np.column_stack([np.zeros(seg, int), i + 1, j + 1]).astype(np.uint32)
    return pos, ind.ravel()


p, k = disc(Z_TOP)
groups.append(group_record('l_top', 'Readout cap (wall between SiPMs)', 'lucid', 'reflector',
                           'mesh', p, k, 1))
p, k = disc(Z_BOT)
groups.append(group_record('l_bottom', 'PMT cap (wall between windows)', 'lucid', 'reflector',
                           'mesh', p, k, 1))

pos_l = npz['positions_mm']
shape = npz['sensor_shape'].astype(str)
half = npz['sensor_half_size'] * 1000.0
z_nexus = (H / 2 - pos_l[:, 2]) + Z_TOP          # LUCiD z -> nexus z
inward = np.where(pos_l[:, 2] > 0, 0.02, -0.02)   # a hair inside the cap, to draw above it

sq = shape == 'square'
corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
xy = (pos_l[sq, None, :2] + half[sq, None, None] * corners[None]).reshape(-1, 2)
zz = np.repeat(z_nexus[sq] + inward[sq], 4)
sp = np.column_stack([xy, zz]).astype(np.float32)
base = 4 * np.arange(sq.sum())[:, None]
si = np.concatenate([base + [0, 1, 2], base + [0, 2, 3]], axis=1).astype(np.uint32)
groups.append(group_record('l_sipm', f'SiPMs: {sq.sum()} flat {2 * half[sq][0]:.1f} mm squares',
                           'lucid', 'sensor', 'mesh', sp, si.ravel(), int(sq.sum())))

dk = ~sq
ds = 64
phi = np.linspace(0.0, 2 * np.pi, ds, endpoint=False)
dpos, dind = [], []
for n_, (c, r, z) in enumerate(zip(pos_l[dk, :2], half[dk], z_nexus[dk] + inward[dk])):
    pts = np.vstack([[c[0], c[1], z], np.column_stack([c[0] + r * np.cos(phi), c[1] + r * np.sin(phi), np.full(ds, z)])])
    o = n_ * (ds + 1)
    a = np.arange(ds)
    dind.append(np.column_stack([np.full(ds, o), o + 1 + a, o + 1 + (a + 1) % ds]))
    dpos.append(pts)
groups.append(group_record('l_pmt', f'PMT windows: {dk.sum()} flat {2 * half[dk][0]:.0f} mm discs',
                           'lucid', 'sensor', 'mesh', np.vstack(dpos).astype(np.float32),
                           np.vstack(dind).astype(np.uint32).ravel(), int(dk.sum())))
groups[-1]['ring_segments'] = ds

meta = dict(frame='nexus Next100OpticalGeometry, mm; gate at z = 0, z towards the cathode',
            lucid=dict(radius_mm=R, height_mm=H, z_top=Z_TOP, z_bottom=Z_BOT,
                       n_sipm=int(sq.sum()), n_pmt=int(dk.sum())))
json.dump(dict(meta=meta, groups=groups), open(OUT, 'w'))
print(f"LUCiD: R {R:.2f} mm, H {H:.2f} mm, caps at z {Z_TOP} and {Z_BOT:.2f}, "
      f"{sq.sum()} SiPMs, {dk.sum()} PMTs")
print(f"wrote {OUT}: {sum(len(g['positions']) + len(g['indices']) for g in groups) / 1e6:.2f} MB of base64")
