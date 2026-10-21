"""PMT maps of the expected rays (direct, indirect = total - direct, total) for the point-source
positions of a point_source_rays.py run: one figure per drift distance."""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

RUN, POSITIONS, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3]
d = np.load(RUN)
pos = json.load(open(POSITIONS))
K = max(int(k.split('K')[1]) for k in d.files if k.startswith('total_K'))
direct, total = d['direct'], d[f'total_K{K}']
indirect = total - direct
pmt = d['sensor_shape'] == 'disc'
xy = d['sensor_xyz'][pmt, :2] * 1000.0
rad = d['sensor_half_size'][pmt] * 1000.0
R = float(d['radius']) * 1000.0

# Sequential one-hue ramp (light -> dark), from the dataviz reference palette.
RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
        '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
cmap = LinearSegmentedColormap.from_list('blue_ramp', RAMP)
INK, INK2, MUTED, LINE = '#0b0b0b', '#52514e', '#898781', '#c3c2b7'
# One colour scale per column, shared by every position and figure: direct spans decades (log),
# indirect and total vary by tens of percent (linear).
dv = direct[:, pmt]
NORMS = {'Direct': LogNorm(vmin=max(dv[dv > 0].min(), 1.0), vmax=dv.max()),
         'Indirect': Normalize(vmin=indirect[:, pmt].min(), vmax=indirect[:, pmt].max()),
         'Total': Normalize(vmin=total[:, pmt].min(), vmax=total[:, pmt].max())}

plt.rcParams.update({'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.edgecolor': LINE})
drifts = sorted({p['drift_m'] for p in pos})
for drift in drifts:
    rows = [i for i, p in enumerate(pos) if p['drift_m'] == drift]
    fig, axes = plt.subplots(len(rows), 3, figsize=(9.2, 3.0 * len(rows) + 1.6),
                             constrained_layout=True)
    mappables = {}
    for r, i in enumerate(rows):
        src = np.array(pos[i]['xyz_m'][:2]) * 1000.0
        for c, (label, arr) in enumerate([('Direct', direct), ('Indirect', indirect), ('Total', total)]):
            ax = axes[r, c]
            v = arr[i, pmt]
            patches = [Circle(p, rr) for p, rr in zip(xy, rad)]
            norm = NORMS[label]
            pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor='white', linewidth=0.6)
            pc.set_array(np.clip(v, norm.vmin, None))
            mappables[label] = pc
            ax.add_collection(pc)
            ax.add_patch(Circle((0, 0), R, fill=False, edgecolor=LINE, linewidth=1.0))
            ax.plot(*src, marker='x', color=INK, markersize=7, mew=1.6)
            ax.set_xlim(-R - 60, R + 60); ax.set_ylim(-R - 60, R + 60)
            ax.set_aspect('equal'); ax.set_xticks([-400, 0, 400]); ax.set_yticks([-400, 0, 400])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.text(0.02, 0.02, f"sum {v.sum():,.0f}", transform=ax.transAxes, color=INK2, fontsize=8.5)
            if r == 0:
                ax.set_title(label, fontsize=11, color=INK, fontweight='semibold')
            if c == 0:
                ax.set_ylabel(f"source x = {src[0]:.0f} mm\n\ny (mm)", fontsize=9.5)
            if r == len(rows) - 1:
                ax.set_xlabel('x (mm)')
    for c, label in enumerate(['Direct', 'Indirect', 'Total']):
        cb = fig.colorbar(mappables[label], ax=axes[:, c], location='bottom', shrink=0.85, pad=0.02,
                          aspect=28)
        cb.set_label(f'{label}: expected rays per PMT (of 10⁶)', color=INK2, fontsize=8.5)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle(f"PMT light, point source {drift:.1f} m from the gate", fontsize=12, color=INK,
                 fontweight='semibold')
    stem = f"{OUTDIR}/pmt_maps_drift{int(drift * 1000)}mm"
    fig.savefig(stem + '.png', dpi=170)
    fig.savefig(stem + '.pdf')
    plt.close(fig)
    print('wrote', stem + '.{png,pdf}')

# Summary table: sums over PMTs and over SiPMs, and the K-convergence of the total.
ks = sorted(int(k.split('K')[1]) for k in d.files if k.startswith('total_K'))
print(f"\n{'position':>16s} | {'PMT direct':>10s} {'PMT indir.':>10s} {'PMT total':>10s} | "
      f"{'SiPM direct':>11s} {'SiPM indir.':>11s} {'SiPM total':>10s} | total K{ks[0]}/K{ks[-1]}")
for i, p in enumerate(pos):
    s = ~pmt
    conv = d[f'total_K{ks[0]}'][i].sum() / d[f'total_K{ks[-1]}'][i].sum()
    print(f"{p['name']:>16s} | {direct[i, pmt].sum():10.0f} {indirect[i, pmt].sum():10.0f} {total[i, pmt].sum():10.0f} | "
          f"{direct[i, s].sum():11.0f} {indirect[i, s].sum():11.0f} {total[i, s].sum():10.0f} | {conv:.4f}")
