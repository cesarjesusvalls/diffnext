"""Video of the point-source gradient fits: residual map on the PMT plane, side view of the path,
and the distance to the true position, one frame per Adam iteration, both cases side by side."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

RUN, OUT, FFMPEG = sys.argv[1], sys.argv[2], sys.argv[3]
plt.rcParams['animation.ffmpeg_path'] = FFMPEG
d = np.load(RUN)
R = float(d['radius']) * 1000.0
H = float(d['height']) * 1000.0
ZTOP, GATE_OFF, CATH = H / 2, 24.13, 1187.37
drift = lambda z_mm: ZTOP - GATE_OFF - z_mm              # LUCiD z (mm) -> distance from the gate
pmt_xy = d['pmt_xy'] * 1000.0
rim = np.hypot(*pmt_xy.T) + 40.0 > R
cases = [c for c in ('A', 'B') if f'{c}_pos' in d.files]

INK, INK2, MUTED, LINE, GRID = '#0b0b0b', '#52514e', '#898781', '#c3c2b7', '#e1e0d9'
DIVERGING = LinearSegmentedColormap.from_list(
    'blue_grey_red', ['#104281', '#3987e5', '#b7d3f6', '#f0efec', '#f3b4b3', '#e34948', '#9c2423'])
NORM = Normalize(-5, 5)
plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': LINE,
                     'xtick.color': MUTED, 'ytick.color': MUTED})

n_it = len(d[f'{cases[0]}_pos'])
fig = plt.figure(figsize=(13.5, 4.3 * len(cases) + 0.8), dpi=110)
gs = fig.add_gridspec(len(cases), 3, width_ratios=[1.0, 1.25, 0.9], wspace=0.28, hspace=0.38,
                      left=0.05, right=0.97, top=0.88, bottom=0.12)
artists = []
for r, c in enumerate(cases):
    pos = d[f'{c}_pos'] * 1000.0
    true = d[f'{c}_true'] * 1000.0
    obs = d[f'{c}_obs']
    light = d[f'{c}_light']
    dist = np.linalg.norm(pos - true, axis=1)
    chi2 = d[f'{c}_chi2']
    # residuals for every PMT; the rim PMTs are outside chi2 and drawn hollow
    inner = ~rim
    resid = (light - obs[None, :]) / np.sqrt(np.maximum(obs[None, :], 1.0))

    ax = fig.add_subplot(gs[r, 0])
    pc = PatchCollection([Circle(p, 40.0) for p in pmt_xy[inner]], cmap=DIVERGING, norm=NORM,
                         edgecolor='white', linewidth=0.6)
    pc.set_array(resid[0, inner]); ax.add_collection(pc)
    ax.add_collection(PatchCollection([Circle(p, 40.0) for p in pmt_xy[rim]], facecolor='none',
                                      edgecolor=MUTED, linewidth=1.1, linestyle=(0, (3, 2))))
    ax.add_patch(Circle((0, 0), R, fill=False, edgecolor=LINE, linewidth=1.0))
    trail_xy, = ax.plot([], [], color=INK, lw=1.3, alpha=0.75)
    est_xy, = ax.plot([], [], 'o', color=INK, ms=6.5, mec='white', mew=1.0)
    ax.plot(*true[:2], marker='x', color=INK, ms=10, mew=2.2)
    ax.set_xlim(-R - 40, R + 40); ax.set_ylim(-R - 40, R + 40); ax.set_aspect('equal')
    ax.set_xticks([-400, 0, 400]); ax.set_yticks([-400, 0, 400])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title(f'Case {c}: PMT residuals, seen from the drift volume', fontsize=10.5, color=INK, loc='left')

    ax2 = fig.add_subplot(gs[r, 1])
    for z, lab, ha in ((-GATE_OFF, 'SiPM plane', 'right'), (0.0, 'gate', 'left'), (CATH, 'cathode', 'right'),
                       (H - GATE_OFF, 'PMT plane', 'right')):
        ax2.axvline(z, color=LINE if lab in ('gate', 'cathode') else MUTED, lw=1.0,
                    ls='--' if lab in ('gate', 'cathode') else '-')
        ax2.text(z + (8 if ha == 'left' else -8), R + 30, lab, ha=ha, va='bottom', fontsize=8.5, color=INK2)
    ax2.axhline(R, color=MUTED, lw=1.0); ax2.axhline(-R, color=MUTED, lw=1.0)
    trail_side, = ax2.plot([], [], color=INK, lw=1.3, alpha=0.75)
    est_side, = ax2.plot([], [], 'o', color=INK, ms=6.5, mec='white', mew=1.0)
    ax2.plot(drift(true[2]), true[0], marker='x', color=INK, ms=10, mew=2.2)
    ax2.set_xlim(-260, H + 20); ax2.set_ylim(-R - 60, R + 110)
    ax2.set_xlabel('distance from the gate (mm)'); ax2.set_ylabel('x (mm)')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.set_title('Side view of the path', fontsize=10.5, color=INK, loc='left')

    ax3 = fig.add_subplot(gs[r, 2])
    ax3.semilogy(np.arange(n_it), dist, color=LINE, lw=1.2)
    conv, = ax3.semilogy([], [], color=INK, lw=1.8)
    dot3, = ax3.semilogy([], [], 'o', color=INK, ms=5.5)
    ax3.set_xlim(0, n_it - 1); ax3.set_ylim(max(dist.min() * 0.5, 0.3), dist.max() * 2)
    ax3.grid(True, which='major', color=GRID, lw=0.8); ax3.set_axisbelow(True)
    ax3.spines[['top', 'right']].set_visible(False)
    ax3.set_xlabel('Adam iteration'); ax3.set_ylabel('distance to the true position (mm)')
    label = ax3.text(0.98, 0.97, '', transform=ax3.transAxes, ha='right', va='top', fontsize=9.5, color=INK)
    ax3.set_title('Convergence', fontsize=10.5, color=INK, loc='left')
    artists.append((pos, true, resid, inner, dist, chi2, pc, trail_xy, est_xy, trail_side, est_side, conv, dot3, label))

cax = fig.add_axes([0.05, 0.032, 0.25, 0.012])
cb = fig.colorbar(artists[0][6], cax=cax, orientation='horizontal', extend='both')
cb.set_label('(predicted − observed) / √observed, per PMT', fontsize=9, color=INK2)
cb.outline.set_visible(False)
fig.text(0.33, 0.03, '×  true position     ●  current estimate     ○  rim PMTs, left out of χ²', fontsize=9.5,
         color=INK2, va='center')
title = fig.suptitle('', fontsize=12.5, color=INK, fontweight='semibold', x=0.05, ha='left')


def update(k):
    out = []
    for (pos, true, resid, inner, dist, chi2, pc, txy, exy, tside, eside, conv, dot3, label) in artists:
        pc.set_array(resid[k, inner])
        txy.set_data(pos[:k + 1, 0], pos[:k + 1, 1]); exy.set_data([pos[k, 0]], [pos[k, 1]])
        tside.set_data(drift(pos[:k + 1, 2]), pos[:k + 1, 0]); eside.set_data([drift(pos[k, 2])], [pos[k, 0]])
        conv.set_data(np.arange(k + 1), dist[:k + 1]); dot3.set_data([k], [dist[k]])
        label.set_text(f"χ²/ndf {chi2[k]:,.3g}\n|Δ| {dist[k]:,.1f} mm")
        out += [pc, txy, exy, tside, eside, conv, dot3, label]
    title.set_text(f"Fitting a point-source position to its PMT light with LUCiD gradients  ·  iteration {k}")
    return out


frames = list(range(n_it)) + [n_it - 1] * 15        # hold the last frame for 1.5 s
anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
anim.save(OUT + '.mp4', writer=animation.FFMpegWriter(fps=10, bitrate=2400,
                                                      extra_args=['-pix_fmt', 'yuv420p']))
update(n_it - 1)
fig.savefig(OUT + '_final.png', dpi=130)
fig.savefig(OUT + '_final.pdf')
print('wrote', OUT + '.mp4', OUT + '_final.{png,pdf}')
