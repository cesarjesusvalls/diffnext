"""Why D_T and the EL gain are off in the closure fit: two panels of evidence.

(a) the D_T column of a `charge_calibration.py --mode check` run, LUCiD's gradient against finite
differences, by the SiPM's distance from the letter's strokes; (b) gain_gauge_check.py's offset of
the profiled-gain gauge at the truth, against the number of forward draws averaged.

  python plot_charge_fit_why.py charge_check_dt.npz next_letters.npz gain_gauge_check.npz <figure stem>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

chk, smp, gg, out = (np.load(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3]), sys.argv[4])
INK, INK2, MUTED, LINE, GRID = '#0b0b0b', '#52514e', '#898781', '#c3c2b7', '#e1e0d9'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': LINE,
                     'xtick.color': MUTED, 'ytick.color': MUTED})


def clean(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 5.6), dpi=120, gridspec_kw=dict(wspace=0.22))
fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.14)

# (a) D_T: how the SiPM light around the T letter changes when D_T goes up, finite differences vs LUCiD AD
sip = chk['sipm']
xy = smp['sensor_xyz'][:, :2] * 1e3
dep = smp['deposits'][smp['letter_of'] == int(chk['source']), :2] * 1e3
dist = np.min(np.linalg.norm(xy[:, None] - dep[None], axis=2), axis=1)
dM = lambda J: 2 * chk['m0'] * J[:, 2]                     # d sqrt(M)/dlog D_T -> dM/dlog D_T
edges = [0, 5, 10, 15, 20, 30, 40, 60, 100, 2000]
labels = ['0–5', '5–10', '10–15', '15–20', '20–30', '30–40', '40–60', '60–100', '>100']
fd = np.array([dM(chk['J_fd'])[sip & (dist >= a) & (dist < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
ad = np.array([dM(chk['J_ad'])[sip & (dist >= a) & (dist < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
x = np.arange(len(labels))
ax_a.bar(x - 0.2, fd / 1e3, 0.38, color=BLUE, label=f'finite differences: sum {fd.sum() / 1e3:+.0f}k')
ax_a.bar(x + 0.2, ad / 1e3, 0.38, color=ORANGE, label=f'LUCiD gradient (AD): sum {ad.sum() / 1e3:+.0f}k')
ax_a.axhline(0, color=MUTED, lw=1.0)
ax_a.set_xticks(x); ax_a.set_xticklabels(labels)
letter = str(smp['word'])[int(chk['source'])] if 'word' in smp else str(int(chk['source']))
ax_a.set_xlabel(f'distance of the SiPM from the {letter} strokes (mm)')
ax_a.set_ylabel('change in SiPM light per unit log D_T (10³ photons)')
ax_a.legend(frameon=False, loc='lower right', fontsize=9)
ax_a.grid(True, axis='y', color=GRID, lw=0.7); ax_a.set_axisbelow(True); clean(ax_a)
ax_a.set_title('Transverse diffusion: LUCiD\'s gradient against finite differences', loc='left',
               fontsize=11.5, pad=24)
pgt = f', pos_grad_threshold {int(chk["pos_grad_threshold"])}' if 'pos_grad_threshold' in chk else ''
ax_a.text(0, 1.02, f'{letter} letter{pgt}.  More diffusion moves light off the strokes onto their surroundings.',
          transform=ax_a.transAxes, fontsize=9, color=INK2)

# (b) EL gain: the profiled-gain gauge offset at the true parameters, by sensor brightness
logk, Q, sipm = gg['logk'], gg['Q'], gg['sipm']
M = Q[None] / np.exp(logk)                                 # the 8 forward draws at the truth, per sensor
ns = [1, 2, 4, 8]
for mask, color, name in ((sipm, BLUE, 'SiPMs'), (~sipm, ORANGE, 'PMTs')):
    means, errs = [], []
    for n in ns:
        groups = [M[i:i + n].mean(0) for i in range(0, len(M), n)]     # independent n-draw averages
        vals = [np.log(Q[mask] / g[mask]).mean() for g in groups]
        means.append(np.mean(vals)); errs.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
    ax_b.errorbar(ns, means, yerr=errs, color=color, lw=1.6, marker='o', ms=6, capsize=0, label=name)
    if name == 'SiPMs':
        for n, m_ in zip(ns, means):
            ax_b.annotate(f'{m_:+.3f}', (n, m_), textcoords='offset points', xytext=(10, 8) if n < 8 else (-10, 8),
                            ha='left' if n < 8 else 'right', fontsize=9, color=INK2)
ax_b.axhline(0, color=MUTED, lw=1.0)
ax_b.set_xscale('log', base=2); ax_b.set_xticks(ns); ax_b.set_xticklabels([str(n) for n in ns])
ax_b.set_xlabel('forward draws averaged into the model (the data average 8)')
ax_b.set_ylabel('mean profiled log-gain at the truth, log(ΣQ / ΣM)')
ax_b.legend(frameon=False, loc='upper right', fontsize=9)
ax_b.grid(True, axis='y', color=GRID, lw=0.7); ax_b.set_axisbelow(True); clean(ax_b)
ax_b.set_title('EL gain: the gain gauge mean(log k) = 0 is pulled by Monte-Carlo noise', loc='left',
               fontsize=11.5, pad=24)
ax_b.text(0, 1.02, f'True parameters, 1 draw as in the fit: all sensors average {float(logk.mean()):+.4f}, '
          f'so EL gain {np.exp(float(logk.mean())) - 1:+.1%}.',
          transform=ax_b.transAxes, fontsize=9, color=INK2)

fig.text(0.06, 0.965, 'Why the closure fit misses D_T and the EL gain', ha='left', va='top', fontsize=14.5,
         fontweight='semibold')
fig.savefig(out + '.png', dpi=130)
fig.savefig(out + '.pdf')
print('wrote', out + '.png/.pdf')
