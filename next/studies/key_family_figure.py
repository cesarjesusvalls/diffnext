"""Figure: the reflectivity pull of single forward draws, by the family of random keys they used.

Reads key_check.npz, key_families.npz, data_noise_sets.npz and the fits joint_fit_16fwd.npz and
joint_fit_loo.npz from <out dir>.

  python key_family_figure.py <out dir> <figure stem>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out, stem = sys.argv[1], sys.argv[2]
kc, kf, ns = (np.load(f'{out}/{f}.npz') for f in ('key_check', 'key_families', 'data_noise_sets'))
fit_a, fit_b = np.load(f'{out}/joint_fit_16fwd.npz'), np.load(f'{out}/joint_fit_loo.npz')

# the 64 draws of the reference family, each against the mean of all 64
names = [str(n) for n in ns['names']]
L, M = ns['log_Lcol'][..., names.index('wall reflectivity')], ns['M']
log_k = 0.1 * np.random.default_rng(6000).standard_normal(M.shape[2])
k = np.exp(log_k - log_k.mean())
pct = lambda d: 100 * 0.06 * (1 - np.exp(d)) / 0.94
ref64 = np.array([pct(np.einsum('gn,gn->', L, k[None] * (M[b] - M.mean(0)))) for b in range(64)])

cols = [('5000\n(pseudo-data)', kc['pseudo-data_keys_5000_+_131_b'])]
cols += [(b, kf[f'{b}_+_131_b']) for b in ('4999', '5001', '5002')]
cols += [('5007', kc['5007_+_131_b'])]
cols += [(b, kf[f'{b}_+_131_b']) for b in ('1000', '2000', '3000', '4000', '9000', '10000', '15000')]
cols += [('20000', kc['20000_+_131_b']), ('60000\n(64 draws)', ref64)]
others = np.concatenate([v for _, v in cols[1:-1]])

INK, INK2, MUTED, LINE, GRID = '#0b0b0b', '#52514e', '#898781', '#c3c2b7', '#e1e0d9'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': LINE,
                     'xtick.color': INK2, 'ytick.color': MUTED})
fig, ax = plt.subplots(figsize=(14, 5.8), dpi=130)
fig.subplots_adjust(left=0.07, right=0.72, top=0.80, bottom=0.16)
sd = ref64.std(ddof=1)
ax.axhspan(-sd, sd, color=GRID, alpha=0.6, lw=0, zorder=0)
ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
rng = np.random.default_rng(1)
for i, (label, v) in enumerate(cols):
    x = i + rng.uniform(-0.16, 0.16, len(v))
    color = ORANGE if i == 0 else BLUE
    ax.scatter(x, v, s=26 if i == 0 else 18, color=color, edgecolor='white', lw=0.6, zorder=3,
               alpha=1.0 if i == 0 else 0.8)
    ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=INK if i == 0 else INK2, lw=1.6, zorder=4)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels([c[0] for c in cols], fontsize=9)
ax.set_xlim(-0.6, len(cols) - 0.4)
ax.set_xlabel('family of random keys:  first key, then + 131 per draw (and + 1000 per letter)')
ax.set_ylabel('wall reflectivity pull of one draw (%)')
ax.grid(True, axis='y', color=GRID, lw=0.7)
ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

# where the fits ended, at the right
R_a = float(fit_a['real_fit'][list(map(str, fit_a['names'])).index('wall reflectivity')] / 0.94 - 1) * 100
R_b = float(fit_b['real_fit'][list(map(str, fit_b['names'])).index('wall reflectivity')] / 0.94 - 1) * 100
pd_mean = float(np.mean(cols[0][1]))
xr = len(cols) - 0.4
labels = sorted(((pd_mean, f'pseudo-data, mean of its 8 draws: {pd_mean:+.2f}%', ORANGE),
                 (R_a, f'where fit A ended (LUCiD, 16 draws): {R_a:+.2f}%', INK2),
                 (R_b, f'where fit B ended (leave-one-out, 4 draws): {R_b:+.2f}%', INK2)), key=lambda t: t[0])
y_text = [labels[0][0] - 0.06]
for y, _, _ in labels[1:]:
    y_text.append(max(y, y_text[-1] + 0.06))
for (y, txt, c), yt in zip(labels, y_text):
    ax.annotate(txt, xy=(xr, y), xytext=(xr + 0.5, yt), textcoords='data', va='center', fontsize=9, color=c,
                annotation_clip=False, arrowprops=dict(arrowstyle='-', color=c, lw=0.8, shrinkA=0, shrinkB=0))
ax.text(xr + 0.5, -sd, f'shaded: ±1 sd of a single draw ({sd:.2f}%)', va='center', fontsize=9, color=MUTED,
        clip_on=False)
ax.set_ylim(-0.55, 0.55)

fig.text(0.07, 0.955, "The pseudo-data's random keys pull the wall reflectivity", fontsize=14,
         fontweight='semibold', ha='left', va='top')
fig.text(0.07, 0.895, "Each dot: one forward draw at the truth (1M rays per letter), projected on the shift of R the "
         "closure fit would make if it were the pseudo-data\n(linearised fixed point of the fit, LUCiD gauge). "
         "lucid.fitting.closure_data draws with keys 5000 + 131 b. Bars: family means.",
         fontsize=9.5, color=INK2, ha='left', va='top', linespacing=1.45)
fig.savefig(f'{stem}.png', dpi=130)
fig.savefig(f'{stem}.pdf')
print('wrote', stem + '.png/.pdf')
