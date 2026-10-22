"""Figures and a video of a charge_calibration.py closure fit.

  python next/plot_charge_calibration.py charge_fit.npz <out stem> [--ffmpeg BIN] [--ref earlier.npz]

<out stem>.png/.pdf: each parameter's fitted / truth - 1 per Gauss-Newton step (with an earlier
fit in grey, given --ref, for the parameters both fitted), the summed S2 waveform at the start and
at the fit against the pseudo-data, the profiled per-sensor gains against the truth (when the
pseudo-data had a gain spread), and the two loss terms.
<out stem>.mp4 (with --ffmpeg): the waveform and the parameters, step by step.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('out')
ap.add_argument('--ffmpeg', help='ffmpeg binary, to write the MP4')
ap.add_argument('--ref', help='an earlier fit of the same problem, drawn in grey')
ap.add_argument('--ref-label', default='earlier fit')
args = ap.parse_args()
OUT, FFMPEG = args.out, args.ffmpeg
d = np.load(args.run)
ref = np.load(args.ref) if args.ref else None
INK, INK2, MUTED, LINE, GRID = '#0b0b0b', '#52514e', '#898781', '#c3c2b7', '#e1e0d9'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': LINE,
                     'xtick.color': MUTED, 'ytick.color': MUTED})

# Reading units: the fit's are cm, us and m.
SCALE = {'drift velocity': (10, 'mm/µs'), 'electron lifetime': (1e-3, 'ms'),
         'transverse diffusion': (1e6, 'cm²/s'), 'longitudinal diffusion': (1e6, 'cm²/s'),
         'EL gain': (1, 'photons/e⁻'), 'wall reflectivity': (1, ''), 'absorption length': (1, 'm')}


def fractions(f):
    """Each parameter's fitted / truth - 1 per step, by name, against what the estimator should
    return (``real_target``, when the run stored it). Runs from before the optics were fitted store
    only theta, the log of every parameter."""
    names = [str(n) for n in f['names']]
    if 'real_history' in f:
        frac = f['real_history'] / f['real_target' if 'real_target' in f else 'real_true'] - 1.0
    else:
        frac = np.exp(f['history'] - f['theta_true']) - 1.0
    return {n: frac[:, i] for i, n in enumerate(names)}


names = [str(n) for n in d['names']]
P = len(names)
frac = fractions(d)
real = lambda k: d['real_' + k] if 'real_' + k in d else np.exp(d['theta_true' if k == 'true' else k])
p_true, p_fit = (np.array([v * SCALE[n][0] for n, v in zip(names, real(k))])
                 for k in ('target' if 'real_target' in d else 'true', 'fit'))
fit = p_fit / p_true - 1.0
steps = np.arange(len(d['history']))
n_steps = len(steps) - 1
polyak = 10
draws = lambda f, k: int(f[k]) if k in f else 1
ref_frac = None if ref is None else fractions(ref)
t_mid = 0.5 * (d['edges_us'][1:] + d['edges_us'][:-1])
has_gains = 'k_true' in d and float(d['gain_spread']) > 0
optics = any(n in ('wall reflectivity', 'absorption length') for n in names)


def clean(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def param_panels(fig, gs_row):
    axes = []
    for i, n in enumerate(names):
        ax = fig.add_subplot(gs_row[0, i])
        ax.axhline(0.0, color=MUTED, lw=1.0)
        ax.axvspan(n_steps - polyak + 0.5, n_steps + 0.5, color=GRID, alpha=0.45, lw=0)
        ax.set_xlim(-0.5, n_steps + 0.5)
        shown = frac[n] if ref_frac is None or n not in ref_frac else np.r_[frac[n], ref_frac[n]]
        lim = max(0.005, 1.15 * np.abs(shown).max())
        ax.set_ylim(-lim * 100, lim * 100)
        ax.set_title(n, loc='left', fontsize=10.5, pad=20)
        ax.set_xlabel('Gauss-Newton step')
        ax.grid(True, axis='y', color=GRID, lw=0.7); ax.set_axisbelow(True)
        clean(ax)
        axes.append(ax)
    axes[0].set_ylabel('fitted / truth − 1 (%)')
    return axes


fig = plt.figure(figsize=(16 * max(1.0, P / 5.5), 9), dpi=120)
top = fig.add_gridspec(1, P, left=0.05, right=0.985, top=0.775, bottom=0.53, wspace=0.32)
bot = fig.add_gridspec(1, 4 if has_gains else 3, left=0.05, right=0.985, top=0.42, bottom=0.08,
                       wspace=0.30)
axes = param_panels(fig, top)
for i, (n, ax) in enumerate(zip(names, axes)):
    if ref_frac is not None and n in ref_frac:
        ax.plot(np.arange(len(ref_frac[n])), ref_frac[n] * 100, color=LINE, lw=1.4, marker='o', ms=2.5,
                label=args.ref_label)
    ax.plot(steps, frac[n] * 100, color=BLUE, lw=1.6, marker='o', ms=3.5, label='this fit')
    unit = ' ' + SCALE[n][1] if SCALE[n][1] else ''
    ax.text(0.0, 1.03, f'truth {p_true[i]:.4g}{unit},  fit {p_fit[i]:.4g} ({fit[i]:+.2%})',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=8.5, color=INK2)

if ref_frac is not None:
    shared = [ax for n, ax in zip(names, axes) if n in ref_frac]
    fig.legend(*shared[0].get_legend_handles_labels(), loc='upper right', bbox_to_anchor=(0.985, 0.875), ncol=2,
               frameon=False, fontsize=9)

ax_w = fig.add_subplot(bot[0, 0:2])
ax_w.plot(t_mid, d['wave_obs'] / 1e6, color=INK, lw=2.4, label='pseudo-data')
ax_w.plot(t_mid, d['wave_start'] / 1e6, color=ORANGE, lw=1.4, label='start')
ax_w.plot(t_mid, d['wave_fit'] / 1e6, color=BLUE, lw=1.4, ls='--', label='fit')
ax_w.set_xlim(t_mid[0], t_mid[-1]); ax_w.set_ylim(0, None)
ax_w.set_xlabel('time after the deposits (µs)'); ax_w.set_ylabel('S2, all sensors (10⁶ photons per µs)')
ax_w.legend(frameon=False, loc='upper right')
ax_w.grid(True, axis='y', color=GRID, lw=0.7); ax_w.set_axisbelow(True); clean(ax_w)
ax_w.set_title('S2 waveform of the four letters (1 µs bins), summed over sensors', loc='left', fontsize=11.5)

if has_gains:
    ax_g = fig.add_subplot(bot[0, 2])
    sipm, k_true, k_fit = d['sipm'], d['k_true'], d['gains_fit']
    lo, hi = 0.97 * min(k_true.min(), k_fit.min()), 1.03 * max(k_true.max(), k_fit.max())
    ax_g.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, zorder=1)
    ax_g.scatter(k_true[sipm], k_fit[sipm], s=3, color=BLUE, alpha=0.35, lw=0, label=f'SiPMs ({sipm.sum()})',
                 zorder=2)
    ax_g.scatter(k_true[~sipm], k_fit[~sipm], s=22, color=ORANGE, edgecolor='white', lw=0.8,
                 label=f'PMTs ({(~sipm).sum()})', zorder=3)
    ax_g.set_xlim(lo, hi); ax_g.set_ylim(lo, hi); ax_g.set_aspect('equal', adjustable='box')
    ax_g.set_xlabel('truth gain'); ax_g.set_ylabel('profiled gain at the fit')
    ax_g.legend(frameon=False, loc='upper left', fontsize=9, markerscale=2)
    rms = lambda k, m: 100 * np.std(k[m] / k_true[m] - 1)
    ax_g.text(0.98, 0.03, f"fitted / truth − 1, rms\nSiPMs {rms(k_fit, sipm):.1f}%  (at the truth "
              f"{rms(d['gains_at_truth'], sipm):.1f}%)\nPMTs {rms(k_fit, ~sipm):.2f}%  (at the truth "
              f"{rms(d['gains_at_truth'], ~sipm):.2f}%)", transform=ax_g.transAxes, ha='right', va='bottom',
              fontsize=8.5, color=INK2, linespacing=1.4)
    ax_g.grid(True, color=GRID, lw=0.7); ax_g.set_axisbelow(True); clean(ax_g)
    ax_g.set_title(f"Per-sensor gains (truth spread {float(d['gain_spread']):.0%})", loc='left', fontsize=11.5)

ax_l = fig.add_subplot(bot[0, 3 if has_gains else 2])
terms = d['terms']
ax_l.semilogy(np.arange(len(terms)), terms[:, 0], color=BLUE, lw=1.6, marker='o', ms=3, label='S2 charge (Neyman χ²)')
ax_l.semilogy(np.arange(len(terms)), float(d['w_time']) * terms[:, 1], color=ORANGE, lw=1.6, marker='o', ms=3,
              label='S2 waveform (Sobolev, weighted)')
ax_l.set_xlabel('Gauss-Newton step'); ax_l.set_ylabel('loss term')
ax_l.legend(frameon=False, loc='center right', fontsize=9)
ax_l.grid(True, which='major', axis='y', color=GRID, lw=0.7); ax_l.set_axisbelow(True); clean(ax_l)
ax_l.set_title('Loss terms', loc='left', fontsize=11.5)

dt_over_v = 'hold_dt_over_v' in d and bool(d['hold_dt_over_v'])
held = ['transverse diffusion / drift velocity' if n == 'transverse diffusion' and dt_over_v else n
        for n in SCALE if n not in names and n not in ('wall reflectivity', 'absorption length')]
absorption = float(d['absorption_length_m']) if 'absorption_length_m' in d else None
loo = str(d['estimator']) == 'leave-one-out' if 'estimator' in d else False
i_gain = names.index('EL gain') if 'EL gain' in names else None
fig.text(0.05, 0.965, 'Closure fit of the charge transport, EL gain and optics on the S2 of the "NEXT" sample'
         if optics else 'Closure fit of the charge transport and EL gain on the S2 of the "NEXT" sample',
         ha='left', va='top', fontsize=14.5, fontweight='semibold')
fig.text(0.05, 0.925, f"4 letters (166 deposits of 41.5 keV), {int(d['n_rays']) // 1000}k rays per letter, "
         f"truth lifetime {float(d['lifetime_ms']):.0f} ms"
         + (f";  held at the truth: {', '.join(held)}" if held else '')
         + (f";  truth optics: wall reflectivity {float(d['wall_reflectivity']):.3g}, absorption length "
            + ('none' if absorption >= 1e6 else f"{absorption:.3g} m") if optics else '')
         + ".\nLUCiD damped Gauss-Newton on the S2 charge per sensor (3130 SiPMs + 60 PMTs, "
         + ("each forward draw's gains from the other draws, LeaveOneOutProblem" if loo else "gains profiled")
         + ") plus JAXTPC's Sobolev loss on the waveform.\n"
         + (f"Wall reflectivity R fitted as log(1 − R).  " if 'wall reflectivity' in names else '')
         + (f"EL gain against the truth's × {float(d['real_target'][i_gain] / d['real_true'][i_gain]):.4f}, "
            f"the light-weighted mean truth gain (that estimator's gauge).  " if loo and i_gain is not None else '')
         + f"{draws(d, 'forward_draws')} forward and {draws(d, 'jacobian_draws')} Jacobian draws per step.  "
         + f"Pseudo-data keys {int(d['data_key']) if 'data_key' in d else 5000} + 131 b.  "
         f"Shaded: the last {polyak} steps, averaged for the fit.", ha='left', va='top', fontsize=9.5, color=INK2,
         linespacing=1.5)
fig.savefig(OUT + '.png', dpi=130)
fig.savefig(OUT + '.pdf')
print('wrote', OUT + '.png/.pdf')

if FFMPEG:
    plt.rcParams['animation.ffmpeg_path'] = FFMPEG
    fig = plt.figure(figsize=(16 * max(1.0, P / 5.5), 9), dpi=120)
    top = fig.add_gridspec(1, P, left=0.05, right=0.985, top=0.83, bottom=0.55, wspace=0.32)
    bot = fig.add_gridspec(1, 1, left=0.05, right=0.985, top=0.43, bottom=0.08)
    axes = param_panels(fig, top)
    lines = [ax.plot([], [], color=BLUE, lw=1.6, marker='o', ms=3.5)[0] for ax in axes]
    ax_w = fig.add_subplot(bot[0, 0])
    ax_w.plot(t_mid, d['wave_obs'] / 1e6, color=INK, lw=2.4, label='pseudo-data')
    model, = ax_w.plot([], [], color=BLUE, lw=1.4, label='model at this step')
    ax_w.set_xlim(t_mid[0], t_mid[-1])
    ax_w.set_ylim(0, 1.1 * max(d['wave_obs'].max(), d['wave_hist'].max()) / 1e6)
    ax_w.set_xlabel('time after the deposits (µs)'); ax_w.set_ylabel('S2, all sensors (10⁶ photons per µs)')
    ax_w.legend(frameon=False, loc='upper right')
    ax_w.grid(True, axis='y', color=GRID, lw=0.7); ax_w.set_axisbelow(True); clean(ax_w)
    title = fig.text(0.05, 0.965, '', ha='left', va='top', fontsize=14.5, fontweight='semibold')
    fig.text(0.05, 0.925, 'LUCiD damped Gauss-Newton on the S2 charge per sensor plus JAXTPC\'s Sobolev loss on the '
             'S2 waveform; parameters shown as fitted / truth − 1.', ha='left', va='top', fontsize=9.5, color=INK2)

    def update(k):
        for n, ln in zip(names, lines):
            ln.set_data(steps[:k + 1], frac[n][:k + 1] * 100)
        model.set_data(t_mid, d['wave_hist'][k] / 1e6)
        title.set_text(f'Closure fit on the "NEXT" sample: Gauss-Newton step {k} of {n_steps}')

    frames = [0] * 4 + list(steps) + [n_steps] * 8
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(OUT + '.mp4', writer=animation.FFMpegWriter(fps=3, bitrate=4000,
                                                          extra_args=['-pix_fmt', 'yuv420p']))
    print('wrote', OUT + '.mp4')
