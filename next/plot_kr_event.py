"""Figures for a kr_event.py run: the anatomy of one event (drift, waveform, S1 and S2 on the
sensors) and the S2 light on the SiPMs against drift distance.

  python next/plot_kr_event.py kr.npz <outdir>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle

RUN, OUT = sys.argv[1], sys.argv[2]
d = np.load(RUN)
INK, INK2, MUTED, LINE, GRID = '#0b0b0b', '#52514e', '#898781', '#c3c2b7', '#e1e0d9'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']           # categorical slots 1-3
RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
        '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
cmap = LinearSegmentedColormap.from_list('blue_ramp', RAMP)
plt.rcParams.update({'font.size': 9.5, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': LINE,
                     'xtick.color': MUTED, 'ytick.color': MUTED})

shape = d['sensor_shape']
sipm, pmt = shape == 'square', shape == 'disc'
xyz = d['sensor_xyz'] * 1000.0
half = d['sensor_half_size'] * 1000.0
R, H = float(d['radius']) * 1000.0, float(d['height']) * 1000.0
z_sipm = H / 2
from_sipm = lambda z_m: z_sipm - z_m * 1000.0            # LUCiD z (m) -> distance from the SiPM plane (mm)
EL_TRANSIT_US = 10.0 / 2.5                             # 10 mm gap at the nexus EL drift velocity 2.5 mm/us


def waveform(i, mask, t):
    """Photons per us on the sensors in mask: S1 within 1 us of t = 0 (gas scintillation lasts
    ~0.1 us), S2 = drift time + Gaussian(longitudinal spread) convolved with the 4 us gap crossing."""
    s1 = d['s1'][i][mask].sum()
    s2 = d['s2'][i][mask].sum()
    dt = t[1] - t[0]
    w = np.zeros_like(t)
    w[np.argmin(np.abs(t))] += s1 / dt
    td, st = d['drift_time_us'][i], d['sigma_time_us'][i]
    u = np.linspace(0.0, EL_TRANSIT_US, 81)
    g = np.exp(-0.5 * ((t[:, None] - td - u[None, :]) / st) ** 2) / (st * np.sqrt(2 * np.pi))
    w += s2 * g.mean(axis=1)
    return w


def sensor_map(ax, values, mask, norm, zoom=None, deposit=None, sigma=None, draw_half=None):
    """Sensors coloured by value. draw_half enlarges the drawn SiPM squares (true size is tiny next
    to the pitch); sigma draws the 1 and 2 sigma circles of the electron cloud at the gate."""
    pts, hs = xyz[mask], half[mask] if draw_half is None else np.full(mask.sum(), draw_half)
    if mask is pmt:
        patches = [Circle(p[:2], h) for p, h in zip(pts, hs)]
    else:
        patches = [Rectangle((p[0] - h, p[1] - h), 2 * h, 2 * h) for p, h in zip(pts, hs)]
    pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor='none')
    pc.set_array(np.clip(values[mask], norm.vmin, None))
    ax.add_collection(pc)
    ax.add_patch(Circle((0, 0), R, fill=False, edgecolor=LINE, lw=1.0))
    if sigma is not None:
        for k_sig, ls in ((1, '-'), (2, '--')):
            ax.add_patch(Circle(deposit[:2], k_sig * sigma, fill=False, edgecolor=SERIES[1], lw=1.4, ls=ls))
    if deposit is not None:
        ax.plot(*deposit[:2], marker='x', color=INK, ms=8, mew=1.8)
    lim = zoom if zoom is not None else (-R - 30, R + 30, -R - 30, R + 30)
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    return pc


# ---------------- Figure 1: one event ----------------
i = len(d['drift_m']) // 2
dep = d['deposit'][i] * 1000.0
fig = plt.figure(figsize=(14.5, 9.4))
gs = fig.add_gridspec(2, 3, height_ratios=[0.85, 1.0], hspace=0.38, wspace=0.28,
                      left=0.055, right=0.975, top=0.9, bottom=0.07)

ax = fig.add_subplot(gs[0, :2])
el0, el1 = from_sipm(d['el_gap_z'][1]), from_sipm(d['el_gap_z'][0])
ax.axvspan(el0, el1, color=SERIES[2], alpha=0.25, lw=0)
for zpos, lab, ls in ((0.0, 'SiPM plane', '-'), (from_sipm(d['gate_z']), 'gate', '--'),
                      (from_sipm(d['cathode_z']), 'cathode', '--'), (H, 'PMT plane', '-')):
    ax.axvline(zpos, color=MUTED if ls == '-' else LINE, lw=1.0, ls=ls)
    ax.text(zpos + (6 if lab == 'gate' else -6), R + 25, lab, ha='left' if lab == 'gate' else 'right',
            va='bottom', fontsize=8.5, color=INK2)
ax.text((el0 + el1) / 2, -R - 40, 'EL gap', ha='center', va='top', fontsize=8.5, color=INK2)
ax.axhline(R, color=MUTED, lw=1.0); ax.axhline(-R, color=MUTED, lw=1.0)
zdep = from_sipm(dep[2] / 1000.0)
ax.annotate('', xy=(from_sipm(d['gate_z']) + 4, dep[0]), xytext=(zdep, dep[0]),
            arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.4))
ax.plot(zdep, dep[0], 'o', color=INK, ms=7)
ax.text(zdep + 25, dep[0], f"{d['energy_MeV']*1e3:.1f} keV deposit, {d['drift_m'][i]:.1f} m from the gate",
        ha='left', va='center', fontsize=9, color=INK)
ax.text(from_sipm(d['gate_z']) + 40, dep[0] - 60,
        f"{d['n_electrons'][i]:,.0f} electrons drift {d['drift_time_us'][i]:.0f} µs to the gate (JAXTPC);\n"
        f"cloud σ_T = {d['sigma_trans_m'][i]*1e3:.1f} mm across, σ_t = {d['sigma_time_us'][i]:.1f} µs in time",
        ha='left', va='top', fontsize=9, color=INK2)
ax.set_xlim(-60, H + 30); ax.set_ylim(-R - 120, R + 90)
ax.set_xlabel('distance from the SiPM plane (mm)'); ax.set_ylabel('x (mm)')
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('The event, side view', loc='left', fontsize=11, color=INK)

ax = fig.add_subplot(gs[0, 2])
t = np.arange(-30.0, d['drift_time_us'][i] + 40.0, 0.25)
for mask, lab, col in ((pmt, '60 PMTs', SERIES[1]), (sipm, '3130 SiPMs', SERIES[0])):
    w = waveform(i, mask, t)
    ax.semilogy(t, np.maximum(w, 1e-3), color=col, lw=1.4)
    k = np.argmax(w)
    ax.text(t[k] - 25, w[k], lab, color=INK2, fontsize=8.5, ha='right', va='center')
ax.set_ylim(1e0, 4e6); ax.set_xlim(t[0], t[-1])
ax.text(8, 1.6e6, 'S1', fontsize=9.5, color=INK, fontweight='semibold')
ax.text(d['drift_time_us'][i] - 25, 1.6e6, 'S2', fontsize=9.5, color=INK, ha='right', fontweight='semibold')
ax.set_xlabel('time after the deposit (µs)'); ax.set_ylabel('photons per µs')
ax.grid(True, which='major', color=GRID, lw=0.7); ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Light against time', loc='left', fontsize=11, color=INK)

s2 = d['s2'][i]
ax = fig.add_subplot(gs[1, 0])
zoom = (dep[0] - 75, dep[0] + 75, dep[1] - 75, dep[1] + 75)
pc = sensor_map(ax, s2, sipm, LogNorm(1.0, s2[sipm].max()), zoom, dep, d['sigma_trans_m'][i] * 1000.0, 6.5)
cb = fig.colorbar(pc, ax=ax, shrink=0.8, pad=0.02); cb.outline.set_visible(False)
cb.set_label('S2 photons per SiPM', color=INK2)
ax.set_title("S2 on the SiPMs (zoom)", loc='left', fontsize=11, color=INK)
ax.text(0.0, -0.2, 'SiPMs drawn as 13 mm squares (1.3 mm real);\norange: electron cloud at the gate, 1σ and 2σ',
        transform=ax.transAxes, fontsize=8, color=INK2, va='top')

ax = fig.add_subplot(gs[1, 1])
pc = sensor_map(ax, s2, pmt, Normalize(s2[pmt].min(), s2[pmt].max()), None, dep)
cb = fig.colorbar(pc, ax=ax, shrink=0.8, pad=0.02); cb.outline.set_visible(False)
cb.set_label('S2 photons per PMT', color=INK2)
ax.set_title(f"S2 on the PMTs: {s2[pmt].sum():,.0f} of {d['n_s2_photons'][i]:,.0f}", loc='left', fontsize=11, color=INK)

s1 = d['s1'][i]
ax = fig.add_subplot(gs[1, 2])
pc = sensor_map(ax, s1, pmt, Normalize(s1[pmt].min(), s1[pmt].max()), None, dep)
cb = fig.colorbar(pc, ax=ax, shrink=0.8, pad=0.02); cb.outline.set_visible(False)
cb.set_label('S1 photons per PMT', color=INK2)
n_s1 = d['energy_MeV'] * d['s1_yield']
ax.set_title(f"S1 on the PMTs: {s1[pmt].sum():,.0f} of {n_s1:,.0f}", loc='left', fontsize=11, color=INK)
fig.suptitle(f"Point deposit in the NEXT-like detector: JAXTPC drift + LUCiD light  ·  S1 on the SiPMs {s1[sipm].sum():.1f}, "
             f"S2 on the SiPMs {s2[sipm].sum():,.0f} photons", x=0.055, ha='left', fontsize=12.5, fontweight='semibold')
fig.savefig(f'{OUT}/kr_event_anatomy.png', dpi=150); fig.savefig(f'{OUT}/kr_event_anatomy.pdf'); plt.close(fig)

# ---------------- Figure 2: S2 on the SiPMs against drift distance ----------------
n = len(d['drift_m'])
fig, axes = plt.subplots(1, n + 1, figsize=(4.1 * (n + 1), 4.6), gridspec_kw=dict(width_ratios=[1] * n + [1.25]))
vmax = max(d['s2'][k][sipm].max() for k in range(n))
norm = LogNorm(1.0, vmax)
for k in range(n):
    depk = d['deposit'][k] * 1000.0
    zoom = (depk[0] - 75, depk[0] + 75, depk[1] - 75, depk[1] + 75)
    pc = sensor_map(axes[k], d['s2'][k], sipm, norm, zoom, depk, d['sigma_trans_m'][k] * 1000.0, 6.5)
    axes[k].set_title(f"{d['drift_m'][k]:.1f} m from the gate, σ_T {d['sigma_trans_m'][k]*1e3:.1f} mm",
                      loc='left', fontsize=10.5, color=INK)
cb = fig.colorbar(pc, ax=axes[:n], location='bottom', shrink=0.5, pad=0.3, aspect=40)
cb.set_label('S2 photons per SiPM', color=INK2); cb.outline.set_visible(False)
ax = axes[n]
for k in range(n):
    depk = d['deposit'][k] * 1000.0
    rr = np.hypot(xyz[sipm, 0] - depk[0], xyz[sipm, 1] - depk[1])
    order = np.argsort(rr)
    ax.semilogy(rr[order], d['s2'][k][sipm][order], 'o', ms=3.2, color=SERIES[k], alpha=0.85,
                label=f"{d['drift_m'][k]:.1f} m")
ax.set_xlim(0, 80); ax.set_ylim(1, vmax * 2)
ax.set_xlabel('SiPM distance from the deposit (x, y) (mm)'); ax.set_ylabel('S2 photons per SiPM')
ax.grid(True, which='major', color=GRID, lw=0.7); ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(title='drift', frameon=False, fontsize=9, title_fontsize=9)
ax.set_title('Spot profile', loc='left', fontsize=10.5, color=INK)
fig.suptitle('S2 on the SiPMs: the electron cloud diffuses as it drifts  (SiPMs drawn as 13 mm squares; '
             'orange: cloud at the gate, 1σ and 2σ)', x=0.02, ha='left', fontsize=12, fontweight='semibold')
fig.subplots_adjust(left=0.04, right=0.985, top=0.83, bottom=0.3, wspace=0.3)
fig.savefig(f'{OUT}/kr_s2_vs_drift.png', dpi=150); fig.savefig(f'{OUT}/kr_s2_vs_drift.pdf'); plt.close(fig)
print('wrote', f'{OUT}/kr_event_anatomy.{{png,pdf}}', f'{OUT}/kr_s2_vs_drift.{{png,pdf}}')

# ---------------- Figure 3: the whole SiPM plane ----------------
# Every SiPM drawn as a 13 mm square on the 15.55 mm pitch (real size 1.3 mm), so the readout reads
# as an image with small gaps: S2 for each drift distance on one linear scale, S1 at the middle
# drift on its own.
PIXEL = 13.0


def plane_map(ax, values, norm, deposit):
    pts = xyz[sipm]
    patches = [Rectangle((p[0] - PIXEL / 2, p[1] - PIXEL / 2), PIXEL, PIXEL) for p in pts]
    pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor='none')
    pc.set_array(np.clip(values[sipm], norm.vmin, None))
    ax.add_collection(pc)
    ax.add_patch(Circle((0, 0), R, fill=False, edgecolor=LINE, lw=1.0))
    ax.plot(*deposit[:2], marker='x', color='white', ms=9, mew=2.4)
    ax.plot(*deposit[:2], marker='x', color=INK, ms=7, mew=1.3)
    ax.set_xlim(-R - 15, R + 15); ax.set_ylim(-R - 15, R + 15); ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel('x (mm)')
    return pc


fig = plt.figure(figsize=(18.5, 6.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.045], hspace=0.28, wspace=0.12,
                      left=0.035, right=0.99, top=0.86, bottom=0.1)
axes = [fig.add_subplot(gs[0, k]) for k in range(4)]
s2max = max(d['s2'][k][sipm].max() for k in range(n))
norm2 = Normalize(0.0, s2max)
for k in range(min(n, 3)):
    pc2 = plane_map(axes[k], d['s2'][k], norm2, d['deposit'][k] * 1000.0)
    axes[k].set_title(f"S2, {d['drift_m'][k]:.1f} m from the gate: {d['s2'][k][sipm].sum():,.0f} photons",
                      loc='left', fontsize=11, color=INK)
axes[0].set_ylabel('y (mm)')
s1 = d['s1'][i]
pc1 = plane_map(axes[3], s1, Normalize(0.0, s1[sipm].max()), dep)
axes[3].set_title(f"S1, {d['drift_m'][i]:.1f} m from the gate: {s1[sipm].sum():.1f} photons",
                  loc='left', fontsize=11, color=INK)
cb = fig.colorbar(pc2, cax=fig.add_subplot(gs[1, 0:3]), orientation='horizontal')
cb.set_label('S2 photons per SiPM', color=INK2); cb.outline.set_visible(False)
cb = fig.colorbar(pc1, cax=fig.add_subplot(gs[1, 3]), orientation='horizontal')
cb.set_label('S1 photons per SiPM', color=INK2); cb.outline.set_visible(False)
fig.suptitle(f"The whole SiPM plane, {d['energy_MeV']*1e3:.1f} keV deposit at x = {dep[0]:.0f}, y = {dep[1]:.0f} mm  "
             "(each SiPM drawn as a 13 mm square on the 15.55 mm pitch; real size 1.3 mm)", x=0.035, ha='left', fontsize=12.5,
             fontweight='semibold')
fig.savefig(f'{OUT}/kr_sipm_plane.png', dpi=150); fig.savefig(f'{OUT}/kr_sipm_plane.pdf'); plt.close(fig)
print('wrote', f'{OUT}/kr_sipm_plane.{{png,pdf}}')
