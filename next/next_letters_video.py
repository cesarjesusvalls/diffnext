"""Video of a next_letters.py run: the ionization of "NEXT" drifts to the EL gap and the S2 image
builds up on the SiPM plane, letter by letter. Time runs 10x slower while a letter crosses the gap.

  python next/next_letters_video.py letters.npz <out stem> <ffmpeg binary>

The drifting electrons drawn are a few samples per deposit, spread with the JAXTPC transverse and
longitudinal widths; the light is the S2 per SiPM of each deposit, spread in time over the
electrons' arrival (longitudinal diffusion) and their crossing of the gap.
"""
import math
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

try:
    from scipy.special import ndtr
except ImportError:
    ndtr = np.vectorize(lambda z: 0.5 * math.erfc(-z / math.sqrt(2.0)))

RUN, OUT, FFMPEG = sys.argv[1], sys.argv[2], sys.argv[3]
plt.rcParams['animation.ffmpeg_path'] = FFMPEG
d = np.load(RUN)
INK, INK2, MUTED, LINE, GRID, GHOST = '#0b0b0b', '#52514e', '#898781', '#c3c2b7', '#e1e0d9', '#cfcec6'
BLUE, AQUA = '#2a78d6', '#1baf7a'
RAMP = ['#f4f8fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#2a78d6', '#1c5cab', '#104281', '#0d366b']
cmap = LinearSegmentedColormap.from_list('blue_ramp', RAMP)
plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': LINE,
                     'xtick.color': MUTED, 'ytick.color': MUTED})

word = str(d['word'])
dep = d['deposits'] * 1000.0                      # (P, 3) mm, LUCiD frame
letter = d['letter_of']
P = len(dep)
sipm = d['sensor_shape'] == 'square'
sxy = d['sensor_xyz'][sipm, :2] * 1000.0
s2 = d['s2'][:, sipm].astype(np.float64)          # (P, n_sipm)
R, H = float(d['radius']) * 1000.0, float(d['height']) * 1000.0
z_sipm = H / 2
gate, cath = float(d['gate_z']) * 1000.0, float(d['cathode_z']) * 1000.0
el0, el1 = z_sipm - d['el_gap_z'][1] * 1000.0, z_sipm - d['el_gap_z'][0] * 1000.0   # from the SiPM plane
v = float(d['velocity_mm_us'])                    # mm/us
D_T = float(d['D_T'])                             # cm^2/us
td, st = d['drift_time_us'], d['sigma_time_us']
T_EL = 10.0 / 2.5                                 # gap crossing at the nexus EL drift velocity (us)

# ---- frame times: 4 us per frame while drifting, 0.4 us while a letter crosses the gap ----
COARSE, FINE = 4.0, 0.4
windows = [(td[letter == k].min() - 3 * st[letter == k].max() - 1.0,
            td[letter == k].max() + T_EL + 3 * st[letter == k].max() + 1.0) for k in range(len(word))]
in_window = lambda tq: np.any([(tq >= a) & (tq <= b) for a, b in windows], axis=0)
coarse = np.arange(0.0, td.max() + T_EL + 3 * st.max() + 30.0, COARSE)
t = np.sort(np.concatenate([coarse[~in_window(coarse)]] + [np.arange(a, b, FINE) for a, b in windows]))
slow = in_window(t)

# ---- S2 emitted per deposit: arrival ~ N(td, st), then uniform emission over the gap crossing ----
u = (np.arange(40) + 0.5) / 40 * T_EL


def emitted(tq):
    """Fraction of each deposit's S2 emitted by each time in tq, and the rate (per us): (T, P)."""
    z = (tq[:, None, None] - td[None, :, None] - u) / st[None, :, None]
    return ndtr(z).mean(axis=2), (np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)).mean(axis=2) / st


F, f = emitted(t)
cum, now = F @ s2, f @ s2                         # (T, n_sipm): photons so far, photons per us
tot = s2.sum(axis=1)                              # S2 photons on the SiPMs per deposit
centre = [td[letter == k].mean() + T_EL / 2 for k in range(len(word))]
pulse_t = [np.linspace(c - 14.0, c + 14.0, 400) for c in centre]
pulse_r = [emitted(tq)[1] @ tot / 1e6 for tq in pulse_t]

# ---- a few sample electrons per deposit, for the drift views ----
rng = np.random.default_rng(3)
NS = 10
nrm = rng.standard_normal((P, NS, 3))
arrive = td[:, None] - st[:, None] * nrm[..., 2]  # electrons ahead of the cloud centre arrive first


def electrons(tt):
    """Sample electrons at time tt: (x, y) and distance from the SiPM plane (mm), with masks of
    those still drifting and of those crossing the EL gap."""
    s = np.clip(np.minimum(tt, td), 0.0, None)
    sig_t = np.sqrt(2.0 * D_T * s) * 10.0                                   # mm
    xy = dep[:, None, :2] + sig_t[:, None, None] * nrm[..., :2]
    z = dep[:, None, 2] + v * s[:, None] + (st * v * np.sqrt(s / td))[:, None] * nrm[..., 2]
    dist = np.maximum(z_sipm - z, z_sipm - gate)
    gap_dist = el1 - (el1 - el0) * (tt - arrive) / T_EL
    return xy, dist, gap_dist, tt < arrive, (tt >= arrive) & (tt < arrive + T_EL)


def clean(ax, keep=('left', 'bottom')):
    for name, sp in ax.spines.items():
        sp.set_visible(name in keep)


fig = plt.figure(figsize=(16, 9), dpi=120)
gs = fig.add_gridspec(2, 3, height_ratios=[0.6, 1.0], hspace=0.3, wspace=0.16,
                      left=0.05, right=0.955, top=0.87, bottom=0.04)

# side view: x against distance from the SiPM plane
ax_s = fig.add_subplot(gs[0, 0:2])
ax_s.axvspan(el0, el1, color=AQUA, alpha=0.3, lw=0)
ax_s.axvline(0.0, color=MUTED, lw=1.0); ax_s.axvline(H, color=MUTED, lw=1.0)
ax_s.axvline(z_sipm - gate, color=LINE, lw=1.0, ls='--'); ax_s.axvline(z_sipm - cath, color=LINE, lw=1.0, ls='--')
ax_s.axhline(R, color=MUTED, lw=1.0); ax_s.axhline(-R, color=MUTED, lw=1.5)
ax_s.text(z_sipm - gate + 8, R + 12, '← SiPM plane, EL gap, gate', ha='left', va='bottom', fontsize=9, color=INK2)
ax_s.text(z_sipm - cath - 8, R + 12, 'cathode', ha='right', va='bottom', fontsize=9, color=INK2)
ax_s.text(H - 8, R + 12, 'PMT plane', ha='right', va='bottom', fontsize=9, color=INK2)
ax_s.scatter(z_sipm - dep[:, 2], dep[:, 0], s=10, color=GHOST, lw=0)
for k, ch in enumerate(word):
    m = letter == k
    ax_s.text(z_sipm - dep[m, 2].mean() + 22, dep[m, 0].max() + 25, ch, fontsize=13, fontweight='bold',
              color=MUTED, ha='left', va='bottom')
side = ax_s.scatter([], [], s=4, color=INK, alpha=0.6, lw=0)
side_gap = ax_s.scatter([], [], s=9, color=AQUA, lw=0)
ax_s.legend(handles=[Line2D([], [], ls='', marker='o', ms=4, color=GHOST, label='deposits, t = 0'),
                     Line2D([], [], ls='', marker='o', ms=3, color=INK, label='drifting electrons (samples)'),
                     Line2D([], [], ls='', marker='o', ms=4, color=AQUA, label='crossing the EL gap: S2 light')],
            loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=9,
            handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
ax_s.set_xlim(-25, H + 15); ax_s.set_ylim(-R, R + 90)
ax_s.set_xlabel('distance from the SiPM plane (mm)'); ax_s.set_ylabel('x (mm)')
clean(ax_s)
ax_s.set_title('Side view: the electrons drift to the EL gap', loc='left', fontsize=11.5)

# S2 pulses, one panel per letter, on a common scale
inner = gs[0, 2].subgridspec(1, len(word), wspace=0.1)
ax_p = [fig.add_subplot(inner[0, k]) for k in range(len(word))]
ymax = max(r.max() for r in pulse_r) * 1.12
cursors, progress = [], []
for k, ax in enumerate(ax_p):
    ax.plot(pulse_t[k], pulse_r[k], color=LINE, lw=1.4)
    progress.append(ax.plot([], [], color=BLUE, lw=1.8)[0])
    cursors.append(ax.axvline(-1e9, color=INK, lw=0.9))
    ax.set_xlim(pulse_t[k][0], pulse_t[k][-1]); ax.set_ylim(0, ymax)
    c = 10 * round(td[letter == k].mean() / 10)
    ax.set_xticks([c - 10, c, c + 10])
    ax.tick_params(labelsize=8.5, labelleft=(k == 0))
    ax.grid(True, axis='y', color=GRID, lw=0.7); ax.set_axisbelow(True)
    clean(ax)
    ax.text(0.06, 0.98, word[k], transform=ax.transAxes, ha='left', va='top', fontsize=12, fontweight='bold',
            color=MUTED)
ax_p[0].set_ylabel('S2 on the SiPMs (10⁶ photons/µs)')
p0, p1 = ax_p[0].get_position(), ax_p[-1].get_position()
fig.text((p0.x0 + p1.x1) / 2, p0.y0 - 0.052, 'time after the deposits (µs)', ha='center', color=INK2)
fig.text(p0.x0, ax_s.get_position().y1 + 0.012, 'S2 pulse of each letter on the SiPMs', ha='left',
         va='bottom', fontsize=11.5)


def plane_axes(ax, title):
    ax.add_patch(Circle((0, 0), R, fill=False, edgecolor=LINE, lw=1.0, zorder=3))
    ax.set_xlim(-R - 15, R + 15); ax.set_ylim(-R - 15, R + 15); ax.set_aspect('equal')
    clean(ax, keep=())
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc='left', fontsize=11.5)


ax_t = fig.add_subplot(gs[1, 0])
plane_axes(ax_t, 'Top view: electrons on their way')
ax_t.scatter(dep[:, 0], dep[:, 1], s=10, color=GHOST, lw=0)
top = ax_t.scatter([], [], s=4, color=INK, alpha=0.6, lw=0)
top_gap = ax_t.scatter([], [], s=9, color=AQUA, lw=0)

PIXEL = 13.0                                       # SiPMs drawn as 13 mm squares on the 15.55 mm pitch
squares = lambda: [Rectangle((p[0] - PIXEL / 2, p[1] - PIXEL / 2), PIXEL, PIXEL) for p in sxy]
maps = []
for col, (title_, data, unit) in enumerate((('S2 collected so far on the SiPM plane', cum, 'photons per SiPM'),
                                            ('S2 arriving now', now, 'photons per SiPM per µs')), start=1):
    ax = fig.add_subplot(gs[1, col])
    plane_axes(ax, title_)
    pc = PatchCollection(squares(), cmap=cmap, norm=Normalize(0.0, data.max()), edgecolor='none')
    pc.set_array(data[0]); ax.add_collection(pc)
    cb = fig.colorbar(pc, ax=ax, fraction=0.045, pad=0.02); cb.outline.set_visible(False)
    cb.set_label(unit, color=INK2)
    maps.append((pc, data))

title = fig.text(0.05, 0.975, '', ha='left', va='top', fontsize=14.5, fontweight='semibold')
fig.text(0.05, 0.935, f"{P} point deposits of {float(d['energy_MeV']) * 1e3:.1f} keV, one letter per drift length "
         f"({', '.join(f'{x:.1f}' for x in d['drift_m'])} m from the gate).  Drift and diffusion: JAXTPC.  "
         "S2 light: LUCiD, 1M rays per deposit.  SiPMs drawn as 13 mm squares (1.3 mm real).",
         ha='left', va='top', fontsize=10, color=INK2)
slow_txt = fig.text(0.985, 0.975, '', ha='right', va='top', fontsize=11, color=INK2)


def update(k):
    tt = t[k]
    xy, dist, gap_dist, drifting, crossing = electrons(tt)
    side.set_offsets(np.c_[dist[drifting], xy[..., 0][drifting]])
    side_gap.set_offsets(np.c_[gap_dist[crossing], xy[..., 0][crossing]])
    top.set_offsets(xy[drifting]); top_gap.set_offsets(xy[crossing])
    for pc, data in maps:
        pc.set_array(data[k])
    for cur, line, pt, pr in zip(cursors, progress, pulse_t, pulse_r):
        cur.set_xdata([tt, tt]); line.set_data(pt[pt <= tt], pr[pt <= tt])
    title.set_text(f'"{word}" written in ionization, drifted and imaged    t = {tt:6.1f} µs')
    slow_txt.set_text('slow motion (×10) while a letter crosses the EL gap' if slow[k] else '')


frames = [0] * 12 + list(range(len(t))) + [len(t) - 1] * 48
anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
anim.save(OUT + '.mp4', writer=animation.FFMpegWriter(fps=24, bitrate=6000, extra_args=['-pix_fmt', 'yuv420p']))
last = int(np.argmax(np.where(t >= windows[-1][0], now.sum(axis=1), -1.0)))
for k, tag in ((last, 'lastS2'), (len(t) - 1, 'final')):
    update(k)
    fig.savefig(f'{OUT}_{tag}.png', dpi=130)
    fig.savefig(f'{OUT}_{tag}.pdf')
print(f'{len(frames)} frames; wrote {OUT}.mp4 and {OUT}_{{lastS2,final}}.{{png,pdf}}')
