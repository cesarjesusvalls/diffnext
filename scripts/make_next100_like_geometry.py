#!/usr/bin/env python3
"""Build the sensors of the NEXT-100-like detector (``config/next100_like_sensors.npz``).

NEXT-100 is a high-pressure xenon TPC. The LUCiD model is a cylinder with the tracking plane
on one cap and the energy plane on the other:

* top cap (``z = +H/2``): the SiPM plane, flat 1.3 mm square SiPMs;
* barrel: the light tube, drift and buffer parts (an 18-panel teflon prism in NEXT-100; here
  the cylinder inscribed in it, radius = the panels' apothem);
* bottom cap (``z = -H/2``): the energy plane at the face of the PMT sapphire windows, with
  the windows as flat 80 mm discs. The cathode mesh between the drift region and the buffer
  is not modelled: LUCiD has no semi-transparent internal surface, so it is fully open.

Numbers from nexus (https://github.com/next-exp/nexus, master):

=============================  ================================================================
``Next100FieldCage.cc``        active_diam 983.3 mm (between opposite light-tube panels);
                               gate -> cathode 1174 + 13.5 - 0.13 mm (the active length)
``Next100.cc``                 gate -> tracking-plane copper 25 mm + 0.13 mm grid;
                               gate -> sapphire windows 1458.8 mm - 0.13 mm grid
``Next100SiPMBoard.cc``        SiPM pitch 15.55 mm; 0.2 mm kapton under the SiPMs
``Next100SiPM.cc``             Hamamatsu S13372-1350TE: 1.3 x 1.3 mm2 sensitive area under a
                               0.5 mm window, package 1.3 mm (+ 2 um coating)
``Next100TrackingPlane.cc``    56 boards of 8x8 SiPMs in columns of 5, 7, 8, 8, 8, 8, 7, 5;
                               boards 122.4 mm wide, 2 mm apart; SiPM margin 6.775 mm
``Next100EnergyPlane.cc``      60 PMTs on concentric hexagonal rings (125 mm pitch) behind
                               80 mm TPB-coated sapphire windows (hole_diam_front)
``Next100.cc``                 fc_displ (-3.7, -6.4) mm: the field cage (light tube, cathode)
                               sits off the vessel axis, on which the tracking and energy
                               planes are centred
=============================  ================================================================

The readout plane is the SiPMs' sensitive surface, 24.1 mm behind the gate:
``25.13 - 0.2 - 1.302 + 0.502``. The cylinder axis is the light-tube axis, so the SiPM grid is
centred at (+3.7, +6.4) mm, where nexus puts the tracking-plane centre relative to the light
tube. Every SiPM is placed as nexus places it (same board layout, same id), and those whose
centre is inside the cylinder are kept. Left out, as finer than this model: the 6 mm teflon
masks around the SiPMs (6x6 mm holes, sensitive area 5.2 mm deep).

The PMT windows are placed the same way, around the energy-plane centre; the six on the
outer ring reach past the cylinder radius, so only their inner part can be hit. Every sensor
carries its own shape and size (``sensor_shape``, ``sensor_half_size``; the geometry config
reads them with ``sensor_model: {"per_sensor": true}``) and keeps its nexus id: 1000 * board +
slot for the SiPMs, the copy number 0-59 for the PMTs.

The npz also records, in the LUCiD frame (m), where the gate (``gate_z``), the electroluminescence
gap between the gate and anode meshes (``el_gap_z``) and the cathode (``cathode_z``) sit, for
charge drift and S2 light; none of them is a surface in the LUCiD geometry.

The npz ``sensor_radius`` is the SiPM half side (0.65 mm): it sizes the cylinder's cell lookup,
and the cylinder classifies a sensor closer than that to the rim as a barrel sensor, so SiPMs
whose centre lies within 0.65 mm of the rim are left out. The PMT windows span several cells
and are found through the closest-sensor fallback, which LUCiD checks when it builds the
propagator.

Usage::

    python scripts/make_next100_like_geometry.py [--out config/next100_like_sensors.npz]
"""
import argparse
import os

import numpy as np

# ---- nexus constants (mm) ----------------------------------------------------------------
GRID_THICKNESS = 0.13                                   # Next100.cc grid_thickness_
GATE_TP_DIST = 25.0 + GRID_THICKNESS                    # Next100.cc gate_tracking_plane_distance_
GATE_CATHODE_DIST = 1174.0 + 13.5 - GRID_THICKNESS      # Next100FieldCage gate_cathode_dist_
GATE_WINDOW_DIST = 1458.8 - GRID_THICKNESS              # Next100.cc gate_sapphire_wdw_distance_
EL_GAP = 10.0                                           # Next100FieldCage el_gap_length_
PMT_WINDOW_DIAM = 80.0                                  # Next100EnergyPlane hole_diam_front_
ACTIVE_DIAM = 983.3                                     # Next100FieldCage active_diam_
TP_CENTRE = (3.7, 6.4)                                  # -Next100.cc fc_displ_x_, fc_displ_y_
SIPM_PITCH = 15.55                                      # Next100SiPMBoard pitch_
BOARD_SIZE = 122.40                                     # Next100SiPMBoard size_
BOARD_GAP = 2.0                                         # Next100TrackingPlane distance_board_board_
BOARD_MARGIN = 6.775                                    # Next100SiPMBoard margin_
# Board columns, left (-x) to right (+x): (number of boards, column offset in board units).
BOARD_COLUMNS = [(5, -3.5), (7, -2.5), (8, -1.5), (8, -0.5),
                 (8, 0.5), (8, 1.5), (7, 2.5), (5, 3.5)]
KAPTON_THICKNESS = 0.2                                  # Next100SiPMBoard board_thickness_
SIPM_ACTIVE = 1.3                                       # Next100SiPM sens_width/length
SIPM_PACKAGE = 1.3 + 0.002                              # Next100SiPM thickness + 2 um coating
SIPM_WINDOW = 0.5 + 0.002                               # window + coating above the sensor

# ---- derived LUCiD geometry --------------------------------------------------------------
SIPM_PLANE_BEHIND_GATE = GATE_TP_DIST - KAPTON_THICKNESS - SIPM_PACKAGE + SIPM_WINDOW   # 24.13 mm
HEIGHT_MM = SIPM_PLANE_BEHIND_GATE + GATE_WINDOW_DIST                                 # 1482.80 mm
RADIUS_MM = ACTIVE_DIAM / 2.0                                                         # 491.65 mm


def nexus_sipms(centre=TP_CENTRE):
    """All 3584 NEXT-100 SiPMs: centres (mm, relative to the light-tube axis) and nexus ids.

    Follows ``Next100TrackingPlane::PlaceSiPMBoardColumns`` and ``Next100SiPMBoard::Construct``,
    with the plane centred at ``centre``; the id is ``1000 * board + slot`` as in
    ``Next100TrackingPlane::PrintSiPMPosInGas``.
    """
    step = BOARD_SIZE + BOARD_GAP
    xy, ids, board = [], [], 1
    for n_boards, column in BOARD_COLUMNS:
        for i in range(n_boards):
            bx, by = column * step, (-0.5 * (n_boards - 1) + i) * step
            for slot in range(64):
                ix, iy = divmod(slot, 8)
                xy.append((centre[0] + bx - BOARD_SIZE / 2 + BOARD_MARGIN + ix * SIPM_PITCH,
                           centre[1] + by - BOARD_SIZE / 2 + BOARD_MARGIN + iy * SIPM_PITCH))
                ids.append(1000 * board + slot)
            board += 1
    return np.asarray(xy), np.asarray(ids, dtype=np.int64)


def nexus_pmts(centre=TP_CENTRE):
    """The 60 NEXT-100 PMT centres (mm, relative to the light-tube axis), in nexus copy order.

    Follows ``Next100EnergyPlane::GeneratePositions`` with the plane centred at ``centre``.
    """
    x_pitch, y_pitch, step = 125.0, 108.3, 60.0
    xy = []
    for circle in range(1, 5):
        rad = circle * x_pitch
        xy += [(rad * np.cos(np.radians(k * step)), rad * np.sin(np.radians(k * step)))
               for k in range(6)]
        for i in range(1, circle):
            sx, sy = (circle - 0.5 * i) * x_pitch, i * y_pitch
            rad, a0 = np.hypot(sx, sy), np.degrees(np.arctan2(sy, sx))
            xy += [(rad * np.cos(np.radians(a0 + k * step)), rad * np.sin(np.radians(a0 + k * step)))
                   for k in range(6)]
    return np.asarray(xy) + np.asarray(centre)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--out', default=os.path.join(here, '..', 'config', 'next100_like_sensors.npz'))
    args = ap.parse_args()

    half = SIPM_ACTIVE / 2.0
    xy_all, ids_all = nexus_sipms()
    keep = np.hypot(xy_all[:, 0], xy_all[:, 1]) <= RADIUS_MM - half
    xy, ids = xy_all[keep], ids_all[keep]
    n = len(xy)
    pmt_xy = nexus_pmts()
    m = len(pmt_xy)

    out = os.path.abspath(args.out)
    np.savez(out,
             positions_mm=np.vstack([np.column_stack([xy, np.full(n, HEIGHT_MM / 2.0)]),
                                     np.column_stack([pmt_xy, np.full(m, -HEIGHT_MM / 2.0)])]),
             directions=np.vstack([np.tile([0.0, 0.0, -1.0], (n, 1)),
                                   np.tile([0.0, 0.0, 1.0], (m, 1))]),
             surfaces=np.array(['top'] * n + ['bottom'] * m),
             pmt_id=np.concatenate([ids, np.arange(m)]),
             sensor_shape=np.array(['square'] * n + ['disc'] * m),
             sensor_half_size=np.concatenate([np.full(n, half), np.full(m, PMT_WINDOW_DIAM / 2.0)]) / 1000.0,
             radius=RADIUS_MM / 1000.0,
             height=HEIGHT_MM / 1000.0,
             gate_z=(HEIGHT_MM / 2 - SIPM_PLANE_BEHIND_GATE) / 1000.0,
             el_gap_z=np.array([HEIGHT_MM / 2 - SIPM_PLANE_BEHIND_GATE + GRID_THICKNESS,
                                HEIGHT_MM / 2 - SIPM_PLANE_BEHIND_GATE + GRID_THICKNESS + EL_GAP]) / 1000.0,
             cathode_z=(HEIGHT_MM / 2 - SIPM_PLANE_BEHIND_GATE - GATE_CATHODE_DIST) / 1000.0,
             sensor_radius=half / 1000.0)

    print(f"wrote {out}")
    print(f"  cylinder: radius {RADIUS_MM:.2f} mm, height {HEIGHT_MM:.2f} mm "
          f"(SiPM plane {SIPM_PLANE_BEHIND_GATE:.2f} mm behind the gate, PMT windows "
          f"{GATE_WINDOW_DIST:.2f} mm past it)")
    print(f"  {n} of {len(xy_all)} NEXT-100 SiPMs inside the cylinder ({SIPM_ACTIVE} mm square, "
          f"pitch {SIPM_PITCH} mm, plane centred at {TP_CENTRE} mm)")
    print(f"  {m} PMT windows, {PMT_WINDOW_DIAM} mm discs; "
          f"{int((np.hypot(*pmt_xy.T) + PMT_WINDOW_DIAM / 2 > RADIUS_MM).sum())} reach past the radius")


if __name__ == '__main__':
    main()
