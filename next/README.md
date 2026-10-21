# NEXT-like detector

A LUCiD model of the NEXT-100 high-pressure xenon TPC and the studies run on it.

## Geometry

`config/NEXT100_like_geom_config.json` with `config/next100_like_sensors.npz`, built by
`scripts/make_next100_like_geometry.py` from the nexus NEXT-100 geometry:

- a cylinder of radius 491.65 mm (the light tube's apothem), 1482.8 mm from the SiPM plane to
  the face of the PMT sapphire windows; the cathode mesh is not modelled;
- 3130 SiPMs, flat 1.3 mm squares, and 60 PMT windows, flat 80 mm discs, at the nexus positions
  and with the nexus ids; the tracking and energy planes are centred (3.7, 6.4) mm off the
  cylinder axis, as nexus centres them off the light-tube axis;
- every sensor is a flat pad (`sensor_model: {"per_sensor": true}`; see
  `lucid.propagation.base.FlatSensorModel`).

`config/NEXT100_like_physics_config.json`: one Lambertian wall reflectivity (0.94), sensors that
absorb every ray (QE 1, reflectivity 0), no scattering or absorption in the gas; material
`config/materials/gxe.json` (scalar mode only).

## Scripts

| Script | What it does |
|---|---|
| `point_source_rays.py` | Expected rays per sensor from an isotropic point source: direct (K=1) and total (K) |
| `plot_pmt_maps.py` | PMT maps of direct, indirect and total light for a `point_source_rays.py` run |
| `validate_sensors.py` | Direct light against exact solid angles, for SiPMs and PMTs |
| `fit_point_source.py` | Autodiff vs finite-difference gradient check (`--mode check`) and an Adam fit of the source position to its own PMT light (`--mode fit`) |
| `fit_video.py` | MP4 of a fit's path (needs an ffmpeg binary) |
| `viewers/` | Interactive pages: the PMT light per source position, and the nexus vs LUCiD 3D comparison |
| `nexus_export/` | Builds the nexus NEXT-100 optical geometry against Geant4 and writes it as VRML meshes |
| `run_gpu_s3df.sh` | Runs any of the above in the LUCiD container on a held S3DF GPU job |

Example:

```bash
HOLD_JOBID=<job> next/run_gpu_s3df.sh next/point_source_rays.py \
    --positions next/positions/drift9.json --k-total 200 --out pmt_drift.npz
```

## Known gradient issues

Source-position gradients are reliable only for direct light, with the wall reflectivity set to
zero (`physics_wallR0.json`, which leaves direct light unchanged), and away from the sensors that
straddle the cylinder rim. The expected-value deposit scales each sensor's gradient surrogate by
(1 - R) of the surface the photon landed on, which suppresses the outer half of the surrogate
under reflective walls; pathwise gradients through many diffuse bounces grow without bound; and
the cap rim has no smooth surrogate.
