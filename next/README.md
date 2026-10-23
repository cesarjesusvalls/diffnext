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

## Charge and light of a point deposit

`lucid.next` simulates a point energy deposit end to end, differentiably:

- S1: E x 25510 photons/MeV, emitted isotropically at the deposit;
- charge: E / 22.4 eV electrons drift to the gate with JAXTPC (`lucid.next.charge`): its
  `tools.drift.compute_drift_to_plane` and `SimParams` (drift velocity 1 mm/us, lifetime, and
  diffusion of 1.0 / 0.3 mm/sqrt(cm) transverse / longitudinal, the nexus NEXT-100 values);
  the cloud reaching the gate is sampled with fixed normals, so it is differentiable;
- S2: each surviving electron makes `el_gain` photons (placeholder, 3264 = the nexus EL yield at
  34.5 kV/cm and 13.5 bar over the 10 mm gap), emitted uniformly across the EL gap at its arrival
  point; S1 and S2 go through the same LUCiD detector (`lucid.next.event.NextEventSimulator`).

The gate, EL gap and cathode positions come from the sensor file. JAXTPC
(https://github.com/OmarAlterkait/JAXTPC) must be on the python path: set `JAXTPC_PATH` for
`run_gpu_s3df.sh`.

```bash
HOLD_JOBID=<job> JAXTPC_PATH=<JAXTPC> next/run_gpu_s3df.sh next/kr_event.py --out kr.npz
python next/plot_kr_event.py kr.npz <figures dir>
```

## Scripts

| Script | What it does |
|---|---|
| `kr_event.py` | S1 and S2 of a 41.5 keV deposit at several drift distances, with gradient checks |
| `plot_kr_event.py` | Event anatomy (drift, waveform, S1 and S2 maps) and the S2 SiPM spot against drift |
| `next_letters.py` | S2 per sensor of point deposits that write a word, one letter per drift distance |
| `next_letters_video.py` | MP4 of a `next_letters.py` run: the letters drift to the EL gap and image on the SiPMs |
| `charge_calibration.py` | Closure fit on the S2 of a sample (`--mode fit`) of any of the charge transport, EL gain, wall reflectivity and absorption length (`--free`), optionally with a per-sensor gain spread in the pseudo-data (`--gain-spread`), D_T held through D_T / v (`--hold-dt-over-v`), LUCiD's charge term or the leave-one-out one (`--estimator`) and the pseudo-data's random keys (`--data-key`); AD vs finite-difference Jacobian check (`check`); charge-only Cramer-Rao bound (`crb`) |
| `plot_charge_calibration.py` | Figures and an MP4 of a `charge_calibration.py` fit |
| `studies/` | The checks behind the calibration findings below; see [Redoing the studies](#redoing-the-studies) |
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

## Calibrating the charge transport

`charge_calibration.py` is a closure test: pseudo-data from the "NEXT" sample (`next_letters.py`,
each letter one calibration source) at known parameters, then a fit of the drift velocity,
electron lifetime, transverse and longitudinal diffusion and EL gain (W fixed: S2 sees only
gain / W) from a start drawn as `lucid.fitting.closure` draws it. It uses existing fitting tools
only (`lucid.next.calibration`): LUCiD's calibration problem on the S2 charge per sensor (Neyman
residual, per-sensor gains profiled, `jacfwd` Jacobian), JAXTPC's Sobolev loss on the S2 waveform
(the charge alone sees only gain / W, v * lifetime and D_T / v, and not D_L), and
`lucid.fitting.minimize` with `damped_gauss_newton`.

```bash
HOLD_JOBID=<job> JAXTPC_PATH=<JAXTPC> next/run_gpu_s3df.sh next/charge_calibration.py \
    --sample next_letters.npz --mode fit --pos-grad-threshold 0 --mu 1e-6 \
    --forward-draws 4 --jacobian-draws 8 --refresh 4 --steps 60 --out charge_fit.npz
python next/plot_charge_calibration.py charge_fit.npz <out stem> --ffmpeg <ffmpeg>
```

With 1M rays per letter this recovers the drift velocity, lifetime, D_L and EL gain to better
than 1% (about an hour on one RTX 2080 Ti). Three things to know:

- `--mu 1e-6`: LUCiD's Levenberg damping is `mu` times the median curvature, which the charge
  term sets; D_L has only the waveform's curvature and stays frozen at the default 0.1.
- The EL gain is defined through the gauge of the profiled sensor gains, mean(log k) = 0, and
  Monte-Carlo noise in the forward pulls that gauge (by the log of noisy dim SiPMs): +2.9% with
  one forward draw per step, about +0.6% with four.
- D_T is not measured at this ray count. EL light spreads about 13 mm before reaching the SiPMs,
  so the 5.5-9.5 mm diffusion changes the image by 1-2.5%, below the per-SiPM Monte-Carlo noise
  (about 20% per draw): along D_T the loss is flat within its noise over -26% to +35%.

## Calibrating the optics and sensor gains as well

`--free` also takes the wall reflectivity, fitted as log(1 - R), and the gas absorption length,
which the simulator then takes as inputs (`NextEventSimulator(optics_input=True)`, the same
forward as the baked-in optics to 1e-7). `--gain-spread` puts per-sensor gains in the pseudo-data
(log-normal, geometric mean 1), which the fit profiles. A global QE is left out: it is exactly
degenerate with the EL gain.

A first joint fit did not close. It freed v, lifetime, D_L, EL gain, R and the absorption length
(truth 20 m), with 10% gains and D_T held, at 4 forward and 8 Jacobian draws for 60 steps. There
were four findings, each with its study in `studies/`:

- **Hold D_T / v, not D_T** (`--hold-dt-over-v`). The SiPM image width is
  sigma_T^2 = 2 (D_T / v) d. With D_T held, the image's Monte-Carlo noise reaches v, whose
  step-to-step jitter grew 12x over the charge-only fit; averaged over that jitter, the waveform
  loss has its D_L minimum at +35% (`dl_drift_check.py`). With D_T / v held, v enters the charge
  only through the electron survival.
- **R and the absorption length are degenerate on this sample**: their correlation in the charge
  metric is 0.985, and the soft direction is 40x weaker than the next (`valley_check.py`). nexus
  has no absorption in the gas, so the later fits free R alone and keep the physics config's
  absorption length (none).
- **LUCiD's charge term is biased by its own Monte-Carlo noise.** `CalibrationProblem` profiles
  the gains, and fixes their gauge mean(log k) = 0, on the residual's own forward draws. Both are
  nonlinear in the draws, so the fixed point sits off the truth by an amount that falls as
  1 / draws (`tests/test_fitting_estimator_unbiased.py`; with 4 draws, `loss_compare.py` finds
  the loss lower at the fit than at the truth). `--estimator leave-one-out`
  (`lucid.next.calibration.LeaveOneOutProblem`) takes each draw's gains and gauge from the other
  draws. The residual is then linear in each draw, and the expected step at the truth is zero for
  any number of draws: with 4 draws it settles where `CalibrationProblem` would with unlimited
  draws (`fixed_point_check.py`). Its gauge weights the gains by light (c = sum model / sum data),
  so its EL gain is the truth's times the light-weighted mean gain (1.019 here); the fit reports
  that as its target.
- **Freeing R biases R and the EL gain together.** Every fit with R free settles near R +0.4%,
  with the EL gain 3-5% low. Raising R puts light on every sensor and a lower EL gain takes it
  back off: their correlation is 0.95 in LUCiD's gauge and 0.99 in the light-weighted one. The
  charge-only fit, with R held at the truth, gets the EL gain to +0.6%. The same fit with only R
  freed (no gain spread, LUCiD's charge term, D_T free) ends at R +0.41% and EL gain -2.7%. It
  came down to there from R +1.5%, while fits B and C crept up to it from +0.1%: this is where
  the fit settles, not where it stopped. The cause is open.

  It is not the pseudo-data's random keys. `lucid.fitting.closure_data` draws the pseudo-data
  with keys 5000 + 131 b (+1000 per source). In the linearised fixed point (`data_noise_sets.py`)
  all 8 of those draws pull R by +0.23% to +0.40%, while the single draws of 13 other key
  families scatter around zero with a standard deviation of about 0.2% (`key_check.py`,
  `key_families.py`; `key_split_check.py` finds the pull needs the electron and the propagation
  streams both from those keys). That map predicted fits A and B (`predict_shift.py`: R +0.32%,
  EL gain -3.0% or -3.3% in the two gauges). But fit C, which is B on pseudo-data from keys
  20000 + 131 b (`--data-key`), was predicted at R -0.03% and EL gain +0.3% and ends at R +0.48%
  and EL gain -5.0% (figure `key_family_figure.py`). B and C share their model draws, so the
  data alone moved the fit by +0.08% in R where the map said -0.35%: the linearised map does not
  describe the fit along R and the EL gain. A candidate it cannot see: the Neyman weight,
  1 / sqrt(Q), is nonlinear in the pseudo-data's own noise, a bias every pseudo-data set shares.

Fitted / truth - 1 (Polyak mean of the last 10 of 60 steps, one RTX 2080 Ti):

| | D_T held, absorption free | A: D_T / v held, no absorption, 16 draws | B: as A, leave-one-out, 4 draws | C: as B, pseudo-data keys 20000 |
|---|---|---|---|---|
| drift velocity | +0.06% | +0.001% | +0.002% | +0.002% |
| electron lifetime | +0.12% | +0.62% | +0.74% | -0.21% |
| longitudinal diffusion | +41% | +0.05% | +0.07% | +0.07% |
| EL gain | -3.4% | -2.88% | -4.20% (against 1.019 x truth) | -4.96% (against 1.019 x truth) |
| wall reflectivity | -0.05% | +0.37% | +0.40% | +0.48% |
| absorption length | +25% | | | |
| SiPM / PMT gains, rms | 10.9% / 0.26% | 9.8% / 0.24% | 9.8% / 0.24% | 9.7% / 0.26% |
| time | 65 min | 2 h 12 min | 61 min | 61 min |

The gains come back to the Monte-Carlo floor, the rms left with the parameters at the truth. For
the PMTs that is 0.24%. The SiPMs are not constrained at all at 1M rays per letter: their floor,
9.9%, is the 10% spread itself.

R alone, added to the charge-only fit (4 forward and 8 Jacobian draws, 60 steps, no gain spread,
LUCiD's charge term; D_T is not measured at this ray count):

| | R held at the truth | R free |
|---|---|---|
| drift velocity | +0.003% | +0.002% |
| electron lifetime | -0.05% | +0.27% |
| transverse diffusion | -15% | -18% |
| longitudinal diffusion | +0.22% | +0.16% |
| EL gain | +0.59% | -2.73% |
| wall reflectivity | | +0.41% |
| time | 61 min | 65 min |

## Known gradient issues

Flat sensors deposit at the sensor's reflectance wherever the photon lands
(`deposit_as_sensor` in the photon step): their forward is a hard step, so this changes no
forward value, and it keeps the outer half of the straight-through gradient, which landing on
the R = 0.94 wall used to scale by 0.06. Still open: pathwise gradients through many diffuse
bounces grow without bound (keep `pos_grad_threshold` small), the cap rim has no smooth
surrogate, and position-type gradients on the 1.3 mm SiPMs are dominated by Monte-Carlo noise
per draw.

## Redoing the studies

Every study here runs on one GPU; the times given are for an RTX 2080 Ti. On S3DF, hold a GPU
and run the scripts into it with `run_gpu_s3df.sh`. The plotting scripts need only numpy and
matplotlib. Elsewhere, run the same scripts with python, with this checkout and JAXTPC on the
python path. Run the commands from the repository root:

```bash
sbatch -p turing -A <account> -q normal -c 8 --mem=64G --gpus=1 -t 10:00:00 --wrap "sleep 36000"
export HOLD_JOBID=<that job> LUCID_IMAGE_PATH=<lucid .sif> JAXTPC_PATH=<JAXTPC checkout>
RUN=next/run_gpu_s3df.sh OUT=<outputs dir> FIG=<figures dir> FFMPEG=<ffmpeg binary>
```

Whatever a command does not set is fixed in the scripts: 1M rays per letter, K = 200, the nexus
transport, a 10 ms truth lifetime, R = 0.94, and the gains of `--gain-spread 0.1` (seed 6000).

**Point-source light and the gradient fit.** The sensor validation, the PMT maps of direct and
reflected light at 9 positions, and the source-position fit on direct light (wall R = 0, 48 inner
PMTs):

```bash
$RUN next/validate_sensors.py
$RUN next/point_source_rays.py --positions next/positions/drift9.json --k-total 200 --out $OUT/pmt_drift.npz
python next/plot_pmt_maps.py $OUT/pmt_drift.npz next/positions/drift9.json $FIG
$RUN next/fit_point_source.py --mode check --K 1 --phys next/physics_wallR0.json --exclude-rim
$RUN next/fit_point_source.py --mode fit --K 1 --phys next/physics_wallR0.json --exclude-rim --out $OUT/fit_direct.npz
python next/fit_video.py $OUT/fit_direct.npz $FIG/fit_point_source_direct $FFMPEG
```

**A krypton event and the "NEXT" sample:**

```bash
$RUN next/kr_event.py --out $OUT/kr_event.npz
python next/plot_kr_event.py $OUT/kr_event.npz $FIG
$RUN next/next_letters.py --out $OUT/next_letters.npz
python next/next_letters_video.py $OUT/next_letters.npz $FIG/next_letters $FFMPEG
```

**The charge-transport closure.** This covers the D_T column of the Jacobian against finite
differences (T letter, 32 draws), and a first fit with 1 forward and 2 Jacobian draws per step,
30 steps (12 min). Then it measures where the gain gauge sits at the truth against the number of
draws averaged, with a figure for both. Last come the fit with 4 forward and 8 Jacobian draws, 60
steps (61 min), and the chi^2 along D_T. The D_T check was first run before the deposit fix (see
Known gradient issues), when the gradient lost the light that diffusion moves onto the SiPMs.
With the current code, the two agree within their noise.

```bash
C="next/charge_calibration.py --sample $OUT/next_letters.npz --pos-grad-threshold 0"
$RUN $C --mode check --draws 32 --fd-step 0.6 --fd-params 2 --out $OUT/charge_check_dt.npz
$RUN $C --mode fit --mu 1e-6 --steps 30 --out $OUT/charge_fit.npz
python next/plot_charge_calibration.py $OUT/charge_fit.npz $FIG/charge_fit
$RUN next/studies/gain_gauge_check.py $OUT/charge_fit.npz $OUT/next_letters.npz $OUT/gain_gauge_check.npz
python next/studies/plot_charge_fit_why.py $OUT/charge_check_dt.npz $OUT/next_letters.npz \
    $OUT/gain_gauge_check.npz $FIG/charge_fit_why
$RUN $C --mode fit --mu 1e-6 --forward-draws 4 --jacobian-draws 8 --refresh 4 --steps 60 \
    --out $OUT/charge_fit_fixed.npz
python next/plot_charge_calibration.py $OUT/charge_fit_fixed.npz $FIG/charge_fit_fixed \
    --ref $OUT/charge_fit.npz --ref-label "1 forward, 2 Jacobian draws"
$RUN next/studies/dt_scan.py $OUT/charge_fit_fixed.npz $OUT/next_letters.npz $OUT/dt_scan.npz
```

**The first joint fit** (absorption free, 65 min). It starts with the optics-input simulator
against the baked-in one and the Jacobian in R and the absorption length against finite
differences. After the fit come its three diagnostics: why D_L drifts, the R-absorption
degeneracy and the SiPM gains by light, and the loss at the fit against at the truth.

```bash
J="--free v lifetime dl gain reflectivity absorption --absorption-length-m 20"
$RUN next/studies/optics_path_check.py $OUT/next_letters.npz
$RUN $C --mode check $J --draws 8 --source 3 --out $OUT/joint_check.npz
$RUN $C --mode fit $J --gain-spread 0.1 --mu 1e-6 --forward-draws 4 --jacobian-draws 8 --refresh 4 \
    --steps 60 --out $OUT/joint_fit.npz
python next/plot_charge_calibration.py $OUT/joint_fit.npz $FIG/joint_fit \
    --ref $OUT/charge_fit_fixed.npz --ref-label "charge-only fit (D_T free)"
$RUN next/studies/dl_drift_check.py $OUT/joint_fit.npz $OUT/charge_fit_fixed.npz $OUT/next_letters.npz \
    $OUT/dl_drift_check.npz
$RUN next/studies/valley_check.py $OUT/joint_fit.npz $OUT/next_letters.npz $OUT/valley_check.npz
$RUN next/studies/loss_compare.py $OUT/joint_fit.npz $OUT/next_letters.npz $OUT/loss_compare.npz
```

**The second round.** Fit A uses LUCiD's charge term with 16 draws (2 h 12 min), and fit B the
leave-one-out term with 4 draws (61 min). Then comes the expected step at the truth for both
charge terms (10 min). `--w-time` is fit A's waveform weight, its `w_time`.

```bash
J2="--free v lifetime dl gain reflectivity --hold-dt-over-v --gain-spread 0.1 --mu 1e-6 \
    --jacobian-draws 8 --refresh 4 --steps 60"
$RUN $C --mode fit $J2 --forward-draws 16 --out $OUT/joint_fit_16fwd.npz
$RUN $C --mode fit $J2 --forward-draws 4 --estimator leave-one-out --out $OUT/joint_fit_loo.npz
python next/plot_charge_calibration.py $OUT/joint_fit_16fwd.npz $FIG/joint_fit_16fwd \
    --ref $OUT/joint_fit_loo.npz --ref-label "fit B: leave-one-out, 4 draws"
$RUN next/studies/fixed_point_check.py $OUT/next_letters.npz $OUT/fixed_point_check.npz --w-time 0.00707
```

**The pseudo-data's keys.** This starts with the linearised fixed point, which gives the closure's
statistical error and this pseudo-data's shift, both from 64 model draws (10 min). Then come
single draws by key family and the streams that carry the pull. Then the linearised fixed point
predicts each fit and pseudo-data from keys 20000 + 131 b, and fit C runs fit B on that
pseudo-data (61 min). The figure, which needs fit C, comes last.

```bash
$RUN next/studies/data_noise_sets.py $OUT/next_letters.npz $OUT/data_noise_sets.npz --w-time 0.00707 \
    --model-draws 64
$RUN next/studies/key_check.py $OUT/next_letters.npz $OUT/data_noise_sets.npz $OUT/key_check.npz
$RUN next/studies/key_families.py $OUT/next_letters.npz $OUT/data_noise_sets.npz $OUT/key_families.npz
$RUN next/studies/key_split_check.py $OUT/next_letters.npz $OUT/data_noise_sets.npz
$RUN next/studies/predict_shift.py $OUT/data_noise_sets.npz $OUT/joint_fit_16fwd.npz $OUT/joint_fit_loo.npz \
    --data-key 20000 --sample $OUT/next_letters.npz
$RUN $C --mode fit $J2 --forward-draws 4 --estimator leave-one-out --data-key 20000 \
    --out $OUT/joint_fit_loo_k20000.npz
python next/plot_charge_calibration.py $OUT/joint_fit_loo_k20000.npz $FIG/joint_fit_loo_k20000 \
    --ref $OUT/joint_fit_loo.npz --ref-label "fit B: pseudo-data keys 5000 + 131 b"
python next/studies/key_family_figure.py $OUT $FIG/pseudo_data_keys
```

**R alone.** This is the charge-only fit above, with R freed and nothing else changed (65 min):

```bash
$RUN $C --mode fit --free v lifetime dt dl gain reflectivity --mu 1e-6 --forward-draws 4 \
    --jacobian-draws 8 --refresh 4 --steps 60 --out $OUT/charge_fit_plus_r.npz
python next/plot_charge_calibration.py $OUT/charge_fit_plus_r.npz $FIG/charge_fit_plus_r \
    --ref $OUT/charge_fit_fixed.npz --ref-label "R held at the truth (the charge-only fit)"
```
