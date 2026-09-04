# IMX219 stereo-depth tuner

This is a diagnostic application for the binocular IMX219 cameras. It does
not import or initialize YOLO, the TCA9548A, DA7280 controllers, or actuators.

It uses the calibrated 1280x720, sensor-mode 4, 30 FPS capture configuration
from `../3.0_imx219/stereo_calibration.npz` by default.

## Easy Mode (recommended)

1. Put a textured box or other flat object exactly **1.0 metre** in front of
   the cameras. Keep it in the centre and keep everything still. Clear away
   hands, cables, table edges, and other objects closer than the target across
   the left, centre, and right thirds of the view.
2. Close the main YOLO/haptic application.
3. Run one command:

```bash
cd /home/orin_nano/Desktop/FYP2/1.1_depth_visualization
./run_easy_tuner.sh
```

4. When the camera windows appear, press `A` once.
5. Wait while Easy Mode captures five separate live stereo pairs, runs 24 VPI
   CUDA disparity configurations on all five, and scores 96 safe combinations
   of P1/P2, uniqueness, confidence, and diagonal paths.
6. It displays the winner, saves evidence under `results/`, and writes the
   validated profile to `../3.0_imx219/vpi_tuned_profile.json`.
7. Press `Q` to finish.

The Easy Mode launcher automatically selects the newest completed calibration
under `../1.0_calibration/images/*/stereo_calibration.npz` and prints the exact
path before opening the cameras.

Easy Mode excludes uniqueness-off and confidence `1`, rejects the last eight
pixels below the 256-disparity search limit, and measures supported false-near
surfaces independently in every left/centre/right zone. A candidate with more
than 3% supported false-near pixels in any zone is rejected. Remaining
false-near and near-limit pixels are penalties, along with target error,
coverage, temporal noise, and processing time.

The selected schema-2 profile is loaded automatically by the `3.0` application, but it
must still be checked at 0.5, 1.5, 2.0 and 3.0 metres before enabling actuators.
The profile contains the calibration SHA-256; `3.0` rejects it if the
calibration file changes.

## Advanced/manual mode

```bash
cd /home/orin_nano/Desktop/FYP2/1.1_depth_visualization
./run_tuner.sh
```

The manual tuner can still compare OpenCV StereoSGBM and VPI. To start with
the production VPI CUDA backend:

```bash
./run_tuner.sh --backend vpi
```

To inspect an already captured pair without opening either camera:

```bash
./run_tuner.sh \
  --left-image /path/to/left.png \
  --right-image /path/to/right.png
```

## Controls

- `SPACE`: freeze the current pair or resume live capture.
- `A`: automatically tune the active backend using the entered known distance
  and selected ROI. On VPI, only native CUDA controls are swept.
- `B`: switch between SGBM and VPI CUDA.
- Click the metric-depth pane to move the measurement ROI.
- `R`: return the ROI to the center.
- `P`: toggle dependency-free top/front 3D projections made from the current
  calibrated disparity and `Q` matrix.
- `O`: open a snapshot in Open3D when the optional `open3d` package is
  installed. If it is absent, the tuner continues normally and explains that
  the built-in view or saved PLY can be used instead.
- `S`: save the rectified pair, heatmaps, float disparity/depth arrays, valid
  mask, settings, timing, ROI metrics and `point_cloud.ply` under `results/`.
- `Q` or `Esc`: close the tuner.

The controls window contains separate SGBM and VPI settings. Settings that
belong to the inactive backend have no effect.

## What the 3D view means

The point cloud is reconstructed from the same stereo disparity using the
calibration `Q` matrix. It is useful for seeing geometry, holes and obvious
outliers, but it does not make an inaccurate disparity map more accurate.
Calibration, rectification, synchronized capture and stereo matching remain
the sources of depth quality.

Each normal `S` save and each successful automatic tune includes a binary PLY
file. That file can be opened later in Open3D, CloudCompare or MeshLab, even
on another computer.

## Important VPI limitation

On the installed VPI 3 CUDA backend, the `window` argument is ignored and a
fixed 9x7 census window is used. The SGBM block-size slider therefore cannot
change VPI output. For VPI, tune confidence, uniqueness, P1/P2, maximum
disparity and diagonal paths.

Easy Mode fixes VPI `maxDisparity` at 256 for navigation close-range coverage,
but disparities from 248 through just below 256 are treated as search-limit
artifacts and never become valid depth.
It computes disparity once per P1/P2, uniqueness and diagonal configuration,
then evaluates several confidence thresholds from the same confidence map.
The live tuner starts with diagonal paths disabled because the full-resolution
256-disparity diagonal payload exceeds the available shared memory on this
Jetson. This does not disable CUDA stereo.

VPI allocates substantial shared CPU/GPU memory at 1280x720. If it reports
`CUDA run-time error: out of memory`, close the main YOLO application, browser
tabs and other GPU-heavy programs, then restart the tuner. The tuner freezes
after an error rather than repeatedly retrying the allocation; press `B` to
continue with SGBM.

## Required validation after the 1 m tune

1. Keep both cameras and every target stationary. Check that matching edges
   in the alignment window lie on the same cyan horizontal guide lines.
2. Put a textured, flat target at a measured distance. Enter that distance in
   the `Known distance cm` control and click the target center.
3. Make sure no real object is closer than 0.65 times the known target distance
   anywhere in the three zones; at a 1 m tune, keep the view clear below 0.65 m.
4. Press `A` once and let the five-frame VPI CUDA sweep finish without moving
   the target or cameras.
5. Reject the profile if the saved report shows unstable ROI depth, excessive
   holes, high frame skew, false-near surfaces, or poor processing speed.
6. Repeat measurements at 0.5, 1.0, 1.5, 2.0 and 3.0 m under representative
   lighting before enabling actuators.

Do not choose a preset only because it fills the most pixels. A dense but
incorrect map is less safe than a sparse confidence-filtered map.
