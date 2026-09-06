# IMX219 stereo-depth tuner

This diagnostic tool evaluates rectification and tunes the stereo matcher. It
does not initialize YOLO, the TCA9548A, DA7280 drivers, or haptic actuators.

## Easy Mode

Place a textured target exactly 1.0 m from the cameras, keep the scene and
cameras still, and clear closer objects from all three image columns. Then run:

```bash
cd /home/orin_nano/Desktop/FYP2/1.1_depth_visualization
./run_easy_tuner.sh
```

The launcher selects the newest completed calibration under
`../1.0_calibration/images/`. Press `A` once to test five live stereo pairs.
The selected schema-2 profile and its calibration SHA-256 are saved as
`results/vpi_recommended_profile.json`; the `2.x`, `4.0`, and `7.0` launchers
load that file automatically. Press `Q` when finished.

Easy Mode keeps VPI maximum disparity at 256 and rejects disparities within
the final eight pixels of that search range. It evaluates P1/P2, uniqueness,
confidence, and diagonal paths while penalizing target error, temporal noise,
false-near surfaces, poor coverage, and slow processing. A dense result is not
automatically a correct result.

## Manual and offline modes

Pass a calibration explicitly when using the manual launcher:

```bash
./run_tuner.sh \
  --calibration ../1.0_calibration/images/SESSION/stereo_calibration.npz \
  --backend vpi
```

To inspect an existing pair without opening the cameras:

```bash
./run_tuner.sh \
  --calibration /path/to/stereo_calibration.npz \
  --left-image /path/to/left.png \
  --right-image /path/to/right.png
```

## Controls

- `Space`: freeze or resume the live stereo pair.
- `A`: automatically tune the active backend.
- `B`: switch between OpenCV StereoSGBM and VPI CUDA.
- Click the metric-depth pane: move the measurement ROI.
- `R`: return the ROI to the image centre.
- `P`: toggle built-in top/front 3D projections.
- `O`: open a snapshot in Open3D when installed.
- `S`: save images, arrays, settings, metrics, valid mask, and a PLY cloud.
- `Q` or `Esc`: quit.

## Validation and memory

Check rectified checkerboard edges on the same horizontal guides, then measure
textured targets at 0.5, 1.0, 1.5, 2.0, and 3.0 m under representative
lighting. Reject a profile with unstable distance, excessive holes,
false-near surfaces, high pair skew, or poor processing speed.

VPI CUDA at 1280x720 and 256 disparities uses substantial shared system/GPU
memory. If it reports an out-of-memory error, close 4.0/7.0, browsers, and other
GPU-heavy programs before restarting the tuner. On VPI 3, the CUDA census
window is fixed at 9x7, so the OpenCV block-size control has no VPI effect.

Run hardware-free checks with:

```bash
python3 -m unittest discover -s tests -v
```
