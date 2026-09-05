# IMX219 centre depth test

This project uses only the selected `1.0_calibration` NPZ. It does not read a
profile, report, image, or setting from `1.1_depth_visualization`.

The pipeline captures and rectifies the two 1280x720 IMX219 streams, computes
one disparity map with VPI CUDA, reconstructs metric Z with the calibration
`Q` matrix, and reports the median distance in the centre box.

## Run

```bash
cd /home/orin_nano/Desktop/FYP2/2.1_testdepthimx219
./run_with_latest_calibration.sh
```

The launcher automatically selects the newest completed calibration under
`../1.0_calibration/images/`. Runtime defaults owned by this project are:

- maximum disparity: 160;
- confidence threshold: 8192;
- uniqueness: 0.4;
- P1/P2: 3/48;
- diagonal paths: disabled.

They are starting values, not automatically certified values. Current values
and valid coverage are displayed while the program runs.

## Controls

- `W/Z`: increase/decrease VPI maximum disparity by 16.
- `E/C`: increase/decrease confidence by 4096.
- `U/J`: increase/decrease uniqueness; `-1` disables it.
- `R`: restore the 2.1 defaults.
- `D`: toggle disparity view.
- `P`: toggle built-in top/front 3D projections.
- `O`: open the latest point cloud in Open3D when available.
- `S`: save current images, arrays, mask, projection, and PLY.
- `Q` or `Esc`: quit.

Changing a VPI setting restarts only the background depth worker. If a new
configuration fails, the last working configuration is restored.

Initial values can also be supplied on the command line:

```bash
./run_with_latest_calibration.sh \
  --vpi-max-disparity 160 \
  --vpi-confidence-threshold 8192 \
  --vpi-uniqueness 0.4
```

OpenCV StereoSGBM remains available for comparison:

```bash
./run_with_latest_calibration.sh --backend opencv
```

## Validation

Test measured targets at 0.5, 1.0, 1.5, 2.0, and 3.0 metres before using the
settings for haptic decisions. Do not interpret a denser heatmap as proof of
accuracy, and do not convert invalid pixels into haptic distances.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```
