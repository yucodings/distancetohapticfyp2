# IMX219 three-zone depth test

This project uses only the selected `1.0_calibration` NPZ. It does not read a
profile, report, image, or setting from `1.1_depth_visualization`.

The VPI/OpenCV matcher runs once per stereo pair. Left, centre, and right
boxes then calculate independent median distances from the same metric depth
map. Each box is 30% of the source image width and height.

## Run

```bash
cd /home/orin_nano/Desktop/FYP2/2.1.1_testdepthimx3frame
./run_with_latest_calibration.sh
```

The launcher selects the newest calibration under `../1.0_calibration/images/`.
Runtime defaults owned by this project are:

- maximum disparity: 160;
- confidence threshold: 8192;
- uniqueness: 0.4;
- P1/P2: 3/48;
- diagonal paths: disabled.

## Controls

- `W/Z`: increase/decrease VPI maximum disparity by 16.
- `E/C`: increase/decrease confidence by 4096.
- `U/J`: increase/decrease uniqueness; `-1` disables it.
- `R`: restore the 2.1.1 defaults.
- `D`: toggle disparity view.
- `P`: toggle built-in top/front 3D projections.
- `O`: open the latest point cloud in Open3D when available.
- `S`: save images, arrays, point cloud, and three-zone JSON report.
- `Q` or `Esc`: quit.

Changing VPI settings restarts only the background depth worker. If a new
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

Validate measured targets in all three boxes at 0.5, 1.0, 1.5, 2.0, and 3.0
metres before connecting depth results to haptic decisions.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```
