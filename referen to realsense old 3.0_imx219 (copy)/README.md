# IMX219 Binocular Distance-to-Haptic System

This project combines two 8 MP IMX219 CSI sensors, calibrated stereo depth,
three DA7280 LRA channels, and informational YOLO detection on a Jetson Orin
Nano. Runtime capture is 1280x720 at 30 FPS because calibration maps are tied
to resolution and sensor mode.

## Data flow

1. Sensor 0 and sensor 1 are captured through low-latency GStreamer pipelines.
2. Frames are paired by closest host arrival time and rectified with
   `stereo_calibration.npz`.
3. A latest-frame worker computes VPI CUDA disparity, confidence and metric Z
   depth from the saved Q matrix. VPI CUDA is the only stereo matcher; failure
   stops depth output and haptics instead of changing to a CPU algorithm.
4. Valid metric depth is split into a 3x3 grid (`LT/CT/RT`, `LM/CM/RM`,
   `LB/CB/RB`). Each cell reports the median of its confidence-supported
   pixels inside a centered 60% sampling ROI, equivalent to the small centered
   ROI method in `test_depth_imx219.py`. A cell needs at least 100 valid pixels.
5. Every cell median comes only from the current depth frame. Each actuator
   column uses the nearest of its three cells, and a different winning cell
   must remain nearest for two results before replacing the current winner.
   The bottom row is treated normally. Missing or stale depth still requests
   an immediate stop.
6. Only these stereo grid distances control the SC2/SC3/SC4 actuators.
7. YOLO runs in an isolated process and is used only for display information.

## Run

For the first physical camera check, set `ENABLE_ACTUATORS = False` in
`config.py`. Then run:

```bash
./run_with_sudo.sh
```

The isolated YOLO process is necessary because JetPack's GStreamer-enabled
OpenCV uses the system NumPy 1.21 build, while the installed Ultralytics stack
uses the newer user-site NumPy build.

## Tests

```bash
cd /home/orin_nano/Desktop/FYP2/3.0_imx219
python3 -m unittest discover -s tests -v
```

The tests verify asset hashes, calibration metadata, disparity-to-depth
conversion, confidence rejection, nine current-frame cell medians, per-column
selection, winner confirmation, haptic hysteresis,
DA7280 register order, fault handling, and final mux shutdown.

## Hardware mapping

- I2C bus: 1
- TCA9548A: `0x70`
- DA7280: `0x4A`
- Left: SC2
- Center: SC3
- Right: SC4

DA7280 actuator voltage/current/impedance/resonant-period registers are not
guessed or overwritten. They must match the exact installed LRA.

## Applied stereo profiles

- `2.1_testdepthimx219`: the exact OpenCV comparison profile, block size 11 and
  160 disparities.
- `3.0_imx219`: VPI CUDA only, with 256 disparities and an 8-pixel upper-limit
  safety margin. Run
  `../1.1_depth_visualization/run_easy_tuner.sh`, place the target at 1 m, and
  press `A` to produce `vpi_tuned_profile.json` automatically.
- Only a multi-frame safety-scored schema-2 VPI profile whose calibration
  SHA-256 matches this application is accepted. Older schema-1 profiles are
  ignored in favor of safe VPI CUDA defaults; rerun Easy Mode to replace them.
  If the profile is absent, documented safe VPI defaults are used. If VPI
  initialization or processing fails, the application reports the error and
  safely stops instead of running CPU stereo depth.

The current copied calibration and profile both come from the 2026-09-04
recalibration/Easy Mode run and share calibration SHA-256 `8198efd...`.
Actuators remain disabled until live distance checks are complete.

With this calibration (`fB` about 68 px·m), the 248-pixel accepted upper bound
has a theoretical near limit around 0.27 m. Values closer to the 256-disparity
search boundary are rejected as artifacts. Depth configured down to 0.1 m
cannot be recovered by this stereo matcher; invalid pixels remain black and
do not drive haptics.
