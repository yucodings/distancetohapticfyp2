# IMX219 stereo calibration

Run `python3 mycalibration.py`, show the complete 9x6-inner-corner board to
both cameras, and press `SPACE` to save a pair. Capture at least 20 sharp pairs
with the board at different distances, rotations, image edges and corners.
Press `Q` to calibrate.

Each timestamped session now contains:

```text
left/ and right/                 unchanged raw stereo pairs
detected_corners/left|right/     accepted pairs with corner overlays
rejected/left|right/             rejected detection evidence
quality_rejected/left|right/     detected pairs rejected by geometry checks
rectified_validation/            every accepted pair with epipolar guides
rectified_preview.png             first rectified validation pair
calibration_report.json           per-pair and aggregate quality metrics
stereo_calibration.npz            complete runtime calibration
```

Review every detected-corner image. Reject the calibration if corners are
ordered incorrectly, the board is blurred, or the rectified checkerboard
corners do not follow the same horizontal guides. The report warns when the
pair-level rectified vertical P95 exceeds 1 pixel.

Before saving the NPZ, normal calibration now automatically removes pairs
that are clearly unreliable. The default checks are:

- checkerboard bounding-box area must be at least 1.2% of each image;
- per-camera pair reprojection RMS must not exceed 1.5 pixels;
- pair epipolar P95 must not exceed 3.0 pixels;
- rectified vertical P95 must not exceed 2.5 pixels.

The calibration is solved again after each rejection round. It fails safely
instead of producing an NPZ if fewer than 20 pairs remain or filtering cannot
converge. Rejection reasons and each calibration round are saved in the JSON
report.

These are deliberately conservative rejection limits for removing clearly bad
pairs. The final report still uses 1.0 pixel as the stricter review warning;
automatic filtering must not silently remove most of a dataset just to make a
single metric look good.

## Recalibrate existing images

No camera capture is needed. To automatically use the newest session that
contains raw `left/` and `right/` images, run:

```bash
cd /home/orin_nano/Desktop/FYP2/1.0_calibration
python3 recalibrate_existing.py
```

To select a particular session:

```bash
python3 recalibrate_existing.py \
  --session images/2026-09-04_12-53-34_357493
```

The original raw images and original NPZ are not overwritten. A new
`*_recalibrated` session is created with its own filtered NPZ, report,
rectified evidence and quality-rejection images. The `1.1` and `2.1` latest
calibration launchers will then select that new NPZ automatically.

Run non-camera tests with:

```bash
python3 -m unittest discover -s tests -v
```
