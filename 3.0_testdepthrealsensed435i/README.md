# Intel RealSense D435i centre-depth test

This diagnostic project requests aligned 1280x720 colour and depth at 30 FPS,
converts the hardware depth frame to metres, and reports the median depth in a
20% by 20% centre ROI. Invalid depth appears black; valid depth is red when
near and blue when far.

## Run

```bash
cd /home/orin_nano/Desktop/FYP2/3.0_testdepthrealsensed435i
./run_with_sudo.sh
```

When multiple RealSense devices are connected, select one with
`--serial CAMERA_SERIAL`. The accepted display range can be changed with
`--min-depth 0.1 --max-depth 4.0`.

Controls are `D` to toggle the heatmap and `Q` or `Esc` to quit. This project
does not run stereo calibration, VPI, YOLO, or haptic actuators.
