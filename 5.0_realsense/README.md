# RealSense distance, object-detection, and haptic prototype

This PySide6 prototype combines Intel RealSense aligned hardware depth, three
full-height screen columns, YOLO object detection, and three DA7280 haptic
outputs. It is independent of the IMX219 calibration and VPI pipeline.

Raw haptic distance is the nearest valid RealSense pixel in each Left, Centre,
and Right third between 0.1 m and 3.0 m. YOLO detections receive an estimated
object depth for display, but tests enforce that detection results cannot be
used directly as actuator commands.

The default detector is `best.engine` at 640x640 with confidence 0.40. This is
the earlier model retained for the RealSense prototype; 7.0 uses the newer 6.1
engine.

## Run

```bash
cd /home/orin_nano/Desktop/FYP2/5.0_realsense
./run_with_sudo.sh
```

The launcher preserves access to the desktop Python packages while using the
system RealSense binding. Review `config.py` before connecting actuators.

Run hardware-free tests with:

```bash
/usr/bin/python3 -m unittest discover -s tests -v
```
