# Low-Cost Binocular Camera Distance-to-Haptic Navigation System

This PySide6 application combines the tested `2.1.1.1` IMX219 three-zone
stereo visualization with three DA7280 haptic outputs behind a TCA9548A.

The stereo core is intentionally unchanged in purpose: synchronized capture,
rectification, VPI CUDA disparity, metric depth, and median distances for the
expanded left, centre, and right zones. No YOLO, object detection, point-cloud
processing, depth fusion, nearest-point search, or extra zone filtering feeds
the haptic decision.

## Haptic policy

Every zone is evaluated independently using its existing metre value:

| Distance | Fixed-strength behavior |
|---|---|
| Invalid or greater than 2.0 m | Off |
| 1.5 m through 2.0 m | 0.20 s on every 1.50 s |
| 0.5 m through less than 1.5 m | 0.20 s on every 0.80 s |
| Less than 0.5 m | 1.00 s on / 0.10 s off, repeating |

All active patterns use level 25/127, matching `test_haptic_pulse.sh`.

The hardware mapping is Left=SC2, Centre=SC3, Right=SC4. Haptics always start
disabled and require the `Enable Haptics` button. Stop, Emergency Stop, stale
depth, worker errors, and application shutdown request all motors off.

## UI

The responsive window follows a 65/35 left/right split. The left camera panel
uses about 77% of its column with three equal metre-only cards below it. The
right column contains equal-height depth heatmap, diagnostics, and log panels.
The camera and heatmap retain the 2.1.1.1 zone shading and labels.

## Run

Test the UI and cameras without enabling haptics:

```bash
cd /home/orin_nano/Desktop/FYP2/3.0_imx219
./run_with_latest_calibration.sh
```

Run with the I2C privileges required to enable the physical haptics:

```bash
./run_with_sudo.sh
```

Always validate Left, Centre, and Right orientation before pressing
`Enable Haptics`.

## Tests

```bash
/usr/bin/python3 -m unittest discover -s tests -v
```

The tests use mock I2C transports and never activate real motors.

