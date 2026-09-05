# Low-Cost Binocular Camera Distance-to-Haptic Navigation System

This PySide6 application combines an 18-zone IMX219 depth backend with the
tested three-column visualization and three DA7280 haptic outputs behind a
TCA9548A.

The stereo core performs synchronized capture, rectification, one VPI CUDA
disparity calculation, metric depth, and 18 median measurements arranged as
six columns by three rows. Cells below 1.6% valid depth report `N/A`. A small
reducer groups each adjacent pair of backend columns and selects the nearest
valid median among its six cells. Only the resulting Left, Centre, and Right
values reach the UI and haptic policy. No YOLO, object detection, point-cloud
processing, depth fusion, nearest-pixel search, or extra filtering feeds the
haptic decision.

```text
 U1  U2  |  U3  U4  |  U5  U6
 M1  M2  |  M3  M4  |  M5  M6
 L1  L2  |  L3  L4  |  L5  L6
    |          |          |
 nearest     nearest     nearest
    |          |          |
  Left       Centre      Right
```

## Haptic policy

Every zone is evaluated independently using its existing metre value:

| Distance | Fixed-strength behavior |
|---|---|
| Invalid or greater than 2.0 m | Off |
| 1.5 m through 2.0 m | 0.50 s on / 1.00 s off, repeating |
| 0.5 m through less than 1.5 m | 0.20 s on / 0.20 s off, repeating |
| Less than 0.5 m | 0.10 s on / 0.10 s off, repeating |

All active patterns use level 25/127, matching `test_haptic_pulse.sh`.

The hardware mapping is Left=SC2, Centre=SC3, Right=SC4. Haptics always start
disabled and require the `Enable Haptics` button. Stop, Emergency Stop, stale
depth, worker errors, and application shutdown request all motors off.

## UI

The responsive window follows a 65/35 left/right split. The left camera panel
uses about 77% of its column with three equal metre-only cards below it. The
right column contains equal-height depth heatmap, diagnostics, and log panels.
The camera and heatmap retain the three-column shading and labels; the 18
backend regions are not drawn in the normal application UI.

## Run

Test the UI and cameras without enabling haptics:

```bash
cd /home/orin_nano/Desktop/FYP2/4.0_imx219
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
