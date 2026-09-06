# RealSense three-zone depth test

This project extends `3.0_testdepthrealsensed435i` with three independent
measurement boxes matching the layout of the IMX219 `2.1` test.

- Left: centred at one sixth of the image width
- Centre: centred at one half of the image width
- Right: centred at five sixths of the image width
- Box size: 30% of the 1280x720 source frame in both dimensions

The RealSense produces one hardware depth frame at 1280x720 and 30 FPS. Depth
is aligned to colour and converted to metres once. All three boxes calculate
independent medians from that shared depth map.

## Run

```bash
cd /home/orin_nano/Desktop/FYP2/3.1_testdepthrealsense3frame
./run_with_sudo.sh
```

When several RealSense devices are connected:

```bash
./run_with_sudo.sh --serial CAMERA_SERIAL
```

To change the accepted and displayed range:

```bash
./run_with_sudo.sh --min-depth 0.2 --max-depth 3.0
```

## Display and controls

The left panel shows aligned colour and the right panel shows fixed-scale
metric depth. Near pixels are red, far pixels are blue, and invalid pixels are
black.

- `D`: show or hide the depth panel
- `S`: save the current evidence bundle
- `Q` or `Esc`: quit

Saved evidence is written under `results/<timestamp>/` and contains annotated
colour, the annotated depth heatmap, metric depth, a valid mask, and
`three_zone_measurements.json`.

## Tests

The tests do not require a connected RealSense camera:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```
