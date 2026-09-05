# IMX219 nine-zone depth evaluation

This project extends `2.1.2_testdepthimx6zone` into nine independent
measurement regions while leaving the six-zone project unchanged.

```text
+----------------+----------------+----------------+
| Upper Left     | Upper Centre   | Upper Right    |
+----------------+----------------+----------------+
| Middle Left    | Middle Centre  | Middle Right   |
+----------------+----------------+----------------+
| Lower Left     | Lower Centre   | Lower Right    |
+----------------+----------------+----------------+
```

The grid covers the complete image width and the middle 80% of image height.
The top and bottom 10% remain excluded to reduce ceiling, floor and
rectification-border interference. At 1280x720, the horizontal boundaries are
0, 426, 853 and 1280 pixels. The vertical boundaries are 72, 264, 456 and 648.

VPI CUDA computes disparity only once for every stereo pair. All nine median
distances reuse that one metric-depth map, so this project does not run nine
stereo matchers or materially increase CUDA memory use. Invalid pixels are
excluded independently, and an invalid region does not affect another region.

## Run

```bash
cd /home/orin_nano/Desktop/FYP2/2.1.3_testdepthimx9zone
./run_with_latest_calibration.sh
```

The launcher selects the newest completed calibration under
`../1.0_calibration/images/` and the VPI profile used by the restored pipeline.

## Controls

- `D`: toggle disparity view.
- `P`: toggle built-in top/front 3D projections.
- `O`: open the latest point cloud in Open3D when installed.
- `S`: save annotated images, arrays, point cloud, and
  `nine_zone_measurements.json` under `results_9zone/`.
- `Q` or `Esc`: stop.

Use `--backend opencv` only for a CPU StereoSGBM comparison.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests cover exact 3x3 geometry, full usable-area coverage, edge dimensions,
independent medians, invalid-depth handling, overlays, point-cloud integration,
profile validation, and nine-zone saved metadata.
