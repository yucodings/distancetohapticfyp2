# IMX219 six-zone depth evaluation

This project extends `2.1.1.1_testdepthimx3zone` into six independent
measurement regions while leaving the source project unchanged.

```text
+----------------+----------------+----------------+
| Upper Left     | Upper Centre   | Upper Right    |
+----------------+----------------+----------------+
| Lower Left     | Lower Centre   | Lower Right    |
+----------------+----------------+----------------+
```

The grid covers the complete image width and the middle 80% of image height.
The top and bottom 10% remain excluded to reduce ceiling, floor and
rectification-border interference. At 1280x720 the horizontal boundaries are
0, 426, 853 and 1280 pixels; the vertical boundaries are 72, 360 and 648.

VPI CUDA computes disparity only once for every stereo pair. All six median
distances reuse that single metric-depth map, so this project does not run six
stereo matchers or multiply CUDA memory usage. Invalid pixels are excluded
independently, and one zone becoming unavailable does not invalidate another.

## Run

```bash
cd /home/orin_nano/Desktop/FYP2/2.1.2_testdepthimx6zone
./run_with_latest_calibration.sh
```

The launcher selects the newest completed calibration under
`../1.0_calibration/images/` and the VPI profile used by the restored pipeline.

## Controls

- `D`: toggle disparity view.
- `P`: toggle built-in top/front 3D projections.
- `O`: open the latest point cloud in Open3D when installed.
- `S`: save annotated images, raw arrays, point cloud and
  `six_zone_measurements.json` under `results_6zone/`.
- `Q` or `Esc`: stop.

Use `--backend opencv` only for a CPU StereoSGBM comparison.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests cover exact 3x2 geometry, full usable-area coverage, edge image
sizes, independent medians, per-zone invalid handling, overlays, point-cloud
integration, profile validation and six-zone saved metadata.
