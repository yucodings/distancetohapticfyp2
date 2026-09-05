# IMX219 18-zone depth evaluation

This project extends `2.1.3_testdepthimx9zone` into 18 independent regions
while leaving the nine-zone project unchanged.

```text
+------+------+------+------+------+------+
| U1   | U2   | U3   | U4   | U5   | U6   |
+------+------+------+------+------+------+
| M1   | M2   | M3   | M4   | M5   | M6   |
+------+------+------+------+------+------+
| L1   | L2   | L3   | L4   | L5   | L6   |
+------+------+------+------+------+------+
```

The 6-column by 3-row grid covers the complete image width and the middle 80%
of image height. The top and bottom 10% remain excluded to reduce ceiling,
floor and rectification-border interference. At 1280x720, the horizontal
boundaries are 0, 213, 426, 640, 853, 1066 and 1280 pixels. The vertical
boundaries are 72, 264, 456 and 648.

VPI CUDA computes disparity only once for every stereo pair. All 18 medians
reuse that metric-depth map; this project does not run 18 stereo matchers or
materially increase CUDA memory use. Each region is independent. A region
below 1.6% valid depth, or below 100 valid samples, reports `N/A` without
calculating a median.

## Run

```bash
cd /home/orin_nano/Desktop/FYP2/2.3_testdepthimx18zone
./run_with_latest_calibration.sh
```

The launcher selects the newest completed calibration under
`../1.0_calibration/images/` and the matching schema-2 VPI profile under
`../1.1_depth_visualization/results/`.

## Controls

- `D`: toggle the disparity view.
- `P`: toggle built-in top/front 3D projections.
- `O`: open the latest point cloud in Open3D when installed.
- `S`: save annotated images, arrays, point cloud, and
  `eighteen_zone_measurements.json` under `results_18zone/`.
- `Q` or `Esc`: stop.

Use `--backend opencv` only for a CPU StereoSGBM comparison.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests cover exact 6x3 geometry, full usable-area coverage, independent
medians, the 1.6% validity gate, overlays, point-cloud integration, profile
validation and 18-zone saved metadata.
