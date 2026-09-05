# IMX219 three-zone depth test

This project keeps the calibrated stereo pipeline from `2.1_testdepthimx219`
and measures three independent regions from each completed depth map:

- **Left** at one sixth of the image width
- **Centre** at one half of the image width
- **Right** at five sixths of the image width

All three boxes are centred vertically and are 30% of the source image width
and height. The VPI/OpenCV matcher runs only once for each stereo pair. Left,
centre, and right distances are median measurements selected from that shared
metric depth map. A zone displays `Depth N/A` when it has fewer than 100 valid
pixels; the other zones continue independently.

## Run

```bash
cd /home/orin_nano/Desktop/FYP2/2.1.1_testdepthin3frame
./run_with_latest_calibration.sh
```

The launcher selects the newest calibration under `../1.0_calibration/images/`
and the schema-2 profile at
`../1.1_depth_visualization/results/vpi_recommended_profile.json`. Startup
stops if the calibration hash in the profile does not match.

Direct execution uses the calibration and profile copied into the original
`2.1_testdepthimx219` project by default:

```bash
python3 test_depth_imx219_3frame.py
```

## Controls

- `D`: toggle disparity view.
- `P`: toggle built-in top/front 3D projections.
- `O`: open the latest point cloud in Open3D when installed.
- `S`: save annotated images, raw arrays, the point cloud, and
  `three_zone_measurements.json` under `results_3d/`.
- `Q` or `Esc`: stop.

Use `--backend opencv` only for comparison with the older SGBM backend.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests cover ROI geometry, independent medians, per-zone invalid handling,
profile validation, point-cloud integration, and the saved measurement report.
