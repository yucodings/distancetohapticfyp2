# IMX219 expanded three-zone depth test

This project extends `2.1.1_testdepthimx3frame` without changing the original
folder. It divides every completed metric-depth map into three contiguous
vertical measurement zones:

```text
+------------+------------+------------+
|    LEFT    |   CENTRE   |   RIGHT    |
|    ZONE    |    ZONE    |    ZONE    |
+------------+------------+------------+
```

The zones cover the complete image width and the middle 80% of image height.
The top and bottom 10% are excluded to reduce ceiling, floor and rectification
border interference. Every zone reports its median valid depth and valid-pixel
percentage independently. A zone displays `Depth N/A` when it has fewer than
100 valid pixels.

The VPI/OpenCV stereo matcher still runs only once for each stereo pair. The
three measurements reuse that one disparity and metric-depth map, so expanding
the measurement regions does not triple CUDA work or memory use.

## Run

```bash
cd /home/orin_nano/Desktop/FYP2/2.1.1.1_testdepthimx3zone
./run_with_latest_calibration.sh
```

The launcher selects the newest calibration under `../1.0_calibration/images/`
and uses the VPI profile expected by the restored 2.1.1 pipeline.

## Controls

- `D`: toggle the disparity view.
- `P`: toggle built-in top/front 3D projections.
- `O`: open the latest point cloud in Open3D when installed.
- `S`: save annotated images, arrays, point cloud and three-zone JSON under
  `results_3zone/`.
- `Q` or `Esc`: stop.

Use `--backend opencv` only when comparing against the CPU StereoSGBM backend.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests cover contiguous zone geometry, small image handling, independent
median measurements, invalid-depth handling, valid percentages, overlays,
point-cloud integration, profile validation and saved report metadata.
