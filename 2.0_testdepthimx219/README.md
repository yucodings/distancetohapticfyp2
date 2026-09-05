# IMX219 tuned depth test

This live test uses the latest recalibrated NPZ and matching `1.1` Easy Mode
visualization profile by default:

- NVIDIA VPI CUDA stereo
- schema-2 profile locked to the calibration SHA-256
- P1/P2, uniqueness, confidence and diagonal setting from Easy Mode
- 256 disparity search with the last 8 pixels rejected as limit artifacts

Run it from the project folder:

```bash
cd /home/orin_nano/Desktop/FYP2/2.1_testdepthimx219
./run_with_latest_calibration.sh
```

This launcher selects the newest completed NPZ under `../1.0_calibration/` and
the recommended profile under `../1.1_depth_visualization/results/`. Startup
stops with an error if their hashes do not match. The matching current files
are also copied beside the Python script, so `python3 test_depth_imx219.py`
uses the same pair until calibration or Easy Mode is run again.

The live controls are:

- `D`: toggle the disparity view.
- `P`: toggle built-in top/front 3D projections. Open3D is not required.
- `S`: save the current rectified image, depth heatmap, float disparity/depth
  arrays, valid mask, orthographic 3D image and binary PLY under `results_3d/`.
- `O`: open the latest cloud in Open3D when that optional package is installed.
- `Q` or `Esc`: stop.

Use `--backend opencv` to deliberately compare the older SGBM block-size 11,
160-disparity profile. `--block-size` and `--num-disparities` apply only to
that comparison backend.

The normal distance calculation remains the efficient Z-depth path. Pressing
`P`, `S` or `O` reconstructs full XYZ from the same disparity and calibration
`Q` matrix for diagnostics. A point cloud does not improve the underlying
stereo depth by itself; it makes geometry and bad matches easier to inspect.

The default VPI path uses the exact matching Easy Mode profile. This folder is
diagnostic only; the current result still contains sparse false-near surfaces,
so it must not be treated as approval to enable the `3.0` actuators.

Run offline checks with:

```bash
python3 -m unittest discover -s tests -v
```
