# IMX219 tuned depth test

This live test uses the copied calibrated NPZ and the automatically selected
approximately 1 m profile by default:

- OpenCV StereoSGBM
- block size 11
- 160 disparities
- no WLS filtering

Run it from the project folder:

```bash
cd /home/orin_nano/Desktop/FYP2/2.1_testdepthimx219
./run_with_latest_calibration.sh
```

This launcher selects the newest completed NPZ under `../1.0_calibration/`
without overwriting the older copied calibration beside the Python script.
To deliberately test that fixed local copy instead, run
`python3 test_depth_imx219.py`.

The live controls are:

- `D`: toggle the disparity view.
- `P`: toggle built-in top/front 3D projections. Open3D is not required.
- `S`: save the current rectified image, depth heatmap, float disparity/depth
  arrays, valid mask, orthographic 3D image and binary PLY under `results_3d/`.
- `O`: open the latest cloud in Open3D when that optional package is installed.
- `Q` or `Esc`: stop.

The parameters can be temporarily overridden with `--block-size` and
`--num-disparities`; the calibrated defaults in the script remain unchanged.

The normal distance calculation remains the efficient Z-depth path. Pressing
`P`, `S` or `O` reconstructs full XYZ from the same disparity and calibration
`Q` matrix for diagnostics. A point cloud does not improve the underlying
stereo depth by itself; it makes geometry and bad matches easier to inspect.

The optional `--backend vpi-cuda` path uses its own 256-disparity CUDA
configuration and is not the profile selected by Easy Mode.

Run offline checks with:

```bash
python3 -m unittest discover -s tests -v
```
