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
python3 test_depth_imx219.py
```

Press `D` to show disparity. Press `Q` or `Esc` to stop. The parameters can be
temporarily overridden with `--block-size` and `--num-disparities`; the
calibrated defaults in the script remain unchanged.

The optional `--backend vpi-cuda` path uses its own 256-disparity CUDA
configuration and is not the profile selected by Easy Mode.

Run offline checks with:

```bash
python3 -m unittest discover -s tests -v
```
