# IMX219 Distance, Object Detection, and Haptic Navigation System

This PySide6 application combines the proven `4.0_imx219` stereo-depth and
haptic project with the custom YOLO11n TensorRT model from `6.0_yolomodel`.

Both IMX219 cameras are always used for stereo depth. Only the rectified left
image is submitted to YOLO. Object detections are informational UI overlays;
they cannot change the three depth values or command an actuator.

```text
Left IMX219 ---- rectify left ----+---- VPI CUDA stereo ---- metric depth
                                  |              ^                  |
                                  |              |              18 medians
                                  |              |                  |
                                  |       rectify right        nearest six
                                  |              |             per column
                                  |        Right IMX219              |
                                  |                           L / C / R depth
                                  |                                  |
                                  |                            haptic policy
                                  |                                  |
                                  |                         three DA7280 LRAs
                                  |
                                  +---- YOLO11n TensorRT ---- UI boxes only
```

## Runtime design

- Stereo: full-resolution 1280x720 NVIDIA VPI CUDA, maximum disparity 256.
- Zones: 18 independent medians arranged as six columns by three rows.
- Invalid cell rule: less than 1.6% valid depth reports unavailable.
- Output: the nearest valid median from six cells for Left, Centre, and Right.
- Detector: `models/best.engine`, static 640x640, batch 1, FP16, 27 classes.
- Detector rate: 2 FPS by default.
- Detector mailbox: one frame; stale waiting frames are replaced.
- CUDA policy: depth blocks until it can run; detection drops a frame when the
  shared GPU is busy.
- Detection failure: stereo and haptics continue without boxes.
- Depth failure/staleness: all haptics are switched off.

The engine was exported with NMS disabled in its TensorRT metadata. The
project's direct TensorRT/CUDA runner performs letterboxing, output decoding,
class-aware NMS, and conversion of boxes back into rectified-left image
coordinates. It deliberately does not import Ultralytics, Torch, Matplotlib,
or SymPy at runtime, avoiding their NumPy conflict with JetPack's
GStreamer-enabled OpenCV.

## Haptic policy

Every displayed depth column is evaluated independently. YOLO results never
enter this policy.

| Distance | Fixed-strength behavior |
|---|---|
| Invalid or greater than 2.0 m | Off |
| 1.5 m through 2.0 m | 0.50 s on / 1.00 s off, repeating |
| 0.5 m through less than 1.5 m | 0.20 s on / 0.20 s off, repeating |
| Less than 0.5 m | 0.10 s on / 0.10 s off, repeating |

All active patterns use level 25/127. Hardware mapping is Left=SC2,
Centre=SC3, Right=SC4. Haptics start disabled and require the UI button.

## Run

Camera, depth, and detection without granting haptic I2C access:

```bash
cd /home/orin_nano/Desktop/FYP2/7.0_distance_to_haptic_system
./run_with_latest_calibration.sh
```

Complete system with physical haptics:

```bash
./run_with_sudo.sh
```

Complete depth and haptic system without loading TensorRT:

```bash
./run_with_sudo.sh --no-detection
```

Change the detector rate only after checking memory and depth performance:

```bash
./run_with_latest_calibration.sh --detection-fps 3
```

Other detector options:

```text
--model PATH
--detection-confidence 0.45
--detection-iou 0.50
--detection-max-results 30
```

## Monitor the Jetson

```bash
sudo tegrastats --interval 1000
```

Prefer at least 700 MB to 1 GB available RAM during sustained use. Continuous
swap growth, falling depth FPS, or CUDA allocation errors mean the detector
rate should be reduced or desktop applications should be closed.

## Tests

The suite is hardware-free and never activates physical motors:

```bash
cd /home/orin_nano/Desktop/FYP2/7.0_distance_to_haptic_system
/usr/bin/python3 -m unittest discover -s tests -v
```

It checks the copied depth/haptic behavior, exact engine metadata, detection
overlays, newest-frame inference, and non-blocking GPU scheduling.

To process one real stereo pair through both CUDA systems without initializing
the haptic hardware:

```bash
/usr/bin/python3 smoke_test_combined.py
```

## Model note

The training notes mention 29 classes, but the actual trained checkpoint and
TensorRT engine contain 27. Engine metadata is authoritative. The training log
reported overall validation mAP50 0.572 and mAP50-95 0.396. Field testing is
still required because several mixed segmentation/detection annotations were
ignored during training.
