# IMX219 depth, object-detection, and haptic system

This is the combined deployment application: binocular IMX219 stereo depth,
three directional haptic outputs, and the 6.1 indoor YOLO11n TensorRT model.
Both cameras are required for depth. Only the rectified left frame is sent to
YOLO, and detection boxes are informational—they never control the motors.

```text
Left IMX219  ----+---- rectification ----+---- VPI CUDA stereo
                 |                       ^             |
Right IMX219 ----+-----------------------+         metric depth
                                                        |
                                                 18 zone medians
                                                        |
                                              nearest 6 per column
                                                        |
                                               Left/Centre/Right
                                                        |
                                                  haptic policy

Rectified left frame ---- YOLO11n TensorRT ---- UI boxes only
```

## Runtime design

- Capture and depth: 1280x720 at 30 FPS, VPI CUDA, maximum disparity 256.
- Backend grid: six columns by three rows (18 independent medians).
- Invalid cell: fewer than 100 valid pixels or less than 1.6% valid coverage.
- UI output: nearest valid median from six cells in each display column.
- Detector: `models/best_indoor_v2.engine`, 640x640, static batch 1, FP16.
- Detector rate: 2 FPS, with a one-frame newest-only mailbox.
- UI stability: confidence 0.40, same-class IoU matching, smoothed boxes, and
  persistence for two missed results; the third miss or 1.5-second age limit
  removes a box.
- Scheduling: depth has priority; detection drops work while CUDA is busy.
- Safety: stale/failed depth switches haptics off; detection failure only
  removes detection overlays.

The application uses its own TensorRT/CUDA runner and postprocessor. It
letterboxes the image, decodes raw YOLO output, applies class-aware NMS, and
maps boxes back to the 1280x720 rectified-left image. Runtime detection does
not import Ultralytics, Torch, Matplotlib, or SymPy.

## Detection model

The active engine comes from
`../6.1_yolo11n_indoor_v2_results/best_indoor_v2.pt` and contains 27 classes:

```text
chair, cleaning_cart, handrail, person, pillars, ramp, squat-toilet,
stairs, urinals, vending_machine, wet_floor_sign, bicycle, bench, cat,
dog, backpack, umbrella, handbag, suitcase, bottle, bed, dining table,
couch, toilet, potted plant, sink, refrigerator
```

The engine was built and verified on this Orin Nano using TensorRT 10.3.0.
Its input is `(1, 3, 640, 640)`, output is `(1, 31, 8400)`, embedded NMS is
disabled, and SHA-256 is
`bce881bfce0e5801582371ae56d01927d4fb5b1ee0a1bb2b7d69548e9d23d627`.
The previous `models/best.engine` remains available only as a rollback.

6.1 evaluation results were:

| Evaluation split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Merged validation | 0.565 | 0.507 | 0.503 | 0.329 |
| Custom-only test | 0.879 | 0.807 | 0.877 | 0.624 |

These are dataset results, not a guarantee of performance in every real room.

## Haptic policy

Each displayed column is evaluated independently at level 25/127:

| Distance | Repeating behavior |
|---|---|
| Invalid or greater than 2.0 m | Off |
| 1.5–2.0 m | 0.50 s on / 1.00 s off |
| 0.5–less than 1.5 m | 0.20 s on / 0.20 s off |
| Less than 0.5 m | 0.10 s on / 0.10 s off |

Hardware mapping is Left=SC2, Centre=SC3, Right=SC4 through the TCA9548A.
Haptics start disabled and require the UI button.

## Run

Camera, depth, and detection without haptic I2C access:

```bash
cd /home/orin_nano/Desktop/FYP2/7.0_distance_to_haptic_system
./run_with_latest_calibration.sh
```

Complete system with physical haptics:

```bash
./run_with_sudo.sh
```

Useful options:

```bash
./run_with_sudo.sh --no-detection
./run_with_latest_calibration.sh --detection-fps 1
./run_with_latest_calibration.sh --model /path/to/model.engine
```

Other detector settings are `--detection-confidence`, `--detection-iou`, and
`--detection-max-results`.

## Verification and monitoring

Hardware-free tests never activate motors:

```bash
/usr/bin/python3 -m unittest discover -s tests -v
```

Process one real stereo pair through VPI and TensorRT without haptics:

```bash
/usr/bin/python3 smoke_test_combined.py
```

Monitor shared Jetson RAM/GPU use with:

```bash
sudo tegrastats --interval 1000
```

If CUDA allocation fails, close memory-heavy desktop applications or lower
the detection rate. Keep at least roughly 700 MB–1 GB available during use.
