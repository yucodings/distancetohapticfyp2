# YOLO11n indoor-v2 results

This is the completed second training run used by the 7.0 IMX219 application.
Training starts from the official COCO-pretrained `yolo11n.pt` for transfer
learning; it does not resume from the old 6.0 custom checkpoint.

## Deployment artifacts

- `best_indoor_v2.pt`: best Ultralytics training checkpoint.
- `best_indoor_v2.onnx`: intermediate ONNX export.
- `best_indoor_v2.engine`: Orin Nano TensorRT deployment engine.
- `export_best_indoor_v2_to_engine.py`: guarded Jetson export and verification.
- `yolo11n_indoor_v2/`: training and merged-validation evidence.
- `yolo11n_indoor_v2_custom_test/`: custom-only evaluation evidence.
- `yolo11n_indoor_v2_test_predictions/`: saved prediction examples.

## Evaluation

| Evaluation split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Merged validation | 0.565 | 0.507 | 0.503 | 0.329 |
| Custom-only test | 0.879 | 0.807 | 0.877 | 0.624 |

The model has 27 classes. The custom-only test is the closest of these two
metrics to the target indoor dataset, but live field validation is still
required for lighting, camera viewpoint, blur, occlusion, and unseen rooms.

## TensorRT export

Run on the target Orin Nano with the camera/depth applications stopped:

```bash
cd /home/orin_nano/Desktop/FYP2/6.1_yolo11n_indoor_v2_results
/usr/bin/python3 export_best_indoor_v2_to_engine.py --force
```

The successful current export is static batch 1, 640x640, FP16, NMS disabled,
8.4 MiB, and uses TensorRT 10.3.0. Structural validation and one blank-frame
inference through the exact 7.0 backend passed. The engine SHA-256 is:

```text
bce881bfce0e5801582371ae56d01927d4fb5b1ee0a1bb2b7d69548e9d23d627
```

Do not move this engine to a different Jetson/TensorRT environment without
rebuilding it there.
