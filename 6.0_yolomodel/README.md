# YOLO11n baseline training and export

This folder contains the original indoor-object YOLO11n experiment and its
legacy deployment artifacts: `best.pt`, `best.onnx`, and `best.engine`.
`colab_train_yolo11n.ipynb` is the original Google Colab workflow.

The newer leakage-aware training workflow is
`colab_train_yolo11n_indoor_v2.ipynb`. Its completed checkpoint, evaluation
reports, predictions, and Orin-built engine are stored in
`../6.1_yolo11n_indoor_v2_results/`. Use 6.1 for the current 7.0 application.

The legacy engine can be rebuilt on the target Jetson with:

```bash
cd /home/orin_nano/Desktop/FYP2/6.0_yolomodel
/usr/bin/python3 export_best_pt_to_engine.py --force
```

TensorRT engine files are device- and TensorRT-version-specific. Build them on
the Orin Nano that will run them. Refer to `COLAB_NOTEBOOK.md` for the original
notebook walkthrough.
