# colab_train_yolo11n.ipynb — Google Colab Training Notebook

## Overview

`colab_train_yolo11n.ipynb` is a Google Colab notebook that handles the **complete
training pipeline** for the YOLO11n indoor accessibility object detection model — from
downloading the dataset all the way to exporting the trained weights. It is the cloud
equivalent of running all three local scripts (`download_roboflow_dataset.py`,
`build_indoor_coco_dataset.py`, and `train_yolo11n.py`) in sequence.

All training logic, hyperparameters, and dataset-building steps are **identical** to
the original `.py` scripts. The notebook simply adapts them to run inside a Colab
environment (no `argparse`, no `__file__`, paths rooted at `/content`).

---

## Why Use This Notebook?

| Reason | Detail |
|---|---|
| **Free GPU** | Google Colab provides a T4 GPU at no cost |
| **No local setup** | No need to install CUDA, PyTorch, or ultralytics locally |
| **Cloud storage** | Dataset is downloaded and built entirely in Colab's environment |
| **Portable** | Output `best.pt` can be downloaded and used anywhere |
| **Jetson-ready** | Produces a `.pt` file compatible with Jetson Nano deployment |

---

## Prerequisites

Before running the notebook:

1. **Google Account** — to access Google Colab
2. **Roboflow Private API Key** — to download the custom dataset
   - Go to [app.roboflow.com](https://app.roboflow.com)
   - Avatar → Settings → Roboflow API → copy Private API Key
3. **Runtime set to GPU** — in Colab: `Runtime → Change runtime type → T4 GPU`

---

## Notebook Structure

The notebook has **7 cells** that must be run top to bottom in order.

---

### Cell 1 — Install Dependencies

**What it does:**
Installs the two required Python packages and verifies that the GPU is available.

```python
!pip install -q ultralytics roboflow
```

**Packages installed:**
| Package | Purpose |
|---|---|
| `ultralytics` | YOLO11n model training and inference |
| `roboflow` | Downloading the custom dataset from Roboflow |

**Output to expect:**
```
PyTorch 2.x.x | CUDA available: True
GPU: Tesla T4
```

> ⚠️ If CUDA shows `False`, stop and change runtime to GPU first.

---

### Cell 2 — Configuration

**What it does:**
Defines all paths and training hyperparameters used by the rest of the notebook.
This is the **only cell you need to edit**.

**You must edit:**
```python
ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"   # ← paste your key here
```

**Paths configured (all under `/content`):**

| Variable | Path | Description |
|---|---|---|
| `PROJECT_ROOT` | `/content` | Root of the working directory |
| `CUSTOM_DIR` | `/content/dataset/my-second-project-...` | Roboflow dataset location |
| `OUTPUT_DIR` | `/content/dataset/indoor_coco_merged` | Merged dataset output |
| `CACHE_DIR` | `/content/dataset/coco_cache` | COCO annotation and image cache |
| `WEIGHTS_PATH` | `/content/yolo11n.pt` | Pretrained base weights |

**Training hyperparameters (identical to `train_yolo11n.py`):**

| Parameter | Value | Notes |
|---|---|---|
| `EPOCHS` | 100 | Maximum training epochs |
| `IMGSZ` | 640 | Input image size in pixels |
| `BATCH` | 16 | Batch size (safe for T4, raise to 32–64 on A100) |
| `WORKERS` | 4 | Dataloader worker threads |
| `PATIENCE` | 20 | Early stopping — stops if no improvement for 20 epochs |
| `RUN_NAME` | `yolo11n_indoor_accessibility` | Name of the training run folder |

---

### Cell 3 — Download Roboflow Dataset

**What it does:**
Connects to your Roboflow account and downloads version 2 of the custom dataset
in YOLOv8 format to `CUSTOM_DIR`.

**Dataset details:**
| Property | Value |
|---|---|
| Workspace | `test-workspace-dwuie` |
| Project | `my-second-project-person-chair-add-chair` |
| Version | 2 |
| Format | YOLOv8 (images + YOLO `.txt` labels) |

**Skip condition:** If `CUSTOM_DIR` already exists, the download is skipped automatically.

---

### Cell 4 — Download Pretrained Weights

**What it does:**
Downloads the official `yolo11n.pt` pretrained checkpoint from the Ultralytics GitHub
releases. This is the starting point for transfer learning.

```
Source: github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

**Skip condition:** If `yolo11n.pt` already exists at `WEIGHTS_PATH`, the download is skipped.

---

### Cell 5 — Build Merged Indoor-COCO Dataset

**What it does (mirrors `build_indoor_coco_dataset.py`):**

1. **Downloads COCO 2017 annotations** — `instances_train2017.json` and `instances_val2017.json`
2. **Filters indoor images** — selects only COCO images that contain at least one "indoor anchor" object (chair, bed, dining table, couch, toilet, potted plant, sink, refrigerator)
3. **Applies per-class quotas** — max 500 COCO train images per class, max 100 val images per class
4. **Downloads selected COCO images** — parallel download using 8 worker threads, cached on re-run
5. **Merges both datasets** — custom images prefixed `custom__`, COCO images prefixed `coco2017__`
6. **Converts COCO labels** — from COCO JSON bbox format to YOLO normalised format
7. **Writes `data.yaml`** — the unified class list and split paths for training

**Dataset composition after build:**

| Type | Train Images | Val Images |
|---|---|---|
| Custom (Roboflow) | All train split images | All valid split images |
| COCO 2017 (filtered) | Up to 5,000 | Up to 1,000 |
| **Total (approx.)** | **~5,000–6,000** | **~1,000–1,200** |

**Total classes: 29** (11 custom + 18 from COCO, with `person` and `chair` shared)

**Skip condition:** If `OUTPUT_DIR` already exists, the entire build step is skipped.

> ⏱️ **This cell takes the longest** — 30–60 min depending on Colab's network speed
> (COCO images are downloaded individually). On re-run it is fast because images are cached.

---

### Cell 6 — Train YOLO11n

**What it does (mirrors `train_yolo11n.py`):**
Runs the full training loop using Ultralytics `model.train()`.

```python
model = YOLO("yolo11n.pt", task="detect")
model.train(
    data      = "dataset/indoor_coco_merged/data.yaml",
    epochs    = 100,
    imgsz     = 640,
    batch     = 16,
    device    = "0",         # Colab GPU
    patience  = 20,
    amp       = True,        # FP16 mixed precision
    seed      = 42,
    ...
)
```

**Training features:**
| Feature | Detail |
|---|---|
| Transfer learning | Fine-tunes from COCO-pretrained `yolo11n.pt` |
| AMP | FP16 mixed precision — speeds up training on GPU |
| Early stopping | Stops automatically if val mAP does not improve for 20 epochs |
| Validation | Runs on val split at every epoch |
| Plots | Training curves (loss, mAP, precision, recall) saved automatically |
| Reproducibility | `seed=42`, `deterministic=True` |

**Output saved to:** `/content/runs/detect/yolo11n_indoor_accessibility/`

| File | Description |
|---|---|
| `weights/best.pt` | Best checkpoint (highest val mAP@0.5) |
| `weights/last.pt` | Final epoch checkpoint |
| `results.csv` | Epoch-by-epoch metrics |
| `confusion_matrix.png` | Class confusion matrix |
| `results.png` | Training curves |

> ⏱️ **Estimated time: 3–6 hours** on a free T4 GPU for 100 epochs at 640px.
> Colab free sessions allow up to ~12 hours — sufficient for a full run.

---

### Cell 7 — Download Weights

**What it does:**
Triggers a browser download of both `best.pt` and `last.pt` to your local machine.

```python
files.download("best.pt")
files.download("last.pt")
```

**Use `best.pt` for deployment.** `last.pt` is a backup in case you want to resume training.

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| `CUDA available: False` | Runtime not set to GPU | Runtime → Change runtime type → T4 GPU |
| Session disconnects mid-training | Free Colab timeout (~12 hrs) | Use `last.pt` + re-run Cell 6 with `exist_ok=True` to resume |
| Cell 5 fails mid-download | Network interruption | Re-run Cell 5 — already-downloaded images are cached and skipped |
| API key error | Wrong or expired key | Re-check key in [Roboflow settings](https://app.roboflow.com) |
| `data.yaml not found` | Cell 5 was skipped or failed | Re-run Cell 5 before Cell 6 |

---

## Relationship to Local Scripts

| Notebook Cell | Equivalent Local Script |
|---|---|
| Cell 3 | [`download_roboflow_dataset.py`](download_roboflow_dataset.py) |
| Cell 5 | [`build_indoor_coco_dataset.py`](build_indoor_coco_dataset.py) |
| Cell 6 | [`train_yolo11n.py`](train_yolo11n.py) |

The notebook combines all three into a single sequential workflow for cloud execution.

---

## After Training — Jetson Nano Deployment

Once you have `best.pt`:

```
best.pt  →  scp to Jetson Nano  →  jetson_setup.sh (once)  →  jetson_infer.py
```

See [`MODEL.md`](MODEL.md) for the full deployment guide.
