import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

# Ultralytics imports matplotlib even for prediction. If matplotlib auto-selects
# Qt after QApplication exists, the Jetson's matplotlib/PySide6 versions clash
# while converting Qt.KeyboardModifier. Detection does not need a GUI backend.
os.environ.setdefault("MPLBACKEND", "Agg")

from ultralytics import YOLO

from config import (
    HEIGHT,
    WIDTH,
    YOLO_CONFIDENCE,
    YOLO_HALF,
    YOLO_IMAGE_SIZE,
    YOLO_IOU,
)
from data_models import DetectionResult


Device = Union[int, str]


class YoloDetector:
    """Owns one YOLO model and performs inference from a single worker thread."""

    def __init__(self, model_path: Path, device: Device):
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self.device = device
        self.uses_cuda = device != "cpu"
        if self.uses_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")

        self.device_label = f"cuda:{device}" if self.uses_cuda else "cpu"
        self.use_half = YOLO_HALF and self.uses_cuda
        self.model = YOLO(str(model_path), task="detect")

    def warmup(self):
        dummy_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.detect(dummy_frame)

    def detect(self, color_image: np.ndarray) -> List[DetectionResult]:
        predictions = self.model.predict(
            source=color_image,
            device=self.device,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU,
            half=self.use_half,
            verbose=False,
        )

        if not predictions or predictions[0].boxes is None:
            return []

        boxes = predictions[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        names = predictions[0].names

        detections: List[DetectionResult] = []
        for coordinates, confidence, class_id in zip(xyxy, confidences, class_ids):
            x1, y1, x2, y2 = (int(round(value)) for value in coordinates.tolist())
            label = names[class_id] if isinstance(names, (dict, list)) else str(class_id)
            detections.append(
                DetectionResult(
                    class_id=int(class_id),
                    label=str(label),
                    confidence=float(confidence),
                    bbox=(x1, y1, x2, y2),
                )
            )

        return detections
