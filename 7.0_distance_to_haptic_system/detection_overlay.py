"""Draw informational YOLO results without changing depth or haptic data."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from data_models import Detection


def _class_color(class_id: int) -> tuple[int, int, int]:
    palette = (
        (255, 191, 0),
        (0, 215, 255),
        (255, 96, 96),
        (180, 105, 255),
        (80, 220, 120),
        (255, 150, 40),
    )
    return palette[class_id % len(palette)]


def draw_detections(
    frame: np.ndarray, detections: Iterable[Detection]
) -> np.ndarray:
    """Draw boxes in-place and return the same image for convenient chaining."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Detection overlay requires an HxWx3 image")
    height, width = frame.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width - 1, int(x2)))
        y2 = max(0, min(height - 1, int(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        color = _class_color(detection.class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{detection.label} {detection.confidence:.0%}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )
        label_top = max(0, y1 - label_height - baseline - 7)
        label_right = min(width - 1, x1 + label_width + 8)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (label_right, y1),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(label_height + 1, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame
