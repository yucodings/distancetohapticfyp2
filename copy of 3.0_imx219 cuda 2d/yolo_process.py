#!/usr/bin/env python3
"""Binary-pipe YOLO server; isolated from the GStreamer NumPy 1.21 process."""

from __future__ import annotations

import json
import os
import struct
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from config import YOLO_ALLOW_CPU_FALLBACK, YOLO_DEVICE, YOLO_MODEL_PATH
from detector import YoloDetector


PROTOCOL_PREFIX = "@@YOLO@@"


def emit(message: dict) -> None:
    print(PROTOCOL_PREFIX + json.dumps(message, separators=(",", ":")), flush=True)


def read_exact(size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sys.stdin.buffer.read(size - len(chunks))
        if not chunk:
            return b""
        chunks.extend(chunk)
    return bytes(chunks)


def create_detector() -> YoloDetector:
    try:
        detector = YoloDetector(YOLO_MODEL_PATH, YOLO_DEVICE)
        detector.warmup()
        return detector
    except Exception as cuda_error:
        if not YOLO_ALLOW_CPU_FALLBACK:
            raise
        detector = YoloDetector(YOLO_MODEL_PATH, "cpu")
        try:
            detector.warmup()
        except Exception as cpu_error:
            raise RuntimeError(
                f"CUDA failed ({cuda_error}); CPU fallback failed ({cpu_error})"
            ) from cpu_error
        return detector


def main() -> int:
    try:
        detector = create_detector()
    except Exception as error:
        emit({"type": "error", "message": str(error)})
        return 1
    emit({"type": "ready", "device": detector.device_label})

    while True:
        header_size_data = read_exact(4)
        if not header_size_data:
            return 0
        header_size = struct.unpack("!I", header_size_data)[0]
        header_data = read_exact(header_size)
        if not header_data:
            return 0
        header = json.loads(header_data.decode("utf-8"))
        payload = read_exact(int(header["payload_size"]))
        if not payload:
            return 0

        try:
            image = np.frombuffer(payload, dtype=np.uint8).reshape(header["shape"])
            started = time.perf_counter()
            detections = detector.detect(image)
            emit(
                {
                    "type": "result",
                    "frame_id": int(header["frame_id"]),
                    "device": detector.device_label,
                    "inference_ms": (time.perf_counter() - started) * 1000.0,
                    "detections": [
                        {
                            "class_id": item.class_id,
                            "label": item.label,
                            "confidence": item.confidence,
                            "bbox": list(item.bbox),
                        }
                        for item in detections
                    ],
                }
            )
        except Exception as error:
            emit({"type": "error", "message": str(error)})


if __name__ == "__main__":
    raise SystemExit(main())
