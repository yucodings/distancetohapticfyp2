#!/usr/bin/env python3
"""Launch the IMX219 distance, YOLO detection, and haptic navigation UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

import stereo_core as core
from config import (
    DETECTION_CONFIDENCE,
    DETECTION_FPS,
    DETECTION_IOU,
    DETECTION_MAX_RESULTS,
    DETECTOR_MODEL_PATH,
)
from main_window import MainWindow
from stream_worker import RuntimeOptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=core.DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--vpi-profile", type=Path, default=core.DEFAULT_VPI_PROFILE_PATH)
    parser.add_argument("--backend", choices=("vpi-cuda", "opencv"), default="vpi-cuda")
    parser.add_argument("--left-id", type=int, default=0)
    parser.add_argument("--right-id", type=int, default=1)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument(
        "--model",
        type=Path,
        default=DETECTOR_MODEL_PATH,
        help="static batch-1 FP16 YOLO11n TensorRT engine",
    )
    parser.add_argument(
        "--detection-fps",
        type=float,
        default=DETECTION_FPS,
        help="maximum YOLO submissions per second (default: 2)",
    )
    parser.add_argument(
        "--detection-confidence", type=float, default=DETECTION_CONFIDENCE
    )
    parser.add_argument("--detection-iou", type=float, default=DETECTION_IOU)
    parser.add_argument(
        "--detection-max-results", type=int, default=DETECTION_MAX_RESULTS
    )
    parser.add_argument(
        "--no-detection",
        action="store_true",
        help="run stereo depth and haptics without loading TensorRT",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.detection_fps <= 0.0:
        raise SystemExit("ERROR: --detection-fps must be positive")
    if not 0.0 < args.detection_confidence <= 1.0:
        raise SystemExit("ERROR: --detection-confidence must be in (0, 1]")
    if not 0.0 < args.detection_iou <= 1.0:
        raise SystemExit("ERROR: --detection-iou must be in (0, 1]")
    if args.detection_max_results < 1:
        raise SystemExit("ERROR: --detection-max-results must be positive")
    options = RuntimeOptions(
        calibration=args.calibration,
        vpi_profile=args.vpi_profile,
        backend=args.backend,
        left_id=args.left_id,
        right_id=args.right_id,
        max_depth_m=args.max_depth,
        detection_enabled=not args.no_detection,
        detector_model=args.model,
        detection_fps=args.detection_fps,
        detection_confidence=args.detection_confidence,
        detection_iou=args.detection_iou,
        detection_max_results=args.detection_max_results,
    )
    application = QApplication(sys.argv[:1])
    window = MainWindow(options)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
