#!/usr/bin/env python3
"""Launch the low-cost IMX219 binocular distance-to-haptic UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

import stereo_core as core
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    options = RuntimeOptions(
        calibration=args.calibration,
        vpi_profile=args.vpi_profile,
        backend=args.backend,
        left_id=args.left_id,
        right_id=args.right_id,
        max_depth_m=args.max_depth,
    )
    application = QApplication(sys.argv[:1])
    window = MainWindow(options)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())

