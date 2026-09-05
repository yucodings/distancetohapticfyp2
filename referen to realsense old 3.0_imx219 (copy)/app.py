#!/usr/bin/env python3
"""Launch the IMX219 binocular distance-to-haptic interface."""

import sys

# Load JetPack's mutually compatible NumPy/OpenCV pair before any model or UI
# module imports NumPy indirectly.
from camera_backend import cv2, np  # noqa: F401
from PySide6.QtWidgets import QApplication

from main_window import MainWindow


def main() -> int:
    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
