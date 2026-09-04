"""Load JetPack's GStreamer-enabled OpenCV and matching NumPy build."""

from __future__ import annotations

import sys
from pathlib import Path


SYSTEM_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
_original_path = list(sys.path)
if SYSTEM_DIST_PACKAGES.is_dir():
    system_path = str(SYSTEM_DIST_PACKAGES)
    if system_path in sys.path:
        sys.path.remove(system_path)
    sys.path.insert(0, system_path)

try:
    import numpy as np
    import cv2
finally:
    sys.path[:] = _original_path


def check_gstreamer() -> None:
    enabled = any(
        "GStreamer" in line and "YES" in line
        for line in cv2.getBuildInformation().splitlines()
    )
    if not enabled:
        raise RuntimeError(
            "The loaded OpenCV has no GStreamer support. "
            f"Loaded OpenCV {cv2.__version__} from {cv2.__file__}; "
            "use JetPack's /usr/lib/python3/dist-packages build."
        )
