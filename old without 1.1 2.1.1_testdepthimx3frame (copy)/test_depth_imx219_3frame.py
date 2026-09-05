#!/usr/bin/env python3
"""Test three-zone live stereo depth from an 8 MP IMX219 binocular camera.

The program captures 1280x720 frames from both CSI sensors, rectifies them
with the selected calibration NPZ, computes full-resolution disparity with
project-owned VPI CUDA settings, calculates metric Z depth from the
saved Q matrix, and reports median distances in left, centre, and right regions. Full XYZ
reconstruction is available as an optional diagnostic and PLY export; it uses
the same disparity and does not replace or improve the stereo matcher.

Controls:
    Q or Esc  quit
    D         toggle the disparity debug window
    W/Z       increase/decrease VPI disparity range
    E/C       increase/decrease VPI confidence
    U/J       increase/decrease VPI uniqueness
    R         restore 2.1.1 VPI defaults
    P         toggle built-in top/front 3D projections
    O         open the latest cloud in Open3D, when installed
    S         save current depth, projections, arrays, and PLY evidence
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Jetson's Ubuntu OpenCV build contains GStreamer support. A pip OpenCV wheel
# often shadows it and normally cannot open nvarguscamerasrc pipelines.
SYSTEM_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
if SYSTEM_DIST_PACKAGES.is_dir():
    system_packages = str(SYSTEM_DIST_PACKAGES)
    if system_packages in sys.path:
        sys.path.remove(system_packages)
    sys.path.insert(0, system_packages)

import cv2
import numpy as np

from pointcloud_utils import (
    disparity_to_xyz,
    make_orthographic_view,
    sample_point_cloud,
    save_binary_ply,
    show_open3d_snapshot,
)


# ---------------------------------------------------------------------------
# User-adjustable settings
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_PROJECT_DIR = SCRIPT_DIR.parent / "2.1_testdepthimx219"
CALIBRATION_ROOT = SCRIPT_DIR.parent / "1.0_calibration" / "images"
CALIBRATION_FALLBACK = SOURCE_PROJECT_DIR / "stereo_calibration.npz"


def newest_calibration_path() -> Path:
    candidates = tuple(CALIBRATION_ROOT.glob("*/stereo_calibration.npz"))
    if not candidates:
        return CALIBRATION_FALLBACK
    return max(candidates, key=lambda path: path.stat().st_mtime)


DEFAULT_CALIBRATION_PATH = newest_calibration_path()

LEFT_SENSOR_ID = 0
RIGHT_SENSOR_ID = 1

# These must agree with the copied calibration for this test.
EXPECTED_WIDTH = 1280
EXPECTED_HEIGHT = 720
EXPECTED_FPS = 30
EXPECTED_SENSOR_MODE = 4

MIN_DEPTH_M = 0.20
MAX_DEPTH_M = 5.00
MIN_ROI_DEPTH_SAMPLES = 100
ROI_WIDTH_FRACTION = 0.30
ROI_HEIGHT_FRACTION = 0.30
ROI_HORIZONTAL_CENTRES = (
    ("Left", 1.0 / 6.0),
    ("Centre", 1.0 / 2.0),
    ("Right", 5.0 / 6.0),
)

SGBM_MIN_DISPARITY = 0
SGBM_NUM_DISPARITIES = 128
SGBM_BLOCK_SIZE = 9

# These settings belong to 2.1.1 and are not loaded from the 1.1 tuner.
VPI_MIN_DISPARITY = 0
VPI_MAX_DISPARITY = 160
VPI_DISPARITY_SAFETY_MARGIN_PX = 8.0
VPI_WINDOW = 5

DEFAULT_STEREO_BACKEND = "vpi-cuda"
# Preserve as many VPI disparities as its Python API allows, then apply a
# configurable confidence filter in Python. VPI 3.2 clamps its CUDA threshold
# to at least 1, so confidence 0 always means invalid.
VPI_INTERNAL_CONFIDENCE_THRESHOLD = 1
VPI_MIN_CONFIDENCE = 8192
VPI_P1 = 3
VPI_P2 = 48
VPI_UNIQUENESS = 0.4
VPI_INCLUDE_DIAGONALS = False
VPI_UNIQUENESS_LEVELS = (-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

CAMERA_FRAME_BUFFER_SIZE = 8
MAX_CONSECUTIVE_CAPTURE_FAILURES = 10

DISPLAY_PANEL_WIDTH = 640
DISPLAY_PANEL_HEIGHT = 360
STATUS_PANEL_HEIGHT = 230
SHOW_DISPARITY_AT_START = False
SHOW_POINT_CLOUD_AT_START = False
POINT_CLOUD_STRIDE = 4
POINT_CLOUD_WINDOW = "IMX219 3D diagnostic: top and front"
RESULTS_3D_DIR = SCRIPT_DIR / "results_3d"


@dataclass(frozen=True)
class Calibration:
    path: Path
    keys: tuple[str, ...]
    width: int
    height: int
    sensor_mode: int
    capture_fps: int
    baseline_m: float
    q_matrix: np.ndarray
    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray


@dataclass(frozen=True)
class RectificationMaps:
    """Fixed-point OpenCV maps generated once during startup."""

    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray


@dataclass(frozen=True)
class VpiRuntimeSettings:
    min_disparity: int
    max_disparity: int
    confidence_threshold: int
    p1: int
    p2: int
    uniqueness: float
    include_diagonals: bool
    disparity_safety_margin_px: float

    def validate(self) -> None:
        if not 0 <= self.min_disparity < self.max_disparity <= 256:
            raise ValueError("VPI disparity range must satisfy 0 <= min < max <= 256")
        if not 0 <= self.confidence_threshold <= 65535:
            raise ValueError("VPI confidence threshold must be 0..65535")
        if not 0 < self.p1 <= self.p2 < 256:
            raise ValueError("VPI penalties must satisfy 0 < P1 <= P2 < 256")
        if self.uniqueness != -1.0 and not 0.0 <= self.uniqueness <= 1.0:
            raise ValueError("VPI uniqueness must be -1 (off) or between 0 and 1")
        if self.max_disparity % 16:
            raise ValueError("VPI maximum disparity must be a multiple of 16")
        if not 0.0 < self.disparity_safety_margin_px < (
            self.max_disparity - self.min_disparity
        ):
            raise ValueError("VPI disparity safety margin must fit inside the range")


@dataclass(frozen=True)
class StereoFrame:
    sequence: int
    captured_at: float
    pair_skew_ms: float
    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True)
class CameraFrame:
    sequence: int
    arrived_at: float
    image: np.ndarray


@dataclass(frozen=True)
class DepthInput:
    sequence: int
    left_rectified: np.ndarray
    right_raw: np.ndarray


@dataclass(frozen=True)
class DepthMeasurement:
    name: str
    distance_m: Optional[float]
    box: tuple[int, int, int, int]
    sample_count: int
    valid_percentage: float


@dataclass(frozen=True)
class DepthResult:
    sequence: int
    disparity: np.ndarray
    depth_map: np.ndarray
    valid_depth_mask: np.ndarray
    left_rectified: np.ndarray
    depth_view: np.ndarray
    measurements: tuple[DepthMeasurement, ...]
    valid_percentage: float
    depth_fps: float
    diagnostic_lines: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="stereo calibration NPZ (default: newest completed 1.0 result)",
    )
    parser.add_argument("--left-id", type=int, default=LEFT_SENSOR_ID)
    parser.add_argument("--right-id", type=int, default=RIGHT_SENSOR_ID)
    parser.add_argument(
        "--max-depth",
        type=float,
        default=MAX_DEPTH_M,
        help="maximum displayed/measured distance in metres",
    )
    parser.add_argument(
        "--backend",
        choices=("vpi-cuda", "opencv"),
        default=DEFAULT_STEREO_BACKEND,
        help="stereo engine; neither backend reads 1.1 tuner data",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=SGBM_BLOCK_SIZE,
        help="OpenCV SGBM block size (odd and at least 3; baseline default: 9)",
    )
    parser.add_argument(
        "--num-disparities",
        type=int,
        default=SGBM_NUM_DISPARITIES,
        help="OpenCV disparity count (multiple of 16; baseline default: 128)",
    )
    parser.add_argument(
        "--vpi-confidence-threshold",
        type=int,
        default=VPI_MIN_CONFIDENCE,
        help="initial VPI confidence threshold, 0-65535",
    )
    parser.add_argument(
        "--vpi-uniqueness",
        type=float,
        default=VPI_UNIQUENESS,
        help="initial VPI uniqueness (-1 disables it; otherwise 0.0-1.0)",
    )
    parser.add_argument(
        "--vpi-max-disparity",
        type=int,
        default=VPI_MAX_DISPARITY,
        help="initial VPI maximum disparity, 16-256 in steps of 16",
    )
    return parser.parse_args()


def _read_scalar(data: Any, key: str, value_type: Any) -> Any:
    value = np.asarray(data[key])
    if value.size != 1:
        raise RuntimeError(
            f"Calibration key '{key}' must be scalar, got shape {value.shape}"
        )
    return value_type(value.reshape(-1)[0])


def load_calibration(path: Path) -> Calibration:
    """Load the saved maps and Q matrix without changing the NPZ."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration NPZ not found: {path}")

    required_keys = (
        "image_width",
        "image_height",
        "sensor_mode",
        "capture_fps",
        "baseline_m",
        "Q",
        "left_map1",
        "left_map2",
        "right_map1",
        "right_map2",
    )

    try:
        with np.load(path, allow_pickle=False) as data:
            keys = tuple(data.files)
            missing = [key for key in required_keys if key not in keys]
            if missing:
                raise RuntimeError(
                    "Calibration NPZ is missing required keys: "
                    + ", ".join(missing)
                )

            calibration = Calibration(
                path=path,
                keys=keys,
                width=_read_scalar(data, "image_width", int),
                height=_read_scalar(data, "image_height", int),
                sensor_mode=_read_scalar(data, "sensor_mode", int),
                capture_fps=_read_scalar(data, "capture_fps", int),
                baseline_m=_read_scalar(data, "baseline_m", float),
                q_matrix=np.asarray(data["Q"], dtype=np.float64).copy(),
                left_map1=data["left_map1"].copy(),
                left_map2=data["left_map2"].copy(),
                right_map1=data["right_map1"].copy(),
                right_map2=data["right_map2"].copy(),
            )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not read calibration NPZ '{path}': {exc}") from exc

    if calibration.q_matrix.shape != (4, 4):
        raise RuntimeError(
            f"Calibration Q must have shape (4, 4), got {calibration.q_matrix.shape}"
        )

    expected_map_shape = (calibration.height, calibration.width)
    for name, rectification_map in (
        ("left_map1", calibration.left_map1),
        ("left_map2", calibration.left_map2),
        ("right_map1", calibration.right_map1),
        ("right_map2", calibration.right_map2),
    ):
        if rectification_map.shape[:2] != expected_map_shape:
            raise RuntimeError(
                f"{name} has shape {rectification_map.shape[:2]}, expected "
                f"{expected_map_shape}"
            )

    validate_calibration_settings(calibration)
    return calibration


def default_vpi_settings() -> VpiRuntimeSettings:
    settings = VpiRuntimeSettings(
        min_disparity=VPI_MIN_DISPARITY,
        max_disparity=VPI_MAX_DISPARITY,
        confidence_threshold=VPI_MIN_CONFIDENCE,
        p1=VPI_P1,
        p2=VPI_P2,
        uniqueness=VPI_UNIQUENESS,
        include_diagonals=VPI_INCLUDE_DIAGONALS,
        disparity_safety_margin_px=VPI_DISPARITY_SAFETY_MARGIN_PX,
    )
    settings.validate()
    return settings


def vpi_settings_from_args(args: argparse.Namespace) -> VpiRuntimeSettings:
    settings = VpiRuntimeSettings(
        min_disparity=VPI_MIN_DISPARITY,
        max_disparity=args.vpi_max_disparity,
        confidence_threshold=args.vpi_confidence_threshold,
        p1=VPI_P1,
        p2=VPI_P2,
        uniqueness=args.vpi_uniqueness,
        include_diagonals=VPI_INCLUDE_DIAGONALS,
        disparity_safety_margin_px=VPI_DISPARITY_SAFETY_MARGIN_PX,
    )
    settings.validate()
    return settings


def adjust_vpi_settings(
    settings: VpiRuntimeSettings,
    action: str,
) -> VpiRuntimeSettings:
    """Return bounded settings for one live keyboard adjustment."""
    if action == "more_disparity":
        updated = replace(settings, max_disparity=min(256, settings.max_disparity + 16))
    elif action == "less_disparity":
        updated = replace(settings, max_disparity=max(16, settings.max_disparity - 16))
    elif action == "more_confidence":
        updated = replace(
            settings,
            confidence_threshold=min(65535, settings.confidence_threshold + 4096),
        )
    elif action == "less_confidence":
        updated = replace(
            settings,
            confidence_threshold=max(0, settings.confidence_threshold - 4096),
        )
    elif action in ("more_uniqueness", "less_uniqueness"):
        index = min(
            range(len(VPI_UNIQUENESS_LEVELS)),
            key=lambda item: abs(VPI_UNIQUENESS_LEVELS[item] - settings.uniqueness),
        )
        delta = 1 if action == "more_uniqueness" else -1
        index = min(len(VPI_UNIQUENESS_LEVELS) - 1, max(0, index + delta))
        updated = replace(settings, uniqueness=VPI_UNIQUENESS_LEVELS[index])
    elif action == "reset":
        updated = default_vpi_settings()
    else:
        raise ValueError(f"Unknown VPI adjustment: {action}")
    updated.validate()
    return updated


def validate_calibration_settings(calibration: Calibration) -> None:
    expected = (
        EXPECTED_WIDTH,
        EXPECTED_HEIGHT,
        EXPECTED_SENSOR_MODE,
        EXPECTED_FPS,
    )
    actual = (
        calibration.width,
        calibration.height,
        calibration.sensor_mode,
        calibration.capture_fps,
    )
    if actual != expected:
        raise RuntimeError(
            "The calibration does not match this 1280x720@30 test. "
            f"Expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}, sensor mode "
            f"{EXPECTED_SENSOR_MODE}, {EXPECTED_FPS} FPS; got "
            f"{calibration.width}x{calibration.height}, sensor mode "
            f"{calibration.sensor_mode}, {calibration.capture_fps} FPS."
        )


def convert_rectification_maps(calibration: Calibration) -> RectificationMaps:
    """Convert float maps to OpenCV's faster fixed-point remap format."""
    left_map1, left_map2 = cv2.convertMaps(
        calibration.left_map1,
        calibration.left_map2,
        cv2.CV_16SC2,
    )
    right_map1, right_map2 = cv2.convertMaps(
        calibration.right_map1,
        calibration.right_map2,
        cv2.CV_16SC2,
    )
    expected_shape = (calibration.height, calibration.width)
    for name, rectification_map in (
        ("left fixed map1", left_map1),
        ("left fixed map2", left_map2),
        ("right fixed map1", right_map1),
        ("right fixed map2", right_map2),
    ):
        if rectification_map.shape[:2] != expected_shape:
            raise RuntimeError(
                f"{name} has shape {rectification_map.shape[:2]}, expected "
                f"{expected_shape}"
            )
    return RectificationMaps(
        left_map1=left_map1,
        left_map2=left_map2,
        right_map1=right_map1,
        right_map2=right_map2,
    )


def check_gstreamer() -> None:
    enabled = any(
        "GStreamer" in line and "YES" in line
        for line in cv2.getBuildInformation().splitlines()
    )
    if not enabled:
        raise RuntimeError(
            "GStreamer is not enabled in the loaded OpenCV build. "
            f"OpenCV {cv2.__version__} was loaded from {cv2.__file__}. "
            "Use JetPack/Ubuntu's python3-opencv package."
        )


def gstreamer_pipeline(sensor_id: int, calibration: Calibration) -> str:
    """Build the low-latency Jetson CSI capture pipeline."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} "
        f"sensor-mode={calibration.sensor_mode} ! "
        f"video/x-raw(memory:NVMM), width=(int){calibration.width}, "
        f"height=(int){calibration.height}, format=(string)NV12, "
        f"framerate=(fraction){calibration.capture_fps}/1 ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "nvvidconv flip-method=0 ! "
        f"video/x-raw, width=(int){calibration.width}, "
        f"height=(int){calibration.height}, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def open_camera(sensor_id: int, calibration: Calibration) -> cv2.VideoCapture:
    pipeline = gstreamer_pipeline(sensor_id, calibration)
    camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open IMX219 sensor-id={sensor_id}. Pipeline: {pipeline}"
        )
    return camera


def open_cameras(
    left_sensor_id: int,
    right_sensor_id: int,
    calibration: Calibration,
) -> tuple[cv2.VideoCapture, cv2.VideoCapture]:
    if left_sensor_id == right_sensor_id:
        raise ValueError("Left and right sensor IDs must be different")

    left_camera: Optional[cv2.VideoCapture] = None
    try:
        left_camera = open_camera(left_sensor_id, calibration)
        right_camera = open_camera(right_sensor_id, calibration)
        return left_camera, right_camera
    except Exception:
        if left_camera is not None:
            left_camera.release()
        raise


class SynchronizedStereoCapture:
    """Pair independently captured frames by their closest arrival times.

    These are host arrival timestamps, not hardware exposure timestamps. This
    reduces software skew but cannot create true sensor trigger
    synchronization. Pair skew is measured for diagnostics and never used to
    reject a frame pair.
    """

    def __init__(
        self,
        left_camera: cv2.VideoCapture,
        right_camera: cv2.VideoCapture,
    ) -> None:
        self.left_camera = left_camera
        self.right_camera = right_camera
        self._condition = threading.Condition()
        self._running = False
        self._start_barrier: Optional[threading.Barrier] = None
        self._threads: list[threading.Thread] = []
        self._left_buffer: deque[CameraFrame] = deque(
            maxlen=CAMERA_FRAME_BUFFER_SIZE
        )
        self._right_buffer: deque[CameraFrame] = deque(
            maxlen=CAMERA_FRAME_BUFFER_SIZE
        )
        self._error: Optional[str] = None
        self._camera_sequences = {"left": 0, "right": 0}
        self._pair_sequence = 0
        self._camera_fps = {"left": 0.0, "right": 0.0}
        self._dropped = {"left": 0, "right": 0}
        self._last_pair_skew_ms = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_barrier = threading.Barrier(3)
        self._threads = [
            threading.Thread(
                target=self._reader,
                args=("left", self.left_camera, self._left_buffer),
                name="imx219-left-capture",
                daemon=True,
            ),
            threading.Thread(
                target=self._reader,
                args=("right", self.right_camera, self._right_buffer),
                name="imx219-right-capture",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()
        try:
            # Release both readers from the same barrier as closely together
            # as the operating system scheduler permits.
            self._start_barrier.wait(timeout=2.0)
        except threading.BrokenBarrierError as exc:
            self.stop()
            raise RuntimeError("Could not start both camera readers") from exc

    def _reader(
        self,
        side: str,
        camera: cv2.VideoCapture,
        frame_buffer: deque[CameraFrame],
    ) -> None:
        previous_capture_time: Optional[float] = None
        consecutive_failures = 0
        try:
            assert self._start_barrier is not None
            self._start_barrier.wait(timeout=2.0)
            while self._running:
                grabbed = camera.grab()
                # Timestamp immediately when the frame becomes available,
                # before BGR retrieval/conversion adds variable CPU latency.
                arrived_at = time.perf_counter()
                retrieved, image = camera.retrieve()
                if not grabbed or not retrieved or image is None:
                    consecutive_failures += 1
                    if consecutive_failures < MAX_CONSECUTIVE_CAPTURE_FAILURES:
                        continue
                    with self._condition:
                        self._error = (
                            f"{side} camera failed "
                            f"{consecutive_failures} consecutive captures"
                        )
                        self._running = False
                        self._condition.notify_all()
                    return

                consecutive_failures = 0
                if previous_capture_time is not None:
                    instantaneous_fps = 1.0 / max(
                        arrived_at - previous_capture_time, 1e-6
                    )
                    self._camera_fps[side] = (
                        instantaneous_fps
                        if self._camera_fps[side] == 0.0
                        else 0.90 * self._camera_fps[side]
                        + 0.10 * instantaneous_fps
                    )
                previous_capture_time = arrived_at

                with self._condition:
                    self._camera_sequences[side] += 1
                    if len(frame_buffer) == frame_buffer.maxlen:
                        self._dropped[side] += 1
                    frame_buffer.append(
                        CameraFrame(
                            sequence=self._camera_sequences[side],
                            arrived_at=arrived_at,
                            image=image,
                        )
                    )
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = (
                    f"{side} capture thread failed "
                    f"({type(exc).__name__}): {exc}"
                )
                self._running = False
                self._condition.notify_all()

    def get_latest(
        self,
        after_sequence: int,
        timeout: float = 2.0,
    ) -> StereoFrame:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError(self._error)
                if not self._running:
                    raise RuntimeError("Stereo capture thread stopped")

                if self._left_buffer and self._right_buffer:
                    # Select the closest timestamps currently available. No
                    # pair is rejected because of skew. Older frames preceding
                    # the chosen pair are stale and are discarded only to keep
                    # the live display from accumulating latency.
                    left_index, right_index = min(
                        (
                            (left_index, right_index)
                            for left_index in range(len(self._left_buffer))
                            for right_index in range(len(self._right_buffer))
                        ),
                        key=lambda indices: (
                            abs(
                                self._left_buffer[indices[0]].arrived_at
                                - self._right_buffer[indices[1]].arrived_at
                            ),
                            -max(
                                self._left_buffer[indices[0]].arrived_at,
                                self._right_buffer[indices[1]].arrived_at,
                            ),
                        ),
                    )
                    for _ in range(left_index):
                        self._left_buffer.popleft()
                        self._dropped["left"] += 1
                    for _ in range(right_index):
                        self._right_buffer.popleft()
                        self._dropped["right"] += 1

                    left = self._left_buffer.popleft()
                    right = self._right_buffer.popleft()
                    absolute_skew_ms = abs(
                        left.arrived_at - right.arrived_at
                    ) * 1000.0
                    self._pair_sequence += 1
                    self._last_pair_skew_ms = absolute_skew_ms
                    if self._pair_sequence > after_sequence:
                        return StereoFrame(
                            sequence=self._pair_sequence,
                            captured_at=(
                                left.arrived_at + right.arrived_at
                            ) / 2.0,
                            pair_skew_ms=absolute_skew_ms,
                            left=left.image,
                            right=right.image,
                        )
                    continue

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        "Timed out waiting for frames from both cameras. "
                        "Frame pairs are not rejected based on timestamp skew, "
                        "so check the camera capture threads and Argus logs."
                    )
                self._condition.wait(remaining)

    @property
    def capture_fps(self) -> float:
        return (self._camera_fps["left"] + self._camera_fps["right"]) / 2.0

    def synchronization_stats(self) -> tuple[float, float, float, int, int]:
        with self._condition:
            return (
                self._camera_fps["left"],
                self._camera_fps["right"],
                self._last_pair_skew_ms,
                self._dropped["left"],
                self._dropped["right"],
            )

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._start_barrier is not None:
            try:
                self._start_barrier.abort()
            except threading.BrokenBarrierError:
                pass
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()


def validate_frame_resolution(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    calibration: Calibration,
) -> None:
    expected_shape = (calibration.height, calibration.width)
    left_shape = left_frame.shape[:2]
    right_shape = right_frame.shape[:2]
    if left_shape != expected_shape or right_shape != expected_shape:
        raise RuntimeError(
            "Camera frames do not match the calibration. Expected "
            f"{calibration.width}x{calibration.height}; received left "
            f"{left_shape[1]}x{left_shape[0]} and right "
            f"{right_shape[1]}x{right_shape[0]}. Images will not be silently "
            "resized because that would invalidate the rectification maps."
        )


def rectify_frames(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    maps: RectificationMaps,
) -> tuple[np.ndarray, np.ndarray]:
    left_rectified = cv2.remap(
        left_frame,
        maps.left_map1,
        maps.left_map2,
        cv2.INTER_LINEAR,
    )
    right_rectified = cv2.remap(
        right_frame,
        maps.right_map1,
        maps.right_map2,
        cv2.INTER_LINEAR,
    )
    return left_rectified, right_rectified


def rectify_left_frame(
    left_frame: np.ndarray,
    maps: RectificationMaps,
) -> np.ndarray:
    """Rectify only the display frame when reusing the previous depth map."""
    return cv2.remap(
        left_frame,
        maps.left_map1,
        maps.left_map2,
        cv2.INTER_LINEAR,
    )


def create_stereo_matcher(
    min_disparity: int = SGBM_MIN_DISPARITY,
    num_disparities: int = SGBM_NUM_DISPARITIES,
    block_size: int = SGBM_BLOCK_SIZE,
) -> Any:
    if num_disparities <= 0 or num_disparities % 16 != 0:
        raise ValueError("num_disparities must be a positive multiple of 16")
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer of at least 3")

    return cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def calculate_disparity(
    matcher: Any,
    left_rectified: np.ndarray,
    right_rectified: np.ndarray,
) -> np.ndarray:
    left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    raw_disparity = matcher.compute(left_gray, right_gray)
    if raw_disparity is None or raw_disparity.size == 0:
        raise RuntimeError("StereoSGBM returned an empty disparity image")

    # StereoSGBM stores disparity with four fractional bits.
    return raw_disparity.astype(np.float32) / 16.0


class OpenCvStereoEngine:
    """CPU fallback retained for diagnostics and result comparisons."""

    name = "OpenCV CPU StereoSGBM"
    uses_cuda = False

    def __init__(
        self,
        min_disparity: int = SGBM_MIN_DISPARITY,
        num_disparities: int = SGBM_NUM_DISPARITIES,
        block_size: int = SGBM_BLOCK_SIZE,
    ) -> None:
        self.minimum_disparity = min_disparity
        self.maximum_disparity = min_disparity + num_disparities
        self.block_size = block_size
        self.matcher = create_stereo_matcher(
            min_disparity, num_disparities, block_size
        )
        self.last_positive_percentage = 0.0
        self.last_median_disparity = float("nan")

    def compute(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        disparity = calculate_disparity(
            self.matcher, left_rectified, right_rectified
        )
        positive = (
            np.isfinite(disparity)
            & (disparity > self.minimum_disparity)
            & (disparity < self.maximum_disparity)
        )
        self.last_positive_percentage = 100.0 * float(np.mean(positive))
        self.last_median_disparity = (
            float(np.median(disparity[positive]))
            if np.any(positive)
            else float("nan")
        )
        return disparity, None

    def warmup(self) -> None:
        return

    def diagnostic_lines(self) -> tuple[str, ...]:
        median_text = (
            f"{self.last_median_disparity:.2f} px"
            if np.isfinite(self.last_median_disparity)
            else "N/A"
        )
        return (
            f"Raw disparity >0: {self.last_positive_percentage:.1f}%",
            f"Median disparity: {median_text}",
        )


class VpiCudaStereoEngine:
    """Full-resolution VPI CUDA disparity with reusable buffers and stream."""

    name = "NVIDIA VPI CUDA Stereo"
    uses_cuda = True

    def __init__(
        self,
        width: int,
        height: int,
        settings: VpiRuntimeSettings,
        profile_source: str,
    ) -> None:
        settings.validate()
        try:
            import vpi
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize NVIDIA VPI. Confirm that python3.10-vpi3 "
                f"is installed and CUDA works in this desktop session: {exc}"
            ) from exc

        self.vpi = vpi
        self.width = width
        self.height = height
        self.settings = settings
        self.profile_source = profile_source
        self.minimum_disparity = settings.min_disparity
        self.maximum_disparity = settings.max_disparity
        self.last_positive_percentage = 0.0
        self.last_confidence_percentage = 0.0
        self.last_near_limit_percentage = 0.0
        self.last_invalid_percentage = 0.0
        self.last_median_disparity = float("nan")
        self.last_median_confidence = float("nan")

        try:
            self.stream = vpi.Stream()
            # Own the U8 input images inside VPI. Each frame is uploaded using
            # an explicit CPU write lock so VPI's CPU/CUDA coherency tracker
            # knows that the image changed after the previous submission.
            self.left_input = vpi.Image((width, height), vpi.Format.U8)
            self.right_input = vpi.Image((width, height), vpi.Format.U8)
            self.disparity_s16 = vpi.Image((width, height), vpi.Format.S16)
            self.confidence_u16 = vpi.Image((width, height), vpi.Format.U16)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to allocate full-resolution VPI CUDA resources: {exc}"
            ) from exc

    def compute(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        try:
            with self.left_input.wlock_cpu() as left_data:
                np.copyto(left_data, left_gray)
            with self.right_input.wlock_cpu() as right_data:
                np.copyto(right_data, right_gray)

            with self.stream, self.vpi.Backend.CUDA:
                # The VPI CUDA stereo backend accepts U8 input directly. This
                # avoids the former U8 -> Y16_ER conversion, which added work
                # and made this path differ unnecessarily from OpenCV.
                self.vpi.stereodisp(
                    self.left_input,
                    self.right_input,
                    out=self.disparity_s16,
                    out_confmap=self.confidence_u16,
                    window=VPI_WINDOW,
                    maxdisp=self.settings.max_disparity,
                    # VPI 3.2's Python API clamps this to at least 1.
                    confthreshold=VPI_INTERNAL_CONFIDENCE_THRESHOLD,
                    conftype=self.vpi.ConfidenceType.ABSOLUTE,
                    mindisp=self.settings.min_disparity,
                    p1=self.settings.p1,
                    p2=self.settings.p2,
                    uniqueness=self.settings.uniqueness,
                    includediagonals=self.settings.include_diagonals,
                )

            # Read locks synchronize the custom VPI stream and make owned
            # copies before the buffers are unlocked and reused next frame.
            with self.disparity_s16.rlock_cpu() as disparity_data:
                disparity = np.array(disparity_data, dtype=np.float32) / 32.0
            with self.confidence_u16.rlock_cpu() as confidence_data:
                confidence = np.array(confidence_data, copy=True)
        except Exception as exc:
            raise RuntimeError(f"VPI CUDA stereo disparity failed: {exc}") from exc

        if disparity.shape != (self.height, self.width):
            raise RuntimeError(
                f"VPI disparity has shape {disparity.shape}, expected "
                f"{(self.height, self.width)}"
            )
        # VPI writes INT16_MAX (32767, or 1023.96875 after Q10.5 conversion)
        # for an invalid disparity. Never treat that positive sentinel as a
        # stereo match. Valid values must remain inside the configured search.
        finite = np.isfinite(disparity)
        plausible = (
            finite
            & (disparity > self.minimum_disparity)
            & (
                disparity
                < self.maximum_disparity
                - self.settings.disparity_safety_margin_px
            )
        )
        near_limit = (
            finite
            & (
                disparity
                >= self.maximum_disparity
                - self.settings.disparity_safety_margin_px
            )
            & (disparity < self.maximum_disparity)
        )
        invalid_sentinel = finite & (disparity >= self.maximum_disparity)
        effective_confidence = max(1, self.settings.confidence_threshold)
        confidence_mask = plausible & (confidence >= effective_confidence)
        self.last_positive_percentage = 100.0 * float(np.mean(plausible))
        self.last_confidence_percentage = 100.0 * float(
            np.mean(confidence_mask)
        )
        self.last_near_limit_percentage = 100.0 * float(np.mean(near_limit))
        self.last_invalid_percentage = 100.0 * float(
            np.mean(invalid_sentinel)
        )
        self.last_median_disparity = (
            float(np.median(disparity[plausible]))
            if np.any(plausible)
            else float("nan")
        )
        self.last_median_confidence = (
            float(np.median(confidence[plausible]))
            if np.any(plausible)
            else float("nan")
        )
        return disparity, confidence_mask

    def warmup(self) -> None:
        """Create/cache the VPI payload before opening the cameras."""
        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.compute(blank, blank)

    def diagnostic_lines(self) -> tuple[str, ...]:
        median_disparity = (
            f"{self.last_median_disparity:.2f} px"
            if np.isfinite(self.last_median_disparity)
            else "N/A"
        )
        median_confidence = (
            f"{self.last_median_confidence:.0f}"
            if np.isfinite(self.last_median_confidence)
            else "N/A"
        )
        return (
            f"VPI runtime max/conf/unique: {self.settings.max_disparity} / "
            f"{self.settings.confidence_threshold} / {self.settings.uniqueness}",
            f"VPI valid/conf pass: {self.last_positive_percentage:.1f}% / "
            f"{self.last_confidence_percentage:.1f}%",
            f"VPI near-limit/invalid: {self.last_near_limit_percentage:.1f}% / "
            f"{self.last_invalid_percentage:.1f}%",
            f"Median disp/conf: {median_disparity} / {median_confidence}",
        )


def create_stereo_engine(
    backend: str,
    calibration: Calibration,
    vpi_settings: VpiRuntimeSettings,
    vpi_profile_source: str,
    sgbm_min_disparity: int = SGBM_MIN_DISPARITY,
    sgbm_num_disparities: int = SGBM_NUM_DISPARITIES,
    sgbm_block_size: int = SGBM_BLOCK_SIZE,
) -> Any:
    if backend == "vpi-cuda":
        return VpiCudaStereoEngine(
            calibration.width,
            calibration.height,
            vpi_settings,
            vpi_profile_source,
        )
    if backend == "opencv":
        return OpenCvStereoEngine(
            sgbm_min_disparity,
            sgbm_num_disparities,
            sgbm_block_size,
        )
    raise ValueError(f"Unsupported stereo backend: {backend}")


def calculate_depth(
    disparity: np.ndarray,
    q_matrix: np.ndarray,
    max_depth_m: float,
    confidence_mask: Optional[np.ndarray] = None,
    min_disparity: int = SGBM_MIN_DISPARITY,
    max_disparity: int = SGBM_MIN_DISPARITY + SGBM_NUM_DISPARITIES,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate only forward Z depth from Q, without allocating X and Y."""
    # OpenCV stereoRectify produces the canonical Q form used below. X and Y
    # would require pixel coordinates, but Z depends only on disparity here.
    if not (
        np.allclose(q_matrix[2, :3], 0.0)
        and np.allclose(q_matrix[3, :2], 0.0)
    ):
        raise RuntimeError(
            "Calibration Q is not in canonical rectified-stereo form; direct "
            "Z-only reconstruction is unsafe for this matrix."
        )

    denominator = (
        float(q_matrix[3, 2]) * disparity + float(q_matrix[3, 3])
    )
    depth_map = np.full(disparity.shape, np.nan, dtype=np.float32)
    can_divide = (
        (disparity > min_disparity)
        & (disparity < max_disparity)
        & np.isfinite(disparity)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-12)
    )
    np.divide(
        float(q_matrix[2, 3]),
        denominator,
        out=depth_map,
        where=can_divide,
    )
    valid_depth_mask = (
        can_divide
        & np.isfinite(depth_map)
        & (depth_map >= MIN_DEPTH_M)
        & (depth_map <= max_depth_m)
    )
    if confidence_mask is not None:
        if confidence_mask.shape != disparity.shape:
            raise RuntimeError(
                f"Confidence map shape {confidence_mask.shape} does not match "
                f"disparity shape {disparity.shape}"
            )
        valid_depth_mask &= confidence_mask
    depth_map[~valid_depth_mask] = np.nan
    return depth_map, valid_depth_mask


def three_rois(
    image_shape: tuple[int, int],
) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    """Return equally sized left, centre, and right measurement regions."""
    if len(image_shape) != 2:
        raise ValueError("Image shape must contain height and width")
    height, width = image_shape
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive")

    roi_width = max(1, int(round(width * ROI_WIDTH_FRACTION)))
    roi_height = max(1, int(round(height * ROI_HEIGHT_FRACTION)))
    centre_y = height // 2
    y1 = min(max(0, centre_y - roi_height // 2), height - roi_height)
    y2 = y1 + roi_height
    regions = []
    for name, horizontal_fraction in ROI_HORIZONTAL_CENTRES:
        centre_x = int(round(width * horizontal_fraction))
        x1 = min(max(0, centre_x - roi_width // 2), width - roi_width)
        regions.append((name, (x1, y1, x1 + roi_width, y2)))
    return tuple(regions)


def empty_measurements(image_shape: tuple[int, int]) -> tuple[DepthMeasurement, ...]:
    """Create unavailable measurements before the first depth result arrives."""
    return tuple(
        DepthMeasurement(name, None, box, 0, 0.0)
        for name, box in three_rois(image_shape)
    )


def measure_roi_depth(
    name: str,
    box: tuple[int, int, int, int],
    depth_map: np.ndarray,
    valid_depth_mask: np.ndarray,
    min_samples: int = MIN_ROI_DEPTH_SAMPLES,
) -> DepthMeasurement:
    """Measure one ROI independently using robust median metric depth."""
    if depth_map.ndim != 2 or valid_depth_mask.shape != depth_map.shape:
        raise ValueError("Depth map and valid mask must be matching 2-D arrays")
    if min_samples < 1:
        raise ValueError("Minimum sample count must be positive")
    x1, y1, x2, y2 = box
    height, width = depth_map.shape
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"ROI {name!r} is outside the depth image")

    roi_mask = valid_depth_mask[y1:y2, x1:x2]
    valid_values = depth_map[y1:y2, x1:x2][roi_mask]
    sample_count = int(valid_values.size)
    valid_percentage = 100.0 * sample_count / max(1, int(roi_mask.size))
    distance_m = (
        None
        if sample_count < min_samples
        else float(np.median(valid_values))
    )
    return DepthMeasurement(
        name=name,
        distance_m=distance_m,
        box=box,
        sample_count=sample_count,
        valid_percentage=valid_percentage,
    )


def measure_three_depths(
    depth_map: np.ndarray,
    valid_depth_mask: np.ndarray,
    min_samples: int = MIN_ROI_DEPTH_SAMPLES,
) -> tuple[DepthMeasurement, ...]:
    """Measure left, centre, and right ROIs from one shared depth map."""
    return tuple(
        measure_roi_depth(
            name,
            box,
            depth_map,
            valid_depth_mask,
            min_samples,
        )
        for name, box in three_rois(depth_map.shape)
    )


def make_depth_view(
    depth_map: np.ndarray,
    valid_depth_mask: np.ndarray,
    max_depth_m: float,
) -> np.ndarray:
    """Colour valid depth red when near and blue when far."""
    scaled = np.zeros(depth_map.shape, dtype=np.uint8)
    if np.any(valid_depth_mask):
        clipped = np.clip(
            depth_map[valid_depth_mask], MIN_DEPTH_M, max_depth_m
        )
        scaled[valid_depth_mask] = np.round(
            (max_depth_m - clipped)
            * 255.0
            / (max_depth_m - MIN_DEPTH_M)
        ).astype(np.uint8)

    colour = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colour[~valid_depth_mask] = 0
    return colour


def make_disparity_view(
    disparity: np.ndarray,
    min_disparity: int = SGBM_MIN_DISPARITY,
    max_disparity: int = SGBM_MIN_DISPARITY + SGBM_NUM_DISPARITIES,
) -> np.ndarray:
    valid = (
        np.isfinite(disparity)
        & (disparity > min_disparity)
        & (disparity < max_disparity)
    )
    scaled = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(disparity[valid], (2.0, 98.0))
        if high > low:
            normalized = np.clip(
                (disparity - low) * 255.0 / (high - low), 0, 255
            )
            scaled[valid] = normalized[valid].astype(np.uint8)
    colour = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colour[~valid] = 0
    return colour


def point_cloud_from_depth_result(result: DepthResult, q_matrix: np.ndarray):
    xyz, xyz_valid = disparity_to_xyz(
        result.disparity,
        q_matrix,
        result.valid_depth_mask,
    )
    return sample_point_cloud(
        xyz,
        result.left_rectified,
        xyz_valid,
        stride=POINT_CLOUD_STRIDE,
    )


def save_3d_evidence(result: DepthResult, cloud: Any) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output = RESULTS_3D_DIR / stamp
    output.mkdir(parents=True, exist_ok=False)
    annotated_left = result.left_rectified.copy()
    annotated_depth = result.depth_view.copy()
    draw_measurements(annotated_left, result.measurements)
    draw_measurements(annotated_depth, result.measurements)
    cv2.imwrite(str(output / "left_rectified.png"), annotated_left)
    cv2.imwrite(str(output / "depth_heatmap.png"), annotated_depth)
    cv2.imwrite(
        str(output / "orthographic_3d.png"), make_orthographic_view(cloud)
    )
    np.save(
        output / "disparity_float32.npy",
        result.disparity.astype(np.float32),
    )
    np.save(
        output / "depth_metres_float32.npy",
        result.depth_map.astype(np.float32),
    )
    np.save(output / "valid_mask.npy", result.valid_depth_mask.astype(np.uint8))
    save_binary_ply(output / "point_cloud.ply", cloud)
    with (output / "three_zone_measurements.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "sequence": result.sequence,
                "stereo_diagnostics": list(result.diagnostic_lines),
                "measurements": [
                    {
                        "name": measurement.name,
                        "distance_m": measurement.distance_m,
                        "box": list(measurement.box),
                        "sample_count": measurement.sample_count,
                        "valid_percentage": measurement.valid_percentage,
                    }
                    for measurement in result.measurements
                ],
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return output


class AsyncDepthProcessor:
    """Continuously process only the newest submitted stereo pair.

    The camera/display thread never waits for disparity. If CUDA is still busy
    when newer input arrives, the pending input is replaced so navigation uses
    fresh imagery instead of building an increasingly delayed queue.
    """

    def __init__(
        self,
        backend: str,
        calibration: Calibration,
        maps: RectificationMaps,
        max_depth_m: float,
        vpi_settings: VpiRuntimeSettings,
        vpi_profile_source: str,
        sgbm_num_disparities: int = SGBM_NUM_DISPARITIES,
        sgbm_block_size: int = SGBM_BLOCK_SIZE,
    ) -> None:
        self.backend = backend
        self.calibration = calibration
        self.maps = maps
        self.max_depth_m = max_depth_m
        self.vpi_settings = vpi_settings
        self.vpi_profile_source = vpi_profile_source
        self.sgbm_num_disparities = sgbm_num_disparities
        self.sgbm_block_size = sgbm_block_size
        self.minimum_disparity = (
            vpi_settings.min_disparity
            if backend == "vpi-cuda"
            else SGBM_MIN_DISPARITY
        )
        self.maximum_disparity = (
            vpi_settings.max_disparity
            if backend == "vpi-cuda"
            else SGBM_MIN_DISPARITY + sgbm_num_disparities
        )
        self.backend_name = (
            VpiCudaStereoEngine.name
            if backend == "vpi-cuda"
            else OpenCvStereoEngine.name
        )
        self._condition = threading.Condition()
        self._ready = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pending: Optional[DepthInput] = None
        self._latest_result: Optional[DepthResult] = None
        self._error: Optional[str] = None
        self._replaced_inputs = 0

    def start(self, timeout: float = 30.0) -> None:
        with self._condition:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="stereo-depth-worker",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            self.stop()
            raise RuntimeError("Timed out initializing the stereo depth worker")
        with self._condition:
            if self._error is not None:
                raise RuntimeError(self._error)

    def submit(
        self,
        sequence: int,
        left_rectified: np.ndarray,
        right_raw: np.ndarray,
    ) -> None:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(self._error)
            if not self._running:
                raise RuntimeError("Stereo depth worker stopped")
            if self._pending is not None:
                self._replaced_inputs += 1
            self._pending = DepthInput(
                sequence=sequence,
                left_rectified=left_rectified,
                right_raw=right_raw,
            )
            self._condition.notify_all()

    def latest_result(self) -> Optional[DepthResult]:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(self._error)
            return self._latest_result

    def status_lines(self) -> tuple[str, ...]:
        with self._condition:
            replaced = self._replaced_inputs
            result = self._latest_result
        queue_line = f"Depth input replacements: {replaced}"
        if result is None:
            return (queue_line, "Waiting for first depth result")
        return (queue_line,) + result.diagnostic_lines

    def _run(self) -> None:
        stereo_engine: Optional[Any] = None
        previous_completion: Optional[float] = None
        smoothed_fps = 0.0
        try:
            stereo_engine = create_stereo_engine(
                self.backend,
                self.calibration,
                self.vpi_settings,
                self.vpi_profile_source,
                SGBM_MIN_DISPARITY,
                self.sgbm_num_disparities,
                self.sgbm_block_size,
            )
            # Construct and warm VPI in the same thread that will use its
            # stream, avoiding cross-thread CUDA/VPI resource ownership.
            stereo_engine.warmup()
            self._ready.set()

            while True:
                with self._condition:
                    while self._running and self._pending is None:
                        self._condition.wait()
                    if not self._running:
                        return
                    depth_input = self._pending
                    self._pending = None
                assert depth_input is not None

                right_rectified = cv2.remap(
                    depth_input.right_raw,
                    self.maps.right_map1,
                    self.maps.right_map2,
                    cv2.INTER_LINEAR,
                )
                disparity, confidence_mask = stereo_engine.compute(
                    depth_input.left_rectified,
                    right_rectified,
                )
                depth_map, valid_depth_mask = calculate_depth(
                    disparity,
                    self.calibration.q_matrix,
                    self.max_depth_m,
                    confidence_mask,
                    stereo_engine.minimum_disparity,
                    stereo_engine.maximum_disparity,
                )
                measurements = measure_three_depths(
                    depth_map,
                    valid_depth_mask,
                )
                depth_view = make_depth_view(
                    depth_map,
                    valid_depth_mask,
                    self.max_depth_m,
                )

                completed_at = time.perf_counter()
                if previous_completion is not None:
                    instantaneous_fps = 1.0 / max(
                        completed_at - previous_completion,
                        1e-6,
                    )
                    smoothed_fps = (
                        instantaneous_fps
                        if smoothed_fps == 0.0
                        else 0.90 * smoothed_fps + 0.10 * instantaneous_fps
                    )
                previous_completion = completed_at

                result = DepthResult(
                    sequence=depth_input.sequence,
                    disparity=disparity,
                    depth_map=depth_map,
                    valid_depth_mask=valid_depth_mask,
                    left_rectified=depth_input.left_rectified,
                    depth_view=depth_view,
                    measurements=measurements,
                    valid_percentage=100.0 * float(
                        np.mean(valid_depth_mask)
                    ),
                    depth_fps=smoothed_fps,
                    diagnostic_lines=stereo_engine.diagnostic_lines(),
                )
                with self._condition:
                    self._latest_result = result
        except Exception as exc:
            with self._condition:
                self._error = (
                    "Asynchronous stereo depth processing failed "
                    f"({type(exc).__name__}): {exc}"
                )
                self._running = False
                self._condition.notify_all()
        finally:
            self._ready.set()

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


def draw_measurements(
    image: np.ndarray,
    measurements: tuple[DepthMeasurement, ...],
) -> None:
    """Draw the three boxes and a compact independent reading above each."""
    for measurement in measurements:
        x1, y1, x2, y2 = measurement.box
        colour = (0, 255, 0) if measurement.distance_m is not None else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        value = (
            f"{measurement.distance_m:.2f} m"
            if measurement.distance_m is not None
            else "Depth N/A"
        )
        text = f"{measurement.name}: {value} ({measurement.sample_count})"
        text_size, baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2
        )
        label_y = y1 - 8
        if label_y - text_size[1] < 0:
            label_y = min(image.shape[0] - baseline - 2, y2 + text_size[1] + 8)
        label_x = min(max(0, x1), max(0, image.shape[1] - text_size[0] - 8))
        cv2.rectangle(
            image,
            (label_x - 3, label_y - text_size[1] - 4),
            (label_x + text_size[0] + 4, label_y + baseline + 3),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            colour,
            2,
            cv2.LINE_AA,
        )


def draw_status(
    image: np.ndarray,
    left_camera_fps: float,
    right_camera_fps: float,
    display_fps: float,
    depth_fps: float,
    valid_percentage: float,
    disparity_visible: bool,
    backend_name: str,
    pair_skew_ms: float,
    dropped_left: int,
    dropped_right: int,
    measurements: tuple[DepthMeasurement, ...],
    stereo_diagnostics: tuple[str, ...],
) -> None:
    if pair_skew_ms <= 5.0:
        sync_status = "EXCELLENT"
    elif pair_skew_ms <= 20.0:
        sync_status = "GOOD"
    else:
        sync_status = "HIGH"
    zone_summary = " | ".join(
        f"{measurement.name[0]}:"
        + (
            f"{measurement.distance_m:.2f}m"
            if measurement.distance_m is not None
            else "N/A"
        )
        for measurement in measurements
    )
    lines = (
        f"Camera FPS L/R: {left_camera_fps:.1f} / {right_camera_fps:.1f}",
        f"Pair skew: {pair_skew_ms:.2f} ms ({sync_status})",
        f"Buffer drops L/R: {dropped_left} / {dropped_right}",
        f"Display FPS: {display_fps:.1f}",
        f"Depth FPS: {depth_fps:.1f}",
        f"Valid depth: {valid_percentage:.1f}%",
        f"Zones {zone_summary}",
        f"Stereo: {backend_name}",
    ) + stereo_diagnostics + (
        "Depth worker: continuous async",
        f"D disparity: {'ON' if disparity_visible else 'OFF'}",
        "W/Z range | E/C confidence | U/J uniqueness | R reset",
        "Q / Esc: quit",
    )
    panel_top = DISPLAY_PANEL_HEIGHT
    image[panel_top:, :] = (16, 16, 16)
    rows_per_column = (len(lines) + 1) // 2
    column_width = image.shape[1] // 2
    for index, line in enumerate(lines):
        column = index // rows_per_column
        row = index % rows_per_column
        cv2.putText(
            image,
            line,
            (14 + column * column_width, panel_top + 27 + row * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def make_primary_display(
    left_rectified: np.ndarray,
    depth_view: np.ndarray,
) -> np.ndarray:
    """Resize only the completed visualizations, never stereo input data."""
    left_preview = cv2.resize(
        left_rectified,
        (DISPLAY_PANEL_WIDTH, DISPLAY_PANEL_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    depth_preview = cv2.resize(
        depth_view,
        (DISPLAY_PANEL_WIDTH, DISPLAY_PANEL_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    cv2.putText(
        left_preview,
        "RECTIFIED LEFT",
        (12, DISPLAY_PANEL_HEIGHT - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        depth_preview,
        "DEPTH: NEAR RED | FAR BLUE",
        (12, DISPLAY_PANEL_HEIGHT - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    image_row = cv2.hconcat((left_preview, depth_preview))
    status_row = np.zeros(
        (STATUS_PANEL_HEIGHT, DISPLAY_PANEL_WIDTH * 2, 3),
        dtype=np.uint8,
    )
    return cv2.vconcat((image_row, status_row))


def print_startup(
    calibration: Calibration,
    left_id: int,
    right_id: int,
    max_depth_m: float,
    backend_name: str,
    vpi_confidence_threshold: int,
    maximum_disparity: int,
    sgbm_block_size: int,
    vpi_profile_source: str,
    vpi_safety_margin_px: float,
) -> None:
    print("\nIMX219 three-zone stereo depth test")
    print(f"  OpenCV version:       {cv2.__version__}")
    print(f"  OpenCV path:          {cv2.__file__}")
    print("  GStreamer enabled:    YES")
    print(f"  Calibration:          {calibration.path}")
    print(f"  Resolution:           {calibration.width}x{calibration.height}")
    print(f"  Sensor mode:          {calibration.sensor_mode}")
    print(f"  Capture FPS:          {calibration.capture_fps}")
    print(f"  Baseline:             {calibration.baseline_m:.6f} m")
    print(f"  Left/right sensors:   {left_id} / {right_id}")
    print(f"  Valid depth range:    {MIN_DEPTH_M:.2f} to {max_depth_m:.2f} m")
    print(f"  Stereo backend:       {backend_name}")
    if backend_name == VpiCudaStereoEngine.name:
        print(f"  VPI settings source:  {vpi_profile_source}")
        print(
            "  VPI confidence:       "
            f">= {vpi_confidence_threshold} / 65535 "
            "(VPI internal minimum 1)"
        )
        print(f"  Disparity margin:     {vpi_safety_margin_px:.1f} px")
    print(f"  Maximum disparity:    {maximum_disparity}")
    if backend_name == OpenCvStereoEngine.name:
        print(f"  Tuned SGBM block:     {sgbm_block_size}")
    print("  Primary measurement:  left / centre / right Z depth")
    print("  Optional 3D:          calibrated XYZ projection and PLY")
    print("  Rectification maps:   fixed-point CV_16SC2")
    print("  Pair skew policy:     measured only, never rejected")
    print("  Sync timestamps:      host arrival time (software pairing)")
    print("  Depth schedule:       continuous async, newest frame")
    print("  Controls:             Q/Esc quit | D disparity | P 3D")
    print("                        O Open3D | S save 3D evidence")
    print("  VPI live controls:    W/Z range | E/C confidence")
    print("                        U/J uniqueness | R reset\n")


def run(args: argparse.Namespace) -> None:
    if args.max_depth <= MIN_DEPTH_M:
        raise ValueError(f"--max-depth must be greater than {MIN_DEPTH_M}")
    if args.num_disparities <= 0 or args.num_disparities % 16 != 0:
        raise ValueError("--num-disparities must be a positive multiple of 16")
    if args.block_size < 3 or args.block_size % 2 == 0:
        raise ValueError("--block-size must be odd and at least 3")

    check_gstreamer()
    calibration = load_calibration(args.calibration)
    rectification_maps = convert_rectification_maps(calibration)
    vpi_settings = vpi_settings_from_args(args)
    vpi_profile_source = "2.1.1 command-line/runtime defaults"

    def make_depth_worker(settings: VpiRuntimeSettings) -> AsyncDepthProcessor:
        return AsyncDepthProcessor(
            args.backend,
            calibration,
            rectification_maps,
            args.max_depth,
            settings,
            vpi_profile_source,
            args.num_disparities,
            args.block_size,
        )

    print(f"Initializing asynchronous stereo backend: {args.backend}...", flush=True)
    depth_worker = make_depth_worker(vpi_settings)

    left_camera: Optional[cv2.VideoCapture] = None
    right_camera: Optional[cv2.VideoCapture] = None
    capture_worker: Optional[SynchronizedStereoCapture] = None
    try:
        depth_worker.start()
        print_startup(
            calibration,
            args.left_id,
            args.right_id,
            args.max_depth,
            depth_worker.backend_name,
            vpi_settings.confidence_threshold,
            depth_worker.maximum_disparity,
            args.block_size,
            vpi_profile_source,
            vpi_settings.disparity_safety_margin_px,
        )
        print("Opening IMX219 cameras...", flush=True)
        left_camera, right_camera = open_cameras(
            args.left_id, args.right_id, calibration
        )
        capture_worker = SynchronizedStereoCapture(
            left_camera,
            right_camera,
        )
        capture_worker.start()

        # Allow automatic exposure/white balance to settle while also proving
        # that the background latest-pair capture loop is healthy.
        last_sequence = 0
        stereo_frame: Optional[StereoFrame] = None
        for _ in range(8):
            stereo_frame = capture_worker.get_latest(last_sequence)
            last_sequence = stereo_frame.sequence
        assert stereo_frame is not None
        validate_frame_resolution(
            stereo_frame.left, stereo_frame.right, calibration
        )
        print(
            f"Camera frames validated at {calibration.width}x{calibration.height}.",
            flush=True,
        )

        window_name = "IMX219 Three-Zone Stereo Depth Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            window_name,
            DISPLAY_PANEL_WIDTH * 2,
            DISPLAY_PANEL_HEIGHT + STATUS_PANEL_HEIGHT,
        )

        show_disparity = SHOW_DISPARITY_AT_START
        show_point_cloud = SHOW_POINT_CLOUD_AT_START
        display_fps = 0.0
        previous_display_time = time.perf_counter()

        latest_disparity: Optional[np.ndarray] = None
        latest_depth_result: Optional[DepthResult] = None
        latest_cloud: Optional[Any] = None
        cloud_sequence = -1
        latest_depth_view = np.zeros(
            (calibration.height, calibration.width, 3),
            dtype=np.uint8,
        )
        latest_measurements = empty_measurements(
            (calibration.height, calibration.width)
        )
        latest_valid_percentage = 0.0
        depth_fps = 0.0
        last_depth_sequence = 0

        while True:
            stereo_frame = capture_worker.get_latest(last_sequence)
            last_sequence = stereo_frame.sequence
            validate_frame_resolution(
                stereo_frame.left, stereo_frame.right, calibration
            )

            # Rectify the live reference frame once. The immutable result is
            # shared with the depth worker; only an annotated copy is modified
            # below for display.
            left_rectified = rectify_left_frame(
                stereo_frame.left,
                rectification_maps,
            )
            depth_worker.submit(
                stereo_frame.sequence,
                left_rectified,
                stereo_frame.right,
            )
            depth_result = depth_worker.latest_result()
            if (
                depth_result is not None
                and depth_result.sequence > last_depth_sequence
            ):
                last_depth_sequence = depth_result.sequence
                latest_depth_result = depth_result
                latest_disparity = depth_result.disparity
                latest_depth_view = depth_result.depth_view
                latest_measurements = depth_result.measurements
                latest_valid_percentage = depth_result.valid_percentage
                depth_fps = depth_result.depth_fps

            annotated_left = left_rectified.copy()
            depth_display = latest_depth_view.copy()
            draw_measurements(annotated_left, latest_measurements)
            draw_measurements(depth_display, latest_measurements)

            now = time.perf_counter()
            instantaneous_display_fps = 1.0 / max(
                now - previous_display_time, 1e-6
            )
            display_fps = (
                instantaneous_display_fps
                if display_fps == 0.0
                else 0.90 * display_fps
                + 0.10 * instantaneous_display_fps
            )
            previous_display_time = now

            display = make_primary_display(annotated_left, depth_display)
            (
                left_camera_fps,
                right_camera_fps,
                pair_skew_ms,
                dropped_left,
                dropped_right,
            ) = capture_worker.synchronization_stats()
            draw_status(
                display,
                left_camera_fps,
                right_camera_fps,
                display_fps,
                depth_fps,
                latest_valid_percentage,
                show_disparity,
                depth_worker.backend_name,
                pair_skew_ms,
                dropped_left,
                dropped_right,
                latest_measurements,
                depth_worker.status_lines(),
            )
            cv2.imshow(window_name, display)

            if show_disparity and latest_disparity is not None:
                cv2.imshow(
                    "Disparity Debug",
                    make_disparity_view(
                        latest_disparity,
                        depth_worker.minimum_disparity,
                        depth_worker.maximum_disparity,
                    ),
                )

            if show_point_cloud and latest_depth_result is not None:
                if cloud_sequence != latest_depth_result.sequence:
                    latest_cloud = point_cloud_from_depth_result(
                        latest_depth_result,
                        calibration.q_matrix,
                    )
                    cloud_sequence = latest_depth_result.sequence
                cv2.imshow(
                    POINT_CLOUD_WINDOW,
                    make_orthographic_view(latest_cloud),
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("d"), ord("D")):
                show_disparity = not show_disparity
                if not show_disparity:
                    try:
                        cv2.destroyWindow("Disparity Debug")
                    except cv2.error:
                        pass
            if key in (ord("p"), ord("P")):
                show_point_cloud = not show_point_cloud
                if not show_point_cloud:
                    try:
                        cv2.destroyWindow(POINT_CLOUD_WINDOW)
                    except cv2.error:
                        pass
                print(
                    "3D projections "
                    f"{'enabled' if show_point_cloud else 'disabled'}."
                )
            if key in (ord("o"), ord("O")):
                if latest_depth_result is None:
                    print("No depth result is available for Open3D yet.")
                else:
                    try:
                        if cloud_sequence != latest_depth_result.sequence:
                            latest_cloud = point_cloud_from_depth_result(
                                latest_depth_result,
                                calibration.q_matrix,
                            )
                            cloud_sequence = latest_depth_result.sequence
                        show_open3d_snapshot(latest_cloud)
                    except RuntimeError as exc:
                        print(f"Open3D unavailable: {exc}")
            if key in (ord("s"), ord("S")):
                if latest_depth_result is None:
                    print("No depth result is available to save yet.")
                else:
                    if cloud_sequence != latest_depth_result.sequence:
                        latest_cloud = point_cloud_from_depth_result(
                            latest_depth_result,
                            calibration.q_matrix,
                        )
                        cloud_sequence = latest_depth_result.sequence
                    output = save_3d_evidence(
                        latest_depth_result,
                        latest_cloud,
                    )
                    print(f"Saved 3D evidence: {output}")
            adjustment = {
                ord("w"): "more_disparity",
                ord("W"): "more_disparity",
                ord("z"): "less_disparity",
                ord("Z"): "less_disparity",
                ord("e"): "more_confidence",
                ord("E"): "more_confidence",
                ord("c"): "less_confidence",
                ord("C"): "less_confidence",
                ord("u"): "more_uniqueness",
                ord("U"): "more_uniqueness",
                ord("j"): "less_uniqueness",
                ord("J"): "less_uniqueness",
                ord("r"): "reset",
                ord("R"): "reset",
            }.get(key)
            if adjustment is not None:
                if args.backend != "vpi-cuda":
                    print("Live VPI controls are disabled for the OpenCV backend.")
                    continue
                candidate = adjust_vpi_settings(vpi_settings, adjustment)
                if candidate == vpi_settings:
                    print(f"VPI setting is already at its limit: {vpi_settings}")
                    continue
                previous_settings = vpi_settings
                depth_worker.stop()
                replacement = make_depth_worker(candidate)
                try:
                    replacement.start()
                except RuntimeError as exc:
                    replacement.stop()
                    print(
                        f"VPI adjustment failed ({exc}); restoring the last "
                        "working settings."
                    )
                    depth_worker = make_depth_worker(previous_settings)
                    depth_worker.start()
                else:
                    depth_worker = replacement
                    vpi_settings = candidate
                    print(f"Applied VPI runtime settings: {vpi_settings}")
    finally:
        if capture_worker is not None:
            capture_worker.stop()
        depth_worker.stop()
        if left_camera is not None:
            left_camera.release()
        if right_camera is not None:
            right_camera.release()
        cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
