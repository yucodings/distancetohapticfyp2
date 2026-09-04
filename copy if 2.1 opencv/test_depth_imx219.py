#!/usr/bin/env python3
"""Test live stereo depth from an 8 MP IMX219 binocular camera.

The program captures 1280x720 frames from both CSI sensors, rectifies them
with the copied calibration NPZ, computes full-resolution disparity with
NVIDIA VPI CUDA, calculates Z-only depth from the saved Q matrix, and reports
the median distance in a centre region.

Controls:
    Q or Esc  quit
    D         toggle the disparity debug window
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# User-adjustable settings
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_PATH = SCRIPT_DIR / "stereo_calibration.npz"

LEFT_SENSOR_ID = 0
RIGHT_SENSOR_ID = 1

# These must agree with the copied calibration for this test.
EXPECTED_WIDTH = 1280
EXPECTED_HEIGHT = 720
EXPECTED_FPS = 30
EXPECTED_SENSOR_MODE = 4

MIN_DEPTH_M = 0.20
MAX_DEPTH_M = 5.00
MIN_CENTRE_DEPTH_SAMPLES = 100
CENTRE_ROI_WIDTH_FRACTION = 0.20
CENTRE_ROI_HEIGHT_FRACTION = 0.20

MIN_DISPARITY = 0
NUM_DISPARITIES = 128  # Keep the full near-range capability.
BLOCK_SIZE = 5         # Used only by the OpenCV diagnostic backend.

DEFAULT_STEREO_BACKEND = "vpi-cuda"
DEPTH_EVERY_N_DISPLAY_FRAMES = 2
VPI_CONFIDENCE_THRESHOLD = 32767
VPI_P1 = 3
VPI_P2 = 48
VPI_UNIQUENESS = -1.0
VPI_INCLUDE_DIAGONALS = False

CAMERA_FRAME_BUFFER_SIZE = 8
MAX_PAIR_SKEW_MS = 12.0
MAX_CONSECUTIVE_CAPTURE_FAILURES = 10

DISPLAY_PANEL_WIDTH = 640
DISPLAY_PANEL_HEIGHT = 360
SHOW_DISPARITY_AT_START = False


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="stereo calibration NPZ (default: copied file beside this script)",
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
        help="stereo engine; VPI CUDA is the accelerated default",
    )
    parser.add_argument(
        "--max-pair-skew-ms",
        type=float,
        default=MAX_PAIR_SKEW_MS,
        help="reject left/right pairs farther apart than this many milliseconds",
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
    reduces software skew and rejects bad pairs, but cannot create true sensor
    trigger synchronization.
    """

    def __init__(
        self,
        left_camera: cv2.VideoCapture,
        right_camera: cv2.VideoCapture,
        max_pair_skew_ms: float,
    ) -> None:
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.max_pair_skew_ms = max_pair_skew_ms
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
        self._last_rejected_skew_ms: Optional[float] = None

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
                    left = self._left_buffer[0]
                    right = self._right_buffer[0]
                    signed_skew_ms = (
                        left.arrived_at - right.arrived_at
                    ) * 1000.0
                    absolute_skew_ms = abs(signed_skew_ms)

                    if absolute_skew_ms <= self.max_pair_skew_ms:
                        self._left_buffer.popleft()
                        self._right_buffer.popleft()
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

                    # The earlier frame can never match the later front frame
                    # or any still newer frame, so discard only that one.
                    self._last_rejected_skew_ms = absolute_skew_ms
                    if signed_skew_ms < 0.0:
                        self._left_buffer.popleft()
                        self._dropped["left"] += 1
                    else:
                        self._right_buffer.popleft()
                        self._dropped["right"] += 1
                    continue

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    rejected = (
                        "none"
                        if self._last_rejected_skew_ms is None
                        else f"{self._last_rejected_skew_ms:.2f} ms"
                    )
                    raise RuntimeError(
                        "Timed out waiting for a synchronized stereo pair. "
                        f"Allowed skew is {self.max_pair_skew_ms:.2f} ms; "
                        f"last rejected skew was {rejected}. If both cameras "
                        "are stable at 30 FPS, try --max-pair-skew-ms 18."
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


def create_stereo_matcher() -> Any:
    if NUM_DISPARITIES <= 0 or NUM_DISPARITIES % 16 != 0:
        raise ValueError("NUM_DISPARITIES must be a positive multiple of 16")
    if BLOCK_SIZE < 3 or BLOCK_SIZE % 2 == 0:
        raise ValueError("BLOCK_SIZE must be an odd integer of at least 3")

    return cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * BLOCK_SIZE * BLOCK_SIZE,
        P2=32 * BLOCK_SIZE * BLOCK_SIZE,
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

    def __init__(self) -> None:
        self.matcher = create_stereo_matcher()

    def compute(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        disparity = calculate_disparity(
            self.matcher, left_rectified, right_rectified
        )
        return disparity, None

    def warmup(self) -> None:
        return


class VpiCudaStereoEngine:
    """Full-resolution VPI CUDA disparity with reusable buffers and stream."""

    name = "NVIDIA VPI CUDA Stereo"
    uses_cuda = True

    def __init__(self, width: int, height: int) -> None:
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
        self.left_host = np.empty((height, width), dtype=np.uint8)
        self.right_host = np.empty((height, width), dtype=np.uint8)

        try:
            self.stream = vpi.Stream()
            # These wrappers keep pointing at the same numpy allocations. Each
            # frame updates their contents without recreating wrapper objects.
            self.left_input = vpi.asimage(self.left_host)
            self.right_input = vpi.asimage(self.right_host)
            self.left_y16 = vpi.Image((width, height), vpi.Format.Y16_ER)
            self.right_y16 = vpi.Image((width, height), vpi.Format.Y16_ER)
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
        np.copyto(self.left_host, left_gray)
        np.copyto(self.right_host, right_gray)

        try:
            with self.stream, self.vpi.Backend.CUDA:
                self.left_input.convert(self.left_y16)
                self.right_input.convert(self.right_y16)
                self.vpi.stereodisp(
                    self.left_y16,
                    self.right_y16,
                    out=self.disparity_s16,
                    out_confmap=self.confidence_u16,
                    window=BLOCK_SIZE,
                    maxdisp=NUM_DISPARITIES,
                    confthreshold=VPI_CONFIDENCE_THRESHOLD,
                    conftype=self.vpi.ConfidenceType.ABSOLUTE,
                    mindisp=MIN_DISPARITY,
                    p1=VPI_P1,
                    p2=VPI_P2,
                    uniqueness=VPI_UNIQUENESS,
                    includediagonals=VPI_INCLUDE_DIAGONALS,
                )

            # VPI disparity is signed Q10.5: five fractional bits.
            disparity = np.asarray(
                self.disparity_s16.cpu(), dtype=np.float32
            ) / 32.0
            confidence = np.asarray(self.confidence_u16.cpu())
        except Exception as exc:
            raise RuntimeError(f"VPI CUDA stereo disparity failed: {exc}") from exc

        if disparity.shape != (self.height, self.width):
            raise RuntimeError(
                f"VPI disparity has shape {disparity.shape}, expected "
                f"{(self.height, self.width)}"
            )
        confidence_mask = confidence >= VPI_CONFIDENCE_THRESHOLD
        return disparity, confidence_mask

    def warmup(self) -> None:
        """Create/cache the VPI payload before opening the cameras."""
        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.compute(blank, blank)


def create_stereo_engine(backend: str, calibration: Calibration) -> Any:
    if NUM_DISPARITIES != 128:
        raise ValueError(
            "This full-resolution configuration requires NUM_DISPARITIES=128; "
            "123 is unsupported."
        )
    if backend == "vpi-cuda":
        return VpiCudaStereoEngine(calibration.width, calibration.height)
    if backend == "opencv":
        return OpenCvStereoEngine()
    raise ValueError(f"Unsupported stereo backend: {backend}")


def calculate_depth(
    disparity: np.ndarray,
    q_matrix: np.ndarray,
    max_depth_m: float,
    confidence_mask: Optional[np.ndarray] = None,
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
        (disparity > MIN_DISPARITY)
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


def centre_roi(image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape
    roi_width = max(1, int(round(width * CENTRE_ROI_WIDTH_FRACTION)))
    roi_height = max(1, int(round(height * CENTRE_ROI_HEIGHT_FRACTION)))
    centre_x = width // 2
    centre_y = height // 2
    x1 = max(0, centre_x - roi_width // 2)
    y1 = max(0, centre_y - roi_height // 2)
    return x1, y1, min(width, x1 + roi_width), min(height, y1 + roi_height)


def measure_centre_depth(
    depth_map: np.ndarray,
    valid_depth_mask: np.ndarray,
) -> tuple[Optional[float], tuple[int, int, int, int], int]:
    box = centre_roi(depth_map.shape)
    x1, y1, x2, y2 = box
    roi_mask = valid_depth_mask[y1:y2, x1:x2]
    valid_values = depth_map[y1:y2, x1:x2][roi_mask]
    sample_count = int(valid_values.size)
    if sample_count < MIN_CENTRE_DEPTH_SAMPLES:
        return None, box, sample_count
    return float(np.median(valid_values)), box, sample_count


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


def make_disparity_view(disparity: np.ndarray) -> np.ndarray:
    valid = disparity > MIN_DISPARITY
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


def draw_measurement(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    distance_m: Optional[float],
    sample_count: int,
) -> None:
    x1, y1, x2, y2 = box
    colour = (0, 255, 0) if distance_m is not None else (0, 0, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
    distance_text = (
        f"Centre: {distance_m:.2f} m"
        if distance_m is not None
        else "Centre: Depth N/A"
    )
    cv2.rectangle(image, (10, 10), (350, 67), (0, 0, 0), -1)
    cv2.putText(
        image,
        distance_text,
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        colour,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Centre valid samples: {sample_count}",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
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
    max_pair_skew_ms: float,
) -> None:
    if pair_skew_ms <= 5.0:
        sync_status = "EXCELLENT"
    elif pair_skew_ms <= max_pair_skew_ms:
        sync_status = "GOOD"
    else:
        sync_status = "POOR"
    lines = (
        f"Camera FPS L/R: {left_camera_fps:.1f} / {right_camera_fps:.1f}",
        f"Pair skew: {pair_skew_ms:.2f} ms ({sync_status})",
        f"Sync drops L/R: {dropped_left} / {dropped_right}",
        f"Display FPS: {display_fps:.1f}",
        f"Depth FPS: {depth_fps:.1f}",
        f"Valid depth: {valid_percentage:.1f}%",
        f"Stereo: {backend_name}",
        "Depth every 2nd display frame",
        f"D disparity: {'ON' if disparity_visible else 'OFF'}",
        "Q / Esc: quit",
    )
    panel_width = 370
    panel_height = len(lines) * 22 + 12
    x1 = image.shape[1] - panel_width - 10
    panel = image[10 : 10 + panel_height, x1 : x1 + panel_width]
    dark = np.zeros_like(panel)
    cv2.addWeighted(dark, 0.65, panel, 0.35, 0, dst=panel)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x1 + 10, 32 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
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
    return cv2.hconcat((left_preview, depth_preview))


def print_startup(
    calibration: Calibration,
    left_id: int,
    right_id: int,
    max_depth_m: float,
    backend_name: str,
    max_pair_skew_ms: float,
) -> None:
    print("\nIMX219 stereo depth test")
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
    print(f"  Maximum disparity:    {NUM_DISPARITIES}")
    print("  3D output:            Z depth only")
    print("  Rectification maps:   fixed-point CV_16SC2")
    print(f"  Maximum pair skew:    {max_pair_skew_ms:.2f} ms")
    print("  Sync timestamps:      host arrival time (software sync)")
    print(f"  Depth schedule:       every {DEPTH_EVERY_N_DISPLAY_FRAMES}nd display frame")
    print("  Controls:             Q/Esc quit | D disparity\n")


def run(args: argparse.Namespace) -> None:
    if args.max_depth <= MIN_DEPTH_M:
        raise ValueError(f"--max-depth must be greater than {MIN_DEPTH_M}")
    if args.max_pair_skew_ms <= 0.0:
        raise ValueError("--max-pair-skew-ms must be greater than zero")

    check_gstreamer()
    calibration = load_calibration(args.calibration)
    rectification_maps = convert_rectification_maps(calibration)

    print(f"Initializing stereo backend: {args.backend}...", flush=True)
    stereo_engine = create_stereo_engine(args.backend, calibration)
    stereo_engine.warmup()
    print_startup(
        calibration,
        args.left_id,
        args.right_id,
        args.max_depth,
        stereo_engine.name,
        args.max_pair_skew_ms,
    )

    left_camera: Optional[cv2.VideoCapture] = None
    right_camera: Optional[cv2.VideoCapture] = None
    capture_worker: Optional[SynchronizedStereoCapture] = None
    try:
        print("Opening IMX219 cameras...", flush=True)
        left_camera, right_camera = open_cameras(
            args.left_id, args.right_id, calibration
        )
        capture_worker = SynchronizedStereoCapture(
            left_camera,
            right_camera,
            args.max_pair_skew_ms,
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

        window_name = "IMX219 Stereo Depth Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            window_name,
            DISPLAY_PANEL_WIDTH * 2,
            DISPLAY_PANEL_HEIGHT,
        )

        show_disparity = SHOW_DISPARITY_AT_START
        display_fps = 0.0
        depth_fps = 0.0
        previous_display_time = time.perf_counter()
        previous_depth_time: Optional[float] = None
        display_frame_counter = 0

        latest_disparity: Optional[np.ndarray] = None
        latest_depth_view: Optional[np.ndarray] = None
        latest_distance_m: Optional[float] = None
        latest_box = centre_roi((calibration.height, calibration.width))
        latest_sample_count = 0
        latest_valid_percentage = 0.0

        while True:
            stereo_frame = capture_worker.get_latest(last_sequence)
            last_sequence = stereo_frame.sequence
            validate_frame_resolution(
                stereo_frame.left, stereo_frame.right, calibration
            )

            calculate_new_depth = (
                latest_depth_view is None
                or display_frame_counter % DEPTH_EVERY_N_DISPLAY_FRAMES == 0
            )
            if calculate_new_depth:
                left_rectified, right_rectified = rectify_frames(
                    stereo_frame.left,
                    stereo_frame.right,
                    rectification_maps,
                )
                disparity, confidence_mask = stereo_engine.compute(
                    left_rectified, right_rectified
                )
                depth_map, valid_depth_mask = calculate_depth(
                    disparity,
                    calibration.q_matrix,
                    args.max_depth,
                    confidence_mask,
                )
                distance_m, box, sample_count = measure_centre_depth(
                    depth_map, valid_depth_mask
                )

                latest_disparity = disparity
                latest_depth_view = make_depth_view(
                    depth_map, valid_depth_mask, args.max_depth
                )
                latest_distance_m = distance_m
                latest_box = box
                latest_sample_count = sample_count
                latest_valid_percentage = 100.0 * float(
                    np.mean(valid_depth_mask)
                )

                depth_now = time.perf_counter()
                if previous_depth_time is not None:
                    instantaneous_depth_fps = 1.0 / max(
                        depth_now - previous_depth_time, 1e-6
                    )
                    depth_fps = (
                        instantaneous_depth_fps
                        if depth_fps == 0.0
                        else 0.90 * depth_fps
                        + 0.10 * instantaneous_depth_fps
                    )
                previous_depth_time = depth_now
            else:
                # Keep the live reference image moving while reusing the most
                # recent full-field depth result on alternating frames.
                left_rectified = rectify_left_frame(
                    stereo_frame.left, rectification_maps
                )

            assert latest_depth_view is not None
            depth_display = latest_depth_view.copy()
            draw_measurement(
                left_rectified,
                latest_box,
                latest_distance_m,
                latest_sample_count,
            )
            draw_measurement(
                depth_display,
                latest_box,
                latest_distance_m,
                latest_sample_count,
            )

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

            display = make_primary_display(left_rectified, depth_display)
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
                stereo_engine.name,
                pair_skew_ms,
                dropped_left,
                dropped_right,
                args.max_pair_skew_ms,
            )
            cv2.imshow(window_name, display)

            if show_disparity and latest_disparity is not None:
                cv2.imshow(
                    "Disparity Debug",
                    make_disparity_view(latest_disparity),
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("d"), ord("D")):
                show_disparity = not show_disparity
                if not show_disparity:
                    cv2.destroyWindow("Disparity Debug")
            display_frame_counter += 1
    finally:
        if capture_worker is not None:
            capture_worker.stop()
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
