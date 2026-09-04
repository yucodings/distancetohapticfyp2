"""Asynchronous disparity and metric-depth processing for rectified IMX219s."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from calibration import Calibration, RectificationMaps, rectify_right
from camera_backend import cv2, np
from config import (
    ALLOW_OPENCV_STEREO_FALLBACK,
    MAX_VALID_DEPTH,
    MIN_VALID_DEPTH,
    SGBM_BLOCK_SIZE,
    SGBM_MIN_DISPARITY,
    SGBM_NUM_DISPARITIES,
    VPI_INCLUDE_DIAGONALS,
    VPI_INTERNAL_CONFIDENCE_THRESHOLD,
    VPI_MIN_CONFIDENCE,
    VPI_MAX_DISPARITY,
    VPI_MIN_DISPARITY,
    VPI_P1,
    VPI_P2,
    VPI_PROFILE_PATH,
    VPI_UNIQUENESS,
    VPI_WINDOW,
)


StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class VpiRuntimeSettings:
    min_disparity: int
    max_disparity: int
    confidence_threshold: int
    p1: int
    p2: int
    uniqueness: float
    include_diagonals: bool

    def validate(self) -> None:
        if not 0 <= self.min_disparity < self.max_disparity <= 256:
            raise ValueError("VPI disparity range must satisfy 0 <= min < max <= 256")
        if not 0 <= self.confidence_threshold <= 65535:
            raise ValueError("VPI confidence threshold must be 0..65535")
        if not 0 < self.p1 <= self.p2 < 256:
            raise ValueError("VPI penalties must satisfy 0 < P1 <= P2 < 256")
        if self.uniqueness != -1.0 and not 0.0 <= self.uniqueness <= 1.0:
            raise ValueError("VPI uniqueness must be -1 or 0..1")


def default_vpi_settings() -> VpiRuntimeSettings:
    settings = VpiRuntimeSettings(
        min_disparity=VPI_MIN_DISPARITY,
        max_disparity=VPI_MAX_DISPARITY,
        confidence_threshold=VPI_MIN_CONFIDENCE,
        p1=VPI_P1,
        p2=VPI_P2,
        uniqueness=VPI_UNIQUENESS,
        include_diagonals=VPI_INCLUDE_DIAGONALS,
    )
    settings.validate()
    return settings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_vpi_runtime_settings(
    calibration: Calibration,
    profile_path: Path = VPI_PROFILE_PATH,
) -> tuple[VpiRuntimeSettings, str]:
    """Load an Easy Mode profile only when it matches this calibration."""
    if not profile_path.is_file():
        return default_vpi_settings(), "built-in VPI defaults (not tuned)"
    try:
        with profile_path.open("r", encoding="utf-8") as stream:
            profile = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read VPI profile {profile_path}: {error}") from error
    if profile.get("schema_version") != 1 or profile.get("backend") != "vpi-cuda":
        raise RuntimeError(f"Unsupported VPI profile format: {profile_path}")
    expected_hash = _sha256(calibration.path)
    if profile.get("calibration_sha256") != expected_hash:
        raise RuntimeError(
            "VPI profile calibration hash does not match stereo_calibration.npz"
        )
    raw = profile.get("settings")
    if not isinstance(raw, dict):
        raise RuntimeError("VPI profile has no settings object")
    required = {
        "min_disparity",
        "max_disparity",
        "confidence_threshold",
        "p1",
        "p2",
        "uniqueness",
        "include_diagonals",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise RuntimeError("VPI profile settings missing: " + ", ".join(missing))
    if not isinstance(raw["include_diagonals"], bool):
        raise RuntimeError("VPI include_diagonals must be true or false")
    try:
        settings = VpiRuntimeSettings(
            min_disparity=int(raw["min_disparity"]),
            max_disparity=int(raw["max_disparity"]),
            confidence_threshold=int(raw["confidence_threshold"]),
            p1=int(raw["p1"]),
            p2=int(raw["p2"]),
            uniqueness=float(raw["uniqueness"]),
            include_diagonals=raw["include_diagonals"],
        )
        settings.validate()
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid VPI profile settings: {error}") from error
    return settings, f"tuned profile {profile_path.name}"


@dataclass(frozen=True)
class DepthInput:
    sequence: int
    captured_at_ms: float
    left_rectified: np.ndarray
    right_raw: np.ndarray


@dataclass(frozen=True)
class DepthResult:
    sequence: int
    captured_at_ms: float
    completed_at_ms: float
    left_rectified: np.ndarray
    disparity: np.ndarray
    depth_map: np.ndarray
    valid_depth_mask: np.ndarray
    depth_view: np.ndarray
    depth_fps: float
    valid_percentage: float
    backend_name: str
    diagnostic_lines: tuple[str, ...]


class OpenCvStereoEngine:
    name = "OpenCV CPU StereoSGBM"

    def __init__(self):
        if SGBM_NUM_DISPARITIES <= 0 or SGBM_NUM_DISPARITIES % 16:
            raise ValueError(
                "SGBM_NUM_DISPARITIES must be a positive multiple of 16"
            )
        if SGBM_BLOCK_SIZE < 3 or SGBM_BLOCK_SIZE % 2 == 0:
            raise ValueError("SGBM_BLOCK_SIZE must be odd and at least 3")
        self.minimum_disparity = SGBM_MIN_DISPARITY
        self.maximum_disparity = (
            SGBM_MIN_DISPARITY + SGBM_NUM_DISPARITIES
        )
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=SGBM_MIN_DISPARITY,
            numDisparities=SGBM_NUM_DISPARITIES,
            blockSize=SGBM_BLOCK_SIZE,
            P1=8 * SGBM_BLOCK_SIZE * SGBM_BLOCK_SIZE,
            P2=32 * SGBM_BLOCK_SIZE * SGBM_BLOCK_SIZE,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self._positive_percentage = 0.0

    def warmup(self) -> None:
        return

    def compute(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        raw = self.matcher.compute(left_gray, right_gray)
        if raw is None or raw.size == 0:
            raise RuntimeError("StereoSGBM returned no disparity")
        disparity = raw.astype(np.float32) / 16.0
        plausible = (
            np.isfinite(disparity)
            & (disparity > self.minimum_disparity)
            & (disparity < self.maximum_disparity)
        )
        self._positive_percentage = 100.0 * float(np.mean(plausible))
        return disparity, None

    def diagnostics(self) -> tuple[str, ...]:
        return (f"SGBM plausible disparity: {self._positive_percentage:.1f}%",)


class VpiCudaStereoEngine:
    name = "NVIDIA VPI CUDA Stereo"

    def __init__(
        self,
        width: int,
        height: int,
        settings: VpiRuntimeSettings,
        profile_source: str,
    ):
        settings.validate()
        try:
            import vpi
        except Exception as error:
            raise RuntimeError(f"NVIDIA VPI import failed: {error}") from error
        self.vpi = vpi
        self.width = width
        self.height = height
        self.settings = settings
        self.profile_source = profile_source
        self.minimum_disparity = settings.min_disparity
        self.maximum_disparity = settings.max_disparity
        self.stream = vpi.Stream()
        self.left_input = vpi.Image((width, height), vpi.Format.U8)
        self.right_input = vpi.Image((width, height), vpi.Format.U8)
        self.disparity_s16 = vpi.Image((width, height), vpi.Format.S16)
        self.confidence_u16 = vpi.Image((width, height), vpi.Format.U16)
        self._plausible_percentage = 0.0
        self._confidence_percentage = 0.0
        self._invalid_percentage = 0.0

    def warmup(self) -> None:
        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.compute(blank, blank)

    def compute(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        try:
            with self.left_input.wlock_cpu() as data:
                np.copyto(data, left_gray)
            with self.right_input.wlock_cpu() as data:
                np.copyto(data, right_gray)
            with self.stream, self.vpi.Backend.CUDA:
                self.vpi.stereodisp(
                    self.left_input,
                    self.right_input,
                    out=self.disparity_s16,
                    out_confmap=self.confidence_u16,
                    window=VPI_WINDOW,
                    maxdisp=self.settings.max_disparity,
                    confthreshold=VPI_INTERNAL_CONFIDENCE_THRESHOLD,
                    conftype=self.vpi.ConfidenceType.ABSOLUTE,
                    mindisp=self.settings.min_disparity,
                    p1=self.settings.p1,
                    p2=self.settings.p2,
                    uniqueness=self.settings.uniqueness,
                    includediagonals=self.settings.include_diagonals,
                )
            with self.disparity_s16.rlock_cpu() as data:
                disparity = np.array(data, dtype=np.float32) / 32.0
            with self.confidence_u16.rlock_cpu() as data:
                confidence = np.array(data, copy=True)
        except Exception as error:
            raise RuntimeError(f"VPI CUDA disparity failed: {error}") from error

        plausible = (
            np.isfinite(disparity)
            & (disparity > self.minimum_disparity)
            & (disparity < self.maximum_disparity)
        )
        invalid = np.isfinite(disparity) & (
            disparity >= self.maximum_disparity
        )
        confidence_mask = plausible & (
            confidence >= max(1, self.settings.confidence_threshold)
        )
        self._plausible_percentage = 100.0 * float(np.mean(plausible))
        self._confidence_percentage = 100.0 * float(np.mean(confidence_mask))
        self._invalid_percentage = 100.0 * float(np.mean(invalid))
        return disparity, confidence_mask

    def diagnostics(self) -> tuple[str, ...]:
        return (
            f"VPI parameters: {self.profile_source}",
            f"VPI plausible disparity: {self._plausible_percentage:.1f}%",
            f"VPI confidence pass: {self._confidence_percentage:.1f}%",
            f"VPI invalid sentinel: {self._invalid_percentage:.1f}%",
        )


def calculate_depth(
    disparity: np.ndarray,
    q_matrix: np.ndarray,
    confidence_mask: Optional[np.ndarray] = None,
    min_disparity: int = SGBM_MIN_DISPARITY,
    max_disparity: int = SGBM_MIN_DISPARITY + SGBM_NUM_DISPARITIES,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate forward Z from canonical Q and reject invalid ranges."""
    if disparity.ndim != 2:
        raise ValueError("Disparity must be a 2-D array")
    if q_matrix.shape != (4, 4):
        raise ValueError("Q matrix must be 4x4")
    if not (
        np.allclose(q_matrix[2, :3], 0.0)
        and np.allclose(q_matrix[3, :2], 0.0)
    ):
        raise RuntimeError("Q matrix is not canonical rectified-stereo form")

    denominator = q_matrix[3, 2] * disparity + q_matrix[3, 3]
    can_divide = (
        np.isfinite(disparity)
        & (disparity > min_disparity)
        & (disparity < max_disparity)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-12)
    )
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    np.divide(q_matrix[2, 3], denominator, out=depth, where=can_divide)
    valid = (
        can_divide
        & np.isfinite(depth)
        & (depth > MIN_VALID_DEPTH)
        & (depth < MAX_VALID_DEPTH)
    )
    if confidence_mask is not None:
        if confidence_mask.shape != disparity.shape:
            raise ValueError("Confidence/disparity shape mismatch")
        valid &= confidence_mask
    depth[~valid] = np.nan
    return depth, valid


def make_depth_view(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth[valid], MIN_VALID_DEPTH, MAX_VALID_DEPTH)
        scaled[valid] = np.round(
            (MAX_VALID_DEPTH - clipped)
            * 255.0
            / (MAX_VALID_DEPTH - MIN_VALID_DEPTH)
        ).astype(np.uint8)
    view = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    view[~valid] = 0
    return view


def create_engine(backend: str, calibration: Calibration) -> Any:
    if backend == "vpi-cuda":
        settings, source = load_vpi_runtime_settings(calibration)
        return VpiCudaStereoEngine(
            calibration.width, calibration.height, settings, source
        )
    if backend == "opencv":
        return OpenCvStereoEngine()
    raise ValueError(f"Unknown stereo backend: {backend}")


class AsyncDepthProcessor:
    """Continuously process only the newest submitted stereo pair."""

    def __init__(
        self,
        backend: str,
        calibration: Calibration,
        maps: RectificationMaps,
        status_callback: Optional[StatusCallback] = None,
    ):
        self.backend = backend
        self.calibration = calibration
        self.maps = maps
        self.status_callback = status_callback
        self.backend_name = backend
        self._condition = threading.Condition()
        self._ready = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pending: Optional[DepthInput] = None
        self._latest: Optional[DepthResult] = None
        self._error: Optional[str] = None
        self._replaced_inputs = 0

    def _status(self, message: str) -> None:
        if self.status_callback is not None:
            self.status_callback(message)

    def start(self, timeout: float = 30.0) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="stereo-depth-worker", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            self.stop()
            raise RuntimeError("Timed out initializing stereo depth")
        with self._condition:
            if self._error is not None:
                raise RuntimeError(self._error)

    def submit(
        self,
        sequence: int,
        captured_at_ms: float,
        left_rectified: np.ndarray,
        right_raw: np.ndarray,
    ) -> None:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(self._error)
            if not self._running:
                raise RuntimeError("Stereo depth worker is stopped")
            if self._pending is not None:
                self._replaced_inputs += 1
            self._pending = DepthInput(
                sequence,
                captured_at_ms,
                left_rectified.copy(),
                right_raw.copy(),
            )
            self._condition.notify_all()

    def latest_after(self, sequence: int) -> Optional[DepthResult]:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(self._error)
            if self._latest is None or self._latest.sequence <= sequence:
                return None
            return self._latest

    def status_lines(self) -> tuple[str, ...]:
        with self._condition:
            return (f"Depth input replacements: {self._replaced_inputs}",)

    def _run(self) -> None:
        engine: Optional[Any] = None
        previous_completion: Optional[float] = None
        smoothed_fps = 0.0
        try:
            try:
                engine = create_engine(self.backend, self.calibration)
                engine.warmup()
            except Exception as primary_error:
                if self.backend != "vpi-cuda" or not ALLOW_OPENCV_STEREO_FALLBACK:
                    raise
                self._status(
                    f"VPI unavailable ({primary_error}); using OpenCV stereo fallback"
                )
                engine = OpenCvStereoEngine()
                engine.warmup()
            self.backend_name = engine.name
            self._status(f"Stereo depth ready | {engine.name}")
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

                right_rectified = rectify_right(depth_input.right_raw, self.maps)
                disparity, confidence = engine.compute(
                    depth_input.left_rectified, right_rectified
                )
                depth, valid = calculate_depth(
                    disparity,
                    self.calibration.q_matrix,
                    confidence,
                    engine.minimum_disparity,
                    engine.maximum_disparity,
                )
                completed = time.monotonic()
                if previous_completion is not None:
                    instantaneous = 1.0 / max(completed - previous_completion, 1e-6)
                    smoothed_fps = (
                        instantaneous
                        if smoothed_fps == 0.0
                        else 0.90 * smoothed_fps + 0.10 * instantaneous
                    )
                previous_completion = completed
                result = DepthResult(
                    sequence=depth_input.sequence,
                    captured_at_ms=depth_input.captured_at_ms,
                    completed_at_ms=completed * 1000.0,
                    left_rectified=depth_input.left_rectified,
                    disparity=disparity,
                    depth_map=depth,
                    valid_depth_mask=valid,
                    depth_view=make_depth_view(depth, valid),
                    depth_fps=smoothed_fps,
                    valid_percentage=100.0 * float(np.mean(valid)),
                    backend_name=engine.name,
                    diagnostic_lines=engine.diagnostics(),
                )
                with self._condition:
                    self._latest = result
        except Exception as error:
            with self._condition:
                self._error = f"Stereo depth failed: {error}"
                self._running = False
                self._condition.notify_all()
        finally:
            self._ready.set()

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._pending = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
