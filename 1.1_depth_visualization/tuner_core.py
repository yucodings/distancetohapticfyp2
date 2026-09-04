"""Core stereo-depth tuning functions for the IMX219 diagnostic tool.

This module deliberately contains no actuator or YOLO code.  It loads
JetPack's GStreamer-enabled OpenCV build so the same image-processing ABI is
used for camera capture, StereoSGBM, and optional WLS filtering.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


SYSTEM_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
_original_path = list(sys.path)
if SYSTEM_DIST_PACKAGES.is_dir():
    _system_path = str(SYSTEM_DIST_PACKAGES)
    if _system_path in sys.path:
        sys.path.remove(_system_path)
    sys.path.insert(0, _system_path)
try:
    import cv2
    import numpy as np
finally:
    sys.path[:] = _original_path


MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 3.00


@dataclass(frozen=True)
class CalibrationData:
    path: Path
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
    stereo_rms: Optional[float]
    sha256: str


@dataclass(frozen=True)
class SgbmSettings:
    min_disparity: int = 0
    num_disparities: int = 160
    block_size: int = 5
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    pre_filter_cap: int = 63
    p1_factor: int = 8
    p2_factor: int = 32
    use_wls: bool = False
    wls_lambda: float = 8000.0
    wls_sigma: float = 1.5

    @property
    def maximum_disparity(self) -> int:
        return self.min_disparity + self.num_disparities

    def validate(self) -> None:
        if self.num_disparities <= 0 or self.num_disparities % 16:
            raise ValueError("SGBM num_disparities must be a positive multiple of 16")
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ValueError("SGBM block_size must be odd and at least 3")
        if self.p1_factor <= 0 or self.p2_factor <= self.p1_factor:
            raise ValueError("SGBM requires positive P1 factor and P2 factor > P1")
        if not 1 <= self.pre_filter_cap <= 63:
            raise ValueError("SGBM pre_filter_cap must be between 1 and 63")


@dataclass(frozen=True)
class VpiSettings:
    min_disparity: int = 0
    max_disparity: int = 160
    confidence_threshold: int = 8192
    p1: int = 3
    p2: int = 48
    uniqueness: float = -1.0
    include_diagonals: bool = False

    def validate(self) -> None:
        if not 0 <= self.min_disparity < self.max_disparity <= 256:
            raise ValueError("VPI disparities must satisfy 0 <= min < max <= 256")
        if not 0 <= self.confidence_threshold <= 65535:
            raise ValueError("VPI confidence threshold must be 0..65535")
        if not 0 < self.p1 <= self.p2 < 256:
            raise ValueError("VPI requires 0 < P1 <= P2 < 256")
        if self.uniqueness != -1.0 and not 0.0 <= self.uniqueness <= 1.0:
            raise ValueError("VPI uniqueness must be -1 (off) or 0..1")


# VPI CUDA fixes its census window at 9x7. These are the native controls that
# Easy Mode can genuinely tune without pretending that SGBM block size applies.
VPI_AUTO_PENALTY_PAIRS = ((1, 32), (3, 48), (5, 64), (8, 96))
VPI_AUTO_UNIQUENESS = (-1.0, 0.80, 0.90, 0.95)
VPI_AUTO_CONFIDENCE_THRESHOLDS = (1, 4096, 8192, 16384, 32767)
VPI_AUTO_DIAGONAL_OPTIONS = (True, False)


def vpi_autotune_candidates(max_disparity: int = 256) -> tuple[VpiSettings, ...]:
    """Return the deterministic VPI CUDA Easy Mode search space."""
    candidates = tuple(
        VpiSettings(
            min_disparity=0,
            max_disparity=max_disparity,
            confidence_threshold=confidence,
            p1=p1,
            p2=p2,
            uniqueness=uniqueness,
            include_diagonals=diagonals,
        )
        for diagonals in VPI_AUTO_DIAGONAL_OPTIONS
        for p1, p2 in VPI_AUTO_PENALTY_PAIRS
        for uniqueness in VPI_AUTO_UNIQUENESS
        for confidence in VPI_AUTO_CONFIDENCE_THRESHOLDS
    )
    for candidate in candidates:
        candidate.validate()
    return candidates


@dataclass(frozen=True)
class RoiStatistics:
    valid_count: int
    total_count: int
    valid_percentage: float
    median_m: Optional[float]
    mad_m: Optional[float]
    error_m: Optional[float]


def automatic_tuning_score(
    statistics: RoiStatistics,
    whole_frame_valid_percentage: float,
    processing_ms: float,
    known_distance_m: float,
) -> Optional[float]:
    """Return a lower-is-better score for a measured flat target.

    Accuracy dominates the score.  Median absolute deviation measures local
    stability, while coverage and processing time are weaker tie-breakers.
    Very sparse ROIs are rejected rather than rewarded for a few lucky pixels.
    """
    if known_distance_m <= 0:
        raise ValueError("Known distance must be positive")
    if (
        statistics.median_m is None
        or statistics.mad_m is None
        or statistics.valid_percentage < 20.0
    ):
        return None
    relative_error = abs(statistics.median_m - known_distance_m) / known_distance_m
    relative_mad = statistics.mad_m / known_distance_m
    roi_missing = 1.0 - min(100.0, max(0.0, statistics.valid_percentage)) / 100.0
    frame_missing = 1.0 - min(100.0, max(0.0, whole_frame_valid_percentage)) / 100.0
    seconds = max(0.0, processing_ms) / 1000.0
    return float(
        4.0 * relative_error
        + 1.5 * relative_mad
        + 0.30 * roi_missing
        + 0.05 * frame_missing
        + 0.03 * seconds
    )


def _scalar(data: Any, key: str, value_type: Any) -> Any:
    value = np.asarray(data[key])
    if value.size != 1:
        raise RuntimeError(f"Calibration '{key}' must be scalar, got {value.shape}")
    return value_type(value.reshape(-1)[0])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_calibration(path: Path) -> CalibrationData:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    required = (
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
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in required if name not in data.files]
        if missing:
            raise RuntimeError("Calibration is missing: " + ", ".join(missing))
        width = _scalar(data, "image_width", int)
        height = _scalar(data, "image_height", int)
        calibration = CalibrationData(
            path=path,
            width=width,
            height=height,
            sensor_mode=_scalar(data, "sensor_mode", int),
            capture_fps=_scalar(data, "capture_fps", int),
            baseline_m=_scalar(data, "baseline_m", float),
            q_matrix=np.asarray(data["Q"], dtype=np.float64).copy(),
            left_map1=np.asarray(data["left_map1"]).copy(),
            left_map2=np.asarray(data["left_map2"]).copy(),
            right_map1=np.asarray(data["right_map1"]).copy(),
            right_map2=np.asarray(data["right_map2"]).copy(),
            stereo_rms=(
                _scalar(data, "stereo_rms", float)
                if "stereo_rms" in data.files
                else None
            ),
            sha256=file_sha256(path),
        )
    if calibration.q_matrix.shape != (4, 4):
        raise RuntimeError(f"Calibration Q must be 4x4, got {calibration.q_matrix.shape}")
    expected = (height, width)
    for name, value in (
        ("left_map1", calibration.left_map1),
        ("left_map2", calibration.left_map2),
        ("right_map1", calibration.right_map1),
        ("right_map2", calibration.right_map2),
    ):
        if value.shape[:2] != expected:
            raise RuntimeError(f"{name} shape {value.shape[:2]} != {expected}")
    if not 0.01 < calibration.baseline_m < 0.50:
        raise RuntimeError(f"Implausible baseline: {calibration.baseline_m:.6f} m")
    return calibration


def convert_rectification_maps(
    calibration: CalibrationData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left1, left2 = cv2.convertMaps(
        calibration.left_map1, calibration.left_map2, cv2.CV_16SC2
    )
    right1, right2 = cv2.convertMaps(
        calibration.right_map1, calibration.right_map2, cv2.CV_16SC2
    )
    return left1, left2, right1, right2


def rectify_pair(
    left: np.ndarray,
    right: np.ndarray,
    calibration: CalibrationData,
    maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    expected = (calibration.height, calibration.width)
    if left.shape[:2] != expected or right.shape[:2] != expected:
        raise ValueError(
            f"Expected {calibration.width}x{calibration.height}; got "
            f"left {left.shape[1]}x{left.shape[0]}, "
            f"right {right.shape[1]}x{right.shape[0]}"
        )
    left1, left2, right1, right2 = maps
    return (
        cv2.remap(left, left1, left2, cv2.INTER_LINEAR),
        cv2.remap(right, right1, right2, cv2.INTER_LINEAR),
    )


def make_sgbm_matcher(settings: SgbmSettings) -> Any:
    settings.validate()
    area = settings.block_size * settings.block_size
    return cv2.StereoSGBM_create(
        minDisparity=settings.min_disparity,
        numDisparities=settings.num_disparities,
        blockSize=settings.block_size,
        P1=settings.p1_factor * area,
        P2=settings.p2_factor * area,
        disp12MaxDiff=settings.disp12_max_diff,
        uniquenessRatio=settings.uniqueness_ratio,
        speckleWindowSize=settings.speckle_window_size,
        speckleRange=settings.speckle_range,
        preFilterCap=settings.pre_filter_cap,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_sgbm(
    left_rectified: np.ndarray,
    right_rectified: np.ndarray,
    settings: SgbmSettings,
) -> tuple[np.ndarray, np.ndarray]:
    left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    matcher = make_sgbm_matcher(settings)
    raw_left = matcher.compute(left_gray, right_gray)
    if settings.use_wls:
        if not hasattr(cv2, "ximgproc"):
            raise RuntimeError("This OpenCV build has no ximgproc WLS support")
        right_matcher = cv2.ximgproc.createRightMatcher(matcher)
        raw_right = right_matcher.compute(right_gray, left_gray)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(settings.wls_lambda)
        wls.setSigmaColor(settings.wls_sigma)
        raw_left = wls.filter(raw_left, left_gray, None, raw_right)
    disparity = raw_left.astype(np.float32) / 16.0
    valid = (
        np.isfinite(disparity)
        & (disparity > settings.min_disparity)
        & (disparity < settings.maximum_disparity)
    )
    return disparity, valid


class VpiCudaMatcher:
    """Small VPI wrapper recreated when payload-affecting controls change."""

    def __init__(self, width: int, height: int, settings: VpiSettings):
        settings.validate()
        try:
            import vpi
        except Exception as error:
            raise RuntimeError(f"Could not initialize NVIDIA VPI: {error}") from error
        self.vpi = vpi
        self.width = width
        self.height = height
        self.payload_key = (settings.max_disparity, settings.include_diagonals)
        self.left_input = vpi.Image((width, height), vpi.Format.U8)
        self.right_input = vpi.Image((width, height), vpi.Format.U8)
        self.disparity_s16 = vpi.Image((width, height), vpi.Format.S16)
        self.confidence_u16 = vpi.Image((width, height), vpi.Format.U16)
        self.stream = vpi.Stream()

    def compute(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
        settings: VpiSettings,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        settings.validate()
        if self.payload_key != (settings.max_disparity, settings.include_diagonals):
            raise RuntimeError("VPI matcher must be recreated after max/diagonal change")
        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
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
                    # VPI 3 CUDA ignores window and always uses a 9x7 census window.
                    window=5,
                    maxdisp=settings.max_disparity,
                    confthreshold=1,
                    conftype=self.vpi.ConfidenceType.ABSOLUTE,
                    mindisp=settings.min_disparity,
                    p1=settings.p1,
                    p2=settings.p2,
                    uniqueness=settings.uniqueness,
                    includediagonals=settings.include_diagonals,
                )
            with self.disparity_s16.rlock_cpu() as data:
                disparity = np.array(data, dtype=np.float32) / 32.0
            with self.confidence_u16.rlock_cpu() as data:
                confidence = np.array(data, copy=True)
        except Exception as error:
            raise RuntimeError(f"VPI CUDA stereo failed: {error}") from error
        plausible = (
            np.isfinite(disparity)
            & (disparity > settings.min_disparity)
            & (disparity < settings.max_disparity)
        )
        valid = plausible & (confidence >= settings.confidence_threshold)
        return disparity, valid, confidence


def depth_from_disparity(
    disparity: np.ndarray,
    q_matrix: np.ndarray,
    disparity_mask: Optional[np.ndarray] = None,
    min_depth_m: float = MIN_DEPTH_M,
    max_depth_m: float = MAX_DEPTH_M,
) -> tuple[np.ndarray, np.ndarray]:
    if disparity.ndim != 2 or q_matrix.shape != (4, 4):
        raise ValueError("Disparity must be 2-D and Q must be 4x4")
    denominator = q_matrix[3, 2] * disparity + q_matrix[3, 3]
    can_divide = (
        np.isfinite(disparity)
        & np.isfinite(denominator)
        & (disparity > 0.0)
        & (np.abs(denominator) > 1e-12)
    )
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    np.divide(q_matrix[2, 3], denominator, out=depth, where=can_divide)
    valid = (
        can_divide
        & np.isfinite(depth)
        & (depth > min_depth_m)
        & (depth < max_depth_m)
    )
    if disparity_mask is not None:
        if disparity_mask.shape != disparity.shape:
            raise ValueError("Disparity mask shape does not match disparity")
        valid &= disparity_mask
    depth[~valid] = np.nan
    return depth, valid


def expected_disparity(q_matrix: np.ndarray, distance_m: float) -> float:
    if distance_m <= 0:
        raise ValueError("Distance must be positive")
    denominator = q_matrix[3, 2] * distance_m
    if abs(denominator) < 1e-12:
        raise ValueError("Q matrix has zero inverse-baseline term")
    return float(q_matrix[2, 3] / denominator - q_matrix[3, 3] / q_matrix[3, 2])


def make_disparity_view(
    disparity: np.ndarray, valid: np.ndarray, minimum: float, maximum: float
) -> np.ndarray:
    scaled = np.zeros(disparity.shape, dtype=np.uint8)
    if maximum <= minimum:
        raise ValueError("Maximum disparity must exceed minimum")
    if np.any(valid):
        values = np.clip(disparity[valid], minimum, maximum)
        scaled[valid] = np.round((values - minimum) * 255.0 / (maximum - minimum)).astype(
            np.uint8
        )
    view = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    view[~valid] = 0
    return view


def make_depth_view(
    depth: np.ndarray,
    valid: np.ndarray,
    min_depth_m: float = MIN_DEPTH_M,
    max_depth_m: float = MAX_DEPTH_M,
) -> np.ndarray:
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        values = np.clip(depth[valid], min_depth_m, max_depth_m)
        scaled[valid] = np.round(
            (max_depth_m - values) * 255.0 / (max_depth_m - min_depth_m)
        ).astype(np.uint8)
    view = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    view[~valid] = 0
    return view


def make_alignment_overlay(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    left_edges = cv2.Canny(left_gray, 60, 150)
    right_edges = cv2.Canny(right_gray, 60, 150)
    overlay = np.zeros((*left_edges.shape, 3), dtype=np.uint8)
    overlay[:, :, 1] = left_edges
    overlay[:, :, 2] = right_edges
    for y in range(0, overlay.shape[0], 60):
        cv2.line(overlay, (0, y), (overlay.shape[1] - 1, y), (255, 255, 0), 1)
    return overlay


def roi_statistics(
    depth: np.ndarray,
    valid: np.ndarray,
    center: tuple[int, int],
    size: int,
    known_distance_m: Optional[float] = None,
) -> RoiStatistics:
    if depth.shape != valid.shape:
        raise ValueError("Depth and valid mask shapes differ")
    radius = max(1, size // 2)
    x, y = center
    x0, x1 = max(0, x - radius), min(depth.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(depth.shape[0], y + radius + 1)
    roi_valid = valid[y0:y1, x0:x1]
    values = depth[y0:y1, x0:x1][roi_valid]
    total = int(roi_valid.size)
    count = int(values.size)
    if count == 0:
        return RoiStatistics(count, total, 0.0, None, None, None)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    error = None if known_distance_m is None else median - known_distance_m
    return RoiStatistics(count, total, 100.0 * count / total, median, mad, error)


def settings_dict(backend: str, settings: Any) -> dict[str, Any]:
    output = asdict(settings)
    output["backend"] = backend
    if backend == "vpi-cuda":
        output["window_note"] = "VPI 3 CUDA fixed 9x7 census window; block size ignored"
    return output


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
