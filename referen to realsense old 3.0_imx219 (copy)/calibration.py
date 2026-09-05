"""Validated loading and conversion of the IMX219 stereo calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from camera_backend import cv2, np
from config import FPS, HEIGHT, SENSOR_MODE, WIDTH


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
    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray


def _read_scalar(data: Any, key: str, value_type: Any) -> Any:
    value = np.asarray(data[key])
    if value.size != 1:
        raise RuntimeError(
            f"Calibration key '{key}' must be scalar, got shape {value.shape}"
        )
    return value_type(value.reshape(-1)[0])


def load_calibration(path: Path) -> Calibration:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stereo calibration not found: {path}")

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
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in required if key not in data.files]
            if missing:
                raise RuntimeError(
                    "Calibration is missing required keys: " + ", ".join(missing)
                )
            calibration = Calibration(
                path=path,
                keys=tuple(data.files),
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
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read calibration '{path}': {error}") from error

    actual = (
        calibration.width,
        calibration.height,
        calibration.sensor_mode,
        calibration.capture_fps,
    )
    expected = (WIDTH, HEIGHT, SENSOR_MODE, FPS)
    if actual != expected:
        raise RuntimeError(
            "Calibration/capture mismatch: expected "
            f"{WIDTH}x{HEIGHT}, sensor mode {SENSOR_MODE}, {FPS} FPS; got "
            f"{calibration.width}x{calibration.height}, sensor mode "
            f"{calibration.sensor_mode}, {calibration.capture_fps} FPS"
        )
    if calibration.q_matrix.shape != (4, 4):
        raise RuntimeError(
            f"Calibration Q must be 4x4, got {calibration.q_matrix.shape}"
        )
    if not (
        np.allclose(calibration.q_matrix[2, :3], 0.0)
        and np.allclose(calibration.q_matrix[3, :2], 0.0)
    ):
        raise RuntimeError("Calibration Q is not canonical rectified-stereo form")
    if not 0.01 < calibration.baseline_m < 0.5:
        raise RuntimeError(
            f"Implausible stereo baseline: {calibration.baseline_m:.6f} m"
        )

    expected_shape = (HEIGHT, WIDTH)
    for name, rectification_map in (
        ("left_map1", calibration.left_map1),
        ("left_map2", calibration.left_map2),
        ("right_map1", calibration.right_map1),
        ("right_map2", calibration.right_map2),
    ):
        if rectification_map.shape[:2] != expected_shape:
            raise RuntimeError(
                f"{name} shape {rectification_map.shape[:2]} != {expected_shape}"
            )
    return calibration


def convert_rectification_maps(calibration: Calibration) -> RectificationMaps:
    left_map1, left_map2 = cv2.convertMaps(
        calibration.left_map1, calibration.left_map2, cv2.CV_16SC2
    )
    right_map1, right_map2 = cv2.convertMaps(
        calibration.right_map1, calibration.right_map2, cv2.CV_16SC2
    )
    return RectificationMaps(left_map1, left_map2, right_map1, right_map2)


def rectify_left(frame: np.ndarray, maps: RectificationMaps) -> np.ndarray:
    return cv2.remap(frame, maps.left_map1, maps.left_map2, cv2.INTER_LINEAR)


def rectify_right(frame: np.ndarray, maps: RectificationMaps) -> np.ndarray:
    return cv2.remap(frame, maps.right_map1, maps.right_map2, cv2.INTER_LINEAR)
