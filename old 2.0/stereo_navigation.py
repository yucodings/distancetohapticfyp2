#!/usr/bin/env python3
"""Live stereo depth, TensorRT YOLO, and generic obstacle fusion on Jetson.

This program intentionally stops at vision output: known/unknown object plus
distance. It contains no haptic, GPIO, serial, audio, or path-planning code.

Controls:
    Q or Esc  quit
    D         toggle disparity debug view
    U         toggle unknown-obstacle display
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# Jetson's Ubuntu OpenCV includes GStreamer. A pip OpenCV wheel often shadows
# it and normally lacks nvarguscamerasrc/GStreamer support.
SYSTEM_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
original_sys_path = sys.path.copy()
try:
    if SYSTEM_DIST_PACKAGES.is_dir():
        system_packages = str(SYSTEM_DIST_PACKAGES)
        if system_packages in sys.path:
            sys.path.remove(system_packages)
        sys.path.insert(0, system_packages)

    # Import only OpenCV and NumPy with system-package priority. Keeping the
    # directory first globally would also select Ubuntu's old SymPy 1.9, while
    # the installed PyTorch/Ultralytics runtime requires the newer user SymPy.
    import cv2
    import numpy as np
finally:
    sys.path[:] = original_sys_path


# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = SCRIPT_DIR / "calibration" / "stereo_calibration.npz"
ENGINE_PATH = SCRIPT_DIR / "model" / "best.engine"

LEFT_SENSOR_ID = 0
RIGHT_SENSOR_ID = 1

MIN_DEPTH_M = 0.20
MAX_DEPTH_M = 5.00

YOLO_CONFIDENCE = 0.40
YOLO_IOU = 0.45
YOLO_IMAGE_SIZE = 640  # The supplied TensorRT engine has a fixed 640x640 input.
DEPTH_ROI_FRACTION = 0.45
MIN_DEPTH_SAMPLES = 50

MIN_DISPARITY = 0
NUM_DISPARITIES = 128  # OpenCV requires a positive multiple of 16.
BLOCK_SIZE = 5

# Conservative first-stage generic obstacle extraction.
GEOMETRY_MAX_DEPTH_M = 3.0
GEOMETRY_DEPTH_BAND_M = 0.50
GEOMETRY_TOP_FRACTION = 0.15
GEOMETRY_BOTTOM_FRACTION = 0.90
GEOMETRY_SIDE_MARGIN_FRACTION = 0.04
MIN_OBSTACLE_AREA = 2500
MIN_OBSTACLE_WIDTH = 45
MIN_OBSTACLE_HEIGHT = 45
MAX_OBSTACLE_AREA_FRACTION = 0.25
MIN_COMPONENT_FILL_RATIO = 0.18
UNKNOWN_MATCH_IOU_THRESHOLD = 0.12
GEOMETRY_NMS_IOU_THRESHOLD = 0.35
MAX_UNKNOWN_OBSTACLES = 12

SHOW_DISPARITY = False
SHOW_DEPTH = False
SHOW_OBSTACLE_MASK = False

KNOWN_BOX_COLOR = (70, 220, 70)       # BGR green
UNKNOWN_BOX_COLOR = (0, 165, 255)     # BGR orange

EXPECTED_CLASSES = (
    "chair",
    "cleaning_cart",
    "handrail",
    "person",
    "pillars",
    "ramp",
    "squat-toilet",
    "stairs",
    "urinals",
    "vending_machine",
    "wet_floor_sign",
)


Box = tuple[int, int, int, int]


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
class KnownDetection:
    box: Box
    class_id: int
    class_name: str
    confidence: float
    distance_m: Optional[float]


@dataclass(frozen=True)
class GeometryObstacle:
    box: Box
    distance_m: float
    area: int


@dataclass(frozen=True)
class FusedDetection:
    box: Box
    label: str
    confidence: Optional[float]
    distance_m: Optional[float]
    known: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--engine", type=Path, default=ENGINE_PATH)
    parser.add_argument("--left-id", type=int, default=LEFT_SENSOR_ID)
    parser.add_argument("--right-id", type=int, default=RIGHT_SENSOR_ID)
    parser.add_argument("--confidence", type=float, default=YOLO_CONFIDENCE)
    parser.add_argument(
        "--no-unknown",
        action="store_true",
        help="start with geometry-based unknown obstacles hidden",
    )
    return parser.parse_args()


def _read_scalar(data: Any, key: str, cast: Any) -> Any:
    value = np.asarray(data[key])
    if value.size != 1:
        raise RuntimeError(f"NPZ key '{key}' must be a scalar; got {value.shape}")
    return cast(value.reshape(-1)[0])


def load_calibration(path: Path) -> Calibration:
    """Load saved maps/Q directly; never recalibrate or alter the NPZ."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration NPZ not found: {path}")

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
            keys = tuple(data.files)
            missing = [key for key in required if key not in keys]
            if missing:
                raise RuntimeError(
                    "Calibration NPZ is missing required keys: "
                    + ", ".join(missing)
                )

            width = _read_scalar(data, "image_width", int)
            height = _read_scalar(data, "image_height", int)
            calibration = Calibration(
                path=path,
                keys=keys,
                width=width,
                height=height,
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

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid calibration resolution: {width}x{height}")
    if calibration.q_matrix.shape != (4, 4):
        raise RuntimeError(
            f"Calibration Q must have shape (4, 4), got {calibration.q_matrix.shape}"
        )

    expected_map_shape = (height, width)
    for name, rectification_map in (
        ("left_map1", calibration.left_map1),
        ("left_map2", calibration.left_map2),
        ("right_map1", calibration.right_map1),
        ("right_map2", calibration.right_map2),
    ):
        if rectification_map.shape[:2] != expected_map_shape:
            raise RuntimeError(
                f"{name} has image shape {rectification_map.shape[:2]}, expected "
                f"{expected_map_shape} from image_width/image_height"
            )
    return calibration


def check_gstreamer() -> None:
    build_info = cv2.getBuildInformation()
    enabled = any(
        "GStreamer" in line and "YES" in line for line in build_info.splitlines()
    )
    if not enabled:
        raise RuntimeError(
            "GStreamer is not enabled in the loaded OpenCV build. "
            f"OpenCV {cv2.__version__} was loaded from {cv2.__file__}. "
            "Use JetPack/Ubuntu's python3-opencv build with GStreamer support."
        )


def cuda_status() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "unavailable according to PyTorch"
        return f"available ({torch.cuda.get_device_name(0)})"
    except Exception as exc:  # Diagnostic only; TensorRT performs the real check.
        return f"could not query PyTorch: {exc}"


def gstreamer_pipeline(
    sensor_id: int,
    width: int,
    height: int,
    fps: int,
    sensor_mode: int,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "nvvidconv flip-method=0 ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, "
        "format=(string)BGRx ! videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def open_cameras(
    left_id: int,
    right_id: int,
    calibration: Calibration,
) -> tuple[cv2.VideoCapture, cv2.VideoCapture]:
    if left_id == right_id:
        raise ValueError("Left and right camera sensor IDs must be different")

    def open_one(sensor_id: int) -> cv2.VideoCapture:
        pipeline = gstreamer_pipeline(
            sensor_id,
            calibration.width,
            calibration.height,
            calibration.capture_fps,
            calibration.sensor_mode,
        )
        camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError(
                f"Could not open IMX219 sensor-id={sensor_id} with GStreamer. "
                f"Pipeline: {pipeline}"
            )
        return camera

    left_camera: Optional[cv2.VideoCapture] = None
    try:
        left_camera = open_one(left_id)
        right_camera = open_one(right_id)
        return left_camera, right_camera
    except Exception:
        if left_camera is not None:
            left_camera.release()
        raise


def capture_stereo_pair(
    left_camera: cv2.VideoCapture,
    right_camera: cv2.VideoCapture,
) -> tuple[np.ndarray, np.ndarray]:
    """Grab both first to reduce skew; the cameras are not hardware-triggered."""
    grabbed_left = left_camera.grab()
    grabbed_right = right_camera.grab()
    ok_left, left_frame = left_camera.retrieve()
    ok_right, right_frame = right_camera.retrieve()
    if not (grabbed_left and grabbed_right and ok_left and ok_right):
        raise RuntimeError("Failed to capture a complete left/right stereo pair")
    if left_frame is None or right_frame is None:
        raise RuntimeError("A camera returned an empty frame")
    return left_frame, right_frame


def validate_frame_resolution(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    calibration: Calibration,
) -> None:
    expected = (calibration.height, calibration.width)
    left_shape = left_frame.shape[:2]
    right_shape = right_frame.shape[:2]
    if left_shape != expected or right_shape != expected:
        raise RuntimeError(
            "Runtime frames do not match the calibration resolution. "
            f"Calibration expects {calibration.width}x{calibration.height}; "
            f"left is {left_shape[1]}x{left_shape[0]} and right is "
            f"{right_shape[1]}x{right_shape[0]}. Processing stopped; the "
            "stereo images must not be silently resized before rectification."
        )


def rectify_frames(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    calibration: Calibration,
) -> tuple[np.ndarray, np.ndarray]:
    left_rectified = cv2.remap(
        left_frame,
        calibration.left_map1,
        calibration.left_map2,
        cv2.INTER_LINEAR,
    )
    right_rectified = cv2.remap(
        right_frame,
        calibration.right_map1,
        calibration.right_map2,
        cv2.INTER_LINEAR,
    )
    return left_rectified, right_rectified


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
    return raw_disparity.astype(np.float32) / 16.0


def calculate_depth(
    disparity: np.ndarray,
    q_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject disparity with Q and return depth plus a strict valid mask."""
    points_3d = cv2.reprojectImageTo3D(disparity, q_matrix)
    # Copy Z so the much larger HxWx3 array can be released immediately.
    depth_map = points_3d[:, :, 2].copy()
    valid = (
        (disparity > 0.0)
        & np.isfinite(depth_map)
        & (depth_map >= MIN_DEPTH_M)
        & (depth_map <= MAX_DEPTH_M)
    )
    depth_map[~valid] = np.nan
    return depth_map, valid


def _normalise_names(names: Any) -> dict[int, str]:
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return {index: str(value) for index, value in enumerate(names)}
    raise RuntimeError(f"Unsupported YOLO class-name format: {type(names).__name__}")


def load_yolo(engine_path: Path) -> Any:
    """Load and warm up the supplied TensorRT engine through Ultralytics."""
    engine_path = engine_path.expanduser().resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(f"YOLO TensorRT engine not found: {engine_path}")
    if engine_path.suffix.lower() != ".engine":
        raise ValueError(f"Expected a TensorRT .engine file, got: {engine_path}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Could not import Ultralytics YOLO. Install/fix the Jetson "
            f"Ultralytics runtime and its dependencies. Original error: {exc}"
        ) from exc

    try:
        model = YOLO(str(engine_path), task="detect")
        # TensorRT deserialisation is lazy, so one warm-up inference catches an
        # incompatible engine/CUDA/TensorRT installation before cameras open.
        warmup_image = np.zeros((YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE, 3), np.uint8)
        warmup_results = model.predict(
            source=warmup_image,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load or execute TensorRT engine '{engine_path}': {exc}"
        ) from exc

    if not warmup_results:
        raise RuntimeError("YOLO warm-up returned no result object")
    names = _normalise_names(warmup_results[0].names)
    ordered_names = tuple(names[index] for index in sorted(names))
    if ordered_names != EXPECTED_CLASSES:
        raise RuntimeError(
            "YOLO engine classes do not match the required 11 classes. "
            f"Expected {EXPECTED_CLASSES}, got {ordered_names}"
        )
    return model


def get_detection_distance(
    depth_map: np.ndarray,
    valid_depth_mask: np.ndarray,
    box: Box,
) -> Optional[float]:
    """Median depth from the central 45% of a detection, never one pixel."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if width <= 1 or height <= 1:
        return None

    roi_width = max(1, int(round(width * DEPTH_ROI_FRACTION)))
    roi_height = max(1, int(round(height * DEPTH_ROI_FRACTION)))
    centre_x = (x1 + x2) // 2
    centre_y = (y1 + y2) // 2
    rx1 = max(x1, centre_x - roi_width // 2)
    ry1 = max(y1, centre_y - roi_height // 2)
    rx2 = min(x2, rx1 + roi_width)
    ry2 = min(y2, ry1 + roi_height)

    roi_mask = valid_depth_mask[ry1:ry2, rx1:rx2]
    values = depth_map[ry1:ry2, rx1:rx2][roi_mask]
    if values.size < MIN_DEPTH_SAMPLES:
        return None
    return float(np.median(values))


def run_yolo(
    model: Any,
    rectified_reference: np.ndarray,
    depth_map: np.ndarray,
    valid_depth_mask: np.ndarray,
    confidence: float,
) -> list[KnownDetection]:
    try:
        results = model.predict(
            source=rectified_reference,
            imgsz=YOLO_IMAGE_SIZE,
            conf=confidence,
            iou=YOLO_IOU,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(f"YOLO TensorRT inference failed: {exc}") from exc

    if not results:
        return []
    result = results[0]
    names = _normalise_names(result.names)
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    height, width = rectified_reference.shape[:2]
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)

    detections: list[KnownDetection] = []
    for coordinates, score, class_id in zip(xyxy, confidences, class_ids):
        x1 = int(np.clip(round(float(coordinates[0])), 0, width - 1))
        y1 = int(np.clip(round(float(coordinates[1])), 0, height - 1))
        x2 = int(np.clip(round(float(coordinates[2])), x1 + 1, width))
        y2 = int(np.clip(round(float(coordinates[3])), y1 + 1, height))
        box = (x1, y1, x2, y2)
        detections.append(
            KnownDetection(
                box=box,
                class_id=int(class_id),
                class_name=names.get(int(class_id), f"class_{class_id}"),
                confidence=float(score),
                distance_m=get_detection_distance(
                    depth_map, valid_depth_mask, box
                ),
            )
        )
    return detections


def calculate_iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def _geometry_nms(obstacles: list[GeometryObstacle]) -> list[GeometryObstacle]:
    """Suppress overlaps and cap output so noisy scenes cannot flood the UI."""
    ordered = sorted(obstacles, key=lambda obstacle: obstacle.area, reverse=True)
    kept: list[GeometryObstacle] = []
    for obstacle in ordered:
        if all(
            calculate_iou(obstacle.box, other.box) < GEOMETRY_NMS_IOU_THRESHOLD
            for other in kept
        ):
            kept.append(obstacle)
    return kept[:MAX_UNKNOWN_OBSTACLES]


def detect_geometry_obstacles(
    depth_map: np.ndarray,
    valid_depth_mask: np.ndarray,
) -> tuple[list[GeometryObstacle], np.ndarray]:
    """Extract conservative coherent near-depth regions.

    The depth is processed in narrow bands so a nearby object is less likely
    to merge into a distant wall. The top, extreme sides, and lowest part of
    the image are excluded, large background-like regions are rejected, and
    morphology removes isolated disparity noise.

    Limitation: this is not full free-space understanding or robust floor-plane
    removal. The function is deliberately isolated so a calibrated ground-plane
    model/RANSAC stage can replace or precede it later.
    """
    height, width = depth_map.shape
    x_start = int(width * GEOMETRY_SIDE_MARGIN_FRACTION)
    x_end = int(width * (1.0 - GEOMETRY_SIDE_MARGIN_FRACTION))
    y_start = int(height * GEOMETRY_TOP_FRACTION)
    y_end = int(height * GEOMETRY_BOTTOM_FRACTION)
    roi_gate = np.zeros((height, width), dtype=bool)
    roi_gate[y_start:y_end, x_start:x_end] = True

    geometry_limit = min(GEOMETRY_MAX_DEPTH_M, MAX_DEPTH_M)
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    obstacles: list[GeometryObstacle] = []
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    image_area = height * width

    band_start = MIN_DEPTH_M
    while band_start < geometry_limit:
        band_end = min(band_start + GEOMETRY_DEPTH_BAND_M, geometry_limit)
        band_mask = (
            valid_depth_mask
            & roi_gate
            & (depth_map >= band_start)
            & (depth_map < band_end)
        ).astype(np.uint8) * 255
        band_start = band_end

        band_mask = cv2.morphologyEx(band_mask, cv2.MORPH_OPEN, open_kernel)
        band_mask = cv2.morphologyEx(band_mask, cv2.MORPH_CLOSE, close_kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            band_mask, connectivity=8
        )

        for label_id in range(1, count):
            x, y, component_width, component_height, area = (
                int(value) for value in stats[label_id]
            )
            if (
                area < MIN_OBSTACLE_AREA
                or component_width < MIN_OBSTACLE_WIDTH
                or component_height < MIN_OBSTACLE_HEIGHT
                or area > image_area * MAX_OBSTACLE_AREA_FRACTION
            ):
                continue

            box_area = component_width * component_height
            if area / max(1, box_area) < MIN_COMPONENT_FILL_RATIO:
                continue

            # Reject wide regions at the bottom of the geometry ROI, which are
            # commonly floor bands. This is intentionally conservative.
            touches_roi_bottom = y + component_height >= y_end - 3
            if touches_roi_bottom and component_width > width * 0.35:
                continue
            if component_width > component_height * 4.0:
                continue

            component_slice = labels[y : y + component_height, x : x + component_width]
            depth_slice = depth_map[y : y + component_height, x : x + component_width]
            component_depths = depth_slice[component_slice == label_id]
            component_depths = component_depths[np.isfinite(component_depths)]
            if component_depths.size < MIN_DEPTH_SAMPLES:
                continue

            box = (x, y, x + component_width, y + component_height)
            obstacles.append(
                GeometryObstacle(
                    box=box,
                    distance_m=float(np.median(component_depths)),
                    area=area,
                )
            )
            combined_region = combined_mask[
                y : y + component_height, x : x + component_width
            ]
            combined_region[component_slice == label_id] = 255

    return _geometry_nms(obstacles), combined_mask


def fuse_detections(
    known_detections: list[KnownDetection],
    geometry_obstacles: list[GeometryObstacle],
    show_unknown: bool,
) -> list[FusedDetection]:
    """Keep reliable YOLO results and add only unmatched physical regions."""
    fused = [
        FusedDetection(
            box=detection.box,
            label=detection.class_name,
            confidence=detection.confidence,
            distance_m=detection.distance_m,
            known=True,
        )
        for detection in known_detections
    ]
    if not show_unknown:
        return fused

    known_boxes = [detection.box for detection in known_detections]
    for obstacle in geometry_obstacles:
        maximum_iou = max(
            (calculate_iou(obstacle.box, known_box) for known_box in known_boxes),
            default=0.0,
        )
        if maximum_iou >= UNKNOWN_MATCH_IOU_THRESHOLD:
            continue
        fused.append(
            FusedDetection(
                box=obstacle.box,
                label="UNKNOWN OBSTACLE",
                confidence=None,
                distance_m=obstacle.distance_m,
                known=False,
            )
        )
    return fused


def _draw_label(image: np.ndarray, box: Box, text: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    font_scale = 0.62
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    label_y = max(text_height + baseline + 5, y1)
    cv2.rectangle(
        image,
        (x1, label_y - text_height - baseline - 5),
        (min(image.shape[1] - 1, x1 + text_width + 8), label_y + 3),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x1 + 4, label_y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (10, 10, 10),
        thickness,
        cv2.LINE_AA,
    )


def draw_results(
    rectified_reference: np.ndarray,
    detections: list[FusedDetection],
    fps: float,
    known_count: int,
    unknown_count: int,
    show_unknown: bool,
) -> np.ndarray:
    output = rectified_reference.copy()
    for detection in detections:
        color = KNOWN_BOX_COLOR if detection.known else UNKNOWN_BOX_COLOR
        if detection.known:
            prefix = f"{detection.label} {detection.confidence:.2f}"
        else:
            prefix = detection.label
        distance = (
            f"{detection.distance_m:.2f} m"
            if detection.distance_m is not None
            else "Depth N/A"
        )
        _draw_label(output, detection.box, f"{prefix} | {distance}", color)

    status_lines = (
        f"FPS {fps:.1f}",
        "Stereo depth: ACTIVE",
        "YOLO TensorRT: ACTIVE",
        f"Known: {known_count}  Unknown: {unknown_count}",
        f"Unknown display: {'ON' if show_unknown else 'OFF'}",
    )
    panel_width = 330
    panel_height = 25 * len(status_lines) + 14
    panel = output[8 : 8 + panel_height, 8 : 8 + panel_width]
    dark = np.zeros_like(panel)
    cv2.addWeighted(dark, 0.62, panel, 0.38, 0, dst=panel)
    for index, line in enumerate(status_lines):
        cv2.putText(
            output,
            line,
            (18, 32 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def make_disparity_view(disparity: np.ndarray) -> np.ndarray:
    valid = disparity > 0.0
    view = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(disparity[valid], (2.0, 98.0))
        if high > low:
            scaled = np.clip((disparity - low) * 255.0 / (high - low), 0, 255)
            view[valid] = scaled[valid].astype(np.uint8)
    return cv2.applyColorMap(view, cv2.COLORMAP_TURBO)


def make_depth_view(depth_map: np.ndarray, valid_depth_mask: np.ndarray) -> np.ndarray:
    view = np.zeros(depth_map.shape, dtype=np.uint8)
    if np.any(valid_depth_mask):
        clipped = np.clip(depth_map[valid_depth_mask], MIN_DEPTH_M, MAX_DEPTH_M)
        view[valid_depth_mask] = (
            (MAX_DEPTH_M - clipped) * 255.0 / (MAX_DEPTH_M - MIN_DEPTH_M)
        ).astype(np.uint8)
    color = cv2.applyColorMap(view, cv2.COLORMAP_TURBO)
    color[~valid_depth_mask] = 0
    return color


def print_startup(
    calibration: Calibration,
    engine_path: Path,
    left_id: int,
    right_id: int,
) -> None:
    print("\nStereo navigation startup")
    print(f"  OpenCV version:       {cv2.__version__}")
    print(f"  OpenCV path:          {cv2.__file__}")
    print("  GStreamer enabled:    YES")
    print(f"  CUDA:                 {cuda_status()}")
    print(f"  Calibration NPZ:      {calibration.path}")
    print(f"  NPZ keys:             {', '.join(calibration.keys)}")
    print(
        f"  Calibration image:    {calibration.width}x{calibration.height}"
    )
    print(f"  Calibration mode:     {calibration.sensor_mode}")
    print(f"  Calibration FPS:      {calibration.capture_fps}")
    print(f"  Calibration baseline: {calibration.baseline_m:.6f} m")
    print(f"  YOLO engine:          {engine_path.expanduser().resolve()}")
    print(f"  YOLO classes:         {', '.join(EXPECTED_CLASSES)}")
    print(
        f"  Camera request:       {calibration.width}x{calibration.height}, "
        f"left sensor {left_id}, right sensor {right_id}"
    )
    print("  Controls:             Q/Esc quit | D disparity | U unknowns\n")


def run(args: argparse.Namespace) -> None:
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence must be greater than 0 and at most 1")
    if MIN_DEPTH_M <= 0.0 or MAX_DEPTH_M <= MIN_DEPTH_M:
        raise ValueError("Configure a valid MIN_DEPTH_M/MAX_DEPTH_M range")

    check_gstreamer()
    calibration = load_calibration(args.calibration)
    engine_path = args.engine.expanduser().resolve()
    print_startup(calibration, engine_path, args.left_id, args.right_id)
    print("Loading and warming up the TensorRT engine...", flush=True)
    model = load_yolo(engine_path)
    print("TensorRT YOLO ready; class list validated.", flush=True)

    left_camera: Optional[cv2.VideoCapture] = None
    right_camera: Optional[cv2.VideoCapture] = None
    try:
        print("Opening stereo IMX219 cameras...", flush=True)
        left_camera, right_camera = open_cameras(
            args.left_id, args.right_id, calibration
        )
        matcher = create_stereo_matcher()

        # Let exposure settle, then validate actual frames before processing.
        left_frame: Optional[np.ndarray] = None
        right_frame: Optional[np.ndarray] = None
        for _ in range(6):
            left_frame, right_frame = capture_stereo_pair(left_camera, right_camera)
        assert left_frame is not None and right_frame is not None
        validate_frame_resolution(left_frame, right_frame, calibration)
        print(
            f"Camera resolution validated: {calibration.width}x{calibration.height}",
            flush=True,
        )

        window_name = "Stereo Navigation - Known and Unknown Obstacles"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 960)

        show_disparity = SHOW_DISPARITY
        show_unknown = not args.no_unknown
        fps = 0.0
        previous_time = time.perf_counter()

        while True:
            loop_start = time.perf_counter()
            left_frame, right_frame = capture_stereo_pair(left_camera, right_camera)
            validate_frame_resolution(left_frame, right_frame, calibration)
            left_rectified, right_rectified = rectify_frames(
                left_frame, right_frame, calibration
            )
            disparity = calculate_disparity(
                matcher, left_rectified, right_rectified
            )
            depth_map, valid_depth_mask = calculate_depth(
                disparity, calibration.q_matrix
            )

            known = run_yolo(
                model,
                left_rectified,
                depth_map,
                valid_depth_mask,
                args.confidence,
            )
            geometry, obstacle_mask = detect_geometry_obstacles(
                depth_map, valid_depth_mask
            )
            fused = fuse_detections(known, geometry, show_unknown)
            unknown_count = sum(not detection.known for detection in fused)

            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - previous_time, 1e-6)
            fps = instantaneous_fps if fps == 0.0 else 0.9 * fps + 0.1 * instantaneous_fps
            previous_time = now

            annotated = draw_results(
                left_rectified,
                fused,
                fps,
                len(known),
                unknown_count,
                show_unknown,
            )
            cv2.imshow(window_name, annotated)

            if show_disparity:
                cv2.imshow("Disparity Debug", make_disparity_view(disparity))
            if SHOW_DEPTH:
                cv2.imshow("Depth Debug", make_depth_view(depth_map, valid_depth_mask))
            if SHOW_OBSTACLE_MASK:
                cv2.imshow("Geometry Obstacle Mask", obstacle_mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("d"), ord("D")):
                show_disparity = not show_disparity
                if not show_disparity:
                    cv2.destroyWindow("Disparity Debug")
            if key in (ord("u"), ord("U")):
                show_unknown = not show_unknown

            # Ensure large per-frame arrays do not survive into the next RHS.
            del depth_map, valid_depth_mask, disparity
            _ = loop_start  # Retained as a convenient future profiling hook.
    finally:
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
    except Exception as exc:
        print(f"UNEXPECTED ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
