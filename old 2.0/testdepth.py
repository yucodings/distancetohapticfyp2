#!/usr/bin/env python3
"""Simple live depth sensing with a calibrated IMX219 stereo camera.

The script loads ``stereo_calibration.npz`` from this directory, rectifies
both camera images, calculates disparity with StereoSGBM, and reports the
median distance in the box at the centre of the image.

Controls:
    q or Esc  Quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Prefer Jetson's OpenCV build because it normally includes GStreamer.
SYSTEM_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
if SYSTEM_DIST_PACKAGES.is_dir():
    system_packages = str(SYSTEM_DIST_PACKAGES)
    if system_packages in sys.path:
        sys.path.remove(system_packages)
    sys.path.insert(0, system_packages)

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = SCRIPT_DIR / "stereo_calibration.npz"

LEFT_SENSOR_ID = 0
RIGHT_SENSOR_ID = 1
MIN_DEPTH_M = 0.20
MAX_DEPTH_M = 4.0
NUM_DISPARITIES = 192  # Must be a multiple of 16.
BLOCK_SIZE = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--left-id", type=int, default=LEFT_SENSOR_ID)
    parser.add_argument("--right-id", type=int, default=RIGHT_SENSOR_ID)
    parser.add_argument("--max-depth", type=float, default=MAX_DEPTH_M)
    return parser.parse_args()


def scalar(data: np.lib.npyio.NpzFile, key: str, default: int) -> int:
    """Read an optional scalar calibration value."""
    return int(data[key]) if key in data.files else default


def load_calibration(path: Path) -> dict[str, np.ndarray | int]:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    required = ("left_map1", "left_map2", "right_map1", "right_map2", "Q")
    with np.load(path) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise RuntimeError(
                f"Calibration file is missing: {', '.join(missing)}"
            )

        calibration: dict[str, np.ndarray | int] = {
            key: data[key].copy() for key in required
        }
        calibration["width"] = scalar(data, "image_width", data["left_map1"].shape[1])
        calibration["height"] = scalar(data, "image_height", data["left_map1"].shape[0])
        calibration["fps"] = scalar(data, "capture_fps", 30)
        calibration["sensor_mode"] = scalar(data, "sensor_mode", 3)

    return calibration


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


def open_camera(
    sensor_id: int,
    width: int,
    height: int,
    fps: int,
    sensor_mode: int,
) -> cv2.VideoCapture:
    pipeline = gstreamer_pipeline(sensor_id, width, height, fps, sensor_mode)
    camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera sensor-id={sensor_id}")
    return camera


def create_matcher() -> cv2.StereoSGBM:
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * BLOCK_SIZE**2,
        P2=32 * BLOCK_SIZE**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def calculate_depth(disparity: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """Convert disparity pixels to Z distance in metres using calibration Q."""
    points_3d = cv2.reprojectImageTo3D(disparity, q_matrix)
    depth_m = points_3d[:, :, 2]
    depth_m[disparity <= 0.0] = np.nan
    return depth_m


def centre_distance(
    depth_m: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> tuple[float | None, tuple[int, int, int, int]]:
    """Return a stable median depth from the middle 20% of the image."""
    height, width = depth_m.shape
    half_width = max(1, width // 10)
    half_height = max(1, height // 10)
    centre_x, centre_y = width // 2, height // 2
    box = (
        centre_x - half_width,
        centre_y - half_height,
        centre_x + half_width,
        centre_y + half_height,
    )

    x1, y1, x2, y2 = box
    region = depth_m[y1:y2, x1:x2]
    valid = region[
        np.isfinite(region) & (region >= min_depth) & (region <= max_depth)
    ]
    if valid.size == 0:
        return None, box
    return float(np.median(valid)), box


def make_depth_view(
    depth_m: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """Create a colour image where red is near and blue is far."""
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= min_depth)
        & (depth_m <= max_depth)
    )
    grey = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth_m[valid], min_depth, max_depth)
        grey[valid] = np.round(
            (max_depth - clipped) * 255.0 / (max_depth - min_depth)
        ).astype(np.uint8)

    colour = cv2.applyColorMap(grey, cv2.COLORMAP_JET)
    colour[~valid] = 0
    return colour


def add_distance_label(
    image: np.ndarray,
    distance_m: float | None,
    box: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    colour = (0, 255, 0) if distance_m is not None else (0, 0, 255)
    text = (
        f"Centre depth: {distance_m:.2f} m"
        if distance_m is not None
        else "Centre depth: no valid match"
    )
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)
    cv2.rectangle(image, (15, 15), (500, 62), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (25, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        colour,
        2,
        cv2.LINE_AA,
    )


def check_gstreamer() -> None:
    enabled = any(
        "GStreamer" in line and "YES" in line
        for line in cv2.getBuildInformation().splitlines()
    )
    if not enabled:
        raise RuntimeError(
            "OpenCV was built without GStreamer support. On Jetson, use the "
            "Ubuntu python3-opencv package instead of the pip OpenCV wheel."
        )


def run(args: argparse.Namespace) -> None:
    if args.left_id == args.right_id:
        raise ValueError("Left and right sensor IDs must be different")
    if args.max_depth <= MIN_DEPTH_M:
        raise ValueError(f"--max-depth must be greater than {MIN_DEPTH_M}")

    check_gstreamer()
    calibration = load_calibration(args.calibration.resolve())
    width = int(calibration["width"])
    height = int(calibration["height"])
    fps = int(calibration["fps"])
    sensor_mode = int(calibration["sensor_mode"])

    print(f"Loaded calibration: {args.calibration}")
    print(f"Capture: {width}x{height}, sensor mode {sensor_mode}, {fps} FPS")
    print("Press Q or Esc to quit")

    left_camera: cv2.VideoCapture | None = None
    right_camera: cv2.VideoCapture | None = None
    try:
        left_camera = open_camera(args.left_id, width, height, fps, sensor_mode)
        right_camera = open_camera(args.right_id, width, height, fps, sensor_mode)
        matcher = create_matcher()

        window_name = "Simple Stereo Depth - Rectified Left | Depth"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 480)

        while True:
            # Grab both frames before retrieving either to reduce capture skew.
            grabbed_left = left_camera.grab()
            grabbed_right = right_camera.grab()
            ok_left, frame_left = left_camera.retrieve()
            ok_right, frame_right = right_camera.retrieve()
            if not (grabbed_left and grabbed_right and ok_left and ok_right):
                print("Warning: could not read both camera frames", file=sys.stderr)
                continue

            actual_size = (frame_left.shape[1], frame_left.shape[0])
            if actual_size != (width, height) or frame_right.shape[:2] != (height, width):
                raise RuntimeError(
                    f"Camera frames must match calibration size {width}x{height}; "
                    f"received left {actual_size[0]}x{actual_size[1]} and right "
                    f"{frame_right.shape[1]}x{frame_right.shape[0]}"
                )

            left_rect = cv2.remap(
                frame_left,
                calibration["left_map1"],
                calibration["left_map2"],
                cv2.INTER_LINEAR,
            )
            right_rect = cv2.remap(
                frame_right,
                calibration["right_map1"],
                calibration["right_map2"],
                cv2.INTER_LINEAR,
            )

            left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
            disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
            depth_m = calculate_depth(disparity, calibration["Q"])
            distance_m, box = centre_distance(depth_m, MIN_DEPTH_M, args.max_depth)

            depth_view = make_depth_view(depth_m, MIN_DEPTH_M, args.max_depth)
            add_distance_label(left_rect, distance_m, box)
            add_distance_label(depth_view, distance_m, box)

            preview_width, preview_height = 640, 480
            left_preview = cv2.resize(left_rect, (preview_width, preview_height))
            depth_preview = cv2.resize(depth_view, (preview_width, preview_height))
            cv2.imshow(window_name, cv2.hconcat((left_preview, depth_preview)))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
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
    except (FileNotFoundError, RuntimeError, ValueError, KeyboardInterrupt) as exc:
        if not isinstance(exc, KeyboardInterrupt):
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
