#!/usr/bin/env python3
"""Rectified IMX219 stereo preview and five-region nearest-depth sensing.

The program captures both cameras at the calibration resolution, continuously
calculates stereo depth, and displays the rectified LEFT image. Each of five
regions reports its nearest reliable distance up to 3 metres.

Controls:
    R       toggle raw/rectified display (depth always uses rectified frames)
    F       toggle fullscreen
    Q/Esc   quit
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# On Jetson, Ubuntu's OpenCV package normally provides GStreamer support while
# pip's opencv-python wheel normally does not. Prefer the system package.
SYSTEM_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
if SYSTEM_DIST_PACKAGES.is_dir():
    package_path = str(SYSTEM_DIST_PACKAGES)
    if package_path in sys.path:
        sys.path.remove(package_path)
    sys.path.insert(0, package_path)

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_DIR = (
    SCRIPT_DIR.parent
    / "testing"
    / "Jetson-Stereo-CSI-Calibration"
    / "1786466847.603000"
)

CAPTURE_WIDTH = 1640
CAPTURE_HEIGHT = 1232
CAPTURE_FPS = 30
SENSOR_MODE = 3

MIN_DEPTH_M = 0.40
MAX_DEPTH_M = 3.00
DEFAULT_DEPTH_SCALE = 0.50
DEFAULT_NUM_DISPARITIES = 128  # Must be divisible by 16.
NEAREST_PERCENTILE = 1.0


def gstreamer_pipeline(
    sensor_id: int,
    width: int,
    height: int,
    framerate: int,
    sensor_mode: int,
    flip_method: int,
) -> str:
    """Return an nvarguscamerasrc pipeline that outputs full-resolution BGR."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, "
        f"format=(string)BGRx ! videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def read_xml_matrix(path: Path, node_name: str) -> np.ndarray:
    """Read one OpenCV matrix from an XML calibration file."""
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"Could not open calibration file: {path}")

    matrix = storage.getNode(node_name).mat()
    storage.release()

    if matrix is None or matrix.size == 0:
        raise RuntimeError(f"Node '{node_name}' is missing from {path}")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"Node '{node_name}' contains invalid numbers in {path}")
    return matrix


@dataclass
class StereoCalibration:
    left_camera_matrix: np.ndarray
    left_distortion: np.ndarray
    right_camera_matrix: np.ndarray
    right_distortion: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    q: np.ndarray
    translation: np.ndarray
    focal_px: float
    baseline_m: float

    @classmethod
    def load(cls, calibration_dir: Path) -> "StereoCalibration":
        left_path = calibration_dir / "left.xml"
        right_path = calibration_dir / "right.xml"
        stereo_path = calibration_dir / "stereo.xml"

        missing = [p.name for p in (left_path, right_path, stereo_path) if not p.is_file()]
        if missing:
            raise RuntimeError(
                f"Missing calibration file(s) in {calibration_dir}: {', '.join(missing)}"
            )

        k1 = read_xml_matrix(left_path, "CameraMatrix")
        d1 = read_xml_matrix(left_path, "Distortion")
        k2 = read_xml_matrix(right_path, "CameraMatrix")
        d2 = read_xml_matrix(right_path, "Distortion")

        r1 = read_xml_matrix(stereo_path, "R1")
        r2 = read_xml_matrix(stereo_path, "R2")
        p1 = read_xml_matrix(stereo_path, "P1")
        p2 = read_xml_matrix(stereo_path, "P2")
        q = read_xml_matrix(stereo_path, "Q")
        translation = read_xml_matrix(stereo_path, "T")

        expected_shapes = {
            "left CameraMatrix": (k1, (3, 3)),
            "right CameraMatrix": (k2, (3, 3)),
            "R1": (r1, (3, 3)),
            "R2": (r2, (3, 3)),
            "P1": (p1, (3, 4)),
            "P2": (p2, (3, 4)),
            "Q": (q, (4, 4)),
            "T": (translation, (3, 1)),
        }
        for name, (matrix, shape) in expected_shapes.items():
            if matrix.shape != shape:
                raise RuntimeError(
                    f"Invalid {name} shape {matrix.shape}; expected {shape}"
                )

        focal_px = float(p1[0, 0])
        if focal_px <= 0.0 or abs(float(p2[0, 0])) < 1e-12:
            raise RuntimeError("Invalid focal length in stereo.xml")

        # This is the baseline encoded by stereoRectify and is consistent with
        # P1/P2/Q. Because calibration square size was in metres, it is metres.
        baseline_m = abs(float(p2[0, 3] / p2[0, 0]))
        if not 0.01 <= baseline_m <= 1.0:
            raise RuntimeError(f"Implausible stereo baseline: {baseline_m:.6f} m")

        return cls(
            left_camera_matrix=k1,
            left_distortion=d1,
            right_camera_matrix=k2,
            right_distortion=d2,
            r1=r1,
            r2=r2,
            p1=p1,
            p2=p2,
            q=q,
            translation=translation,
            focal_px=focal_px,
            baseline_m=baseline_m,
        )


class CameraStream:
    """Continuously retain only the newest frame from one CSI camera."""

    def __init__(self, sensor_id: int, pipeline: str) -> None:
        self.sensor_id = sensor_id
        self.pipeline = pipeline
        self.capture: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.frame_time = 0.0
        self.sequence = 0
        self.error: Optional[str] = None
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.capture = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open CSI camera sensor-id={self.sensor_id}.\n"
                f"Pipeline:\n{self.pipeline}"
            )

        self.running = True
        self.thread = threading.Thread(
            target=self._read_frames,
            name=f"camera-{self.sensor_id}",
            daemon=True,
        )
        self.thread.start()

    def _read_frames(self) -> None:
        assert self.capture is not None
        while self.running:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                with self.lock:
                    self.error = f"Failed to read sensor-id={self.sensor_id}"
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = frame
                self.frame_time = time.monotonic()
                self.sequence += 1
                self.error = None

    def latest(self) -> tuple[Optional[np.ndarray], float, int, Optional[str]]:
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return frame, self.frame_time, self.sequence, self.error

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.capture is not None:
            self.capture.release()


@dataclass
class RegionDepth:
    name: str
    rect: tuple[int, int, int, int]
    depth_m: Optional[float]
    valid_fraction: float
    nearest_point: Optional[tuple[int, int]]


class StereoDepthProcessor:
    """Compute stereo depth and the nearest reliable depth in five regions."""

    def __init__(
        self,
        calibration: StereoCalibration,
        scale: float,
        num_disparities: int,
        block_size: int,
    ) -> None:
        self.calibration = calibration
        self.scale = scale
        self.process_width = max(1, int(round(CAPTURE_WIDTH * scale)))
        self.process_height = max(1, int(round(CAPTURE_HEIGHT * scale)))
        self.process_size = (self.process_width, self.process_height)
        self.scaled_focal_px = calibration.focal_px * scale

        # Q belongs to the full-resolution rectified images. The depth images
        # use a uniform scale, so scale Q's pixel-coordinate/focal terms while
        # retaining its metric baseline term. This preserves XYZ output in m.
        self.scaled_q = calibration.q.copy()
        self.scaled_q[0, 3] *= scale
        self.scaled_q[1, 3] *= scale
        self.scaled_q[2, 3] *= scale
        self.scaled_q[3, 3] *= scale

        scaled_p1 = calibration.p1.copy()
        scaled_p2 = calibration.p2.copy()
        scaled_p1[0:2, :] *= scale
        scaled_p2[0:2, :] *= scale

        # Rectify directly from the full-resolution camera frames into the
        # smaller processing image. This avoids expensive full-size remaps while
        # retaining the calibration's field of view and metric geometry.
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            calibration.left_camera_matrix,
            calibration.left_distortion,
            calibration.r1,
            scaled_p1[:, :3],
            self.process_size,
            cv2.CV_16SC2,
        )
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            calibration.right_camera_matrix,
            calibration.right_distortion,
            calibration.r2,
            scaled_p2[:, :3],
            self.process_size,
            cv2.CV_16SC2,
        )

        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
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

    def resize_for_depth(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2] == (self.process_height, self.process_width):
            return frame
        return cv2.resize(frame, self.process_size, interpolation=cv2.INTER_AREA)

    def compute(
        self, left_rectified: np.ndarray, right_rectified: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[RegionDepth]]:
        left_small = self.resize_for_depth(left_rectified)
        right_small = self.resize_for_depth(right_rectified)

        left_gray = cv2.cvtColor(left_small, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_small, cv2.COLOR_BGR2GRAY)

        disparity = (
            self.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        )

        valid_disparity = disparity > 0.0
        points_3d = cv2.reprojectImageTo3D(disparity, self.scaled_q)
        depth_m = points_3d[:, :, 2]

        valid_depth = (
            valid_disparity
            & np.isfinite(depth_m)
            & (depth_m >= MIN_DEPTH_M)
            & (depth_m <= MAX_DEPTH_M)
        )
        depth_m[~valid_depth] = np.nan

        return left_small, depth_m, self.compute_regions(depth_m)

    @staticmethod
    def region_rectangles(width: int, height: int) -> list[tuple[str, tuple[int, int, int, int]]]:
        x_left = width // 3
        x_right = (2 * width) // 3
        y_middle = height // 2
        return [
            ("LEFT UP", (0, 0, x_left, y_middle)),
            ("LEFT DOWN", (0, y_middle, x_left, height)),
            ("MIDDLE", (x_left, 0, x_right, height)),
            ("RIGHT UP", (x_right, 0, width, y_middle)),
            ("RIGHT DOWN", (x_right, y_middle, width, height)),
        ]

    def compute_regions(self, depth_m: np.ndarray) -> list[RegionDepth]:
        height, width = depth_m.shape
        results: list[RegionDepth] = []

        for name, rect in self.region_rectangles(width, height):
            x1, y1, x2, y2 = rect
            region = depth_m[y1:y2, x1:x2]
            valid_values = region[np.isfinite(region)]
            valid_fraction = (
                float(valid_values.size) / float(region.size) if region.size else 0.0
            )

            # Require enough supporting pixels, then use the nearest one percent
            # rather than the absolute minimum. A single bad disparity pixel
            # therefore cannot become a false nearest-obstacle measurement.
            minimum_count = max(200, int(region.size * 0.005))
            value = None
            nearest_point = None
            if valid_values.size >= minimum_count:
                value = float(np.percentile(valid_values, NEAREST_PERCENTILE))

                # Mark the valid pixel whose depth is closest to the robust
                # nearest-depth estimate. This provides a stable on-frame point
                # without selecting an isolated absolute-minimum outlier.
                distance_from_value = np.where(
                    np.isfinite(region), np.abs(region - value), np.inf
                )
                flat_index = int(np.argmin(distance_from_value))
                local_y, local_x = np.unravel_index(flat_index, region.shape)
                nearest_point = (x1 + int(local_x), y1 + int(local_y))

            results.append(
                RegionDepth(
                    name=name,
                    rect=rect,
                    depth_m=value,
                    valid_fraction=valid_fraction,
                    nearest_point=nearest_point,
                )
            )

        return results


class ObjectDetector:
    """Future local object-detection integration point.

    Replace detect() later with the local model. Coordinates returned by the
    model should refer to the same image passed into this method.
    """

    def detect(self, left_rectified: np.ndarray) -> list[dict]:
        del left_rectified
        return []


def depth_color(depth_m: Optional[float]) -> tuple[int, int, int]:
    if depth_m is None:
        return (180, 180, 180)
    if depth_m <= 0.75:
        return (0, 0, 255)
    if depth_m <= 1.50:
        return (0, 140, 255)
    if depth_m <= 2.25:
        return (0, 220, 255)
    return (0, 200, 0)


def draw_regions(frame: np.ndarray, regions: list[RegionDepth]) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()

    for result in regions:
        x1, y1, x2, y2 = result.rect
        color = depth_color(result.depth_m)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    cv2.addWeighted(overlay, 0.12, output, 0.88, 0, output)

    for result in regions:
        x1, y1, x2, y2 = result.rect
        color = depth_color(result.depth_m)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

        if result.depth_m is None:
            value_text = "no reliable depth <= 3 m"
        else:
            value_text = f"{result.depth_m:.2f} m"
        label = f"{result.name}: {value_text}"

        font_scale = max(0.55, output.shape[1] / 2200.0)
        thickness = max(1, int(round(font_scale * 2)))
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_x = x1 + 12
        label_y = y1 + text_height + 18
        cv2.rectangle(
            output,
            (label_x - 6, label_y - text_height - 8),
            (label_x + text_width + 6, label_y + baseline + 6),
            (15, 15, 15),
            -1,
        )
        cv2.putText(
            output,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        if result.nearest_point is not None:
            point_x, point_y = result.nearest_point
            cv2.circle(output, (point_x, point_y), 11, (255, 255, 255), 3)
            cv2.drawMarker(
                output,
                (point_x, point_y),
                color,
                cv2.MARKER_CROSS,
                24,
                3,
                cv2.LINE_AA,
            )

    return output


def add_status_panel(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Return the camera frame with a separate status panel below it."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 2
    line_height = 28
    panel_height = 14 + line_height * len(lines)
    panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (12, 25 + index * line_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return cv2.vconcat([frame, panel])


def resize_preview(frame: np.ndarray, preview_width: int) -> np.ndarray:
    if preview_width <= 0 or frame.shape[1] <= preview_width:
        return frame
    ratio = preview_width / float(frame.shape[1])
    preview_height = max(1, int(round(frame.shape[0] * ratio)))
    return cv2.resize(frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)


def validate_runtime(args: argparse.Namespace) -> None:
    if not 0.0 < args.depth_scale <= 1.0:
        raise RuntimeError("--depth-scale must be greater than 0 and at most 1")
    if args.num_disparities <= 0 or args.num_disparities % 16 != 0:
        raise RuntimeError("--num-disparities must be positive and divisible by 16")
    if args.block_size < 3 or args.block_size % 2 == 0:
        raise RuntimeError("--block-size must be an odd number of at least 3")
    if args.left_id == args.right_id:
        raise RuntimeError("Left and right sensor IDs must be different")

    if not args.check_calibration:
        gstreamer_enabled = any(
            "GStreamer" in line and "YES" in line
            for line in cv2.getBuildInformation().splitlines()
        )
        if not gstreamer_enabled:
            raise RuntimeError(
                "This OpenCV build has no GStreamer support. "
                f"Loaded OpenCV {cv2.__version__} from {cv2.__file__}. "
                "Use Jetson's Ubuntu python3-opencv package."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IMX219 rectified left preview and five-region stereo depth"
    )
    parser.add_argument("--left-id", type=int, default=0)
    parser.add_argument("--right-id", type=int, default=1)
    parser.add_argument("--fps", type=int, default=CAPTURE_FPS)
    parser.add_argument("--sensor-mode", type=int, default=SENSOR_MODE)
    parser.add_argument("--flip-method", type=int, choices=range(8), default=0)
    parser.add_argument("--preview-width", type=int, default=960)
    parser.add_argument("--depth-scale", type=float, default=DEFAULT_DEPTH_SCALE)
    parser.add_argument(
        "--num-disparities", type=int, default=DEFAULT_NUM_DISPARITIES
    )
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Folder containing left.xml, right.xml, and stereo.xml",
    )
    parser.add_argument(
        "--check-calibration",
        action="store_true",
        help="Load and validate the XML files without opening cameras",
    )
    return parser.parse_args()


def print_calibration(calibration: StereoCalibration, directory: Path) -> None:
    translation = calibration.translation.reshape(-1)
    print(f"Calibration directory: {directory.resolve()}")
    print(f"Calibration resolution: {CAPTURE_WIDTH} x {CAPTURE_HEIGHT}")
    print(f"Rectified focal length: {calibration.focal_px:.3f} pixels")
    print(f"Rectified baseline: {calibration.baseline_m:.6f} m")
    print(
        "Translation T: "
        f"[{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}] m"
    )


def main() -> int:
    args = parse_args()

    try:
        validate_runtime(args)
        calibration = StereoCalibration.load(args.calibration_dir)
    except (RuntimeError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_calibration(calibration, args.calibration_dir)
    if args.check_calibration:
        print("Calibration XML check passed.")
        return 0

    processor = StereoDepthProcessor(
        calibration=calibration,
        scale=args.depth_scale,
        num_disparities=args.num_disparities,
        block_size=args.block_size,
    )
    theoretical_nearest = (
        processor.scaled_focal_px
        * calibration.baseline_m
        / float(args.num_disparities)
    )
    print(
        f"Depth processing resolution: {processor.process_width} x "
        f"{processor.process_height}"
    )
    print(
        f"Requested depth range: {MIN_DEPTH_M:.2f} to {MAX_DEPTH_M:.2f} m"
    )
    print(
        "Approximate matcher near limit: "
        f"{theoretical_nearest:.2f} m (depends on scene and texture)"
    )

    left_pipeline = gstreamer_pipeline(
        args.left_id,
        CAPTURE_WIDTH,
        CAPTURE_HEIGHT,
        args.fps,
        args.sensor_mode,
        args.flip_method,
    )
    right_pipeline = gstreamer_pipeline(
        args.right_id,
        CAPTURE_WIDTH,
        CAPTURE_HEIGHT,
        args.fps,
        args.sensor_mode,
        args.flip_method,
    )

    left_camera = CameraStream(args.left_id, left_pipeline)
    right_camera = CameraStream(args.right_id, right_pipeline)

    window_name = "IMX219 Stereo Depth - Rectified Left View"
    show_rectified = True
    fullscreen = False
    last_left_sequence = -1
    last_right_sequence = -1
    latest_regions: list[RegionDepth] = []
    latest_depth_time_ms: Optional[float] = None
    latest_left_rectified: Optional[np.ndarray] = None

    shown_fps = 0.0
    fps_frames = 0
    fps_started = time.monotonic()
    shown_depth_fps = 0.0
    depth_frames = 0
    depth_fps_started = time.monotonic()

    try:
        left_camera.start()
        right_camera.start()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            window_name,
            args.preview_width,
            int(
                round(
                    args.preview_width
                    * (processor.process_height + 98)
                    / processor.process_width
                )
            ),
        )

        print("Camera view and continuous depth sensing started. Q or Esc quits.")

        while True:
            left_frame, left_time, left_sequence, left_error = left_camera.latest()
            right_frame, right_time, right_sequence, right_error = right_camera.latest()

            if left_frame is None or right_frame is None:
                waiting = np.zeros((480, 800, 3), dtype=np.uint8)
                message = left_error or right_error or "Waiting for both cameras..."
                cv2.putText(
                    waiting,
                    message,
                    (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, waiting)
                key = cv2.waitKey(10) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            expected_shape = (CAPTURE_HEIGHT, CAPTURE_WIDTH)
            if left_frame.shape[:2] != expected_shape or right_frame.shape[:2] != expected_shape:
                raise RuntimeError(
                    "Camera frames do not match the XML calibration resolution. "
                    f"Expected {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}; got left "
                    f"{left_frame.shape[1]}x{left_frame.shape[0]} and right "
                    f"{right_frame.shape[1]}x{right_frame.shape[0]}."
                )

            new_pair = (
                left_sequence != last_left_sequence
                and right_sequence != last_right_sequence
            )

            if new_pair:
                last_left_sequence = left_sequence
                last_right_sequence = right_sequence

                latest_left_rectified = cv2.remap(
                    left_frame,
                    processor.left_map1,
                    processor.left_map2,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )

                right_rectified = cv2.remap(
                    right_frame,
                    processor.right_map1,
                    processor.right_map2,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                started = time.monotonic()
                _, _, latest_regions = processor.compute(
                    latest_left_rectified, right_rectified
                )
                latest_depth_time_ms = (time.monotonic() - started) * 1000.0
                depth_frames += 1

                depth_elapsed = time.monotonic() - depth_fps_started
                if depth_elapsed >= 1.0:
                    shown_depth_fps = depth_frames / depth_elapsed
                    depth_frames = 0
                    depth_fps_started = time.monotonic()

            if show_rectified and latest_left_rectified is not None:
                display_source = latest_left_rectified
            else:
                display_source = left_frame

            display_frame = processor.resize_for_depth(display_source)
            if latest_regions:
                display_frame = draw_regions(display_frame, latest_regions)
            else:
                display_frame = display_frame.copy()

            fps_frames += 1
            now = time.monotonic()
            fps_elapsed = now - fps_started
            if fps_elapsed >= 1.0:
                shown_fps = fps_frames / fps_elapsed
                fps_frames = 0
                fps_started = now

            offset_ms = abs(left_time - right_time) * 1000.0
            view_text = "RECTIFIED LEFT" if show_rectified else "RAW LEFT"
            depth_speed = (
                ""
                if latest_depth_time_ms is None
                else (
                    f" | depth {shown_depth_fps:.1f} FPS "
                    f"({latest_depth_time_ms:.0f} ms)"
                )
            )
            display_frame = add_status_panel(
                display_frame,
                [
                    f"{view_text} | DEPTH ACTIVE | range {MIN_DEPTH_M:.1f}-{MAX_DEPTH_M:.1f} m",
                    f"display {shown_fps:.1f} FPS | pair offset {offset_ms:.1f} ms{depth_speed}",
                    "R raw/rectified | F fullscreen | Q quit",
                ],
            )

            cv2.imshow(window_name, resize_preview(display_frame, args.preview_width))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                show_rectified = not show_rectified
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) == 0:
                    break
            except cv2.error:
                pass

    except KeyboardInterrupt:
        pass
    except (RuntimeError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        left_camera.stop()
        right_camera.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
