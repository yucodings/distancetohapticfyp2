#!/usr/bin/env python3
"""Test aligned metric depth from an Intel RealSense camera.

The program requests 1280x720 color and depth at 30 FPS, aligns depth to the
color coordinate system, measures median depth in a centre ROI, and displays
annotated color and metric depth views side by side.

Controls:
    Q or Esc  quit
    D         toggle the depth visualization (depth processing stays active)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Use Jetson/Ubuntu's mutually compatible OpenCV, NumPy, and RealSense
# bindings. This test does not need the pip Ultralytics/PyTorch environment.
SYSTEM_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
if SYSTEM_DIST_PACKAGES.is_dir():
    system_packages = str(SYSTEM_DIST_PACKAGES)
    if system_packages in sys.path:
        sys.path.remove(system_packages)
    sys.path.insert(0, system_packages)

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as exc:  # Report a focused startup error from main().
    rs = None
    REALSENSE_IMPORT_ERROR: Optional[Exception] = exc
else:
    REALSENSE_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# User-adjustable settings
# ---------------------------------------------------------------------------
WIDTH = 1280
HEIGHT = 720
FPS = 30

MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 4.00
CENTRE_ROI_WIDTH_FRACTION = 0.20
CENTRE_ROI_HEIGHT_FRACTION = 0.20
MIN_CENTRE_DEPTH_SAMPLES = 100

START_RETRIES = 5
START_RETRY_DELAY_S = 1.0
FRAME_TIMEOUT_MS = 5000
WARMUP_FRAMES = 10

DISPLAY_PANEL_WIDTH = 640
DISPLAY_PANEL_HEIGHT = 360
SHOW_DEPTH_AT_START = True


@dataclass(frozen=True)
class DepthMeasurement:
    distance_m: Optional[float]
    box: tuple[int, int, int, int]
    sample_count: int
    valid_percentage: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial",
        default=None,
        help="optional RealSense serial number when multiple devices are connected",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=MIN_DEPTH_M,
        help="minimum accepted depth in metres",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=MAX_DEPTH_M,
        help="maximum accepted/displayed depth in metres",
    )
    return parser.parse_args()


def require_realsense() -> None:
    if rs is None:
        raise RuntimeError(
            "Could not import the system pyrealsense2 binding from "
            f"{SYSTEM_DIST_PACKAGES}. Original error: {REALSENSE_IMPORT_ERROR}"
        )


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
    depth_m: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[DepthMeasurement, np.ndarray]:
    valid_mask = (
        np.isfinite(depth_m)
        & (depth_m >= min_depth_m)
        & (depth_m <= max_depth_m)
    )
    box = centre_roi(depth_m.shape)
    x1, y1, x2, y2 = box
    roi_values = depth_m[y1:y2, x1:x2]
    roi_valid = valid_mask[y1:y2, x1:x2]
    valid_values = roi_values[roi_valid]
    sample_count = int(valid_values.size)
    distance_m = (
        float(np.median(valid_values))
        if sample_count >= MIN_CENTRE_DEPTH_SAMPLES
        else None
    )
    measurement = DepthMeasurement(
        distance_m=distance_m,
        box=box,
        sample_count=sample_count,
        valid_percentage=100.0 * float(np.mean(valid_mask)),
    )
    return measurement, valid_mask


def make_depth_view(
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Render valid metric depth with near red and far blue."""
    scaled = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid_mask):
        clipped = np.clip(depth_m[valid_mask], min_depth_m, max_depth_m)
        scaled[valid_mask] = np.round(
            (max_depth_m - clipped)
            * 255.0
            / (max_depth_m - min_depth_m)
        ).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    color[~valid_mask] = 0
    return color


def draw_measurement(
    image: np.ndarray,
    measurement: DepthMeasurement,
) -> None:
    x1, y1, x2, y2 = measurement.box
    color = (0, 255, 0) if measurement.distance_m is not None else (0, 0, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    distance_text = (
        f"Centre: {measurement.distance_m:.2f} m"
        if measurement.distance_m is not None
        else "Centre: Depth N/A"
    )
    cv2.rectangle(image, (10, 10), (350, 68), (0, 0, 0), -1)
    cv2.putText(
        image,
        distance_text,
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Centre valid samples: {measurement.sample_count}",
        (18, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def make_primary_display(
    color_image: np.ndarray,
    depth_view: np.ndarray,
    show_depth: bool,
) -> np.ndarray:
    color_preview = cv2.resize(
        color_image,
        (DISPLAY_PANEL_WIDTH, DISPLAY_PANEL_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    if show_depth:
        depth_preview = cv2.resize(
            depth_view,
            (DISPLAY_PANEL_WIDTH, DISPLAY_PANEL_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
    else:
        depth_preview = np.zeros_like(color_preview)
        cv2.putText(
            depth_preview,
            "DEPTH VIEW OFF - PRESS D",
            (145, DISPLAY_PANEL_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        color_preview,
        "ALIGNED COLOR",
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
    return cv2.hconcat((color_preview, depth_preview))


def draw_status(
    display: np.ndarray,
    stream_fps: float,
    display_fps: float,
    measurement: DepthMeasurement,
    depth_scale: float,
    show_depth: bool,
) -> None:
    lines = (
        f"Stream FPS: {stream_fps:.1f}",
        f"Display FPS: {display_fps:.1f}",
        f"Valid depth: {measurement.valid_percentage:.1f}%",
        f"Depth scale: {depth_scale:.6f} m/unit",
        "Alignment: depth -> color",
        "Depth source: RealSense hardware",
        f"D depth view: {'ON' if show_depth else 'OFF'}",
        "Q / Esc: quit",
    )
    panel_width = 350
    panel_height = len(lines) * 22 + 12
    x1 = display.shape[1] - panel_width - 10
    panel = display[10 : 10 + panel_height, x1 : x1 + panel_width]
    dark = np.zeros_like(panel)
    cv2.addWeighted(dark, 0.68, panel, 0.32, 0, dst=panel)
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (x1 + 10, 32 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def safe_device_info(device: Any, field: Any) -> str:
    try:
        return device.get_info(field)
    except Exception:
        return "N/A"


def start_pipeline(serial: Optional[str]) -> tuple[Any, Any]:
    """Start a new pipeline with retries and return it with its profile."""
    last_error: Optional[Exception] = None
    for attempt in range(1, START_RETRIES + 1):
        pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        try:
            profile = pipeline.start(config)
            return pipeline, profile
        except Exception as exc:
            last_error = exc
            try:
                pipeline.stop()
            except Exception:
                pass
            print(
                f"RealSense start attempt {attempt}/{START_RETRIES} failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < START_RETRIES:
                time.sleep(START_RETRY_DELAY_S)
    raise RuntimeError(
        f"Could not start RealSense after {START_RETRIES} attempts: {last_error}"
    )


def validate_profiles(profile: Any) -> None:
    for stream_type, name in (
        (rs.stream.color, "color"),
        (rs.stream.depth, "depth"),
    ):
        video = profile.get_stream(stream_type).as_video_stream_profile()
        if video.width() != WIDTH or video.height() != HEIGHT or video.fps() != FPS:
            raise RuntimeError(
                f"RealSense {name} profile is {video.width()}x{video.height()}@"
                f"{video.fps()}, expected {WIDTH}x{HEIGHT}@{FPS}."
            )


def print_startup(profile: Any, depth_scale: float) -> None:
    device = profile.get_device()
    print("\nRealSense depth test")
    print(f"  Effective user:       UID {os.geteuid()}")
    print(f"  OpenCV version:       {cv2.__version__}")
    print(f"  OpenCV path:          {cv2.__file__}")
    print(f"  pyrealsense2 path:    {getattr(rs, '__file__', 'N/A')}")
    print(
        "  Device:               "
        f"{safe_device_info(device, rs.camera_info.name)}"
    )
    print(
        "  Serial:               "
        f"{safe_device_info(device, rs.camera_info.serial_number)}"
    )
    print(
        "  Firmware:             "
        f"{safe_device_info(device, rs.camera_info.firmware_version)}"
    )
    print(
        "  USB type:             "
        f"{safe_device_info(device, rs.camera_info.usb_type_descriptor)}"
    )
    print(f"  Streams:              {WIDTH}x{HEIGHT}@{FPS} color + depth")
    print(f"  Depth scale:          {depth_scale:.6f} m/unit")
    print("  Alignment:            depth to color")
    print("  Depth processing:     RealSense hardware (no VPI/CUDA)")
    print("  Controls:             Q/Esc quit | D depth view\n")


def run(args: argparse.Namespace) -> None:
    require_realsense()
    if args.min_depth <= 0.0:
        raise ValueError("--min-depth must be greater than zero")
    if args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth")

    pipeline: Optional[Any] = None
    try:
        print("Connecting to RealSense...", flush=True)
        pipeline, profile = start_pipeline(args.serial)
        validate_profiles(profile)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        if not np.isfinite(depth_scale) or depth_scale <= 0.0:
            raise RuntimeError(f"Invalid RealSense depth scale: {depth_scale}")

        align_to_color = rs.align(rs.stream.color)
        print_startup(profile, depth_scale)
        print(f"Warming up with {WARMUP_FRAMES} frames...", flush=True)
        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames(FRAME_TIMEOUT_MS)

        window_name = "RealSense Depth Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            window_name,
            DISPLAY_PANEL_WIDTH * 2,
            DISPLAY_PANEL_HEIGHT,
        )

        show_depth = SHOW_DEPTH_AT_START
        stream_fps = 0.0
        display_fps = 0.0
        previous_stream_time: Optional[float] = None
        previous_display_time = time.perf_counter()

        while True:
            frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
            aligned = align_to_color.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                print("Frame skipped: aligned color or depth is missing", flush=True)
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            expected_shape = (HEIGHT, WIDTH)
            if color_image.shape[:2] != expected_shape or depth_raw.shape != expected_shape:
                raise RuntimeError(
                    "Aligned frame resolution changed. Expected "
                    f"{WIDTH}x{HEIGHT}; received color {color_image.shape[:2]} "
                    f"and depth {depth_raw.shape}."
                )

            depth_m = depth_raw.astype(np.float32) * np.float32(depth_scale)
            measurement, valid_mask = measure_centre_depth(
                depth_m,
                args.min_depth,
                args.max_depth,
            )
            depth_view = make_depth_view(
                depth_m,
                valid_mask,
                args.min_depth,
                args.max_depth,
            )

            annotated_color = color_image.copy()
            annotated_depth = depth_view.copy()
            draw_measurement(annotated_color, measurement)
            draw_measurement(annotated_depth, measurement)

            stream_now = time.perf_counter()
            if previous_stream_time is not None:
                instantaneous_stream_fps = 1.0 / max(
                    stream_now - previous_stream_time,
                    1e-6,
                )
                stream_fps = (
                    instantaneous_stream_fps
                    if stream_fps == 0.0
                    else 0.90 * stream_fps + 0.10 * instantaneous_stream_fps
                )
            previous_stream_time = stream_now

            display = make_primary_display(
                annotated_color,
                annotated_depth,
                show_depth,
            )
            display_now = time.perf_counter()
            instantaneous_display_fps = 1.0 / max(
                display_now - previous_display_time,
                1e-6,
            )
            display_fps = (
                instantaneous_display_fps
                if display_fps == 0.0
                else 0.90 * display_fps + 0.10 * instantaneous_display_fps
            )
            previous_display_time = display_now

            draw_status(
                display,
                stream_fps,
                display_fps,
                measurement,
                depth_scale,
                show_depth,
            )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("d"), ord("D")):
                show_depth = not show_depth
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
                print("RealSense pipeline stopped.", flush=True)
            except Exception:
                pass
        cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
