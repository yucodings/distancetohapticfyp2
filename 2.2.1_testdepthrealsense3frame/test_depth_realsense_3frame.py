#!/usr/bin/env python3
"""Measure aligned RealSense depth in left, centre, and right boxes.

The camera produces one hardware depth frame. After aligning it to colour, the
same metric depth map is sampled independently by all three measurement boxes.

Controls:
    D         toggle the depth panel
    S         save the current evidence bundle
    Q or Esc  quit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


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
except Exception as exc:
    rs = None
    REALSENSE_IMPORT_ERROR: Optional[Exception] = exc
else:
    REALSENSE_IMPORT_ERROR = None


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

WIDTH = 1280
HEIGHT = 720
FPS = 30
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 4.00
MIN_ROI_DEPTH_SAMPLES = 100

ROI_WIDTH_FRACTION = 0.30
ROI_HEIGHT_FRACTION = 0.30
ROI_HORIZONTAL_CENTRES = (
    ("Left", 1.0 / 6.0),
    ("Centre", 1.0 / 2.0),
    ("Right", 5.0 / 6.0),
)

START_RETRIES = 5
START_RETRY_DELAY_S = 1.0
FRAME_TIMEOUT_MS = 5000
WARMUP_FRAMES = 10

DISPLAY_PANEL_WIDTH = 640
DISPLAY_PANEL_HEIGHT = 360
STATUS_PANEL_HEIGHT = 150
SHOW_DEPTH_AT_START = True

ZONE_COLOURS = {
    "Left": (255, 200, 0),
    "Centre": (0, 255, 0),
    "Right": (255, 0, 255),
}


@dataclass(frozen=True)
class DepthMeasurement:
    name: str
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


def three_rois(
    image_shape: tuple[int, int],
) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
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


def measure_three_depths(
    depth_m: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    min_samples: int = MIN_ROI_DEPTH_SAMPLES,
) -> tuple[tuple[DepthMeasurement, ...], np.ndarray]:
    if depth_m.ndim != 2:
        raise ValueError("Metric depth must be a 2-D array")
    if min_depth_m <= 0 or max_depth_m <= min_depth_m:
        raise ValueError("Depth limits must satisfy 0 < minimum < maximum")
    if min_samples < 1:
        raise ValueError("Minimum sample count must be positive")

    valid_mask = (
        np.isfinite(depth_m)
        & (depth_m >= min_depth_m)
        & (depth_m <= max_depth_m)
    )
    measurements = []
    for name, box in three_rois(depth_m.shape):
        x1, y1, x2, y2 = box
        roi_depth = depth_m[y1:y2, x1:x2]
        roi_valid = valid_mask[y1:y2, x1:x2]
        values = roi_depth[roi_valid]
        sample_count = int(values.size)
        distance_m = (
            float(np.median(values)) if sample_count >= min_samples else None
        )
        measurements.append(
            DepthMeasurement(
                name=name,
                distance_m=distance_m,
                box=box,
                sample_count=sample_count,
                valid_percentage=100.0 * float(np.mean(roi_valid)),
            )
        )
    return tuple(measurements), valid_mask


def make_depth_view(
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    if depth_m.shape != valid_mask.shape:
        raise ValueError("Depth and validity arrays must have matching shapes")
    scaled = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid_mask):
        clipped = np.clip(depth_m[valid_mask], min_depth_m, max_depth_m)
        scaled[valid_mask] = np.round(
            (max_depth_m - clipped) * 255.0 / (max_depth_m - min_depth_m)
        ).astype(np.uint8)
    colour = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colour[~valid_mask] = 0
    return colour


def draw_measurements(
    image: np.ndarray,
    measurements: tuple[DepthMeasurement, ...],
) -> None:
    for measurement in measurements:
        x1, y1, x2, y2 = measurement.box
        colour = (
            ZONE_COLOURS[measurement.name]
            if measurement.distance_m is not None
            else (0, 0, 255)
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)
        value = (
            f"{measurement.distance_m:.2f} m"
            if measurement.distance_m is not None
            else "Depth N/A"
        )
        label = f"{measurement.name}: {value}"
        text_y = max(28, y1 - 10)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2
        )
        cv2.rectangle(
            image,
            (x1, text_y - text_height - 7),
            (min(image.shape[1] - 1, x1 + text_width + 10), text_y + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 5, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            colour,
            2,
            cv2.LINE_AA,
        )


def make_primary_display(
    annotated_colour: np.ndarray,
    annotated_depth: np.ndarray,
    show_depth: bool,
) -> np.ndarray:
    colour_panel = cv2.resize(
        annotated_colour,
        (DISPLAY_PANEL_WIDTH, DISPLAY_PANEL_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    if show_depth:
        depth_panel = cv2.resize(
            annotated_depth,
            (DISPLAY_PANEL_WIDTH, DISPLAY_PANEL_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
    else:
        depth_panel = np.zeros_like(colour_panel)
        cv2.putText(
            depth_panel,
            "DEPTH VIEW OFF - PRESS D",
            (145, DISPLAY_PANEL_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        colour_panel,
        "ALIGNED COLOUR",
        (12, DISPLAY_PANEL_HEIGHT - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        depth_panel,
        "DEPTH: NEAR RED | FAR BLUE",
        (12, DISPLAY_PANEL_HEIGHT - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    images = cv2.hconcat((colour_panel, depth_panel))
    status = np.full(
        (STATUS_PANEL_HEIGHT, images.shape[1], 3), 18, dtype=np.uint8
    )
    return cv2.vconcat((images, status))


def draw_status(
    display: np.ndarray,
    stream_fps: float,
    display_fps: float,
    depth_scale: float,
    valid_percentage: float,
    measurements: tuple[DepthMeasurement, ...],
    show_depth: bool,
) -> None:
    values = " | ".join(
        f"{item.name}: "
        + (f"{item.distance_m:.2f} m" if item.distance_m is not None else "N/A")
        + f" ({item.sample_count} samples)"
        for item in measurements
    )
    lines = (
        values,
        f"Stream {stream_fps:.1f} FPS | Display {display_fps:.1f} FPS | "
        f"Global valid depth {valid_percentage:.1f}%",
        f"Depth scale {depth_scale:.6f} m/unit | Alignment depth -> colour | "
        f"Depth panel {'ON' if show_depth else 'OFF'}",
        "D depth view | S save evidence | Q/Esc quit",
    )
    start_y = DISPLAY_PANEL_HEIGHT + 30
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (14, start_y + index * 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def save_evidence(
    annotated_colour: np.ndarray,
    annotated_depth: np.ndarray,
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    measurements: tuple[DepthMeasurement, ...],
    depth_scale: float,
    min_depth_m: float,
    max_depth_m: float,
    serial: str,
) -> Path:
    output = RESULTS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(output / "annotated_colour.png"), annotated_colour):
        raise RuntimeError("Could not save annotated colour image")
    if not cv2.imwrite(str(output / "depth_heatmap.png"), annotated_depth):
        raise RuntimeError("Could not save depth heatmap")
    np.save(output / "depth_metres_float32.npy", depth_m.astype(np.float32))
    np.save(output / "valid_mask.npy", valid_mask.astype(np.uint8))
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "camera_serial": serial,
        "resolution": [WIDTH, HEIGHT],
        "fps": FPS,
        "alignment": "depth-to-colour",
        "depth_scale_m_per_unit": depth_scale,
        "depth_range_m": [min_depth_m, max_depth_m],
        "measurements": [asdict(item) for item in measurements],
    }
    (output / "three_zone_measurements.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return output


def safe_device_info(device: Any, field: Any) -> str:
    try:
        return device.get_info(field)
    except Exception:
        return "N/A"


def start_pipeline(serial: Optional[str]) -> tuple[Any, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(1, START_RETRIES + 1):
        pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        try:
            return pipeline, pipeline.start(config)
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
                f"{video.fps()}, expected {WIDTH}x{HEIGHT}@{FPS}"
            )


def print_startup(profile: Any, depth_scale: float) -> str:
    device = profile.get_device()
    serial = safe_device_info(device, rs.camera_info.serial_number)
    print("\nRealSense three-zone depth test")
    print(f"  Effective user:    UID {os.geteuid()}")
    print(f"  Device:            {safe_device_info(device, rs.camera_info.name)}")
    print(f"  Serial:            {serial}")
    print(f"  Firmware:          {safe_device_info(device, rs.camera_info.firmware_version)}")
    print(f"  USB type:          {safe_device_info(device, rs.camera_info.usb_type_descriptor)}")
    print(f"  Streams:           {WIDTH}x{HEIGHT}@{FPS} colour + depth")
    print(f"  Depth scale:       {depth_scale:.6f} m/unit")
    print("  Measurement boxes: left | centre | right")
    print("  Controls:          D depth | S save | Q/Esc quit\n")
    return serial


def run(args: argparse.Namespace) -> None:
    require_realsense()
    if args.min_depth <= 0:
        raise ValueError("--min-depth must be greater than zero")
    if args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth")

    pipeline: Optional[Any] = None
    try:
        print("Connecting to RealSense...", flush=True)
        pipeline, profile = start_pipeline(args.serial)
        validate_profiles(profile)
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise RuntimeError(f"Invalid RealSense depth scale: {depth_scale}")
        serial = print_startup(profile, depth_scale)
        align_to_colour = rs.align(rs.stream.color)

        print(f"Warming up with {WARMUP_FRAMES} frames...", flush=True)
        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames(FRAME_TIMEOUT_MS)

        window_name = "RealSense Three-Zone Depth Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            window_name,
            DISPLAY_PANEL_WIDTH * 2,
            DISPLAY_PANEL_HEIGHT + STATUS_PANEL_HEIGHT,
        )

        show_depth = SHOW_DEPTH_AT_START
        stream_fps = 0.0
        display_fps = 0.0
        previous_stream_time: Optional[float] = None
        previous_display_time = time.perf_counter()

        while True:
            frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
            aligned = align_to_colour.process(frames)
            depth_frame = aligned.get_depth_frame()
            colour_frame = aligned.get_color_frame()
            if not depth_frame or not colour_frame:
                print("Frame skipped: aligned colour or depth is missing", flush=True)
                continue

            colour_image = np.asanyarray(colour_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            expected_shape = (HEIGHT, WIDTH)
            if colour_image.shape[:2] != expected_shape or depth_raw.shape != expected_shape:
                raise RuntimeError(
                    f"Expected {WIDTH}x{HEIGHT}; received colour "
                    f"{colour_image.shape[:2]} and depth {depth_raw.shape}"
                )

            depth_m = depth_raw.astype(np.float32) * np.float32(depth_scale)
            measurements, valid_mask = measure_three_depths(
                depth_m, args.min_depth, args.max_depth
            )
            depth_view = make_depth_view(
                depth_m, valid_mask, args.min_depth, args.max_depth
            )
            annotated_colour = colour_image.copy()
            annotated_depth = depth_view.copy()
            draw_measurements(annotated_colour, measurements)
            draw_measurements(annotated_depth, measurements)

            stream_now = time.perf_counter()
            if previous_stream_time is not None:
                instant = 1.0 / max(stream_now - previous_stream_time, 1e-6)
                stream_fps = instant if stream_fps == 0 else 0.9 * stream_fps + 0.1 * instant
            previous_stream_time = stream_now

            display = make_primary_display(
                annotated_colour, annotated_depth, show_depth
            )
            display_now = time.perf_counter()
            instant = 1.0 / max(display_now - previous_display_time, 1e-6)
            display_fps = instant if display_fps == 0 else 0.9 * display_fps + 0.1 * instant
            previous_display_time = display_now
            draw_status(
                display,
                stream_fps,
                display_fps,
                depth_scale,
                100.0 * float(np.mean(valid_mask)),
                measurements,
                show_depth,
            )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("d"), ord("D")):
                show_depth = not show_depth
            elif key in (ord("s"), ord("S")):
                output = save_evidence(
                    annotated_colour,
                    annotated_depth,
                    depth_m,
                    valid_mask,
                    measurements,
                    depth_scale,
                    args.min_depth,
                    args.max_depth,
                    serial,
                )
                print(f"Saved evidence: {output}", flush=True)
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
                print("RealSense pipeline stopped.", flush=True)
            except Exception:
                pass
        cv2.destroyAllWindows()


def main() -> int:
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
