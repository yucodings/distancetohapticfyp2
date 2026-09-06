#!/usr/bin/env python3
"""Process one camera pair through TensorRT and VPI without enabling haptics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import stereo_core as core
from config import (
    DETECTION_CONFIDENCE,
    DETECTION_FPS,
    DETECTION_IOU,
    DETECTION_MAX_RESULTS,
    DETECTOR_MODEL_PATH,
)
from detection_worker import LatestFrameDetectionWorker
from gpu_scheduler import GpuScheduler


def latest_calibration(project_root: Path) -> Path:
    candidates = list(
        (project_root / "1.0_calibration" / "images").glob(
            "*/stereo_calibration.npz"
        )
    )
    if not candidates:
        raise FileNotFoundError("No completed calibration was found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration", type=Path, default=latest_calibration(project_root)
    )
    parser.add_argument(
        "--vpi-profile",
        type=Path,
        default=project_root
        / "1.1_depth_visualization"
        / "results"
        / "vpi_recommended_profile.json",
    )
    parser.add_argument("--model", type=Path, default=DETECTOR_MODEL_PATH)
    parser.add_argument("--timeout", type=float, default=15.0)
    return parser.parse_args()


def wait_for_result(fetch, timeout: float, description: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = fetch()
        if result is not None:
            return result
        time.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for {description}")


def main() -> int:
    args = parse_args()
    scheduler = GpuScheduler()
    depth_worker = None
    detector_worker = None
    capture = None
    left_camera = None
    right_camera = None
    try:
        core.check_gstreamer()
        calibration = core.load_calibration(args.calibration)
        maps = core.convert_rectification_maps(calibration)
        settings, profile_source = core.load_vpi_profile(
            calibration, args.vpi_profile
        )
        depth_worker = core.AsyncDepthProcessor(
            "vpi-cuda",
            calibration,
            maps,
            5.0,
            settings,
            profile_source,
            gpu_scheduler=scheduler,
        )
        depth_worker.start()
        print("VPI CUDA warmup passed", flush=True)

        detector_worker = LatestFrameDetectionWorker(
            args.model,
            scheduler,
            DETECTION_FPS,
            DETECTION_CONFIDENCE,
            DETECTION_IOU,
            DETECTION_MAX_RESULTS,
            status_callback=lambda message: print(
                f"Detection: {message}", flush=True
            ),
        )
        if not detector_worker.start(timeout=args.timeout):
            raise RuntimeError("TensorRT detector initialization failed")

        left_camera, right_camera = core.open_cameras(0, 1, calibration)
        capture = core.SynchronizedStereoCapture(left_camera, right_camera)
        capture.start()
        stereo_frame = capture.get_latest(0, timeout=args.timeout)
        current_left = core.rectify_left_frame(stereo_frame.left, maps)

        detector_worker.submit(
            stereo_frame.sequence, stereo_frame.captured_at, current_left
        )
        detection = wait_for_result(
            detector_worker.latest_result, args.timeout, "TensorRT result"
        )
        print(
            f"TensorRT result passed: {detection.inference_ms:.1f} ms, "
            f"{len(detection.detections)} objects",
            flush=True,
        )

        depth_worker.submit(
            stereo_frame.sequence, current_left, stereo_frame.right
        )
        depth = wait_for_result(
            depth_worker.latest_result, args.timeout, "VPI depth result"
        )
        print(
            f"VPI result passed: {depth.depth_fps:.1f} FPS, "
            f"{depth.valid_percentage:.1f}% valid depth",
            flush=True,
        )
        print("Combined smoke test passed; haptics were never initialized.")
        return 0
    finally:
        if capture is not None:
            capture.stop()
        if detector_worker is not None:
            detector_worker.stop()
        if depth_worker is not None:
            depth_worker.stop()
        if left_camera is not None:
            left_camera.release()
        if right_camera is not None:
            right_camera.release()


if __name__ == "__main__":
    raise SystemExit(main())
