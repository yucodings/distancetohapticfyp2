#!/usr/bin/env python3
"""Live/offline IMX219 stereo-depth tuner with no haptic or YOLO access."""

from __future__ import annotations

import argparse
import gc
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

from tuner_core import (
    MAX_DEPTH_M,
    MIN_DEPTH_M,
    SgbmSettings,
    VpiCudaMatcher,
    VpiSettings,
    VPI_AUTO_DIAGONAL_OPTIONS,
    VPI_AUTO_PENALTY_PAIRS,
    VPI_AUTO_UNIQUENESS,
    automatic_tuning_score,
    compute_sgbm,
    convert_rectification_maps,
    cv2,
    depth_from_disparity,
    expected_disparity,
    load_calibration,
    make_alignment_overlay,
    make_depth_view,
    make_disparity_view,
    np,
    rectify_pair,
    roi_statistics,
    safety_tuning_score,
    save_json,
    settings_dict,
    vpi_autotune_candidates,
    zone_safety_statistics,
)
from pointcloud_utils import (
    disparity_to_xyz,
    make_orthographic_view,
    sample_point_cloud,
    save_binary_ply,
    show_open3d_snapshot,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = (
    SCRIPT_DIR.parent / "2.0_testdepthimx219" / "stereo_calibration.npz"
)
RESULTS_DIR = SCRIPT_DIR / "results"
CONTROL_WINDOW = "Stereo tuner controls"
DASHBOARD_WINDOW = "IMX219 stereo depth tuner"
ALIGNMENT_WINDOW = "Rectification check: LEFT green | RIGHT red"
AUTO_WINDOW = "Easy Mode automatic tuning"
POINT_CLOUD_WINDOW = "3D diagnostic: top and front projections"
# The supported consumers read RESULTS_DIR/vpi_recommended_profile.json.
# Do not write a second deployment copy; supported consumers read RESULTS_DIR.
PRODUCTION_VPI_PROFILE = None
VPI_EASY_FRAME_COUNT = 5


@dataclass(frozen=True)
class CapturedPair:
    sequence: int
    skew_ms: float
    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True)
class TimedFrame:
    sequence: int
    timestamp: float
    image: np.ndarray


def point_cloud_from_result(
    disparity: np.ndarray,
    valid: np.ndarray,
    color_bgr: np.ndarray,
    q_matrix: np.ndarray,
):
    xyz, xyz_valid = disparity_to_xyz(disparity, q_matrix, valid)
    return sample_point_cloud(xyz, color_bgr, xyz_valid, stride=4)


def gstreamer_pipeline(sensor_id: int, calibration) -> str:
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


def open_camera(sensor_id: int, calibration):
    pipeline = gstreamer_pipeline(sensor_id, calibration)
    camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open sensor-id={sensor_id}\n{pipeline}")
    return camera


class StereoCapture:
    """Pair independently captured frames by closest host arrival time."""

    def __init__(self, left_camera, right_camera):
        self.cameras = {"left": left_camera, "right": right_camera}
        self.buffers = {"left": deque(maxlen=8), "right": deque(maxlen=8)}
        self.condition = threading.Condition()
        self.running = False
        self.error: Optional[str] = None
        self.threads: list[threading.Thread] = []
        self.sequences = {"left": 0, "right": 0}
        self.pair_sequence = 0

    def start(self) -> None:
        self.running = True
        self.threads = [
            threading.Thread(target=self._reader, args=(side,), daemon=True)
            for side in ("left", "right")
        ]
        for thread in self.threads:
            thread.start()

    def _reader(self, side: str) -> None:
        camera = self.cameras[side]
        failures = 0
        try:
            while self.running:
                grabbed = camera.grab()
                timestamp = time.monotonic()
                ok, image = camera.retrieve()
                if not grabbed or not ok or image is None:
                    failures += 1
                    if failures < 10:
                        continue
                    raise RuntimeError(f"10 consecutive {side} camera failures")
                failures = 0
                with self.condition:
                    self.sequences[side] += 1
                    self.buffers[side].append(
                        TimedFrame(self.sequences[side], timestamp, image)
                    )
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = str(error)
                self.running = False
                self.condition.notify_all()

    def get(self, timeout: float = 2.0) -> CapturedPair:
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if self.error:
                    raise RuntimeError(self.error)
                if self.buffers["left"] and self.buffers["right"]:
                    left_index, right_index = min(
                        (
                            (li, ri)
                            for li in range(len(self.buffers["left"]))
                            for ri in range(len(self.buffers["right"]))
                        ),
                        key=lambda pair: abs(
                            self.buffers["left"][pair[0]].timestamp
                            - self.buffers["right"][pair[1]].timestamp
                        ),
                    )
                    for _ in range(left_index):
                        self.buffers["left"].popleft()
                    for _ in range(right_index):
                        self.buffers["right"].popleft()
                    left = self.buffers["left"].popleft()
                    right = self.buffers["right"].popleft()
                    self.pair_sequence += 1
                    return CapturedPair(
                        self.pair_sequence,
                        abs(left.timestamp - right.timestamp) * 1000.0,
                        left.image,
                        right.image,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Timed out waiting for both IMX219 cameras")
                self.condition.wait(remaining)

    def close(self) -> None:
        with self.condition:
            self.running = False
            self.condition.notify_all()
        for thread in self.threads:
            thread.join(timeout=2.0)
        for camera in self.cameras.values():
            camera.release()


def collect_live_tuning_pairs(
    capture: StereoCapture,
    count: int = VPI_EASY_FRAME_COUNT,
) -> list[CapturedPair]:
    if count < 3:
        raise ValueError("VPI safety tuning requires at least three live frames")
    pairs: list[CapturedPair] = []
    while len(pairs) < count:
        candidate = capture.get()
        if not pairs or candidate.sequence != pairs[-1].sequence:
            pairs.append(candidate)
            print(
                f"Collected live tuning frame {len(pairs)}/{count} "
                f"(skew {candidate.skew_ms:.2f} ms)",
                flush=True,
            )
    return pairs


def check_gstreamer() -> None:
    if not any(
        "GStreamer" in line and "YES" in line
        for line in cv2.getBuildInformation().splitlines()
    ):
        raise RuntimeError(
            f"OpenCV {cv2.__version__} at {cv2.__file__} has no GStreamer support"
        )


def nothing(_value: int) -> None:
    return


def create_controls(default_backend: str, known_distance_m: Optional[float]) -> None:
    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 620, 850)
    controls = (
        ("Backend 0=SGBM 1=VPI", 1 if default_backend == "vpi" else 0, 1),
        (
            "Known distance cm (0=off)",
            0 if known_distance_m is None else round(known_distance_m * 100),
            300,
        ),
        ("ROI size px", 61, 151),
        ("SGBM block size (odd)", 5, 15),
        ("SGBM disparities x16", 10, 16),
        ("SGBM min disparity", 0, 64),
        ("SGBM uniqueness", 10, 30),
        ("SGBM speckle window", 100, 300),
        ("SGBM speckle range", 2, 32),
        ("SGBM LR max diff", 1, 32),
        ("SGBM P1 factor", 8, 32),
        ("SGBM P2 factor", 32, 96),
        ("SGBM WLS", 0, 1),
        ("VPI max disparity x16", 16, 16),
        ("VPI min disparity", 0, 64),
        ("VPI confidence percent", 0, 100),
        ("VPI unique pct (101=off)", 101, 101),
        ("VPI P1", 8, 64),
        ("VPI P2", 96, 255),
        ("VPI diagonals", 0, 1),
    )
    for name, initial, maximum in controls:
        cv2.createTrackbar(name, CONTROL_WINDOW, initial, maximum, nothing)


def position(name: str) -> int:
    return cv2.getTrackbarPos(name, CONTROL_WINDOW)


def read_controls() -> tuple[str, object, int, Optional[float]]:
    backend = "vpi-cuda" if position("Backend 0=SGBM 1=VPI") else "opencv-sgbm"
    roi_size = max(3, position("ROI size px"))
    if roi_size % 2 == 0:
        roi_size += 1
    known_cm = position("Known distance cm (0=off)")
    known_distance = known_cm / 100.0 if known_cm else None
    if backend == "opencv-sgbm":
        p1_factor = max(1, position("SGBM P1 factor"))
        p2_factor = max(p1_factor + 1, position("SGBM P2 factor"))
        block_size = max(3, position("SGBM block size (odd)"))
        if block_size % 2 == 0:
            block_size += 1
        settings: object = SgbmSettings(
            min_disparity=position("SGBM min disparity"),
            num_disparities=max(1, position("SGBM disparities x16")) * 16,
            block_size=min(15, block_size),
            uniqueness_ratio=position("SGBM uniqueness"),
            speckle_window_size=position("SGBM speckle window"),
            speckle_range=position("SGBM speckle range"),
            disp12_max_diff=position("SGBM LR max diff"),
            p1_factor=p1_factor,
            p2_factor=p2_factor,
            use_wls=bool(position("SGBM WLS")),
        )
    else:
        unique_control = position("VPI unique pct (101=off)")
        p1 = max(1, position("VPI P1"))
        p2 = max(p1, position("VPI P2"))
        settings = VpiSettings(
            min_disparity=position("VPI min disparity"),
            max_disparity=max(1, position("VPI max disparity x16")) * 16,
            confidence_threshold=(
                1
                if position("VPI confidence percent") == 0
                else round(position("VPI confidence percent") * 65535 / 100)
            ),
            p1=p1,
            p2=p2,
            uniqueness=(-1.0 if unique_control == 101 else unique_control / 100.0),
            include_diagonals=bool(position("VPI diagonals")),
        )
    return backend, settings, roi_size, known_distance


def set_sgbm_controls(settings: SgbmSettings) -> None:
    cv2.setTrackbarPos("Backend 0=SGBM 1=VPI", CONTROL_WINDOW, 0)
    cv2.setTrackbarPos("SGBM block size (odd)", CONTROL_WINDOW, settings.block_size)
    cv2.setTrackbarPos(
        "SGBM disparities x16", CONTROL_WINDOW, settings.num_disparities // 16
    )
    cv2.setTrackbarPos("SGBM min disparity", CONTROL_WINDOW, settings.min_disparity)
    cv2.setTrackbarPos("SGBM uniqueness", CONTROL_WINDOW, settings.uniqueness_ratio)
    cv2.setTrackbarPos(
        "SGBM speckle window", CONTROL_WINDOW, settings.speckle_window_size
    )
    cv2.setTrackbarPos("SGBM speckle range", CONTROL_WINDOW, settings.speckle_range)
    cv2.setTrackbarPos("SGBM LR max diff", CONTROL_WINDOW, settings.disp12_max_diff)
    cv2.setTrackbarPos("SGBM P1 factor", CONTROL_WINDOW, settings.p1_factor)
    cv2.setTrackbarPos("SGBM P2 factor", CONTROL_WINDOW, settings.p2_factor)
    cv2.setTrackbarPos("SGBM WLS", CONTROL_WINDOW, int(settings.use_wls))


def set_vpi_controls(settings: VpiSettings) -> None:
    cv2.setTrackbarPos("Backend 0=SGBM 1=VPI", CONTROL_WINDOW, 1)
    cv2.setTrackbarPos(
        "VPI max disparity x16", CONTROL_WINDOW, settings.max_disparity // 16
    )
    cv2.setTrackbarPos("VPI min disparity", CONTROL_WINDOW, settings.min_disparity)
    cv2.setTrackbarPos(
        "VPI confidence percent",
        CONTROL_WINDOW,
        round(settings.confidence_threshold * 100 / 65535),
    )
    cv2.setTrackbarPos(
        "VPI unique pct (101=off)",
        CONTROL_WINDOW,
        101 if settings.uniqueness == -1.0 else round(settings.uniqueness * 100),
    )
    cv2.setTrackbarPos("VPI P1", CONTROL_WINDOW, settings.p1)
    cv2.setTrackbarPos("VPI P2", CONTROL_WINDOW, settings.p2)
    cv2.setTrackbarPos(
        "VPI diagonals", CONTROL_WINDOW, int(settings.include_diagonals)
    )


def add_label(image: np.ndarray, title: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (10, 10, 10), -1)
    cv2.putText(
        output, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
    )
    return output


def fit_pane(image: np.ndarray, width: int = 640, height: int = 360) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def text_panel(width: int, lines: list[str]) -> np.ndarray:
    height = 30 + 26 * len(lines)
    panel = np.full((height, width, 3), 18, dtype=np.uint8)
    for index, line in enumerate(lines):
        color = (80, 220, 255) if index == 0 else (230, 230, 230)
        cv2.putText(
            panel,
            line,
            (12, 24 + 26 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )
    return panel


def draw_roi(
    view: np.ndarray, center: tuple[int, int], roi_size: int, depth_shape: tuple[int, int]
) -> None:
    sx = view.shape[1] / depth_shape[1]
    sy = view.shape[0] / depth_shape[0]
    radius = roi_size // 2
    x, y = center
    p1 = (round((x - radius) * sx), round((y - radius) * sy))
    p2 = (round((x + radius) * sx), round((y + radius) * sy))
    cv2.rectangle(view, p1, p2, (255, 255, 255), 2)
    cv2.drawMarker(
        view,
        (round(x * sx), round(y * sy)),
        (255, 255, 255),
        cv2.MARKER_CROSS,
        16,
        2,
    )


def save_result(
    calibration,
    pair: CapturedPair,
    backend: str,
    settings,
    roi_size: int,
    known_distance: Optional[float],
    left_rectified: np.ndarray,
    right_rectified: np.ndarray,
    disparity: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    disparity_view: np.ndarray,
    depth_view: np.ndarray,
    alignment: np.ndarray,
    stats,
    elapsed_ms: float,
) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output = RESULTS_DIR / stamp
    output.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(output / "left_rectified.png"), left_rectified)
    cv2.imwrite(str(output / "right_rectified.png"), right_rectified)
    cv2.imwrite(str(output / "disparity_heatmap.png"), disparity_view)
    cv2.imwrite(str(output / "depth_heatmap.png"), depth_view)
    cv2.imwrite(str(output / "alignment_edges.png"), alignment)
    np.save(str(output / "disparity_float32.npy"), disparity.astype(np.float32))
    np.save(str(output / "depth_metres_float32.npy"), depth.astype(np.float32))
    np.save(str(output / "valid_mask.npy"), valid.astype(np.uint8))
    cloud = point_cloud_from_result(
        disparity, valid, left_rectified, calibration.q_matrix
    )
    save_binary_ply(output / "point_cloud.ply", cloud)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "calibration": {
            "path": str(calibration.path),
            "sha256": calibration.sha256,
            "resolution": [calibration.width, calibration.height],
            "sensor_mode": calibration.sensor_mode,
            "capture_fps": calibration.capture_fps,
            "baseline_m": calibration.baseline_m,
            "stereo_rms": calibration.stereo_rms,
        },
        "capture": {"sequence": pair.sequence, "host_pair_skew_ms": pair.skew_ms},
        "settings": settings_dict(backend, settings),
        "measurement": {
            **asdict(stats),
            "known_distance_m": known_distance,
            "roi_size_px": roi_size,
            "processing_ms": elapsed_ms,
            "whole_frame_valid_percentage": 100.0 * float(np.mean(valid)),
        },
    }
    save_json(output / "report.json", report)
    return output


def show_auto_progress(
    completed: int,
    total: int,
    settings: SgbmSettings,
    score: Optional[float],
    best_score: Optional[float],
) -> None:
    panel = np.full((260, 900, 3), 20, dtype=np.uint8)
    lines = [
        f"Easy Mode: testing {completed}/{total}",
        f"block={settings.block_size}, disparities={settings.num_disparities}, WLS={settings.use_wls}",
        "candidate rejected" if score is None else f"candidate score: {score:.5f}",
        "no valid candidate yet" if best_score is None else f"best score: {best_score:.5f}",
        "Please wait; the frozen stereo pair is unchanged.",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (24, 42 + index * 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (80, 230, 255) if index == 0 else (235, 235, 235),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    cv2.imshow(AUTO_WINDOW, panel)
    cv2.waitKey(1)


def show_vpi_auto_progress(
    completed: int,
    total: int,
    settings: VpiSettings,
    score: Optional[float],
    best_score: Optional[float],
) -> None:
    panel = np.full((300, 960, 3), 20, dtype=np.uint8)
    lines = [
        f"VPI CUDA Easy Mode: scoring {completed}/{total}",
        (
            f"P1/P2={settings.p1}/{settings.p2}, "
            f"unique={settings.uniqueness}, diag={settings.include_diagonals}"
        ),
        f"confidence threshold={settings.confidence_threshold}",
        "candidate rejected" if score is None else f"candidate score: {score:.5f}",
        "no valid candidate yet" if best_score is None else f"best score: {best_score:.5f}",
        f"Scoring {VPI_EASY_FRAME_COUNT} separately captured live stereo pairs.",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (24, 38 + index * 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (80, 230, 255) if index == 0 else (235, 235, 235),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    cv2.imshow(AUTO_WINDOW, panel)
    cv2.waitKey(1)


def run_easy_autotune(
    calibration,
    maps,
    pair: CapturedPair,
    center: tuple[int, int],
    roi_size: int,
    known_distance_m: float,
    output_root: Path = RESULTS_DIR,
    display_progress: bool = True,
):
    """Sweep safe SGBM candidates on one frozen, measured target pair."""
    left_rectified, right_rectified = rectify_pair(
        pair.left, pair.right, calibration, maps
    )
    raw_candidates = [
        SgbmSettings(num_disparities=num_disparities, block_size=block_size)
        for num_disparities in (160, 192, 256)
        for block_size in (3, 5, 7, 9, 11)
    ]
    total = len(raw_candidates) + 3
    records: list[dict] = []
    payloads: list[tuple[float, SgbmSettings, tuple]] = []

    def evaluate(settings: SgbmSettings) -> None:
        started = time.monotonic()
        disparity, disparity_valid = compute_sgbm(
            left_rectified, right_rectified, settings
        )
        depth, valid = depth_from_disparity(
            disparity, calibration.q_matrix, disparity_valid
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        statistics = roi_statistics(
            depth, valid, center, roi_size, known_distance_m
        )
        whole_valid = 100.0 * float(np.mean(valid))
        score = automatic_tuning_score(
            statistics, whole_valid, elapsed_ms, known_distance_m
        )
        # Preserve thin obstacle boundaries when two candidates are otherwise
        # nearly equal, and avoid selecting slower WLS merely for density.
        if score is not None:
            score += 0.002 * ((settings.block_size - 3) / 2)
            if settings.use_wls:
                score += 0.005
            payloads.append(
                (
                    score,
                    settings,
                    (
                        disparity,
                        depth,
                        valid,
                        elapsed_ms,
                        statistics,
                        whole_valid,
                    ),
                )
            )
        records.append(
            {
                "settings": settings_dict("opencv-sgbm", settings),
                "score": score,
                "processing_ms": elapsed_ms,
                "whole_frame_valid_percentage": whole_valid,
                "roi": asdict(statistics),
            }
        )
        current_best = min((item[0] for item in payloads), default=None)
        if display_progress:
            show_auto_progress(len(records), total, settings, score, current_best)
        median = "none" if statistics.median_m is None else f"{statistics.median_m:.3f}m"
        print(
            f"Auto {len(records):02d}/{total}: block={settings.block_size}, "
            f"disp={settings.num_disparities}, WLS={settings.use_wls} | "
            f"median={median}, ROI valid={statistics.valid_percentage:.1f}%, "
            f"score={score}"
        )

    if display_progress:
        cv2.namedWindow(AUTO_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(AUTO_WINDOW, 900, 260)
    for candidate in raw_candidates:
        evaluate(candidate)
    if not payloads:
        if display_progress:
            cv2.destroyWindow(AUTO_WINDOW)
        raise RuntimeError(
            "Easy Mode found no candidate with at least 20% valid target pixels"
        )

    best_raw = sorted(payloads, key=lambda item: item[0])[:3]
    for _score, candidate, _payload in best_raw:
        evaluate(replace(candidate, use_wls=True))

    payloads.sort(key=lambda item: item[0])
    best_score, best_settings, best_payload = payloads[0]
    disparity, depth, valid, elapsed_ms, statistics, whole_valid = best_payload
    disparity_view = make_disparity_view(
        disparity,
        np.isfinite(disparity)
        & (disparity > best_settings.min_disparity)
        & (disparity < best_settings.maximum_disparity),
        best_settings.min_disparity,
        best_settings.maximum_disparity,
    )
    depth_view = make_depth_view(depth, valid)
    alignment = make_alignment_overlay(left_rectified, right_rectified)

    stamp = datetime.now().strftime("auto_%Y-%m-%d_%H-%M-%S_%f")
    output = output_root / stamp
    output.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(output / "best_left_rectified.png"), left_rectified)
    cv2.imwrite(str(output / "best_right_rectified.png"), right_rectified)
    cv2.imwrite(str(output / "best_disparity_heatmap.png"), disparity_view)
    cv2.imwrite(str(output / "best_depth_heatmap.png"), depth_view)
    cv2.imwrite(str(output / "alignment_edges.png"), alignment)
    np.save(str(output / "best_disparity_float32.npy"), disparity.astype(np.float32))
    np.save(str(output / "best_depth_metres_float32.npy"), depth.astype(np.float32))
    np.save(str(output / "best_valid_mask.npy"), valid.astype(np.uint8))
    save_binary_ply(
        output / "best_point_cloud.ply",
        point_cloud_from_result(
            disparity, valid, left_rectified, calibration.q_matrix
        ),
    )
    ranked = sorted(
        records,
        key=lambda item: float("inf") if item["score"] is None else item["score"],
    )
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "Measured flat-target SGBM automatic sweep",
        "warning": "Recommendation is valid only after live checks at all navigation distances.",
        "calibration": {
            "path": str(calibration.path),
            "sha256": calibration.sha256,
            "resolution": [calibration.width, calibration.height],
            "baseline_m": calibration.baseline_m,
        },
        "capture": {"sequence": pair.sequence, "host_pair_skew_ms": pair.skew_ms},
        "known_distance_m": known_distance_m,
        "roi_center": list(center),
        "roi_size_px": roi_size,
        "best": {
            "score": best_score,
            "settings": settings_dict("opencv-sgbm", best_settings),
            "roi": asdict(statistics),
            "processing_ms": elapsed_ms,
            "whole_frame_valid_percentage": whole_valid,
        },
        "ranked_candidates": ranked,
    }
    save_json(output / "autotune_report.json", report)
    if display_progress:
        show_auto_progress(total, total, best_settings, best_score, best_score)
        cv2.waitKey(250)
        cv2.destroyWindow(AUTO_WINDOW)
    computed = (
        pair,
        "opencv-sgbm",
        best_settings,
        left_rectified,
        right_rectified,
        disparity,
        depth,
        valid,
        disparity_view,
        depth_view,
        alignment,
        elapsed_ms,
        None,
    )
    return best_settings, computed, output, best_score


def run_vpi_easy_autotune(
    calibration,
    maps,
    pairs: list[CapturedPair],
    center: tuple[int, int],
    roi_size: int,
    known_distance_m: float,
    output_root: Path = RESULTS_DIR,
    display_progress: bool = True,
    production_profile_path: Optional[Path] = PRODUCTION_VPI_PROFILE,
):
    """Tune native VPI controls across several live safety-check frames."""
    if len(pairs) < 3:
        raise ValueError("VPI Easy Mode requires at least three live stereo pairs")
    rectified_pairs = [
        (pair, *rectify_pair(pair.left, pair.right, calibration, maps))
        for pair in pairs
    ]
    candidates = vpi_autotune_candidates(max_disparity=256)
    total = len(candidates)
    records: list[dict] = []
    best: Optional[tuple[float, VpiSettings, object]] = None
    completed = 0

    if display_progress:
        cv2.namedWindow(AUTO_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(AUTO_WINDOW, 960, 300)

    for diagonals in VPI_AUTO_DIAGONAL_OPTIONS:
        matcher: Optional[VpiCudaMatcher] = None
        payload_error: Optional[str] = None
        for p1, p2 in VPI_AUTO_PENALTY_PAIRS:
            for uniqueness in VPI_AUTO_UNIQUENESS:
                base = VpiSettings(
                    min_disparity=0,
                    max_disparity=256,
                    confidence_threshold=1,
                    p1=p1,
                    p2=p2,
                    uniqueness=uniqueness,
                    include_diagonals=diagonals,
                )
                if matcher is None and payload_error is None:
                    try:
                        matcher = VpiCudaMatcher(
                            calibration.width, calibration.height, base
                        )
                        # Exclude one-time VPI payload creation from candidate timing.
                        matcher.compute(
                            rectified_pairs[0][1], rectified_pairs[0][2], base
                        )
                    except Exception as error:
                        payload_error = str(error)
                        matcher = None
                frame_payloads: list[tuple] = []
                try:
                    if payload_error is not None:
                        raise RuntimeError(payload_error)
                    assert matcher is not None
                    for captured, left_rectified, right_rectified in rectified_pairs:
                        started = time.monotonic()
                        disparity, plausible, confidence = matcher.compute(
                            left_rectified, right_rectified, base
                        )
                        elapsed_ms = (time.monotonic() - started) * 1000.0
                        frame_payloads.append(
                            (
                                captured,
                                disparity,
                                plausible,
                                confidence,
                                elapsed_ms,
                            )
                        )
                    compute_error = None
                except Exception as error:
                    compute_error = str(error)
                    frame_payloads = []

                group_best: Optional[tuple[float, int]] = None
                for candidate in (
                    item
                    for item in candidates
                    if item.include_diagonals == diagonals
                    and item.p1 == p1
                    and item.p2 == p2
                    and item.uniqueness == uniqueness
                ):
                    completed += 1
                    if compute_error is not None:
                        score = None
                        evaluation = None
                        records.append(
                            {
                                "settings": settings_dict("vpi-cuda", candidate),
                                "score": None,
                                "error": compute_error,
                            }
                        )
                    else:
                        roi_frames = []
                        zone_frames = []
                        frame_valid_percentages = []
                        processing_times_ms = []
                        for (
                            _captured,
                            disparity,
                            plausible,
                            confidence,
                            elapsed_ms,
                        ) in frame_payloads:
                            confidence_mask = confidence >= (
                                candidate.confidence_threshold
                            )
                            disparity_valid = plausible & confidence_mask
                            depth, valid = depth_from_disparity(
                                disparity,
                                calibration.q_matrix,
                                disparity_valid,
                            )
                            roi_frames.append(
                                roi_statistics(
                                    depth,
                                    valid,
                                    center,
                                    roi_size,
                                    known_distance_m,
                                )
                            )
                            zone_frames.append(
                                zone_safety_statistics(
                                    depth,
                                    valid,
                                    disparity,
                                    confidence_mask,
                                    candidate.max_disparity,
                                    candidate.disparity_safety_margin_px,
                                    known_distance_m,
                                )
                            )
                            frame_valid_percentages.append(
                                100.0 * float(np.mean(valid))
                            )
                            processing_times_ms.append(elapsed_ms)
                        evaluation = safety_tuning_score(
                            roi_frames,
                            zone_frames,
                            frame_valid_percentages,
                            processing_times_ms,
                            known_distance_m,
                        )
                        score = evaluation.score
                        if score is not None:
                            if group_best is None or score < group_best[0]:
                                group_best = (score, candidate.confidence_threshold)
                            if best is None or score < best[0]:
                                best = (score, candidate, evaluation)
                        records.append(
                            {
                                "settings": settings_dict("vpi-cuda", candidate),
                                "score": score,
                                "rejection_reason": evaluation.rejection_reason,
                                "processing_ms_mean": float(
                                    np.mean(processing_times_ms)
                                ),
                                "frame_count": len(frame_payloads),
                                "roi": asdict(evaluation.aggregate_roi),
                                "safety": asdict(evaluation),
                                "zone_frames": [
                                    [asdict(zone) for zone in frame]
                                    for frame in zone_frames
                                ],
                            }
                        )
                    best_score = None if best is None else best[0]
                    if display_progress:
                        show_vpi_auto_progress(
                            completed, total, candidate, score, best_score
                        )

                summary = (
                    "failed: " + compute_error
                    if compute_error is not None
                    else (
                        "no accepted confidence threshold"
                        if group_best is None
                        else f"best conf={group_best[1]}, score={group_best[0]:.5f}"
                    )
                )
                print(
                    f"VPI auto {completed:03d}/{total}: P1/P2={p1}/{p2}, "
                    f"unique={uniqueness}, diag={diagonals} | {summary}"
                )
        matcher = None
        gc.collect()

    if best is None:
        if display_progress:
            cv2.destroyWindow(AUTO_WINDOW)
        errors = [record.get("error") for record in records if record.get("error")]
        rejections = [
            record.get("rejection_reason")
            for record in records
            if record.get("rejection_reason")
        ]
        detail = (
            errors[0]
            if errors
            else (
                rejections[0]
                if rejections
                else "insufficient stable target coverage"
            )
        )
        raise RuntimeError(f"VPI Easy Mode found no valid candidate: {detail}")

    best_score, best_settings, _sweep_evaluation = best
    stamp = datetime.now().strftime("vpi_auto_%Y-%m-%d_%H-%M-%S_%f")
    output = output_root / stamp
    validation_output = output / "validation_frames"
    validation_output.mkdir(parents=True, exist_ok=False)

    best_matcher = VpiCudaMatcher(
        calibration.width, calibration.height, best_settings
    )
    best_matcher.compute(
        rectified_pairs[0][1], rectified_pairs[0][2], best_settings
    )
    roi_frames = []
    zone_frames = []
    frame_valid_percentages = []
    processing_times_ms = []
    final_frame_records = []
    for frame_index, (captured, left_rectified, right_rectified) in enumerate(
        rectified_pairs, start=1
    ):
        started = time.monotonic()
        disparity, plausible, confidence = best_matcher.compute(
            left_rectified, right_rectified, best_settings
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        confidence_mask = confidence >= best_settings.confidence_threshold
        disparity_valid = plausible & confidence_mask
        depth, valid = depth_from_disparity(
            disparity, calibration.q_matrix, disparity_valid
        )
        statistics = roi_statistics(
            depth, valid, center, roi_size, known_distance_m
        )
        zones = zone_safety_statistics(
            depth,
            valid,
            disparity,
            confidence_mask,
            best_settings.max_disparity,
            best_settings.disparity_safety_margin_px,
            known_distance_m,
        )
        whole_valid = 100.0 * float(np.mean(valid))
        roi_frames.append(statistics)
        zone_frames.append(zones)
        frame_valid_percentages.append(whole_valid)
        processing_times_ms.append(elapsed_ms)
        final_frame_records.append(
            {
                "sequence": captured.sequence,
                "host_pair_skew_ms": captured.skew_ms,
                "processing_ms": elapsed_ms,
                "whole_frame_valid_percentage": whole_valid,
                "roi": asdict(statistics),
                "zones": [asdict(zone) for zone in zones],
            }
        )
        prefix = validation_output / f"frame_{frame_index:02d}"
        cv2.imwrite(str(prefix.with_name(prefix.name + "_left.png")), left_rectified)
        cv2.imwrite(
            str(prefix.with_name(prefix.name + "_depth.png")),
            make_depth_view(depth, valid),
        )

    final_evaluation = safety_tuning_score(
        roi_frames,
        zone_frames,
        frame_valid_percentages,
        processing_times_ms,
        known_distance_m,
    )
    if final_evaluation.score is None:
        raise RuntimeError(
            "Best VPI candidate failed final multi-frame safety verification: "
            + str(final_evaluation.rejection_reason)
        )
    best_score = final_evaluation.score
    disparity_view = make_disparity_view(
        disparity,
        disparity_valid,
        best_settings.min_disparity,
        best_settings.max_disparity,
    )
    depth_view = make_depth_view(depth, valid)
    alignment = make_alignment_overlay(left_rectified, right_rectified)

    cv2.imwrite(str(output / "best_left_rectified.png"), left_rectified)
    cv2.imwrite(str(output / "best_right_rectified.png"), right_rectified)
    cv2.imwrite(str(output / "best_disparity_heatmap.png"), disparity_view)
    cv2.imwrite(str(output / "best_depth_heatmap.png"), depth_view)
    cv2.imwrite(str(output / "alignment_edges.png"), alignment)
    np.save(str(output / "best_disparity_float32.npy"), disparity.astype(np.float32))
    np.save(str(output / "best_depth_metres_float32.npy"), depth.astype(np.float32))
    np.save(str(output / "best_valid_mask.npy"), valid.astype(np.uint8))
    np.save(str(output / "best_confidence_u16.npy"), confidence.astype(np.uint16))
    save_binary_ply(
        output / "best_point_cloud.ply",
        point_cloud_from_result(
            disparity, valid, left_rectified, calibration.q_matrix
        ),
    )

    ranked = sorted(
        records,
        key=lambda item: float("inf") if item.get("score") is None else item["score"],
    )
    profile = {
        "schema_version": 2,
        "backend": "vpi-cuda",
        "calibration_sha256": calibration.sha256,
        "known_distance_m": known_distance_m,
        "score": best_score,
        "settings": asdict(best_settings),
    }
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "Multi-frame safety-scored VPI CUDA native-parameter sweep",
        "window_note": "VPI 3 CUDA uses a fixed 9x7 census window.",
        "warning": "Verify the profile live at every navigation threshold.",
        "calibration": {
            "path": str(calibration.path),
            "sha256": calibration.sha256,
            "resolution": [calibration.width, calibration.height],
            "baseline_m": calibration.baseline_m,
        },
        "captures": final_frame_records,
        "known_distance_m": known_distance_m,
        "roi_center": list(center),
        "roi_size_px": roi_size,
        "best": {
            "score": best_score,
            "settings": settings_dict("vpi-cuda", best_settings),
            "roi": asdict(final_evaluation.aggregate_roi),
            "temporal_roi_mad_m": final_evaluation.temporal_roi_mad_m,
            "processing_ms_mean": float(np.mean(processing_times_ms)),
            "whole_frame_valid_percentage_mean": (
                final_evaluation.mean_frame_valid_percentage
            ),
            "mean_supported_near_percentage": (
                final_evaluation.mean_supported_near_percentage
            ),
            "worst_supported_near_percentage": (
                final_evaluation.worst_supported_near_percentage
            ),
            "mean_near_limit_percentage": (
                final_evaluation.mean_near_limit_percentage
            ),
        },
        "ranked_candidates": ranked,
    }
    save_json(output / "autotune_report.json", report)
    save_json(output / "vpi_profile.json", profile)
    save_json(output_root / "vpi_recommended_profile.json", profile)
    if production_profile_path is not None:
        save_json(production_profile_path, profile)

    if display_progress:
        show_vpi_auto_progress(total, total, best_settings, best_score, best_score)
        cv2.waitKey(250)
        cv2.destroyWindow(AUTO_WINDOW)
    computed = (
        pairs[-1],
        "vpi-cuda",
        best_settings,
        left_rectified,
        right_rectified,
        disparity,
        depth,
        valid,
        disparity_view,
        depth_view,
        alignment,
        elapsed_ms,
        confidence,
    )
    return best_settings, computed, output, best_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--left-sensor", type=int, default=0)
    parser.add_argument("--right-sensor", type=int, default=1)
    parser.add_argument("--backend", choices=("sgbm", "vpi"), default="sgbm")
    parser.add_argument("--known-distance-m", type=float)
    parser.add_argument("--left-image", type=Path)
    parser.add_argument("--right-image", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.left_image is None) != (args.right_image is None):
        raise SystemExit("--left-image and --right-image must be supplied together")
    if args.known_distance_m is not None and not 0.1 <= args.known_distance_m <= 3.0:
        raise SystemExit("--known-distance-m must be between 0.1 and 3.0")
    calibration = load_calibration(args.calibration)
    maps = convert_rectification_maps(calibration)
    print(
        f"Calibration: {calibration.width}x{calibration.height}, "
        f"mode {calibration.sensor_mode}, {calibration.capture_fps} FPS, "
        f"baseline {calibration.baseline_m * 1000:.2f} mm"
    )
    print(
        "Expected disparity: "
        + ", ".join(
            f"{distance:.1f}m={expected_disparity(calibration.q_matrix, distance):.1f}px"
            for distance in (0.5, 1.0, 1.5, 2.0, 3.0)
        )
    )
    print("No actuator or YOLO modules are imported by this tool.")

    capture: Optional[StereoCapture] = None
    if args.left_image:
        left = cv2.imread(str(args.left_image), cv2.IMREAD_COLOR)
        right = cv2.imread(str(args.right_image), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise RuntimeError("Could not read the supplied offline image pair")
        pair = CapturedPair(1, 0.0, left, right)
        frozen = True
        offline = True
    else:
        check_gstreamer()
        left_camera = open_camera(args.left_sensor, calibration)
        try:
            right_camera = open_camera(args.right_sensor, calibration)
        except Exception:
            left_camera.release()
            raise
        capture = StereoCapture(left_camera, right_camera)
        capture.start()
        pair = capture.get()
        frozen = False
        offline = False

    create_controls(args.backend, args.known_distance_m)
    cv2.namedWindow(DASHBOARD_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DASHBOARD_WINDOW, 1280, 900)
    cv2.namedWindow(ALIGNMENT_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ALIGNMENT_WINDOW, 960, 540)

    inspection = [calibration.width // 2, calibration.height // 2]

    def mouse(event, x, y, _flags, _parameter):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Depth occupies the lower-right 640x360 pane of the 1280px dashboard.
        if 640 <= x < 1280 and 360 <= y < 720:
            inspection[0] = min(calibration.width - 1, max(0, (x - 640) * 2))
            inspection[1] = min(calibration.height - 1, max(0, (y - 360) * 2))

    cv2.setMouseCallback(DASHBOARD_WINDOW, mouse)
    print(
        "A automatic Easy Mode | SPACE freeze/live | B backend | "
        "P 3D projections | O Open3D snapshot | S save evidence | "
        "R reset ROI | Q quit"
    )

    vpi_matcher: Optional[VpiCudaMatcher] = None
    last_signature = None
    last_pair_sequence = -1
    computed = None
    status_message = "Ready"
    show_3d = False
    cloud_cache_key = None
    cloud_cache = None
    try:
        while True:
            if not frozen and capture is not None:
                pair = capture.get()
            backend, settings, roi_size, known_distance = read_controls()
            signature = (backend, settings, pair.sequence)
            if frozen:
                should_compute = signature != last_signature
            else:
                should_compute = pair.sequence != last_pair_sequence or signature != last_signature
            if should_compute:
                try:
                    left_rectified, right_rectified = rectify_pair(
                        pair.left, pair.right, calibration, maps
                    )
                    started = time.monotonic()
                    confidence = None
                    if backend == "opencv-sgbm":
                        assert isinstance(settings, SgbmSettings)
                        disparity, disparity_valid = compute_sgbm(
                            left_rectified, right_rectified, settings
                        )
                        minimum, maximum = (
                            settings.min_disparity,
                            settings.maximum_disparity,
                        )
                    else:
                        assert isinstance(settings, VpiSettings)
                        payload_key = (settings.max_disparity, settings.include_diagonals)
                        if vpi_matcher is None or vpi_matcher.payload_key != payload_key:
                            # Release the old payload before allocating the new one.  Jetson
                            # CPU/GPU memory is shared, so holding both can cause avoidable OOM.
                            vpi_matcher = None
                            gc.collect()
                            vpi_matcher = VpiCudaMatcher(
                                calibration.width, calibration.height, settings
                            )
                        disparity, disparity_valid, confidence = vpi_matcher.compute(
                            left_rectified, right_rectified, settings
                        )
                        minimum, maximum = settings.min_disparity, settings.max_disparity
                    depth, valid = depth_from_disparity(
                        disparity, calibration.q_matrix, disparity_valid
                    )
                    elapsed_ms = (time.monotonic() - started) * 1000.0
                    disparity_view = make_disparity_view(
                        disparity, disparity_valid, minimum, maximum
                    )
                    depth_view = make_depth_view(depth, valid)
                    alignment = make_alignment_overlay(left_rectified, right_rectified)
                    computed = (
                        pair,
                        backend,
                        settings,
                        left_rectified,
                        right_rectified,
                        disparity,
                        depth,
                        valid,
                        disparity_view,
                        depth_view,
                        alignment,
                        elapsed_ms,
                        confidence,
                    )
                    status_message = "Ready"
                    last_signature = signature
                    last_pair_sequence = pair.sequence
                except Exception as error:
                    status_message = f"ERROR: {error}"
                    print(status_message)
                    # Do not hammer CUDA/CPU with the same failing configuration.
                    # Moving a control or changing backend triggers another attempt.
                    frozen = True
                    last_signature = signature
                    last_pair_sequence = pair.sequence
                    vpi_matcher = None
                    gc.collect()

            if computed is not None:
                (
                    processed_pair,
                    shown_backend,
                    shown_settings,
                    left_rectified,
                    right_rectified,
                    disparity,
                    depth,
                    valid,
                    disparity_view,
                    depth_view,
                    alignment,
                    elapsed_ms,
                    _confidence,
                ) = computed
                stats = roi_statistics(
                    depth,
                    valid,
                    (inspection[0], inspection[1]),
                    roi_size,
                    known_distance,
                )
                current_cloud_key = (
                    processed_pair.sequence,
                    shown_backend,
                    shown_settings,
                )
                if show_3d:
                    if cloud_cache_key != current_cloud_key:
                        cloud_cache = point_cloud_from_result(
                            disparity,
                            valid,
                            left_rectified,
                            calibration.q_matrix,
                        )
                        cloud_cache_key = current_cloud_key
                    cv2.imshow(
                        POINT_CLOUD_WINDOW,
                        make_orthographic_view(cloud_cache),
                    )
                depth_pane = fit_pane(depth_view)
                draw_roi(depth_pane, tuple(inspection), roi_size, depth.shape)
                top = cv2.hconcat(
                    [
                        add_label(fit_pane(left_rectified), "Rectified LEFT"),
                        add_label(fit_pane(right_rectified), "Rectified RIGHT"),
                    ]
                )
                bottom = cv2.hconcat(
                    [
                        add_label(fit_pane(disparity_view), "Disparity"),
                        add_label(depth_pane, "Metric depth: near red | far blue"),
                    ]
                )
                median_text = (
                    "no valid depth"
                    if stats.median_m is None
                    else f"median {stats.median_m:.3f}m, MAD {stats.mad_m:.3f}m"
                )
                error_text = (
                    "known distance disabled"
                    if stats.error_m is None
                    else f"known error {stats.error_m:+.3f}m"
                )
                if shown_backend == "vpi-cuda":
                    setting_text = (
                        f"VPI max={shown_settings.max_disparity}, "
                        f"conf={shown_settings.confidence_threshold}, "
                        f"P1/P2={shown_settings.p1}/{shown_settings.p2}, "
                        f"unique={shown_settings.uniqueness}, diag={shown_settings.include_diagonals}"
                    )
                    warning = "VPI CUDA window is fixed 9x7; SGBM block slider has no effect"
                else:
                    setting_text = (
                        f"SGBM block={shown_settings.block_size}, "
                        f"range={shown_settings.min_disparity}..{shown_settings.maximum_disparity}, "
                        f"unique={shown_settings.uniqueness_ratio}, "
                        f"speckle={shown_settings.speckle_window_size}/{shown_settings.speckle_range}, "
                        f"WLS={shown_settings.use_wls}"
                    )
                    warning = "Tune on a frozen pair first; then verify temporal stability live"
                lines = [
                    f"{shown_backend} | {elapsed_ms:.1f} ms | valid {100*np.mean(valid):.1f}% | "
                    f"pair skew {processed_pair.skew_ms:.2f} ms | {'FROZEN' if frozen else 'LIVE'}",
                    setting_text,
                    f"ROI ({inspection[0]}, {inspection[1]}) {roi_size}px: {median_text}, "
                    f"valid {stats.valid_percentage:.1f}%, {error_text}",
                    warning,
                    f"{status_message} | A auto | P 3D | O Open3D | S save | Q quit",
                ]
                dashboard = cv2.vconcat([top, bottom, text_panel(1280, lines)])
                cv2.imshow(DASHBOARD_WINDOW, dashboard)
                cv2.imshow(ALIGNMENT_WINDOW, fit_pane(alignment, 960, 540))
            else:
                blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "No depth result yet",
                    (400, 330),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                dashboard = cv2.vconcat(
                    [
                        blank,
                        text_panel(
                            1280,
                            [
                                status_message,
                                "Change a control to retry | B backend | Q quit",
                            ],
                        ),
                    ]
                )
                cv2.imshow(DASHBOARD_WINDOW, dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and not offline:
                frozen = not frozen
                status_message = "Frozen pair" if frozen else "Live capture"
            elif key == ord("b"):
                current = position("Backend 0=SGBM 1=VPI")
                cv2.setTrackbarPos("Backend 0=SGBM 1=VPI", CONTROL_WINDOW, 1 - current)
            elif key == ord("r"):
                inspection[:] = [calibration.width // 2, calibration.height // 2]
            elif key == ord("p"):
                show_3d = not show_3d
                cloud_cache_key = None
                if not show_3d:
                    try:
                        cv2.destroyWindow(POINT_CLOUD_WINDOW)
                    except cv2.error:
                        pass
                status_message = (
                    "3D projections enabled"
                    if show_3d
                    else "3D projections disabled"
                )
            elif key == ord("o") and computed is not None:
                try:
                    cloud_cache = point_cloud_from_result(
                        disparity,
                        valid,
                        left_rectified,
                        calibration.q_matrix,
                    )
                    cloud_cache_key = current_cloud_key
                    show_open3d_snapshot(cloud_cache)
                    status_message = "Open3D snapshot closed"
                except Exception as error:
                    status_message = f"Open3D unavailable: {error}"
                    print(status_message)
            elif key == ord("a"):
                if known_distance is None:
                    status_message = (
                        "Easy Mode needs Known distance cm (use 100 for a 1m target)"
                    )
                    print(status_message)
                else:
                    vpi_matcher = None
                    gc.collect()
                    try:
                        if backend == "vpi-cuda":
                            if capture is None:
                                raise RuntimeError(
                                    "VPI Easy Mode requires live cameras; offline images "
                                    "cannot measure temporal stability"
                                )
                            status_message = (
                                f"Collecting {VPI_EASY_FRAME_COUNT} live stereo pairs"
                            )
                            print(status_message)
                            tuning_pairs = collect_live_tuning_pairs(capture)
                            pair = tuning_pairs[-1]
                            frozen = True
                            best_settings, computed, output, best_score = (
                                run_vpi_easy_autotune(
                                    calibration,
                                    maps,
                                    tuning_pairs,
                                    tuple(inspection),
                                    roi_size,
                                    known_distance,
                                )
                            )
                            set_vpi_controls(best_settings)
                            recommendation = (
                                f"VPI CUDA P1/P2={best_settings.p1}/{best_settings.p2}, "
                                f"unique={best_settings.uniqueness}, "
                                f"confidence={best_settings.confidence_threshold}, "
                                f"diagonals={best_settings.include_diagonals}"
                            )
                            selected_backend = "vpi-cuda"
                        else:
                            frozen = True
                            best_settings, computed, output, best_score = (
                                run_easy_autotune(
                                    calibration,
                                    maps,
                                    pair,
                                    tuple(inspection),
                                    roi_size,
                                    known_distance,
                                )
                            )
                            set_sgbm_controls(best_settings)
                            recommendation = (
                                f"SGBM block {best_settings.block_size}, "
                                f"{best_settings.num_disparities} disparities, "
                                f"WLS={best_settings.use_wls}"
                            )
                            selected_backend = "opencv-sgbm"
                        last_signature = (
                            selected_backend,
                            best_settings,
                            pair.sequence,
                        )
                        last_pair_sequence = pair.sequence
                        status_message = (
                            f"Easy Mode recommends {recommendation}; "
                            f"score {best_score:.5f}; "
                            f"saved {output}"
                        )
                        print(status_message)
                    except Exception as error:
                        status_message = f"Easy Mode failed: {error}"
                        print(status_message)
            elif key == ord("s") and computed is not None:
                output = save_result(
                    calibration,
                    processed_pair,
                    shown_backend,
                    shown_settings,
                    roi_size,
                    known_distance,
                    left_rectified,
                    right_rectified,
                    disparity,
                    depth,
                    valid,
                    disparity_view,
                    depth_view,
                    alignment,
                    stats,
                    elapsed_ms,
                )
                status_message = f"Saved {output}"
                print(status_message)
    finally:
        if capture is not None:
            capture.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
