#!/usr/bin/env python3
"""IMX219 stereo capture, calibration, and visual quality evidence."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Jetson's system OpenCV includes GStreamer support.
SYSTEM_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
if SYSTEM_DIST_PACKAGES.is_dir():
    system_packages = str(SYSTEM_DIST_PACKAGES)
    if system_packages in sys.path:
        sys.path.remove(system_packages)
    sys.path.insert(0, system_packages)

import cv2
import numpy as np

from calibration_filter import (
    PairSample,
    QualityThresholds,
    filter_calibration_pairs,
)


# ============================================================
# SETTINGS — kept the same as requested
# ============================================================
LEFT_SENSOR_ID = 0
RIGHT_SENSOR_ID = 1

WIDTH = 1280
HEIGHT = 720
FPS = 30
SENSOR_MODE = 4

CHECKERBOARD = (9, 6)       # 9 x 6 inner corners
SQUARE_SIZE_M = 0.025       # 25 mm squares
CORNER_WINDOW = (11, 11)
MIN_VALID_PAIRS = 20
EPIPOLAR_WARNING_P95_PX = 1.0

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGES_ROOT = SCRIPT_DIR / "images"


@dataclass(frozen=True)
class SessionFolders:
    root: Path
    left_raw: Path
    right_raw: Path
    corners_left: Path
    corners_right: Path
    rejected_left: Path
    rejected_right: Path
    quality_rejected_left: Path
    quality_rejected_right: Path
    rectified: Path


# ============================================================
# PART 1 — OPEN BOTH CAMERAS
# ============================================================
def gstreamer_pipeline(sensor_id: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={SENSOR_MODE} ! "
        f"video/x-raw(memory:NVMM), width=(int){WIDTH}, height=(int){HEIGHT}, "
        f"format=(string)NV12, framerate=(fraction){FPS}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width=(int){WIDTH}, height=(int){HEIGHT}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def open_camera(sensor_id: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(
        gstreamer_pipeline(sensor_id),
        cv2.CAP_GSTREAMER,
    )
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera sensor-id={sensor_id}")
    return camera


def check_gstreamer() -> None:
    enabled = any(
        "GStreamer" in line and "YES" in line
        for line in cv2.getBuildInformation().splitlines()
    )
    if not enabled:
        raise RuntimeError(
            "This OpenCV build has no GStreamer support. "
            f"Loaded OpenCV from {cv2.__file__}"
        )


# ============================================================
# PART 2, 3 AND 4 — SPACE SAVES; Q ENDS; DATE/LEFT/RIGHT FOLDERS
# ============================================================
def create_image_folders() -> SessionFolders:
    session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    session_dir = IMAGES_ROOT / session_name
    folders = SessionFolders(
        root=session_dir,
        left_raw=session_dir / "left",
        right_raw=session_dir / "right",
        corners_left=session_dir / "detected_corners" / "left",
        corners_right=session_dir / "detected_corners" / "right",
        rejected_left=session_dir / "rejected" / "left",
        rejected_right=session_dir / "rejected" / "right",
        quality_rejected_left=session_dir / "quality_rejected" / "left",
        quality_rejected_right=session_dir / "quality_rejected" / "right",
        rectified=session_dir / "rectified_validation",
    )
    for directory in (
        folders.left_raw,
        folders.right_raw,
        folders.corners_left,
        folders.corners_right,
        folders.rejected_left,
        folders.rejected_right,
        folders.quality_rejected_left,
        folders.quality_rejected_right,
        folders.rectified,
    ):
        directory.mkdir(parents=True)
    return folders


def add_status_panel(
    camera_view: np.ndarray,
    saved_pairs: int,
    message: str,
) -> np.ndarray:
    """Add status information below, without covering camera pixels."""
    panel_height = 82
    panel = np.zeros((panel_height, camera_view.shape[1], 3), dtype=np.uint8)

    cv2.putText(
        panel,
        "LEFT CAMERA",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "RIGHT CAMERA",
        (PREVIEW_WIDTH + 15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"Saved pairs: {saved_pairs} | {message}",
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return cv2.vconcat([camera_view, panel])


def capture_images(
    left_dir: Path,
    right_dir: Path,
) -> int:
    left_camera = None
    right_camera = None
    saved_pairs = 0
    saved_message_until = 0.0

    try:
        left_camera = open_camera(LEFT_SENSOR_ID)
        right_camera = open_camera(RIGHT_SENSOR_ID)

        print(f"Left camera:  sensor-id={LEFT_SENSOR_ID}")
        print(f"Right camera: sensor-id={RIGHT_SENSOR_ID}")
        print(f"Capture: {WIDTH}x{HEIGHT}, mode {SENSOR_MODE}, {FPS} FPS")
        print("SPACE = save both images")
        print("Q = finish capture and start calibration")

        window_name = "Stereo Capture - LEFT | RIGHT"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, PREVIEW_WIDTH * 2, PREVIEW_HEIGHT + 82)

        # Allow camera exposure and white balance to settle.
        for _ in range(10):
            left_camera.grab()
            right_camera.grab()
            left_camera.retrieve()
            right_camera.retrieve()

        while True:
            # Grab both first, then retrieve, to reduce left/right time offset.
            grabbed_left = left_camera.grab()
            grabbed_right = right_camera.grab()
            ok_left, frame_left = left_camera.retrieve()
            ok_right, frame_right = right_camera.retrieve()

            if not grabbed_left or not grabbed_right or not ok_left or not ok_right:
                print("Could not read both cameras.")
                continue

            expected_shape = (HEIGHT, WIDTH)
            if frame_left.shape[:2] != expected_shape or frame_right.shape[:2] != expected_shape:
                raise RuntimeError(
                    f"Expected {WIDTH}x{HEIGHT}; got left "
                    f"{frame_left.shape[1]}x{frame_left.shape[0]} and right "
                    f"{frame_right.shape[1]}x{frame_right.shape[0]}"
                )

            left_preview = cv2.resize(
                frame_left,
                (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            right_preview = cv2.resize(
                frame_right,
                (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

            message = "SPACE: save pair | Q: finish and calibrate"
            if time.monotonic() < saved_message_until:
                message = f"SAVED PAIR #{saved_pairs}"

            camera_view = cv2.hconcat([left_preview, right_preview])
            combined = add_status_panel(camera_view, saved_pairs, message)
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # Space
                saved_pairs += 1
                filename = f"{saved_pairs:04d}.png"
                left_path = left_dir / filename
                right_path = right_dir / filename

                left_saved = cv2.imwrite(str(left_path), frame_left)
                right_saved = cv2.imwrite(str(right_path), frame_right)

                if not left_saved or not right_saved:
                    raise RuntimeError(f"Failed to save stereo pair #{saved_pairs}")

                saved_message_until = time.monotonic() + 0.8
                print(f"Saved pair #{saved_pairs}: {filename}")

            elif key == ord("q"):
                print("Capture finished. Starting offline calibration.")
                break

    finally:
        if left_camera is not None:
            left_camera.release()
        if right_camera is not None:
            right_camera.release()
        cv2.destroyAllWindows()

    return saved_pairs


# ============================================================
# PART 5 — LOAD SAVED IMAGES AND DETECT CHECKERBOARDS
# ============================================================
def make_object_points() -> np.ndarray:
    points = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    points[:, :2] = np.mgrid[
        0 : CHECKERBOARD[0],
        0 : CHECKERBOARD[1],
    ].T.reshape(-1, 2)
    points *= SQUARE_SIZE_M
    return points


def detect_checkerboard(image: np.ndarray) -> tuple[bool, np.ndarray | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

    if found:
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners = cv2.cornerSubPix(
            gray,
            corners,
            CORNER_WINDOW,
            (-1, -1),
            criteria,
        )

    return found, corners


def annotated_checkerboard(
    image: np.ndarray,
    found: bool,
    corners: np.ndarray | None,
    label: str,
) -> np.ndarray:
    """Return an evidence image without modifying the raw capture."""
    output = image.copy()
    if corners is not None:
        cv2.drawChessboardCorners(output, CHECKERBOARD, corners, found)
    colour = (0, 190, 0) if found else (0, 0, 220)
    cv2.rectangle(output, (0, 0), (output.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (14, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        colour,
        2,
        cv2.LINE_AA,
    )
    return output


def save_detection_evidence(
    folders: SessionFolders,
    filename: str,
    left_image: np.ndarray,
    right_image: np.ndarray,
    found_left: bool,
    found_right: bool,
    corners_left: np.ndarray | None,
    corners_right: np.ndarray | None,
) -> None:
    accepted = found_left and found_right
    status = "ACCEPTED" if accepted else "REJECTED"
    left_output = annotated_checkerboard(
        left_image,
        found_left,
        corners_left,
        f"{status} LEFT | corners {'found' if found_left else 'missing'}",
    )
    right_output = annotated_checkerboard(
        right_image,
        found_right,
        corners_right,
        f"{status} RIGHT | corners {'found' if found_right else 'missing'}",
    )
    left_dir = folders.corners_left if accepted else folders.rejected_left
    right_dir = folders.corners_right if accepted else folders.rejected_right
    if not cv2.imwrite(str(left_dir / filename), left_output):
        raise RuntimeError(f"Could not save corner evidence for {filename} left")
    if not cv2.imwrite(str(right_dir / filename), right_output):
        raise RuntimeError(f"Could not save corner evidence for {filename} right")


def make_pair_samples(
    object_points: list[np.ndarray],
    left_points: list[np.ndarray],
    right_points: list[np.ndarray],
    accepted_paths: list[tuple[Path, Path]],
) -> list[PairSample]:
    return [
        PairSample(
            filename=left_path.name,
            object_points=objects,
            left_corners=left_corners,
            right_corners=right_corners,
            left_path=left_path,
            right_path=right_path,
        )
        for objects, left_corners, right_corners, (left_path, right_path) in zip(
            object_points,
            left_points,
            right_points,
            accepted_paths,
        )
    ]


def save_quality_rejection_evidence(
    folders: SessionFolders,
    rejected: tuple[dict[str, object], ...],
    all_samples: list[PairSample],
) -> None:
    samples_by_name = {sample.filename: sample for sample in all_samples}
    for record in rejected:
        filename = str(record["filename"])
        sample = samples_by_name[filename]
        left_image = cv2.imread(str(sample.left_path), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(sample.right_path), cv2.IMREAD_COLOR)
        if left_image is None or right_image is None:
            raise RuntimeError(
                f"Could not reload automatically rejected pair {filename}"
            )
        stage = str(record.get("stage", "quality filter"))
        left_output = annotated_checkerboard(
            left_image,
            True,
            sample.left_corners,
            f"AUTO REJECTED LEFT | {stage}",
        )
        right_output = annotated_checkerboard(
            right_image,
            True,
            sample.right_corners,
            f"AUTO REJECTED RIGHT | {stage}",
        )
        if not cv2.imwrite(
            str(folders.quality_rejected_left / filename), left_output
        ):
            raise RuntimeError(f"Could not save quality rejection: {filename}")
        if not cv2.imwrite(
            str(folders.quality_rejected_right / filename), right_output
        ):
            raise RuntimeError(f"Could not save quality rejection: {filename}")


def apply_quality_rejections_to_records(
    records: list[dict[str, object]],
    rejected: tuple[dict[str, object], ...],
) -> None:
    records_by_name = {str(record["filename"]): record for record in records}
    for rejection in rejected:
        filename = str(rejection["filename"])
        record = records_by_name[filename]
        record["accepted"] = False
        record["reason"] = "automatic quality rejection: " + str(
            rejection["reason"]
        )
        record["quality_rejection_stage"] = rejection["stage"]


def load_calibration_points(
    folders: SessionFolders,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    tuple[int, int],
    list[tuple[Path, Path]],
    list[dict[str, object]],
]:
    object_template = make_object_points()
    object_points: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    accepted_paths: list[tuple[Path, Path]] = []
    records: list[dict[str, object]] = []
    image_size: tuple[int, int] | None = None

    left_files = sorted(folders.left_raw.glob("*.png"))
    if not left_files:
        raise RuntimeError(f"No saved images found in {folders.left_raw}")

    print("\nDetecting checkerboards in saved images...")

    for left_path in left_files:
        right_path = folders.right_raw / left_path.name
        if not right_path.is_file():
            print(f"Skipping {left_path.name}: right image is missing")
            records.append(
                {
                    "filename": left_path.name,
                    "accepted": False,
                    "reason": "right image missing",
                }
            )
            continue

        left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)

        if left_image is None or right_image is None:
            print(f"Skipping {left_path.name}: image could not be loaded")
            records.append(
                {
                    "filename": left_path.name,
                    "accepted": False,
                    "reason": "image could not be loaded",
                }
            )
            continue

        current_left_size = (left_image.shape[1], left_image.shape[0])
        current_right_size = (right_image.shape[1], right_image.shape[0])
        expected_size = (WIDTH, HEIGHT)

        if current_left_size != expected_size or current_right_size != expected_size:
            print(f"Skipping {left_path.name}: resolution is not {WIDTH}x{HEIGHT}")
            records.append(
                {
                    "filename": left_path.name,
                    "accepted": False,
                    "reason": "resolution mismatch",
                    "left_size": list(current_left_size),
                    "right_size": list(current_right_size),
                }
            )
            continue

        found_left, corners_left = detect_checkerboard(left_image)
        found_right, corners_right = detect_checkerboard(right_image)
        save_detection_evidence(
            folders,
            left_path.name,
            left_image,
            right_image,
            found_left,
            found_right,
            corners_left,
            corners_right,
        )

        if found_left and found_right:
            object_points.append(object_template.copy())
            left_points.append(corners_left)
            right_points.append(corners_right)
            accepted_paths.append((left_path, right_path))
            image_size = current_left_size
            records.append(
                {
                    "filename": left_path.name,
                    "accepted": True,
                    "reason": "checkerboard found in both images",
                    "corner_count": int(corners_left.shape[0]),
                }
            )
            print(f"Accepted {left_path.name}")
        else:
            records.append(
                {
                    "filename": left_path.name,
                    "accepted": False,
                    "reason": "checkerboard missing from one or both images",
                    "left_found": bool(found_left),
                    "right_found": bool(found_right),
                }
            )
            print(
                f"Rejected {left_path.name}: "
                f"left={'found' if found_left else 'not found'}, "
                f"right={'found' if found_right else 'not found'}"
            )

    if image_size is None:
        raise RuntimeError("Checkerboard was not detected in any stereo pair")

    print(
        f"Checkerboard detected in {len(object_points)} of "
        f"{len(left_files)} saved pairs."
    )

    if len(object_points) < MIN_VALID_PAIRS:
        raise RuntimeError(
            f"Need at least {MIN_VALID_PAIRS} valid stereo pairs, but only "
            f"{len(object_points)} passed checkerboard detection. Raw images were kept."
        )

    return (
        object_points,
        left_points,
        right_points,
        image_size,
        accepted_paths,
        records,
    )


# ============================================================
# PART 6 — CALIBRATE AND OUTPUT THE MATRICES
# ============================================================
def print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"\n{name} =")
    print(matrix)


def calibrate_stereo(
    object_points: list[np.ndarray],
    left_points: list[np.ndarray],
    right_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> dict[str, np.ndarray | float]:
    print("\nCalibrating left camera...")
    left_rms, left_matrix, left_distortion, left_rvecs, left_tvecs = cv2.calibrateCamera(
        object_points,
        left_points,
        image_size,
        None,
        None,
    )

    print("Calibrating right camera...")
    right_rms, right_matrix, right_distortion, right_rvecs, right_tvecs = cv2.calibrateCamera(
        object_points,
        right_points,
        image_size,
        None,
        None,
    )

    print("Calibrating stereo cameras...")
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5,
    )
    stereo_rms, left_matrix, left_distortion, right_matrix, right_distortion, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        left_matrix,
        left_distortion,
        right_matrix,
        right_distortion,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    baseline_m = float(np.linalg.norm(translation))

    print(f"\nLeft RMS:   {left_rms:.6f} pixels")
    print(f"Right RMS:  {right_rms:.6f} pixels")
    print(f"Stereo RMS: {stereo_rms:.6f} pixels")
    print(f"Baseline:   {baseline_m:.6f} m")

    print_matrix("Left camera matrix", left_matrix)
    print_matrix("Left distortion", left_distortion)
    print_matrix("Right camera matrix", right_matrix)
    print_matrix("Right distortion", right_distortion)
    print_matrix("Rotation R", rotation)
    print_matrix("Translation T", translation)
    print_matrix("Essential E", essential)
    print_matrix("Fundamental F", fundamental)

    return {
        "left_rms": float(left_rms),
        "right_rms": float(right_rms),
        "stereo_rms": float(stereo_rms),
        "left_matrix": left_matrix,
        "left_distortion": left_distortion,
        "right_matrix": right_matrix,
        "right_distortion": right_distortion,
        "rotation": rotation,
        "translation": translation,
        "essential": essential,
        "fundamental": fundamental,
        "baseline_m": baseline_m,
        "left_rvecs": left_rvecs,
        "left_tvecs": left_tvecs,
        "right_rvecs": right_rvecs,
        "right_tvecs": right_tvecs,
    }


def per_view_reprojection_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for objects, observed, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(
            objects, rvec, tvec, camera_matrix, distortion
        )
        difference = observed.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(difference * difference, axis=1)))))
    return errors


def epipolar_errors(
    left_corners: np.ndarray,
    right_corners: np.ndarray,
    fundamental: np.ndarray,
) -> np.ndarray:
    """Return symmetric point-to-epipolar-line errors in pixels."""
    left = left_corners.reshape(-1, 2)
    right = right_corners.reshape(-1, 2)
    right_lines = cv2.computeCorrespondEpilines(
        left.reshape(-1, 1, 2), 1, fundamental
    ).reshape(-1, 3)
    left_lines = cv2.computeCorrespondEpilines(
        right.reshape(-1, 1, 2), 2, fundamental
    ).reshape(-1, 3)

    def distances(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
        numerator = np.abs(
            lines[:, 0] * points[:, 0]
            + lines[:, 1] * points[:, 1]
            + lines[:, 2]
        )
        denominator = np.hypot(lines[:, 0], lines[:, 1])
        return numerator / np.maximum(denominator, 1e-12)

    return 0.5 * (distances(right, right_lines) + distances(left, left_lines))


def rectified_vertical_errors(
    left_corners: np.ndarray,
    right_corners: np.ndarray,
    calibration: dict[str, np.ndarray | float],
    rectification: dict[str, np.ndarray],
) -> np.ndarray:
    left_rectified = cv2.undistortPoints(
        left_corners,
        calibration["left_matrix"],
        calibration["left_distortion"],
        R=rectification["r1"],
        P=rectification["p1"],
    ).reshape(-1, 2)
    right_rectified = cv2.undistortPoints(
        right_corners,
        calibration["right_matrix"],
        calibration["right_distortion"],
        R=rectification["r2"],
        P=rectification["p2"],
    ).reshape(-1, 2)
    return np.abs(left_rectified[:, 1] - right_rectified[:, 1])


# ============================================================
# PART 7 AND 8 — RECTIFY AND OUTPUT THE STEREO MAPS
# ============================================================
def create_rectification(
    calibration: dict[str, np.ndarray | float],
    image_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    print("\nCreating stereo rectification...")

    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        calibration["left_matrix"],
        calibration["left_distortion"],
        calibration["right_matrix"],
        calibration["right_distortion"],
        image_size,
        calibration["rotation"],
        calibration["translation"],
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        calibration["left_matrix"],
        calibration["left_distortion"],
        r1,
        p1,
        image_size,
        cv2.CV_32FC1,
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        calibration["right_matrix"],
        calibration["right_distortion"],
        r2,
        p2,
        image_size,
        cv2.CV_32FC1,
    )

    print_matrix("Left rectification R1", r1)
    print_matrix("Right rectification R2", r2)
    print_matrix("Left projection P1", p1)
    print_matrix("Right projection P2", p2)
    print_matrix("Disparity-to-depth Q", q)

    # The maps contain millions of values, so print their useful description
    # and store the complete arrays in the NPZ file below.
    print("\nStereo map output:")
    print(f"left_map1:  shape={left_map1.shape}, dtype={left_map1.dtype}")
    print(f"left_map2:  shape={left_map2.shape}, dtype={left_map2.dtype}")
    print(f"right_map1: shape={right_map1.shape}, dtype={right_map1.dtype}")
    print(f"right_map2: shape={right_map2.shape}, dtype={right_map2.dtype}")

    return {
        "r1": r1,
        "r2": r2,
        "p1": p1,
        "p2": p2,
        "q": q,
        "roi1": np.asarray(roi1),
        "roi2": np.asarray(roi2),
        "left_map1": left_map1,
        "left_map2": left_map2,
        "right_map1": right_map1,
        "right_map2": right_map2,
    }


def save_rectified_preview(
    folders: SessionFolders,
    accepted_paths: list[tuple[Path, Path]],
    left_points: list[np.ndarray],
    right_points: list[np.ndarray],
    calibration: dict[str, np.ndarray | float],
    rectification: dict[str, np.ndarray],
) -> tuple[Path, list[dict[str, float | str]]]:
    pair_metrics: list[dict[str, float | str]] = []
    preview_path = folders.root / "rectified_preview.png"
    for index, ((left_path, right_path), corners_left, corners_right) in enumerate(
        zip(accepted_paths, left_points, right_points)
    ):
        left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_image is None or right_image is None:
            raise RuntimeError(f"Could not reload accepted pair {left_path.name}")
        left_rectified = cv2.remap(
            left_image,
            rectification["left_map1"],
            rectification["left_map2"],
            cv2.INTER_LINEAR,
        )
        right_rectified = cv2.remap(
            right_image,
            rectification["right_map1"],
            rectification["right_map2"],
            cv2.INTER_LINEAR,
        )
        transformed_left = cv2.undistortPoints(
            corners_left,
            calibration["left_matrix"],
            calibration["left_distortion"],
            R=rectification["r1"],
            P=rectification["p1"],
        )
        transformed_right = cv2.undistortPoints(
            corners_right,
            calibration["right_matrix"],
            calibration["right_distortion"],
            R=rectification["r2"],
            P=rectification["p2"],
        )
        cv2.drawChessboardCorners(
            left_rectified, CHECKERBOARD, transformed_left, True
        )
        cv2.drawChessboardCorners(
            right_rectified, CHECKERBOARD, transformed_right, True
        )
        preview = cv2.hconcat([left_rectified, right_rectified])
        for y in range(0, preview.shape[0], 60):
            cv2.line(preview, (0, y), (preview.shape[1], y), (0, 255, 255), 1)
        output_path = folders.rectified / left_path.name
        if not cv2.imwrite(str(output_path), preview):
            raise RuntimeError(f"Could not save rectified evidence: {output_path}")
        if index == 0 and not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"Could not save rectified preview: {preview_path}")

        vertical = rectified_vertical_errors(
            corners_left, corners_right, calibration, rectification
        )
        pair_metrics.append(
            {
                "filename": left_path.name,
                "rectified_vertical_mean_px": float(np.mean(vertical)),
                "rectified_vertical_p95_px": float(np.percentile(vertical, 95)),
                "rectified_vertical_max_px": float(np.max(vertical)),
            }
        )
    return preview_path, pair_metrics


def build_calibration_report(
    folders: SessionFolders,
    object_points: list[np.ndarray],
    left_points: list[np.ndarray],
    right_points: list[np.ndarray],
    accepted_paths: list[tuple[Path, Path]],
    detection_records: list[dict[str, object]],
    calibration: dict[str, np.ndarray | float],
    rectified_metrics: list[dict[str, float | str]],
) -> dict[str, object]:
    left_errors = per_view_reprojection_errors(
        object_points,
        left_points,
        calibration["left_rvecs"],
        calibration["left_tvecs"],
        calibration["left_matrix"],
        calibration["left_distortion"],
    )
    right_errors = per_view_reprojection_errors(
        object_points,
        right_points,
        calibration["right_rvecs"],
        calibration["right_tvecs"],
        calibration["right_matrix"],
        calibration["right_distortion"],
    )
    rectified_by_name = {item["filename"]: item for item in rectified_metrics}
    accepted_metrics: list[dict[str, object]] = []
    all_epipolar: list[np.ndarray] = []
    for index, ((left_path, _), corners_left, corners_right) in enumerate(
        zip(accepted_paths, left_points, right_points)
    ):
        epipolar = epipolar_errors(
            corners_left, corners_right, calibration["fundamental"]
        )
        all_epipolar.append(epipolar)
        rectified = rectified_by_name[left_path.name]
        accepted_metrics.append(
            {
                "filename": left_path.name,
                "left_reprojection_rms_px": left_errors[index],
                "right_reprojection_rms_px": right_errors[index],
                "epipolar_mean_px": float(np.mean(epipolar)),
                "epipolar_p95_px": float(np.percentile(epipolar, 95)),
                **rectified,
            }
        )

    vertical_p95_values = [
        float(item["rectified_vertical_p95_px"]) for item in rectified_metrics
    ]
    epipolar_values = np.concatenate(all_epipolar)
    rectified_p95 = float(np.percentile(vertical_p95_values, 95))
    warnings: list[str] = []
    if rectified_p95 > EPIPOLAR_WARNING_P95_PX:
        warnings.append(
            f"Rectified vertical P95 {rectified_p95:.3f}px exceeds "
            f"the {EPIPOLAR_WARNING_P95_PX:.3f}px review threshold."
        )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session": str(folders.root),
        "capture": {
            "resolution": [WIDTH, HEIGHT],
            "fps": FPS,
            "sensor_mode": SENSOR_MODE,
            "checkerboard_inner_corners": list(CHECKERBOARD),
            "square_size_m": SQUARE_SIZE_M,
        },
        "summary": {
            "saved_pairs": len(detection_records),
            "accepted_pairs": len(accepted_paths),
            "rejected_pairs": sum(
                1 for record in detection_records if not record.get("accepted", False)
            ),
            "left_rms_px": calibration["left_rms"],
            "right_rms_px": calibration["right_rms"],
            "stereo_rms_px": calibration["stereo_rms"],
            "baseline_m": calibration["baseline_m"],
            "epipolar_mean_px": float(np.mean(epipolar_values)),
            "epipolar_p95_px": float(np.percentile(epipolar_values, 95)),
            "rectified_vertical_pair_p95_px": rectified_p95,
        },
        "warnings": warnings,
        "detections": detection_records,
        "accepted_pair_metrics": accepted_metrics,
    }


def save_calibration_report(session_dir: Path, report: dict[str, object]) -> Path:
    path = session_dir / "calibration_report.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


# ============================================================
# PART 9 — SAVE EVERYTHING TO ONE NPZ FILE
# ============================================================
def save_npz(
    session_dir: Path,
    image_size: tuple[int, int],
    valid_pair_count: int,
    calibration: dict[str, np.ndarray | float],
    rectification: dict[str, np.ndarray],
) -> Path:
    output_path = session_dir / "stereo_calibration.npz"

    np.savez_compressed(
        output_path,
        image_width=image_size[0],
        image_height=image_size[1],
        sensor_mode=SENSOR_MODE,
        capture_fps=FPS,
        checkerboard_columns=CHECKERBOARD[0],
        checkerboard_rows=CHECKERBOARD[1],
        square_size_m=SQUARE_SIZE_M,
        valid_pair_count=valid_pair_count,
        left_rms=calibration["left_rms"],
        right_rms=calibration["right_rms"],
        stereo_rms=calibration["stereo_rms"],
        camera_matrix_left=calibration["left_matrix"],
        dist_coeffs_left=calibration["left_distortion"],
        camera_matrix_right=calibration["right_matrix"],
        dist_coeffs_right=calibration["right_distortion"],
        R=calibration["rotation"],
        T=calibration["translation"],
        E=calibration["essential"],
        F=calibration["fundamental"],
        baseline_m=calibration["baseline_m"],
        R1=rectification["r1"],
        R2=rectification["r2"],
        P1=rectification["p1"],
        P2=rectification["p2"],
        Q=rectification["q"],
        roi1=rectification["roi1"],
        roi2=rectification["roi2"],
        left_map1=rectification["left_map1"],
        left_map2=rectification["left_map2"],
        right_map1=rectification["right_map1"],
        right_map2=rectification["right_map2"],
    )

    return output_path


def main() -> int:
    try:
        check_gstreamer()

        folders = create_image_folders()
        print(f"Session folder: {folders.root}")

        saved_count = capture_images(folders.left_raw, folders.right_raw)
        if saved_count == 0:
            print("No images were saved. Calibration was not started.")
            return 1

        (
            object_points,
            left_points,
            right_points,
            image_size,
            accepted_paths,
            detection_records,
        ) = load_calibration_points(folders)

        all_samples = make_pair_samples(
            object_points, left_points, right_points, accepted_paths
        )
        quality_thresholds = QualityThresholds()
        print("\nAutomatically rejecting weak calibration pairs...")
        filtered = filter_calibration_pairs(
            all_samples,
            image_size,
            MIN_VALID_PAIRS,
            quality_thresholds,
        )
        save_quality_rejection_evidence(
            folders, filtered.rejected, all_samples
        )
        apply_quality_rejections_to_records(
            detection_records, filtered.rejected
        )

        object_points = [sample.object_points for sample in filtered.accepted]
        left_points = [sample.left_corners for sample in filtered.accepted]
        right_points = [sample.right_corners for sample in filtered.accepted]
        accepted_paths = [
            (sample.left_path, sample.right_path)
            for sample in filtered.accepted
        ]
        calibration = filtered.calibration
        rectification = filtered.rectification

        print(
            f"Quality filter retained {len(filtered.accepted)} pairs and "
            f"rejected {len(filtered.rejected)}."
        )
        print(f"Left RMS:   {calibration['left_rms']:.6f} pixels")
        print(f"Right RMS:  {calibration['right_rms']:.6f} pixels")
        print(f"Stereo RMS: {calibration['stereo_rms']:.6f} pixels")
        print(f"Baseline:   {calibration['baseline_m']:.6f} m")
        for rejection in filtered.rejected:
            print(f"Rejected {rejection['filename']}: {rejection['reason']}")

        preview_path, rectified_metrics = save_rectified_preview(
            folders,
            accepted_paths,
            left_points,
            right_points,
            calibration,
            rectification,
        )
        npz_path = save_npz(
            folders.root,
            image_size,
            len(object_points),
            calibration,
            rectification,
        )
        report = build_calibration_report(
            folders,
            object_points,
            left_points,
            right_points,
            accepted_paths,
            detection_records,
            calibration,
            rectified_metrics,
        )
        report["automatic_quality_filter"] = {
            "thresholds": asdict(quality_thresholds),
            "history": list(filtered.history),
            "rejected_pairs": list(filtered.rejected),
        }
        report_path = save_calibration_report(folders.root, report)

        print("\nCalibration completed successfully.")
        print(f"Raw left images:  {folders.left_raw}")
        print(f"Raw right images: {folders.right_raw}")
        print(f"Corner evidence:  {folders.root / 'detected_corners'}")
        print(f"Rejected pairs:   {folders.root / 'rejected'}")
        print(f"Quality rejects:  {folders.root / 'quality_rejected'}")
        print(f"Rectified checks: {folders.rectified}")
        print(f"Rectified preview: {preview_path}")
        print(f"Quality report:    {report_path}")
        print(f"NPZ output: {npz_path}")
        return 0

    except (RuntimeError, cv2.error) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        print("Any images already captured were kept.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
