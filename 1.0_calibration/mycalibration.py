#!/usr/bin/env python3
"""Simple IMX219 stereo capture and offline calibration.

Capture does not detect the checkerboard. Space always saves both images.
When Q is pressed, capture ends and calibration starts from the saved images.
"""

from __future__ import annotations

import sys
import time
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

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGES_ROOT = SCRIPT_DIR / "images"


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
def create_image_folders() -> tuple[Path, Path, Path]:
    session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    session_dir = IMAGES_ROOT / session_name
    left_dir = session_dir / "left"
    right_dir = session_dir / "right"

    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)
    return session_dir, left_dir, right_dir


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


def load_calibration_points(
    left_dir: Path,
    right_dir: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], tuple[int, int], list[tuple[Path, Path]]]:
    object_template = make_object_points()
    object_points: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    accepted_paths: list[tuple[Path, Path]] = []
    image_size: tuple[int, int] | None = None

    left_files = sorted(left_dir.glob("*.png"))
    if not left_files:
        raise RuntimeError(f"No saved images found in {left_dir}")

    print("\nDetecting checkerboards in saved images...")

    for left_path in left_files:
        right_path = right_dir / left_path.name
        if not right_path.is_file():
            print(f"Skipping {left_path.name}: right image is missing")
            continue

        left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)

        if left_image is None or right_image is None:
            print(f"Skipping {left_path.name}: image could not be loaded")
            continue

        current_left_size = (left_image.shape[1], left_image.shape[0])
        current_right_size = (right_image.shape[1], right_image.shape[0])
        expected_size = (WIDTH, HEIGHT)

        if current_left_size != expected_size or current_right_size != expected_size:
            print(f"Skipping {left_path.name}: resolution is not {WIDTH}x{HEIGHT}")
            continue

        found_left, corners_left = detect_checkerboard(left_image)
        found_right, corners_right = detect_checkerboard(right_image)

        if found_left and found_right:
            object_points.append(object_template.copy())
            left_points.append(corners_left)
            right_points.append(corners_right)
            accepted_paths.append((left_path, right_path))
            image_size = current_left_size
            print(f"Accepted {left_path.name}")
        else:
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

    return object_points, left_points, right_points, image_size, accepted_paths


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
    left_rms, left_matrix, left_distortion, _, _ = cv2.calibrateCamera(
        object_points,
        left_points,
        image_size,
        None,
        None,
    )

    print("Calibrating right camera...")
    right_rms, right_matrix, right_distortion, _, _ = cv2.calibrateCamera(
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
    }


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
    session_dir: Path,
    accepted_paths: list[tuple[Path, Path]],
    rectification: dict[str, np.ndarray],
) -> Path:
    left_image = cv2.imread(str(accepted_paths[0][0]), cv2.IMREAD_COLOR)
    right_image = cv2.imread(str(accepted_paths[0][1]), cv2.IMREAD_COLOR)

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

    preview = cv2.hconcat([left_rectified, right_rectified])
    for y in range(0, preview.shape[0], 80):
        cv2.line(preview, (0, y), (preview.shape[1], y), (0, 255, 0), 1)

    preview_path = session_dir / "rectified_preview.png"
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Could not save rectified preview: {preview_path}")
    return preview_path


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

        session_dir, left_dir, right_dir = create_image_folders()
        print(f"Session folder: {session_dir}")

        saved_count = capture_images(left_dir, right_dir)
        if saved_count == 0:
            print("No images were saved. Calibration was not started.")
            return 1

        object_points, left_points, right_points, image_size, accepted_paths = load_calibration_points(
            left_dir,
            right_dir,
        )

        calibration = calibrate_stereo(
            object_points,
            left_points,
            right_points,
            image_size,
        )
        rectification = create_rectification(calibration, image_size)

        preview_path = save_rectified_preview(
            session_dir,
            accepted_paths,
            rectification,
        )
        npz_path = save_npz(
            session_dir,
            image_size,
            len(object_points),
            calibration,
            rectification,
        )

        print("\nCalibration completed successfully.")
        print(f"Raw left images:  {left_dir}")
        print(f"Raw right images: {right_dir}")
        print(f"Rectified preview: {preview_path}")
        print(f"NPZ output: {npz_path}")
        return 0

    except (RuntimeError, cv2.error) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        print("Any images already captured were kept.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
