import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# =========================
# Camera settings
# =========================
LEFT_SENSOR_ID = 0
RIGHT_SENSOR_ID = 1
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30

# Capture remains 1280x720 so the saved calibration maps match the main app.
# Corner detection and preview are downscaled to reduce CPU/RAM pressure.
DETECTION_SCALE = 0.50
DISPLAY_SCALE = 0.50
DETECT_EVERY_N_FRAMES = 3

# =========================
# Checkerboard settings
# =========================
CHECKERBOARD = (9, 6)          # 9 x 6 inner corners
SQUARE_SIZE_M = 0.025          # 25 mm squares
MIN_PAIRS = 20
OUTPUT_FILE = Path(__file__).with_name("stereo_calibration_imx219.npz")


def gstreamer_pipeline(
    sensor_id: int,
    width: int = CAPTURE_WIDTH,
    height: int = CAPTURE_HEIGHT,
    framerate: int = CAPTURE_FPS,
    flip_method: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"queue max-size-buffers=1 leaky=downstream ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def open_camera(sensor_id: int) -> cv2.VideoCapture:
    pipeline = gstreamer_pipeline(sensor_id)
    print(f"Opening sensor-id={sensor_id}", flush=True)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open camera sensor-id={sensor_id}")
    return cap


def detect_corners_scaled(frame: np.ndarray):
    """Detect at reduced resolution, then scale corners back to 1280x720."""
    small = cv2.resize(
        frame,
        None,
        fx=DETECTION_SCALE,
        fy=DETECTION_SCALE,
        interpolation=cv2.INTER_AREA,
    )
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners_small = cv2.findChessboardCorners(
        gray_small,
        CHECKERBOARD,
        flags,
    )

    if not found:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        20,
        0.01,
    )
    corners_small = cv2.cornerSubPix(
        gray_small,
        corners_small,
        (7, 7),
        (-1, -1),
        criteria,
    )

    corners_full = corners_small / DETECTION_SCALE
    return True, corners_full.astype(np.float32)


def make_preview(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    found_left: bool,
    corners_left,
    found_right: bool,
    corners_right,
    saved_pairs: int,
) -> np.ndarray:
    left = frame_left.copy()
    right = frame_right.copy()

    if found_left and corners_left is not None:
        cv2.drawChessboardCorners(left, CHECKERBOARD, corners_left, True)
    if found_right and corners_right is not None:
        cv2.drawChessboardCorners(right, CHECKERBOARD, corners_right, True)

    cv2.putText(
        left,
        "LEFT: FOUND" if found_left else "LEFT: NOT FOUND",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0) if found_left else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        right,
        "RIGHT: FOUND" if found_right else "RIGHT: NOT FOUND",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0) if found_right else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    combined = np.hstack((left, right))
    cv2.putText(
        combined,
        f"Saved pairs: {saved_pairs}/{MIN_PAIRS}+ | S=save  C=calibrate  Q=quit",
        (20, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if DISPLAY_SCALE != 1.0:
        combined = cv2.resize(
            combined,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_AREA,
        )
    return combined


def calibrate_and_save(objpoints, imgpoints_left, imgpoints_right, image_size):
    print("Calibrating left camera...", flush=True)
    rms_left, matrix_left, dist_left, _, _ = cv2.calibrateCamera(
        objpoints,
        imgpoints_left,
        image_size,
        None,
        None,
    )

    print("Calibrating right camera...", flush=True)
    rms_right, matrix_right, dist_right, _, _ = cv2.calibrateCamera(
        objpoints,
        imgpoints_right,
        image_size,
        None,
        None,
    )

    print("Running stereo calibration...", flush=True)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5,
    )
    stereo_rms, matrix_left, dist_left, matrix_right, dist_right, R, T, E, F = (
        cv2.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            matrix_left,
            dist_left,
            matrix_right,
            dist_right,
            image_size,
            criteria=criteria,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
    )

    print("Creating rectification maps...", flush=True)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        matrix_left,
        dist_left,
        matrix_right,
        dist_right,
        image_size,
        R,
        T,
        alpha=0,
    )

    # CV_16SC2 maps use less memory and remap faster than two CV_32FC1 maps.
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        matrix_left,
        dist_left,
        R1,
        P1,
        image_size,
        cv2.CV_16SC2,
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        matrix_right,
        dist_right,
        R2,
        P2,
        image_size,
        cv2.CV_16SC2,
    )

    baseline_m = float(np.linalg.norm(T))

    np.savez_compressed(
        OUTPUT_FILE,
        image_width=image_size[0],
        image_height=image_size[1],
        camera_matrix_left=matrix_left,
        dist_coeffs_left=dist_left,
        camera_matrix_right=matrix_right,
        dist_coeffs_right=dist_right,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        left_map1=left_map1,
        left_map2=left_map2,
        right_map1=right_map1,
        right_map2=right_map2,
        baseline_m=baseline_m,
        left_roi=np.asarray(roi1),
        right_roi=np.asarray(roi2),
        left_rms=float(rms_left),
        right_rms=float(rms_right),
        stereo_rms=float(stereo_rms),
    )

    print(f"Left RMS: {rms_left:.4f}")
    print(f"Right RMS: {rms_right:.4f}")
    print(f"Stereo RMS: {stereo_rms:.4f}")
    print(f"Estimated baseline: {baseline_m:.6f} m")
    print(f"Saved calibration file: {OUTPUT_FILE}")


def main():
    # Limiting OpenCV worker threads helps the desktop remain responsive on Jetson.
    cv2.setNumThreads(2)

    cap_left = None
    cap_right = None

    try:
        cap_left = open_camera(LEFT_SENSOR_ID)
        cap_right = open_camera(RIGHT_SENSOR_ID)

        print("Stereo calibration started.")
        print("Use a 9x6-inner-corner checkerboard with 25 mm squares.")
        print("Press S only when both views show FOUND.")
        print("Collect 20-30 varied poses, then press C.")
        print("Press Q to quit.")

        objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:CHECKERBOARD[0],
            0:CHECKERBOARD[1],
        ].T.reshape(-1, 2)
        objp *= SQUARE_SIZE_M

        objpoints = []
        imgpoints_left = []
        imgpoints_right = []
        image_size = (CAPTURE_WIDTH, CAPTURE_HEIGHT)

        found_left = False
        found_right = False
        corners_left = None
        corners_right = None
        frame_index = 0
        last_status_time = time.monotonic()

        cv2.namedWindow("Stereo Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            "Stereo Calibration",
            int(CAPTURE_WIDTH * 2 * DISPLAY_SCALE),
            int(CAPTURE_HEIGHT * DISPLAY_SCALE),
        )

        while True:
            ok_left, frame_left = cap_left.read()
            ok_right, frame_right = cap_right.read()

            if not ok_left or frame_left is None or not ok_right or frame_right is None:
                print("Frame grab failed; retrying...", flush=True)
                time.sleep(0.02)
                continue

            image_size = (frame_left.shape[1], frame_left.shape[0])
            frame_index += 1

            if frame_index % DETECT_EVERY_N_FRAMES == 0:
                found_left, corners_left = detect_corners_scaled(frame_left)
                found_right, corners_right = detect_corners_scaled(frame_right)

            preview = make_preview(
                frame_left,
                frame_right,
                found_left,
                corners_left,
                found_right,
                corners_right,
                len(objpoints),
            )
            cv2.imshow("Stereo Calibration", preview)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("s"), ord("S")):
                # Refresh detection on the exact frames being saved.
                save_found_left, save_corners_left = detect_corners_scaled(frame_left)
                save_found_right, save_corners_right = detect_corners_scaled(frame_right)

                if save_found_left and save_found_right:
                    objpoints.append(objp.copy())
                    imgpoints_left.append(save_corners_left.copy())
                    imgpoints_right.append(save_corners_right.copy())
                    print(f"Saved pair #{len(objpoints)}", flush=True)
                    time.sleep(0.10)
                else:
                    print("Not saved: checkerboard must be detected in both views.", flush=True)

            elif key in (ord("c"), ord("C")):
                if len(objpoints) < MIN_PAIRS:
                    print(
                        f"Need at least {MIN_PAIRS} pairs; currently {len(objpoints)}.",
                        flush=True,
                    )
                else:
                    calibrate_and_save(
                        objpoints,
                        imgpoints_left,
                        imgpoints_right,
                        image_size,
                    )
                    break

            elif key in (ord("q"), ord("Q"), 27):
                break

            # Occasional status output proves the event loop is still alive.
            now = time.monotonic()
            if now - last_status_time >= 5.0:
                print(
                    f"Running | pairs={len(objpoints)} | "
                    f"left={'FOUND' if found_left else 'not found'} | "
                    f"right={'FOUND' if found_right else 'not found'}",
                    flush=True,
                )
                last_status_time = now

    finally:
        if cap_left is not None:
            cap_left.release()
        if cap_right is not None:
            cap_right.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
