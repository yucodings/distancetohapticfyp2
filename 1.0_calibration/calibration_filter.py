"""Conservative automatic quality filtering for stereo calibration pairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PairSample:
    filename: str
    object_points: np.ndarray
    left_corners: np.ndarray
    right_corners: np.ndarray
    left_path: Path
    right_path: Path


@dataclass(frozen=True)
class QualityThresholds:
    min_board_area_percent: float = 1.2
    max_reprojection_rms_px: float = 1.5
    max_epipolar_p95_px: float = 3.0
    max_rectified_vertical_p95_px: float = 2.5
    max_rounds: int = 4


@dataclass(frozen=True)
class FilterResult:
    accepted: tuple[PairSample, ...]
    rejected: tuple[dict[str, Any], ...]
    calibration: dict[str, Any]
    rectification: dict[str, np.ndarray]
    final_metrics: tuple[dict[str, Any], ...]
    history: tuple[dict[str, Any], ...]


def board_area_percent(corners: np.ndarray, image_size: tuple[int, int]) -> float:
    points = corners.reshape(-1, 2)
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    area = float(np.prod(np.maximum(high - low, 0.0)))
    image_area = float(image_size[0] * image_size[1])
    return 100.0 * area / image_area


def calibrate_once(
    samples: list[PairSample],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    object_points = [sample.object_points for sample in samples]
    left_points = [sample.left_corners for sample in samples]
    right_points = [sample.right_corners for sample in samples]

    left_rms, left_matrix, left_distortion, left_rvecs, left_tvecs = (
        cv2.calibrateCamera(object_points, left_points, image_size, None, None)
    )
    right_rms, right_matrix, right_distortion, right_rvecs, right_tvecs = (
        cv2.calibrateCamera(object_points, right_points, image_size, None, None)
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        60,
        1e-5,
    )
    (
        stereo_rms,
        left_matrix,
        left_distortion,
        right_matrix,
        right_distortion,
        rotation,
        translation,
        essential,
        fundamental,
    ) = cv2.stereoCalibrate(
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
        "baseline_m": float(np.linalg.norm(translation)),
        "left_rvecs": left_rvecs,
        "left_tvecs": left_tvecs,
        "right_rvecs": right_rvecs,
        "right_tvecs": right_tvecs,
    }


def rectify_once(
    calibration: dict[str, Any],
    image_size: tuple[int, int],
) -> dict[str, np.ndarray]:
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


def _reprojection_rms(
    objects: np.ndarray,
    corners: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(objects, rvec, tvec, matrix, distortion)
    difference = corners.reshape(-1, 2) - projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def _line_distances(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    numerator = np.abs(
        lines[:, 0] * points[:, 0]
        + lines[:, 1] * points[:, 1]
        + lines[:, 2]
    )
    denominator = np.hypot(lines[:, 0], lines[:, 1])
    return numerator / np.maximum(denominator, 1e-12)


def measure_pairs(
    samples: list[PairSample],
    calibration: dict[str, Any],
    rectification: dict[str, np.ndarray],
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        left = sample.left_corners.reshape(-1, 2)
        right = sample.right_corners.reshape(-1, 2)
        right_lines = cv2.computeCorrespondEpilines(
            left.reshape(-1, 1, 2), 1, calibration["fundamental"]
        ).reshape(-1, 3)
        left_lines = cv2.computeCorrespondEpilines(
            right.reshape(-1, 1, 2), 2, calibration["fundamental"]
        ).reshape(-1, 3)
        epipolar = 0.5 * (
            _line_distances(right, right_lines)
            + _line_distances(left, left_lines)
        )
        left_rectified = cv2.undistortPoints(
            sample.left_corners,
            calibration["left_matrix"],
            calibration["left_distortion"],
            R=rectification["r1"],
            P=rectification["p1"],
        ).reshape(-1, 2)
        right_rectified = cv2.undistortPoints(
            sample.right_corners,
            calibration["right_matrix"],
            calibration["right_distortion"],
            R=rectification["r2"],
            P=rectification["p2"],
        ).reshape(-1, 2)
        vertical = np.abs(left_rectified[:, 1] - right_rectified[:, 1])
        output.append(
            {
                "filename": sample.filename,
                "left_board_area_percent": board_area_percent(
                    sample.left_corners, image_size
                ),
                "right_board_area_percent": board_area_percent(
                    sample.right_corners, image_size
                ),
                "left_reprojection_rms_px": _reprojection_rms(
                    sample.object_points,
                    sample.left_corners,
                    calibration["left_rvecs"][index],
                    calibration["left_tvecs"][index],
                    calibration["left_matrix"],
                    calibration["left_distortion"],
                ),
                "right_reprojection_rms_px": _reprojection_rms(
                    sample.object_points,
                    sample.right_corners,
                    calibration["right_rvecs"][index],
                    calibration["right_tvecs"][index],
                    calibration["right_matrix"],
                    calibration["right_distortion"],
                ),
                "epipolar_p95_px": float(np.percentile(epipolar, 95)),
                "rectified_vertical_p95_px": float(
                    np.percentile(vertical, 95)
                ),
                "rectified_vertical_max_px": float(np.max(vertical)),
            }
        )
    return output


def metric_violations(
    metrics: dict[str, Any],
    thresholds: QualityThresholds,
) -> tuple[list[str], float]:
    checks = (
        (
            "left reprojection RMS",
            float(metrics["left_reprojection_rms_px"]),
            thresholds.max_reprojection_rms_px,
        ),
        (
            "right reprojection RMS",
            float(metrics["right_reprojection_rms_px"]),
            thresholds.max_reprojection_rms_px,
        ),
        (
            "epipolar P95",
            float(metrics["epipolar_p95_px"]),
            thresholds.max_epipolar_p95_px,
        ),
        (
            "rectified vertical P95",
            float(metrics["rectified_vertical_p95_px"]),
            thresholds.max_rectified_vertical_p95_px,
        ),
    )
    violations = [
        f"{name} {value:.3f}px > {limit:.3f}px"
        for name, value, limit in checks
        if value > limit
    ]
    score = max(value / limit for _, value, limit in checks)
    return violations, score


def filter_calibration_pairs(
    samples: list[PairSample],
    image_size: tuple[int, int],
    min_pairs: int,
    thresholds: QualityThresholds | None = None,
) -> FilterResult:
    """Reject clearly weak pairs, recalibrating after every rejection round."""
    limits = thresholds or QualityThresholds()
    accepted: list[PairSample] = []
    rejected: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []

    for sample in samples:
        left_area = board_area_percent(sample.left_corners, image_size)
        right_area = board_area_percent(sample.right_corners, image_size)
        if min(left_area, right_area) < limits.min_board_area_percent:
            rejected.append(
                {
                    "filename": sample.filename,
                    "stage": "board-size prefilter",
                    "reason": (
                        f"board area {min(left_area, right_area):.3f}% < "
                        f"{limits.min_board_area_percent:.3f}%"
                    ),
                    "left_board_area_percent": left_area,
                    "right_board_area_percent": right_area,
                }
            )
        else:
            accepted.append(sample)

    if len(accepted) < min_pairs:
        raise RuntimeError(
            f"Only {len(accepted)} pairs remain after the board-size filter; "
            f"at least {min_pairs} are required."
        )

    for round_index in range(1, limits.max_rounds + 1):
        calibration = calibrate_once(accepted, image_size)
        rectification = rectify_once(calibration, image_size)
        metrics = measure_pairs(
            accepted, calibration, rectification, image_size
        )
        failing: list[tuple[PairSample, dict[str, Any], list[str], float]] = []
        for sample, pair_metrics in zip(accepted, metrics):
            violations, score = metric_violations(pair_metrics, limits)
            if violations:
                failing.append((sample, pair_metrics, violations, score))
        history.append(
            {
                "round": round_index,
                "input_pairs": len(accepted),
                "failing_pairs": len(failing),
                "left_rms_px": calibration["left_rms"],
                "right_rms_px": calibration["right_rms"],
                "stereo_rms_px": calibration["stereo_rms"],
                "baseline_m": calibration["baseline_m"],
            }
        )
        if not failing:
            return FilterResult(
                tuple(accepted),
                tuple(rejected),
                calibration,
                rectification,
                tuple(metrics),
                tuple(history),
            )

        removable = len(accepted) - min_pairs
        if removable <= 0:
            raise RuntimeError(
                "Bad pairs remain, but rejecting them would leave fewer than "
                f"the required {min_pairs} pairs. Capture more varied images."
            )
        failing.sort(key=lambda item: item[3], reverse=True)
        remove_now = failing[:removable]
        remove_names = {item[0].filename for item in remove_now}
        for sample, pair_metrics, violations, score in remove_now:
            rejected.append(
                {
                    "filename": sample.filename,
                    "stage": f"geometry round {round_index}",
                    "reason": "; ".join(violations),
                    "quality_score": score,
                    **pair_metrics,
                }
            )
        accepted = [
            sample for sample in accepted if sample.filename not in remove_names
        ]

    calibration = calibrate_once(accepted, image_size)
    rectification = rectify_once(calibration, image_size)
    metrics = measure_pairs(accepted, calibration, rectification, image_size)
    still_failing = [
        metric["filename"]
        for metric in metrics
        if metric_violations(metric, limits)[0]
    ]
    if still_failing:
        raise RuntimeError(
            "Automatic rejection did not converge after "
            f"{limits.max_rounds} rounds. Remaining failures: "
            + ", ".join(still_failing)
        )
    return FilterResult(
        tuple(accepted),
        tuple(rejected),
        calibration,
        rectification,
        tuple(metrics),
        tuple(history),
    )
