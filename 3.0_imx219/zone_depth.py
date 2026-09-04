"""Robust left/center/right distances from confidence-filtered stereo depth."""

from __future__ import annotations

from typing import Dict, Optional

from camera_backend import np
from config import (
    DISPLAY_LABELS,
    MAX_VALID_DEPTH,
    MIN_VALID_DEPTH,
    ZONE_CANDIDATE_LIMIT,
    ZONE_DEPTH_TOLERANCE_M,
    ZONE_DEPTH_TOLERANCE_RATIO,
    ZONE_KEYS,
    ZONE_MIN_SUPPORT_PIXELS,
    ZONE_SUPPORT_RADIUS,
)
from data_models import ZoneResult
from hazard_policy import alert_message, tone_from_depth


def _nearest_supported_point(
    region: np.ndarray,
    valid: np.ndarray,
) -> Optional[tuple[float, int, int]]:
    """Reject isolated near outliers and return the closest supported surface."""
    coordinates = np.argwhere(valid)
    if coordinates.size == 0:
        return None

    values = region[valid]
    candidate_count = min(int(values.size), ZONE_CANDIDATE_LIMIT)
    if candidate_count < values.size:
        candidate_indices = np.argpartition(values, candidate_count - 1)[
            :candidate_count
        ]
        candidate_indices = candidate_indices[np.argsort(values[candidate_indices])]
    else:
        candidate_indices = np.argsort(values)

    height, width = region.shape
    radius = ZONE_SUPPORT_RADIUS
    for candidate_index in candidate_indices:
        y, x = coordinates[int(candidate_index)]
        candidate_depth = float(region[y, x])
        y1, y2 = max(0, y - radius), min(height, y + radius + 1)
        x1, x2 = max(0, x - radius), min(width, x + radius + 1)
        patch_depth = region[y1:y2, x1:x2]
        patch_valid = valid[y1:y2, x1:x2]
        tolerance = max(
            ZONE_DEPTH_TOLERANCE_M,
            candidate_depth * ZONE_DEPTH_TOLERANCE_RATIO,
        )
        support = patch_valid & (
            np.abs(patch_depth - candidate_depth) <= tolerance
        )
        if int(np.count_nonzero(support)) < ZONE_MIN_SUPPORT_PIXELS:
            continue

        supported_depths = patch_depth[support]
        estimated_depth = float(np.median(supported_depths))
        supported_coordinates = np.argwhere(support)
        closest = int(
            np.argmin(np.abs(supported_depths - estimated_depth))
        )
        point_y, point_x = supported_coordinates[closest]
        return estimated_depth, x1 + int(point_x), y1 + int(point_y)
    return None


def compute_stereo_depth_zones(
    depth_map: np.ndarray,
    valid_depth_mask: np.ndarray,
) -> Dict[str, ZoneResult]:
    if depth_map.ndim != 2 or valid_depth_mask.shape != depth_map.shape:
        raise ValueError("Depth map and valid mask must be matching 2-D arrays")
    height, width = depth_map.shape
    if height == 0 or width < len(ZONE_KEYS):
        raise ValueError("Depth map is empty or too narrow for three zones")

    results: Dict[str, ZoneResult] = {}
    zone_count = len(ZONE_KEYS)
    finite_valid = (
        valid_depth_mask
        & np.isfinite(depth_map)
        & (depth_map > MIN_VALID_DEPTH)
        & (depth_map < MAX_VALID_DEPTH)
    )
    for index, zone_name in enumerate(ZONE_KEYS):
        x1 = width * index // zone_count
        x2 = width * (index + 1) // zone_count
        result = ZoneResult(
            key=zone_name,
            display_name=DISPLAY_LABELS[zone_name],
            rect=(x1, 0, x2, height),
        )
        nearest = _nearest_supported_point(
            depth_map[:, x1:x2], finite_valid[:, x1:x2]
        )
        if nearest is None:
            result.alert_message = (
                f"[{result.display_name}] No supported stereo depth"
            )
        else:
            depth_m, local_x, y = nearest
            result.depth_m = depth_m
            result.nearest_point = (x1 + local_x, y)
            result.alert_message = alert_message(depth_m, result.display_name)
            result.tone = tone_from_depth(depth_m)
        results[zone_name] = result
    return results


def empty_stereo_zones(
    image_shape: tuple[int, int], reason: str = "Stereo depth unavailable"
) -> Dict[str, ZoneResult]:
    height, width = image_shape
    results: Dict[str, ZoneResult] = {}
    for index, zone_name in enumerate(ZONE_KEYS):
        x1 = width * index // len(ZONE_KEYS)
        x2 = width * (index + 1) // len(ZONE_KEYS)
        results[zone_name] = ZoneResult(
            key=zone_name,
            display_name=DISPLAY_LABELS[zone_name],
            rect=(x1, 0, x2, height),
            alert_message=f"[{DISPLAY_LABELS[zone_name]}] {reason}",
        )
    return results
