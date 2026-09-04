"""Raw RealSense depth-zone extraction, independent of YOLO and the GUI."""

from typing import Dict

import numpy as np

from config import DISPLAY_LABELS, MAX_VALID_DEPTH, MIN_VALID_DEPTH, ZONE_KEYS
from data_models import ZoneResult
from hazard_policy import alert_message, tone_from_depth


def compute_raw_depth_zones(
    depth_in_meters: np.ndarray,
) -> Dict[str, ZoneResult]:
    """Return the nearest strictly-valid raw pixel in each vertical third."""
    if depth_in_meters.ndim != 2:
        raise ValueError(
            "Aligned depth frame must be a 2-D array, got "
            f"shape {depth_in_meters.shape!r}"
        )

    height, width = depth_in_meters.shape
    if height == 0 or width < len(ZONE_KEYS):
        raise ValueError(
            f"Depth frame must be non-empty and at least {len(ZONE_KEYS)} pixels wide"
        )

    results: Dict[str, ZoneResult] = {}
    zone_count = len(ZONE_KEYS)
    for index, zone_name in enumerate(ZONE_KEYS):
        x1 = width * index // zone_count
        x2 = width * (index + 1) // zone_count
        region = depth_in_meters[:, x1:x2]
        result = ZoneResult(
            key=zone_name,
            display_name=DISPLAY_LABELS[zone_name],
            rect=(x1, 0, x2, height),
        )

        valid_mask = (region > MIN_VALID_DEPTH) & (region < MAX_VALID_DEPTH)
        if not np.any(valid_mask):
            result.alert_message = f"[{result.display_name}] No valid raw depth"
            results[zone_name] = result
            continue

        valid_depths = np.where(valid_mask, region, np.inf)
        flat_index = int(np.argmin(valid_depths))
        local_y, local_x = np.unravel_index(flat_index, region.shape)
        depth_m = float(region[local_y, local_x])
        result.depth_m = depth_m
        result.nearest_point = (x1 + int(local_x), int(local_y))
        result.alert_message = alert_message(depth_m, result.display_name)
        result.tone = tone_from_depth(depth_m)
        results[zone_name] = result

    return results
