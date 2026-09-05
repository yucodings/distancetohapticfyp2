from dataclasses import replace
from typing import Iterable, List, Optional, Tuple

import numpy as np

from config import (
    MAX_VALID_DEPTH,
    MIN_VALID_DEPTH,
    YOLO_BOX_DEPTH_CROP_RATIO,
    YOLO_BOX_DEPTH_PERCENTILE,
    YOLO_ZONE_MIN_OVERLAP,
    ZONE_KEYS,
)
from data_models import DetectionResult
from hazard_policy import tone_from_depth


def clip_bbox(
    bbox: Tuple[int, int, int, int], image_width: int, image_height: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), image_width))
    x2 = max(0, min(int(x2), image_width))
    y1 = max(0, min(int(y1), image_height))
    y2 = max(0, min(int(y2), image_height))
    return x1, y1, x2, y2


def object_depth(
    depth_in_meters: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Optional[float]:
    """Estimate foreground distance from the central area of a detection box."""
    height, width = depth_in_meters.shape
    x1, y1, x2, y2 = clip_bbox(bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return None

    crop_ratio = max(0.1, min(YOLO_BOX_DEPTH_CROP_RATIO, 1.0))
    inset_x = int((x2 - x1) * (1.0 - crop_ratio) / 2.0)
    inset_y = int((y2 - y1) * (1.0 - crop_ratio) / 2.0)
    center_x1, center_x2 = x1 + inset_x, x2 - inset_x
    center_y1, center_y2 = y1 + inset_y, y2 - inset_y

    region = depth_in_meters[center_y1:center_y2, center_x1:center_x2]
    valid = region[(region > MIN_VALID_DEPTH) & (region < MAX_VALID_DEPTH)]
    if valid.size == 0:
        return None

    return float(np.percentile(valid, YOLO_BOX_DEPTH_PERCENTILE))


def detection_zones(
    bbox: Tuple[int, int, int, int], image_width: int
) -> Tuple[str, ...]:
    x1, _, x2, _ = bbox
    x1 = max(0, min(x1, image_width))
    x2 = max(0, min(x2, image_width))
    box_width = x2 - x1
    if box_width <= 0:
        return ()

    zones = []
    zone_count = len(ZONE_KEYS)
    for index, zone_name in enumerate(ZONE_KEYS):
        zone_x1 = image_width * index / zone_count
        zone_x2 = image_width * (index + 1) / zone_count
        overlap = max(0.0, min(x2, zone_x2) - max(x1, zone_x1)) / box_width
        if overlap >= YOLO_ZONE_MIN_OVERLAP:
            zones.append(zone_name)

    if not zones:
        center_x = (x1 + x2) / 2.0
        index = min(zone_count - 1, int(center_x * zone_count / image_width))
        zones.append(ZONE_KEYS[index])
    return tuple(zones)


def fuse_detections_with_depth(
    detections: Iterable[DetectionResult], depth_in_meters: np.ndarray
) -> List[DetectionResult]:
    height, width = depth_in_meters.shape
    fused: List[DetectionResult] = []

    for detection in detections:
        bbox = clip_bbox(detection.bbox, width, height)
        depth_m = object_depth(depth_in_meters, bbox)
        fused.append(
            replace(
                detection,
                bbox=bbox,
                depth_m=depth_m,
                zones=detection_zones(bbox, width),
                tone=tone_from_depth(depth_m),
            )
        )

    return fused
