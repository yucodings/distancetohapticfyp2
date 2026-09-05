from typing import Optional

from config import HAPTIC_DISTANCE_HYSTERESIS_M
from data_models import (
    MotorPattern,
    PATTERN_CONTINUOUS,
    PATTERN_MEDIUM,
    PATTERN_OFF,
    PATTERN_SLOW,
)


def alert_message(depth_m: float, zone_name: str) -> str:
    if depth_m > 2.0:
        return f"[{zone_name}] Clear path (> 2.0 m)"
    if depth_m >= 1.5:
        return f"[{zone_name}] Distant obstacle ({depth_m:.2f} m) - caution"
    if depth_m >= 0.5:
        return f"[{zone_name}] Mid-range obstacle ({depth_m:.2f} m) - attention required"
    return f"[{zone_name}] Imminent obstacle ({depth_m:.2f} m) - immediate action"


def tone_from_depth(depth_m: Optional[float]) -> str:
    if depth_m is None:
        return "invalid"
    if depth_m > 2.0:
        return "clear"
    if depth_m >= 1.5:
        return "green"
    if depth_m >= 0.5:
        return "orange"
    return "red"


def pattern_from_depth(depth_m: Optional[float]) -> MotorPattern:
    if depth_m is None or depth_m > 2.0:
        return PATTERN_OFF
    if depth_m >= 1.5:
        return PATTERN_SLOW
    if depth_m >= 0.5:
        return PATTERN_MEDIUM
    return PATTERN_CONTINUOUS


def pattern_from_depth_hysteresis(
    depth_m: Optional[float], previous: MotorPattern
) -> MotorPattern:
    """Prevent threshold chatter while keeping invalid-depth shutdown immediate."""
    if depth_m is None:
        return PATTERN_OFF
    margin = HAPTIC_DISTANCE_HYSTERESIS_M
    if previous == PATTERN_CONTINUOUS and depth_m < 0.5 + margin:
        return PATTERN_CONTINUOUS
    if previous == PATTERN_MEDIUM:
        if depth_m < 0.5 - margin:
            return PATTERN_CONTINUOUS
        if depth_m < 1.5 + margin:
            return PATTERN_MEDIUM
    if previous == PATTERN_SLOW and 1.5 - margin <= depth_m <= 2.0 + margin:
        return PATTERN_SLOW
    if previous == PATTERN_OFF and depth_m > 2.0 - margin:
        return PATTERN_OFF
    return pattern_from_depth(depth_m)
