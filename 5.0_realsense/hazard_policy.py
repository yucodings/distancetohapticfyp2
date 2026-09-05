from typing import Optional

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
