"""Map an existing zone distance directly to one haptic timing pattern."""

from typing import Optional

from config import (
    CLEAR_DEPTH_M,
    FAST_MIN_DEPTH_M,
    FAST_OFF_SECONDS,
    FAST_ON_SECONDS,
    HAPTIC_LEVEL,
    SLOW_MIN_DEPTH_M,
    SLOW_OFF_SECONDS,
    SLOW_ON_SECONDS,
    URGENT_OFF_SECONDS,
    URGENT_ON_SECONDS,
)
from data_models import HazardBand, MotorPattern, PATTERN_OFF


PATTERN_SLOW = MotorPattern(
    "slow: 0.50 s on / 1.00 s off",
    HAPTIC_LEVEL,
    SLOW_ON_SECONDS,
    SLOW_OFF_SECONDS,
)
PATTERN_FAST = MotorPattern(
    "fast: 0.20 s on / 0.20 s off",
    HAPTIC_LEVEL,
    FAST_ON_SECONDS,
    FAST_OFF_SECONDS,
)
PATTERN_URGENT = MotorPattern(
    "urgent: 0.10 s on / 0.10 s off",
    HAPTIC_LEVEL,
    URGENT_ON_SECONDS,
    URGENT_OFF_SECONDS,
)


def band_from_depth(depth_m: Optional[float]) -> HazardBand:
    """Classify one already-computed zone value; no depth processing occurs."""
    if depth_m is None or depth_m > CLEAR_DEPTH_M:
        return HazardBand.CLEAR
    if depth_m >= SLOW_MIN_DEPTH_M:
        return HazardBand.SLOW
    if depth_m >= FAST_MIN_DEPTH_M:
        return HazardBand.FAST
    return HazardBand.URGENT


def pattern_from_depth(depth_m: Optional[float]) -> MotorPattern:
    return {
        HazardBand.CLEAR: PATTERN_OFF,
        HazardBand.SLOW: PATTERN_SLOW,
        HazardBand.FAST: PATTERN_FAST,
        HazardBand.URGENT: PATTERN_URGENT,
    }[band_from_depth(depth_m)]
