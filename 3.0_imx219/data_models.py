"""Small shared models that do not depend on camera or UI hardware."""

from dataclasses import dataclass
from enum import Enum


class HazardBand(str, Enum):
    CLEAR = "clear"
    SLOW = "slow"
    FAST = "fast"
    URGENT = "urgent"


@dataclass(frozen=True)
class MotorPattern:
    name: str
    level: int
    on_time: float = 0.0
    off_time: float = 0.0


PATTERN_OFF = MotorPattern("off", 0x00, 0.0, 0.0)

