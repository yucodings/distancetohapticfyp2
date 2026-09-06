"""Small shared models that do not depend on camera or UI hardware."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


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


@dataclass(frozen=True)
class Detection:
    """One object detection in rectified-left image coordinates."""

    class_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionScene:
    """Newest complete detector output; it never participates in haptics."""

    source_sequence: int
    captured_at: float
    completed_at: float
    detections: tuple[Detection, ...]
    inference_ms: float
    detector_fps: float
    device: str
    error: Optional[str] = None
