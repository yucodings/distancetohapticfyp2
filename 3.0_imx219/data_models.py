from dataclasses import dataclass
from typing import List, Optional, Tuple

from camera_backend import np


@dataclass
class ZoneResult:
    key: str
    display_name: str
    rect: Tuple[int, int, int, int]
    depth_m: Optional[float] = None
    alert_message: str = "No valid depth"
    tone: str = "invalid"
    nearest_point: Optional[Tuple[int, int]] = None
    source_cell: Optional[str] = None


@dataclass
class GridCellResult:
    key: str
    display_name: str
    row: int
    column: int
    rect: Tuple[int, int, int, int]
    sample_rect: Tuple[int, int, int, int]
    depth_m: Optional[float] = None
    valid_count: int = 0
    total_count: int = 0
    valid_percentage: float = 0.0
    representative_point: Optional[Tuple[int, int]] = None
    selected: bool = False


@dataclass(frozen=True)
class DetectionResult:
    class_id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    depth_m: Optional[float] = None
    zones: Tuple[str, ...] = ()
    tone: str = "invalid"


@dataclass
class FramePacket:
    frame_id: int
    captured_at_ms: float
    color_image: np.ndarray
    depth_in_meters: np.ndarray


@dataclass
class SceneResult:
    frame_id: int
    captured_at_ms: float
    completed_at_ms: float
    detections: List[DetectionResult]
    inference_ms: float
    device: str


@dataclass(frozen=True)
class MotorPattern:
    name: str
    level: int
    on_time: float = 0.0
    off_time: float = 0.0


PATTERN_OFF = MotorPattern("off", 0x00, 0.0, 0.0)
PATTERN_SLOW = MotorPattern(
    "slow pulse (1 Hz envelope)", 0x30, 0.50, 0.50
)  # ~38%
PATTERN_MEDIUM = MotorPattern(
    "medium pulse (5 Hz envelope)", 0x60, 0.08, 0.12
)  # ~76%
PATTERN_CONTINUOUS = MotorPattern("continuous", 0x7F, 0.0, 0.0)  # 100%
