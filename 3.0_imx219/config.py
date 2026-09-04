from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CALIBRATION_PATH = PROJECT_ROOT / "stereo_calibration.npz"
YOLO_MODEL_PATH = PROJECT_ROOT / "best.engine"

# The selected calibration is valid only for this exact capture mode.
WIDTH = 1280
HEIGHT = 720
FPS = 30
SENSOR_MODE = 4
LEFT_SENSOR_ID = 0
RIGHT_SENSOR_ID = 1
FRAME_ASPECT_RATIO = WIDTH / HEIGHT
CAMERA_FRAME_BUFFER_SIZE = 8
CAMERA_PAIR_TIMEOUT_SECONDS = 2.0
MAX_CONSECUTIVE_CAPTURE_FAILURES = 10

# Stereo depth: the calibrated VPI CUDA profile is the only matcher. A VPI
# failure is a safety error and never silently changes the depth algorithm.

# Built-in VPI values are used only until Easy Mode creates the JSON profile.
VPI_MIN_DISPARITY = 0
VPI_MAX_DISPARITY = 256
VPI_DISPARITY_SAFETY_MARGIN_PX = 8.0
VPI_WINDOW = 5
VPI_INTERNAL_CONFIDENCE_THRESHOLD = 1
VPI_MIN_CONFIDENCE = 8192
VPI_P1 = 3
VPI_P2 = 48
VPI_UNIQUENESS = 0.90
VPI_INCLUDE_DIAGONALS = False
VPI_PROFILE_PATH = PROJECT_ROOT / "vpi_tuned_profile.json"
DEPTH_RESULT_MAX_AGE_MS = 500.0

# Accepted navigation depth and robust 3x3 median-grid selection.
MIN_VALID_DEPTH = 0.1
MAX_VALID_DEPTH = 3.0
DEPTH_GRID_ROWS = 3
DEPTH_GRID_COLUMNS = 3
GRID_CELL_MIN_VALID_PIXELS = 100
GRID_CELL_ROI_WIDTH_FRACTION = 0.60
GRID_CELL_ROI_HEIGHT_FRACTION = 0.60
ZONE_BORDER_MARGIN_X = 24
ZONE_BORDER_MARGIN_Y = 24
GRID_WINNER_CONFIRM_FRAMES = 2
HAPTIC_DISTANCE_HYSTERESIS_M = 0.08

LOG_EVERY_N_RESULTS = 10
ZONE_KEYS = ["left", "center", "right"]
DISPLAY_LABELS = {
    "left": "Left",
    "center": "Center",
    "right": "Right",
}

# YOLO is informational and never feeds the haptic policy.
ENABLE_YOLO = False
YOLO_DEVICE = 0
YOLO_ALLOW_CPU_FALLBACK = True
YOLO_HALF = True
YOLO_IMAGE_SIZE = 640
YOLO_CONFIDENCE = 0.40
YOLO_IOU = 0.45
YOLO_FRAME_INTERVAL = 1
YOLO_RESULT_MAX_AGE_MS = 750.0
YOLO_BOX_DEPTH_CROP_RATIO = 0.60
YOLO_BOX_DEPTH_PERCENTILE = 20.0
YOLO_ZONE_MIN_OVERLAP = 0.25

# TCA9548A + three DA7280s.
# Keep motors off for the first live stereo validation. Set True only after
# confirming that left/right orientation and displayed zone depths are correct.
ENABLE_ACTUATORS = False
I2C_BUS = 1
TCA9548A_ADDR = 0x70
TCA_CHANNELS = {"left": 2, "center": 3, "right": 4}
DA7280_ADDR = 0x4A
REG_CHIP_REV = 0x00
REG_IRQ_EVENT1 = 0x03
REG_TOP_CTL1 = 0x22
REG_OVERRIDE_VAL = 0x23
