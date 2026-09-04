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

# Stereo depth: VPI CUDA is primary. OpenCV SGBM remains the safety fallback;
# its earlier 1 m tune selected block size 11. Both retain a 256-disparity
# navigation range for better close-range coverage.
STEREO_BACKEND = "vpi-cuda"
ALLOW_OPENCV_STEREO_FALLBACK = True
SGBM_MIN_DISPARITY = 0
SGBM_NUM_DISPARITIES = 256
SGBM_BLOCK_SIZE = 11

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

# Accepted navigation depth and robust near-surface support.
MIN_VALID_DEPTH = 0.1
MAX_VALID_DEPTH = 3.0
ZONE_SUPPORT_RADIUS = 3
ZONE_MIN_SUPPORT_PIXELS = 6
ZONE_MIN_CONNECTED_SURFACE_PIXELS = 24
ZONE_BORDER_MARGIN_X = 8
ZONE_BORDER_MARGIN_Y = 8
ZONE_DEPTH_TOLERANCE_M = 0.06
ZONE_DEPTH_TOLERANCE_RATIO = 0.10
ZONE_CANDIDATE_LIMIT = 128

LOG_EVERY_N_RESULTS = 10
ZONE_KEYS = ["left", "center", "right"]
DISPLAY_LABELS = {
    "left": "Left",
    "center": "Center",
    "right": "Right",
}

# YOLO is informational and never feeds the haptic policy.
ENABLE_YOLO = True
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
