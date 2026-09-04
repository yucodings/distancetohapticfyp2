from pathlib import Path


WIDTH = 1280
HEIGHT = 720
FPS = 30
REALSENSE_START_RETRIES = 5
REALSENSE_RETRY_DELAY_SECONDS = 1.0
MIN_VALID_DEPTH = 0.1
MAX_VALID_DEPTH = 3.0
LOG_EVERY_N_FRAMES = 10
FRAME_ASPECT_RATIO = WIDTH / HEIGHT
ENABLE_ACTUATORS = True  # Drive three LRAs through the TCA9548A and DA7280s

# =========================
# YOLO object detection
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_MODEL_PATH = PROJECT_ROOT / "best.engine"
ENABLE_YOLO = True
YOLO_DEVICE = 0  # First CUDA device on the Jetson
YOLO_ALLOW_CPU_FALLBACK = True
YOLO_HALF = True  # FP16 is used only when inference is running on CUDA
YOLO_IMAGE_SIZE = 640
YOLO_CONFIDENCE = 0.40
YOLO_IOU = 0.45
YOLO_FRAME_INTERVAL = 1
YOLO_RESULT_MAX_AGE_MS = 500.0
YOLO_BOX_DEPTH_CROP_RATIO = 0.60
YOLO_BOX_DEPTH_PERCENTILE = 20.0
YOLO_ZONE_MIN_OVERLAP = 0.25

ZONE_KEYS = ["left", "center", "right"]
DISPLAY_LABELS = {
    "left": "Left",
    "center": "Center",
    "right": "Right",
}

# =========================
# TCA9548A + DA7280 actuator settings
# =========================
I2C_BUS = 1  # TCA9548A detected on /dev/i2c-1 (Jetson I2C0 wiring)
TCA9548A_ADDR = 0x70
TCA_CHANNEL_LEFT = 2
TCA_CHANNEL_CENTER = 3
TCA_CHANNEL_RIGHT = 4
TCA_CHANNELS = {
    "left": TCA_CHANNEL_LEFT,
    "center": TCA_CHANNEL_CENTER,
    "right": TCA_CHANNEL_RIGHT,
}
DA7280_ADDR = 0x4A
REG_CHIP_REV = 0x00
REG_IRQ_EVENT1 = 0x03
REG_TOP_CTL1 = 0x22
REG_OVERRIDE_VAL = 0x23

# Do not add actuator voltage/current/impedance/resonant-period values here
# until they have been calculated from the exact LRA manufacturer's datasheet.
