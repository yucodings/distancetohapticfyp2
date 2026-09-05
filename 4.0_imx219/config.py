"""Application, stereo and haptic settings for the IMX219 system."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

WIDTH = 1280
HEIGHT = 720
FPS = 30
SENSOR_MODE = 4
LEFT_SENSOR_ID = 0
RIGHT_SENSOR_ID = 1
FRAME_ASPECT_RATIO = WIDTH / HEIGHT

STEREO_BACKEND = "vpi-cuda"
MAX_VALID_DEPTH_M = 5.0
DEPTH_RESULT_MAX_AGE_SECONDS = 1.0

ZONE_KEYS = ("left", "center", "right")
DISPLAY_LABELS = {
    "left": "LEFT",
    "center": "CENTRE",
    "right": "RIGHT",
}

# One fixed strength is used for every non-clear range. It matches the initial
# low amplitude already exercised by test_haptic_pulse.sh.
HAPTIC_LEVEL = 25
SLOW_MIN_DEPTH_M = 1.5
FAST_MIN_DEPTH_M = 0.5
CLEAR_DEPTH_M = 2.0
SLOW_ON_SECONDS = 0.50
SLOW_OFF_SECONDS = 1.00
FAST_ON_SECONDS = 0.20
FAST_OFF_SECONDS = 0.20
URGENT_ON_SECONDS = 0.10
URGENT_OFF_SECONDS = 0.10

# TCA9548A + three same-address DA7280 devices.
I2C_BUS = 1
TCA9548A_ADDR = 0x70
TCA_CHANNELS = {"left": 2, "center": 3, "right": 4}
DA7280_ADDR = 0x4A
REG_CHIP_REV = 0x00
REG_IRQ_EVENT1 = 0x03
REG_TOP_CTL1 = 0x22
REG_OVERRIDE_VAL = 0x23

WINDOW_MIN_WIDTH = 1000
WINDOW_MIN_HEIGHT = 680
WINDOW_DEFAULT_WIDTH = 1280
WINDOW_DEFAULT_HEIGHT = 800
