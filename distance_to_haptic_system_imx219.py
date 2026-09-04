import os
import sys
import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from smbus2 import SMBus
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QGridLayout,
)

# =========================
# Camera settings
# =========================
LEFT_SENSOR_ID = 0
RIGHT_SENSOR_ID = 1
WIDTH = 1280
HEIGHT = 720
FPS = 30

# UI window size (landscape 4:3 assumption)
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 1050

# Depth range
MIN_VALID_DEPTH = 0.1
MAX_VALID_DEPTH = 3.0

LOG_EVERY_N_FRAMES = 10

ZONE_KEYS = ["left", "right"]
DISPLAY_LABELS = {
    "left": "Left",
    "right": "Right",
}

# =========================
# DA7280 actuator settings
# =========================
ADDR = 0x4A
BUS_LEFT = 7
BUS_RIGHT = 1
REG_EVENT = 0x03
REG_CFG1 = 0x13
REG_MODE = 0x22
REG_LEVEL = 0x23
REG_CHIP_REV = 0x00
LRA_DEFAULT_CFG1 = 0x1E


@dataclass
class ZoneResult:
    key: str
    display_name: str
    rect: Tuple[int, int, int, int]
    depth_m: Optional[float] = None
    alert_message: str = "No valid depth"
    tone: str = "invalid"
    nearest_point: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class MotorPattern:
    name: str
    level: int
    on_time: float = 0.0
    off_time: float = 0.0


PATTERN_OFF = MotorPattern("off", 0x00, 0.0, 0.0)
PATTERN_SLOW = MotorPattern("slow pulse (1 Hz)", 0x30, 0.50, 0.50)
PATTERN_MEDIUM = MotorPattern("medium pulse (5 Hz)", 0x60, 0.08, 0.12)
PATTERN_CONTINUOUS = MotorPattern("continuous", 0x7F, 0.0, 0.0)


def gstreamer_pipeline(sensor_id, width=1280, height=720, framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


class DA7280Motor:
    def __init__(self, bus_num: int, name: str):
        self.bus_num = bus_num
        self.name = name
        self.bus = None

    def open(self):
        self.bus = SMBus(self.bus_num)

    def close(self):
        if self.bus is not None:
            try:
                self.stop()
            except Exception:
                pass
            self.bus.close()
            self.bus = None

    def read_reg(self, reg: int) -> int:
        return self.bus.read_byte_data(ADDR, reg)

    def write_reg(self, reg: int, value: int):
        self.bus.write_byte_data(ADDR, reg, value & 0xFF)

    def init_lra(self):
        chip = self.read_reg(REG_CHIP_REV)
        if chip != 0xBA:
            raise RuntimeError(f"{self.name}: unexpected CHIP_REV 0x{chip:02X}, expected 0xBA")

        self.write_reg(REG_EVENT, 0xFF)
        self.write_reg(REG_CFG1, LRA_DEFAULT_CFG1)
        self.stop()

    def set_level(self, level: int):
        level = max(0, min(level, 0x7F))
        self.write_reg(REG_LEVEL, level)

    def start(self):
        self.write_reg(REG_MODE, 0x01)

    def stop(self):
        self.write_reg(REG_MODE, 0x00)


class MotorController:
    def __init__(self, motor: DA7280Motor):
        self.motor = motor
        self._cmd = PATTERN_OFF
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.motor.open()
        self.motor.init_lra()
        self._thread.start()

    def set_pattern(self, pattern: MotorPattern):
        with self._lock:
            self._cmd = pattern

    def get_pattern(self) -> MotorPattern:
        with self._lock:
            return self._cmd

    def _pattern_changed(self, old: MotorPattern) -> bool:
        return self.get_pattern() != old

    def _wait_with_change_check(self, duration: float, old: MotorPattern) -> bool:
        step = 0.02
        elapsed = 0.0
        while elapsed < duration and not self._stop_event.is_set():
            if self._pattern_changed(old):
                return True
            time.sleep(step)
            elapsed += step
        return self._stop_event.is_set() or self._pattern_changed(old)

    def _run(self):
        try:
            while not self._stop_event.is_set():
                cmd = self.get_pattern()

                if cmd.name == "off":
                    self.motor.stop()
                    time.sleep(0.05)
                    continue

                if cmd.name == "continuous":
                    self.motor.set_level(cmd.level)
                    self.motor.start()
                    while not self._stop_event.is_set() and not self._pattern_changed(cmd):
                        time.sleep(0.05)
                    self.motor.stop()
                    continue

                self.motor.set_level(cmd.level)
                self.motor.start()
                changed = self._wait_with_change_check(cmd.on_time, cmd)
                self.motor.stop()
                if changed:
                    continue
                self._wait_with_change_check(cmd.off_time, cmd)
        finally:
            try:
                self.motor.stop()
            except Exception:
                pass
            try:
                self.motor.close()
            except Exception:
                pass

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)


class ActuatorManager:
    def __init__(self):
        self.left_controller = MotorController(DA7280Motor(BUS_LEFT, "LEFT"))
        self.right_controller = MotorController(DA7280Motor(BUS_RIGHT, "RIGHT"))
        self.started = False

    def start(self):
        self.left_controller.start()
        self.right_controller.start()
        self.started = True

    def stop(self):
        if self.started:
            self.left_controller.stop()
            self.right_controller.stop()
            self.started = False

    def update_patterns(self, left_pattern: MotorPattern, right_pattern: MotorPattern):
        if not self.started:
            return
        self.left_controller.set_pattern(left_pattern)
        self.right_controller.set_pattern(right_pattern)


class StereoMatcher:
    def __init__(self):
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,   # must be divisible by 16
            blockSize=5,
            P1=8 * 3 * 5 * 5,
            P2=32 * 3 * 5 * 5,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        disparity = self.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disparity


class StreamWorker(QObject):
    live_ready = Signal(QImage)
    left_ready = Signal(QImage)
    right_ready = Signal(QImage)
    depth_ready = Signal(QImage)
    log_ready = Signal(str)
    camera_status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self.cap_left = None
        self.cap_right = None
        self.actuators: Optional[ActuatorManager] = None
        self.stereo = StereoMatcher()
        self.last_pattern_names = {"left": None, "right": None}

        self.calibration_file = "stereo_calibration_imx219.npz"
        self.fx = None
        self.baseline_m = None
        self.left_map1 = None
        self.left_map2 = None
        self.right_map1 = None
        self.right_map2 = None

    def stop(self):
        self._running = False

    def _log(self, message: str):
        self.log_ready.emit(message)

    def load_calibration(self):
        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(
                f"Calibration file '{self.calibration_file}' not found. "
                f"Run stereo_calibrate_imx219.py first."
            )

        data = np.load(self.calibration_file, allow_pickle=True)
        self.left_map1 = data["left_map1"]
        self.left_map2 = data["left_map2"]
        self.right_map1 = data["right_map1"]
        self.right_map2 = data["right_map2"]
        self.baseline_m = float(data["baseline_m"])
        camera_matrix_left = data["camera_matrix_left"]
        self.fx = float(camera_matrix_left[0, 0])

        self._log(f"Calibration loaded | fx={self.fx:.3f} px | baseline={self.baseline_m:.4f} m")

    def open_cameras(self):
        self.cap_left = cv2.VideoCapture(
            gstreamer_pipeline(LEFT_SENSOR_ID, WIDTH, HEIGHT, FPS),
            cv2.CAP_GSTREAMER
        )
        self.cap_right = cv2.VideoCapture(
            gstreamer_pipeline(RIGHT_SENSOR_ID, WIDTH, HEIGHT, FPS),
            cv2.CAP_GSTREAMER
        )

        if not self.cap_left.isOpened():
            raise RuntimeError("Failed to open LEFT camera (sensor-id=0)")
        if not self.cap_right.isOpened():
            raise RuntimeError("Failed to open RIGHT camera (sensor-id=1)")

    @staticmethod
    def get_alert_message(depth_m: float, zone_name: str) -> str:
        if depth_m > 2.0:
            return f"[{zone_name}] Clear path (> 2.0 m)"
        elif 1.5 <= depth_m <= 2.0:
            return f"[{zone_name}] Distant obstacle ({depth_m:.2f} m) - caution"
        elif 0.5 <= depth_m < 1.5:
            return f"[{zone_name}] Mid-range obstacle ({depth_m:.2f} m) - attention required"
        else:
            return f"[{zone_name}] Imminent obstacle ({depth_m:.2f} m) - immediate action"

    @staticmethod
    def get_tone(depth_m: Optional[float]) -> str:
        if depth_m is None:
            return "invalid"
        if depth_m > 2.0:
            return "clear"
        if depth_m >= 1.5:
            return "green"
        if depth_m >= 0.5:
            return "orange"
        return "red"

    @staticmethod
    def pattern_from_depth(depth_m: Optional[float]) -> MotorPattern:
        if depth_m is None or depth_m > 2.0:
            return PATTERN_OFF
        if depth_m >= 1.5:
            return PATTERN_SLOW
        if depth_m >= 0.5:
            return PATTERN_MEDIUM
        return PATTERN_CONTINUOUS

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        valid = disparity > 0.1
        depth[valid] = (self.fx * self.baseline_m) / disparity[valid]
        return depth

    def compute_zones(self, depth_in_meters: np.ndarray) -> Dict[str, ZoneResult]:
        h, w = depth_in_meters.shape
        col_step = w // 2
        results: Dict[str, ZoneResult] = {}

        for col, zone_name in enumerate(ZONE_KEYS):
            x1 = col * col_step
            x2 = (col + 1) * col_step if col < 1 else w
            y1, y2 = 0, h
            zone = depth_in_meters[y1:y2, x1:x2]

            result = ZoneResult(
                key=zone_name,
                display_name=DISPLAY_LABELS[zone_name],
                rect=(x1, y1, x2, y2),
            )

            valid_mask = np.isfinite(zone) & (zone > MIN_VALID_DEPTH) & (zone < MAX_VALID_DEPTH)
            if not np.any(valid_mask):
                result.alert_message = f"[{zone_name}] No valid depth"
                result.tone = "invalid"
                results[zone_name] = result
                continue

            masked = np.where(valid_mask, zone, np.inf)
            flat_index = int(np.argmin(masked))
            local_y, local_x = np.unravel_index(flat_index, zone.shape)
            depth_m = float(masked[local_y, local_x])

            result.depth_m = depth_m
            result.nearest_point = (x1 + int(local_x), y1 + int(local_y))
            result.alert_message = self.get_alert_message(depth_m, zone_name)
            result.tone = self.get_tone(depth_m)
            results[zone_name] = result

        return results

    @staticmethod
    def blend_zone(frame: np.ndarray, rect: Tuple[int, int, int, int], color: Tuple[int, int, int], alpha: float):
        x1, y1, x2, y2 = rect
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return
        overlay = np.full_like(roi, color)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)

    def draw_camera_overlay(self, color_frame: np.ndarray, zones: Dict[str, ZoneResult]) -> np.ndarray:
        output = color_frame.copy()

        fill_map = {
            "green": ((80, 190, 120), 0.18),
            "orange": ((0, 165, 255), 0.24),
            "red": ((40, 55, 255), 0.30),
        }

        border_map = {
            "clear": (255, 255, 255),
            "green": (110, 220, 150),
            "orange": (0, 190, 255),
            "red": (80, 90, 255),
            "invalid": (210, 210, 210),
        }

        for zone in zones.values():
            if zone.tone in fill_map:
                color, alpha = fill_map[zone.tone]
                self.blend_zone(output, zone.rect, color, alpha)

        for zone in zones.values():
            x1, y1, x2, y2 = zone.rect
            border = border_map.get(zone.tone, (255, 255, 255))
            cv2.rectangle(output, (x1, y1), (x2, y2), border, 2)

            label = f"{zone.display_name}: --" if zone.depth_m is None else f"{zone.display_name}: {zone.depth_m:.2f} m"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            label_x = x1 + 10
            label_y = y1 + 32

            cv2.rectangle(
                output,
                (label_x - 6, label_y - th - 8),
                (label_x + tw + 8, label_y + baseline + 6),
                (20, 20, 20),
                -1,
            )
            cv2.putText(
                output,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if zone.nearest_point is not None:
                px, py = zone.nearest_point
                cv2.circle(output, (px, py), 8, (255, 255, 255), 2)
                cv2.circle(output, (px, py), 4, (255, 0, 0), -1)

        return output

    def depth_to_colormap(self, depth_m: np.ndarray) -> np.ndarray:
        valid = np.isfinite(depth_m) & (depth_m >= MIN_VALID_DEPTH) & (depth_m <= MAX_VALID_DEPTH)

        normalized = np.zeros(depth_m.shape, dtype=np.uint8)
        if np.any(valid):
            clipped = np.clip(depth_m, MIN_VALID_DEPTH, MAX_VALID_DEPTH)
            normalized[valid] = ((MAX_VALID_DEPTH - clipped[valid]) / (MAX_VALID_DEPTH - MIN_VALID_DEPTH) * 255).astype(np.uint8)

        color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        color[~valid] = (0, 0, 0)
        return color

    @staticmethod
    def to_qimage_bgr(frame: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

    @Slot()
    def run(self):
        self._running = True

        try:
            self.camera_status.emit("Connecting...")
            self._log("Loading stereo calibration")
            self.load_calibration()

            self._log("Opening IMX219 binocular cameras")
            self.open_cameras()

            self.actuators = ActuatorManager()
            self.actuators.start()
            self._log(f"Actuators ready | LEFT bus={BUS_LEFT}, RIGHT bus={BUS_RIGHT}")

            for _ in range(10):
                self.cap_left.read()
                self.cap_right.read()

            self.camera_status.emit("Connected")
            self._log("Stereo capture connected")

            frame_counter = 0

            while self._running:
                ret_l, frame_left = self.cap_left.read()
                ret_r, frame_right = self.cap_right.read()

                if not ret_l or not ret_r:
                    self._log("Frame skipped: missing left or right frame")
                    continue

                left_rect = cv2.remap(frame_left, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
                right_rect = cv2.remap(frame_right, self.right_map1, self.right_map2, cv2.INTER_LINEAR)

                gray_left = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

                disparity = self.stereo.compute(gray_left, gray_right)
                depth_m = self.disparity_to_depth(disparity)
                zones = self.compute_zones(depth_m)

                left_pattern = self.pattern_from_depth(zones["left"].depth_m if "left" in zones else None)
                right_pattern = self.pattern_from_depth(zones["right"].depth_m if "right" in zones else None)

                if self.actuators is not None:
                    self.actuators.update_patterns(left_pattern, right_pattern)

                if self.last_pattern_names["left"] != left_pattern.name:
                    self._log(f"LEFT actuator -> {left_pattern.name}")
                    self.last_pattern_names["left"] = left_pattern.name

                if self.last_pattern_names["right"] != right_pattern.name:
                    self._log(f"RIGHT actuator -> {right_pattern.name}")
                    self.last_pattern_names["right"] = right_pattern.name

                live_frame = self.draw_camera_overlay(left_rect, zones)
                depth_colormap = self.depth_to_colormap(depth_m)

                self.live_ready.emit(self.to_qimage_bgr(live_frame))
                self.left_ready.emit(self.to_qimage_bgr(left_rect))
                self.right_ready.emit(self.to_qimage_bgr(right_rect))
                self.depth_ready.emit(self.to_qimage_bgr(depth_colormap))

                frame_counter += 1
                if frame_counter % LOG_EVERY_N_FRAMES == 0:
                    self._log("Frame received")
                    active_alerts = [z for z in zones.values() if z.depth_m is not None and z.depth_m <= 2.0]
                    if active_alerts:
                        nearest = min(active_alerts, key=lambda z: z.depth_m)
                        self._log(f"Alert triggered: {nearest.display_name} | {nearest.alert_message.split('] ', 1)[1]}")
                    else:
                        self._log("Alert triggered: none (clear path)")

        except Exception as exc:
            self.camera_status.emit("Error")
            self.error.emit(str(exc))
            self._log(f"ERROR: {exc}")
        finally:
            try:
                if self.actuators is not None:
                    self.actuators.stop()
                    self._log("Actuators stopped")
            except Exception:
                pass

            try:
                if self.cap_left is not None:
                    self.cap_left.release()
            except Exception:
                pass

            try:
                if self.cap_right is not None:
                    self.cap_right.release()
            except Exception:
                pass

            self.camera_status.emit("Stopped")
            self.finished.emit()


class AspectRatioVideoLabel(QLabel):
    def __init__(self, aspect_ratio: float = WIDTH / HEIGHT):
        super().__init__("Waiting for stream...")
        self.aspect_ratio = aspect_ratio
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("videoFrame")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width: int):
        return int(width / self.aspect_ratio)

    def set_frame(self, image: QImage):
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            return
        self._pixmap = pixmap
        self._update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._pixmap is None or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)


class VideoPanel(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")

        self.video_label = AspectRatioVideoLabel()

        layout.addWidget(title_label)
        layout.addWidget(self.video_label)

    def set_image(self, image: QImage):
        self.video_label.set_frame(image)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Distance-to-Haptic System Interface - IMX219 Binocular Stereo")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.thread: Optional[QThread] = None
        self.worker: Optional[StreamWorker] = None

        self._build_ui()
        self._apply_styles()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Header
        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 10, 14, 10)

        title = QLabel("Distance to haptic System Interface")
        title.setObjectName("mainTitle")

        self.status_badge = QLabel("Stopped")
        self.status_badge.setObjectName("statusBadge")

        self.start_button = QPushButton("Start Stream")
        self.stop_button = QPushButton("Stop Stream")
        self.stop_button.setEnabled(False)

        header_layout.addWidget(title, 1)
        header_layout.addWidget(self.status_badge)
        header_layout.addWidget(self.start_button)
        header_layout.addWidget(self.stop_button)

        root.addWidget(header)

        # Main content
        content = QHBoxLayout()
        content.setSpacing(10)
        root.addLayout(content, 1)

        # Left column
        left_column = QVBoxLayout()
        left_column.setSpacing(10)
        content.addLayout(left_column, 58)

        self.live_panel = VideoPanel("Life camera stream")
        left_column.addWidget(self.live_panel, 5)

        log_panel = QFrame()
        log_panel.setObjectName("panel")
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(10, 10, 10, 10)
        log_layout.setSpacing(6)

        log_title = QLabel("Log")
        log_title.setObjectName("panelTitle")

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setObjectName("logBox")
        self.log_box.setMaximumBlockCount(300)

        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log_box, 1)

        left_column.addWidget(log_panel, 2)

        # Right column
        right_column = QVBoxLayout()
        right_column.setSpacing(10)
        content.addLayout(right_column, 42)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        right_column.addLayout(top_row, 2)

        self.right_panel = VideoPanel("Right camera")
        self.left_panel = VideoPanel("Left camera")
        top_row.addWidget(self.right_panel, 1)
        top_row.addWidget(self.left_panel, 1)

        self.depth_panel = VideoPanel("colourized depth map")
        right_column.addWidget(self.depth_panel, 3)

        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)

        self._append_log("UI ready")
        self._append_log("Run stereo_calibrate_imx219.py first if calibration file does not exist")
        self._append_log("Press 'Start Stream' to begin")

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QWidget {
                background: #f2f2f2;
                color: #000000;
                font-family: Arial;
                font-size: 12px;
            }
            QFrame#header, QFrame#panel {
                background: #7aa0cd;
                border: 1px solid #5a84b7;
            }
            QLabel#mainTitle {
                font-size: 18px;
                font-weight: 500;
                color: #000000;
            }
            QLabel#panelTitle {
                font-size: 14px;
                font-weight: 500;
                color: #000000;
            }
            QLabel#statusBadge {
                background: #e2e8f0;
                color: #0f172a;
                border-radius: 12px;
                padding: 6px 12px;
                font-weight: 700;
            }
            QPushButton {
                background: #1f2937;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 14px;
                min-width: 110px;
                font-weight: 700;
            }
            QPushButton:disabled {
                background: #94a3b8;
                color: #e2e8f0;
            }
            QLabel#videoFrame {
                background: #1e293b;
                color: #ffffff;
                border: 1px solid #5a84b7;
            }
            QPlainTextEdit#logBox {
                background: #e8eef7;
                color: #000000;
                border: 1px solid #5a84b7;
                font-family: Consolas, Menlo, monospace;
                font-size: 11px;
            }
            """
        )

    def _append_log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{ts}] {message}")
        bar = self.log_box.verticalScrollBar()
        bar.setValue(bar.maximum())

    @Slot()
    def start_stream(self):
        if self.thread is not None:
            return

        self.thread = QThread()
        self.worker = StreamWorker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.live_ready.connect(self.update_live_frame)
        self.worker.left_ready.connect(self.update_left_frame)
        self.worker.right_ready.connect(self.update_right_frame)
        self.worker.depth_ready.connect(self.update_depth_frame)
        self.worker.log_ready.connect(self._append_log)
        self.worker.camera_status.connect(self.set_camera_status)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.set_camera_status("Starting...")
        self._append_log("Start button pressed")

    @Slot()
    def stop_stream(self):
        if self.worker is not None:
            self.worker.stop()
            self._append_log("Stop requested")
        self.stop_button.setEnabled(False)

    @Slot(QImage)
    def update_live_frame(self, image: QImage):
        self.live_panel.set_image(image)

    @Slot(QImage)
    def update_left_frame(self, image: QImage):
        self.left_panel.set_image(image)

    @Slot(QImage)
    def update_right_frame(self, image: QImage):
        self.right_panel.set_image(image)

    @Slot(QImage)
    def update_depth_frame(self, image: QImage):
        self.depth_panel.set_image(image)

    @Slot(str)
    def set_camera_status(self, status: str):
        self.status_badge.setText(status)
        style_map = {
            "Connected": "background:#dcfce7; color:#166534;",
            "Connecting...": "background:#dbeafe; color:#1d4ed8;",
            "Starting...": "background:#dbeafe; color:#1d4ed8;",
            "Stopped": "background:#e2e8f0; color:#0f172a;",
            "Error": "background:#fee2e2; color:#b91c1c;",
        }
        self.status_badge.setStyleSheet(
            f"QLabel#statusBadge {{ border-radius:12px; padding:6px 12px; font-weight:700; {style_map.get(status, 'background:#e2e8f0; color:#0f172a;')} }}"
        )

    @Slot(str)
    def on_worker_error(self, message: str):
        self._append_log(f"Worker error: {message}")

    @Slot()
    def on_thread_finished(self):
        self.thread = None
        self.worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.set_camera_status("Stopped")

    def closeEvent(self, event):
        self.stop_stream()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(3000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()