import sys
import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from smbus2 import SMBus
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


WIDTH = 640
HEIGHT = 480
FPS = 30
MIN_VALID_DEPTH = 0.1
MAX_VALID_DEPTH = 4.0
LOG_EVERY_N_FRAMES = 10
FRAME_ASPECT_RATIO = WIDTH / HEIGHT

ZONE_KEYS = ["left", "right"]
DISPLAY_LABELS = {
    "left": "Left",
    "right": "Right",
}

# =========================
# DA7280 actuator settings from vibrate.py
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
PATTERN_SLOW = MotorPattern("slow pulse (1 Hz)", 0x30, 0.50, 0.50)       # ~38% intensity
PATTERN_MEDIUM = MotorPattern("medium pulse (5 Hz)", 0x60, 0.08, 0.12)   # ~75% intensity
PATTERN_CONTINUOUS = MotorPattern("continuous", 0x7F, 0.0, 0.0)          # 100% intensity


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

                # Pulse mode
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


class StreamWorker(QObject):
    color_ready = Signal(QImage)
    depth_ready = Signal(QImage)
    zones_ready = Signal(object)
    log_ready = Signal(str)
    camera_status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self.pipeline = None
        self.actuators: Optional[ActuatorManager] = None
        self.last_pattern_names = {"left": None, "right": None}

    def stop(self):
        self._running = False

    def _log(self, message: str):
        self.log_ready.emit(message)

    @staticmethod
    def get_alert_message(depth_m: float, zone_name: str) -> str:
        # Thresholds from the image the user provided
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

            valid_mask = (zone > MIN_VALID_DEPTH) & (zone < MAX_VALID_DEPTH)
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
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
            label_x = x1 + 10
            label_y = y1 + 28
            box_end_x = min(label_x + tw + 12, x2 - 10)
            cv2.rectangle(
                output,
                (label_x - 4, label_y - th - 6),
                (box_end_x, label_y + baseline + 4),
                (20, 20, 20),
                -1,
            )
            cv2.putText(
                output,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if zone.nearest_point is not None:
                px, py = zone.nearest_point
                cv2.circle(output, (px, py), 7, (255, 255, 255), 2)
                cv2.circle(output, (px, py), 4, (255, 0, 0), -1)

        cv2.rectangle(output, (8, output.shape[0] - 38), (205, output.shape[0] - 8), (20, 20, 20), -1)
        cv2.putText(output, "Blue dot = nearest pixel", (16, output.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(output, (output.shape[1] - 205, output.shape[0] - 38), (output.shape[1] - 8, output.shape[0] - 8), (20, 20, 20), -1)
        cv2.putText(output, "Clear (> 2.0 m) = no fill", (output.shape[1] - 193, output.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return output

    @staticmethod
    def to_qimage_bgr(frame: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

    @Slot()
    def run(self):
        self._running = True
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        align = rs.align(rs.stream.color)

        try:
            self.camera_status.emit("Connecting...")
            self._log("Starting RealSense pipeline")
            self.pipeline.start(config)

            # Start actuators
            self.actuators = ActuatorManager()
            self.actuators.start()
            self._log(f"Actuators ready | LEFT bus={BUS_LEFT}, RIGHT bus={BUS_RIGHT}")

            for _ in range(10):
                self.pipeline.wait_for_frames()

            profile = self.pipeline.get_active_profile()
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            self.camera_status.emit("Connected")
            self._log(f"Camera connected | depth scale = {depth_scale:.6f} m/unit")

            frame_counter = 0
            while self._running:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    self._log("Frame skipped: missing color or depth frame")
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_in_meters = depth_image * depth_scale
                zones = self.compute_zones(depth_in_meters)

                # Update actuators
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

                overlay_frame = self.draw_camera_overlay(color_image, zones)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET,
                )

                self.color_ready.emit(self.to_qimage_bgr(overlay_frame))
                self.depth_ready.emit(self.to_qimage_bgr(depth_colormap))
                self.zones_ready.emit(zones)

                frame_counter += 1
                if frame_counter % LOG_EVERY_N_FRAMES == 0:
                    self._log("Frame received")
                    self._log("Depth updated")
                    self._log("Zone distance computed")

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
                if self.pipeline is not None:
                    self.pipeline.stop()
                    self._log("RealSense pipeline stopped")
            except Exception:
                pass
            self.camera_status.emit("Stopped")
            self.finished.emit()


class AspectRatioVideoLabel(QLabel):
    def __init__(self, aspect_ratio: float = FRAME_ASPECT_RATIO):
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

    def minimumSizeHint(self):
        return QPixmap(WIDTH, HEIGHT).size()

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
    def __init__(self, title: str, subtitle: str):
        super().__init__()
        self.setObjectName("panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("panelSubtitle")

        self.video_label = AspectRatioVideoLabel()

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addWidget(self.video_label)

    def set_image(self, image: QImage):
        self.video_label.set_frame(image)


class ZoneCard(QFrame):
    def __init__(self, zone_name: str):
        super().__init__()
        self.zone_name = zone_name
        self.setObjectName("zoneCard")
        self.setMinimumHeight(88)
        self.setMaximumHeight(96)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.name_label = QLabel(DISPLAY_LABELS[zone_name])
        self.name_label.setObjectName("zoneName")
        self.value_label = QLabel("--")
        self.value_label.setObjectName("zoneValue")

        layout.addWidget(self.name_label)
        layout.addWidget(self.value_label)

        self.set_tone("invalid")

    def set_tone(self, tone: str):
        style_map = {
            "clear": "background:#f8fafc; border:1px solid #d7dee7;",
            "green": "background:#ecfdf5; border:1px solid #a7f3d0;",
            "orange": "background:#fff7ed; border:1px solid #fdba74;",
            "red": "background:#fef2f2; border:1px solid #fca5a5;",
            "invalid": "background:#f8fafc; border:1px solid #d7dee7;",
        }
        self.setStyleSheet(f"QFrame#zoneCard {{{style_map.get(tone, style_map['invalid'])} border-radius:14px;}}")

    def update_zone(self, result: ZoneResult):
        if result.depth_m is None:
            self.value_label.setText("--")
            self.set_tone("invalid")
            return

        self.value_label.setText(f"{result.depth_m:.2f} m")
        self.set_tone(result.tone)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Distance-to-Haptic System Interface - FYP1")
        self.resize(1366, 768)

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

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 10, 14, 10)

        header_left = QVBoxLayout()
        header_left.setSpacing(2)
        title = QLabel("Distance-to-Haptic System Interface")
        title.setObjectName("mainTitle")
        subtitle = QLabel("FYP1 : VIP Indoor Navigation Assist Interface")
        subtitle.setObjectName("mainSubtitle")
        header_left.addWidget(title)
        header_left.addWidget(subtitle)

        header_right = QHBoxLayout()
        header_right.setSpacing(8)
        self.status_badge = QLabel("Stopped")
        self.status_badge.setObjectName("statusBadge")
        self.start_button = QPushButton("Start Stream")
        self.stop_button = QPushButton("Stop Stream")
        self.stop_button.setEnabled(False)
        header_right.addWidget(self.status_badge)
        header_right.addWidget(self.start_button)
        header_right.addWidget(self.stop_button)

        header_layout.addLayout(header_left, 1)
        header_layout.addLayout(header_right)
        root.addWidget(header)

        dashboard = QGridLayout()
        dashboard.setHorizontalSpacing(10)
        dashboard.setVerticalSpacing(10)
        dashboard.setColumnStretch(0, 5)
        dashboard.setColumnStretch(1, 3)
        dashboard.setRowStretch(0, 5)
        dashboard.setRowStretch(1, 3)
        root.addLayout(dashboard, 1)

        self.camera_panel = VideoPanel(
            "Live Camera Stream",
            "2-column overlay with actuator-triggered alerts",
        )
        dashboard.addWidget(self.camera_panel, 0, 0)

        self.depth_panel = VideoPanel(
            "Depth Video",
            "Depth colormap stream",
        )
        dashboard.addWidget(self.depth_panel, 0, 1)

        log_panel = QFrame()
        log_panel.setObjectName("panel")
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(12, 12, 12, 12)
        log_layout.setSpacing(6)

        log_title = QLabel("Stream Logging")
        log_title.setObjectName("panelTitle")
        log_subtitle = QLabel("Camera, alerts, and actuator status")
        log_subtitle.setObjectName("panelSubtitle")

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setObjectName("logBox")
        self.log_box.setMaximumBlockCount(300)

        log_layout.addWidget(log_title)
        log_layout.addWidget(log_subtitle)
        log_layout.addWidget(self.log_box, 1)
        dashboard.addWidget(log_panel, 1, 0)

        zone_panel = QFrame()
        zone_panel.setObjectName("panel")
        zone_layout = QVBoxLayout(zone_panel)
        zone_layout.setContentsMargins(12, 12, 12, 12)
        zone_layout.setSpacing(6)

        zone_title = QLabel("Zone Values")
        zone_title.setObjectName("panelTitle")
        zone_subtitle = QLabel("2-column distance values (thresholds: 2.0 / 1.5 / 0.5 m)")
        zone_subtitle.setObjectName("panelSubtitle")
        zone_grid = QGridLayout()
        zone_grid.setHorizontalSpacing(8)
        zone_grid.setVerticalSpacing(8)

        self.zone_cards: Dict[str, ZoneCard] = {}
        for idx, zone_key in enumerate(ZONE_KEYS):
            card = ZoneCard(zone_key)
            self.zone_cards[zone_key] = card
            zone_grid.addWidget(card, 0, idx)

        zone_layout.addWidget(zone_title)
        zone_layout.addWidget(zone_subtitle)
        zone_layout.addLayout(zone_grid)
        dashboard.addWidget(zone_panel, 1, 1)

        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)

        self._append_log("UI ready")
        self._append_log("Press 'Start Stream' to begin")

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QWidget {
                background: #f3f6fa;
                color: #0f172a;
                font-family: Arial;
                font-size: 12px;
            }
            QFrame#header, QFrame#panel {
                background: white;
                border: 1px solid #dbe3ec;
                border-radius: 16px;
            }
            QLabel#mainTitle {
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#mainSubtitle {
                color: #64748b;
                font-size: 11px;
            }
            QLabel#panelTitle {
                font-size: 14px;
                font-weight: 700;
            }
            QLabel#panelSubtitle {
                color: #64748b;
                font-size: 10px;
            }
            QLabel#statusBadge {
                background: #e2e8f0;
                color: #0f172a;
                border-radius: 12px;
                padding: 6px 12px;
                font-weight: 700;
            }
            QPushButton {
                background: #111827;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 8px 14px;
                font-weight: 700;
                min-width: 100px;
            }
            QPushButton:disabled {
                background: #94a3b8;
                color: #e2e8f0;
            }
            QLabel#videoFrame {
                background: #0f172a;
                color: #e2e8f0;
                border: 1px solid #dbe3ec;
                border-radius: 14px;
            }
            QLabel#zoneName {
                color: #64748b;
                font-size: 10px;
            }
            QLabel#zoneValue {
                font-size: 21px;
                font-weight: 700;
                color: #0f172a;
            }
            QPlainTextEdit#logBox {
                background: #020617;
                color: #86efac;
                border: 1px solid #0f172a;
                border-radius: 12px;
                font-family: Consolas, Menlo, monospace;
                font-size: 11px;
                padding: 6px;
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
        self.worker.color_ready.connect(self.update_camera_frame)
        self.worker.depth_ready.connect(self.update_depth_frame)
        self.worker.zones_ready.connect(self.update_zones)
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
    def update_camera_frame(self, image: QImage):
        self.camera_panel.set_image(image)

    @Slot(QImage)
    def update_depth_frame(self, image: QImage):
        self.depth_panel.set_image(image)

    @Slot(object)
    def update_zones(self, zones: Dict[str, ZoneResult]):
        for zone_key, card in self.zone_cards.items():
            if zone_key in zones:
                card.update_zone(zones[zone_key])

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
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
