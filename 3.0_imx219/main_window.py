import time
from typing import Dict, Optional

from PySide6.QtCore import QThread, Slot
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import ZONE_KEYS
from data_models import SceneResult, ZoneResult
from stream_worker import StreamWorker
from ui_components import VideoPanel, ZoneCard


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Distance-to-Haptic System - IMX219 Stereo")
        self.setMinimumSize(800, 520)

        self.thread: Optional[QThread] = None
        self.worker: Optional[StreamWorker] = None

        self._build_ui()
        self._apply_styles()
        self._set_initial_window_geometry()

    def _set_initial_window_geometry(self):
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1200, 700)
            return

        available = screen.availableGeometry()
        width = min(1200, max(self.minimumWidth(), available.width() - 80))
        height = min(760, max(self.minimumHeight(), available.height() - 80))
        self.resize(width, height)

        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

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
        subtitle = QLabel("FYP2 · 8 MP IMX219 Binocular Navigation Assist")
        subtitle.setObjectName("mainSubtitle")
        header_left.addWidget(title)
        header_left.addWidget(subtitle)

        header_right = QHBoxLayout()
        header_right.setSpacing(8)
        self.status_badge = QLabel("Stopped")
        self.status_badge.setObjectName("statusBadge")
        self.stereo_badge = QLabel("Stereo stopped")
        self.stereo_badge.setObjectName("stereoBadge")
        self.stereo_badge.setMaximumWidth(190)
        self.detector_badge = QLabel("YOLO stopped")
        self.detector_badge.setObjectName("detectorBadge")
        self.detector_badge.setMaximumWidth(190)
        self.start_button = QPushButton("Start Stream")
        self.stop_button = QPushButton("Stop Stream")
        self.stop_button.setEnabled(False)
        header_right.addWidget(self.status_badge)
        header_right.addWidget(self.stereo_badge)
        header_right.addWidget(self.detector_badge)
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
            "Rectified Left IMX219",
            "Confidence-supported stereo zones and informational YOLO overlays",
        )
        dashboard.addWidget(self.camera_panel, 0, 0)

        self.depth_panel = VideoPanel(
            "Stereo Depth",
            "Metric stereo depth: near red, far blue, invalid black",
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
        zone_subtitle = QLabel(
            "Nearest supported stereo depth (thresholds: 2.0 / 1.5 / 0.5 m)"
        )
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
            QLabel#detectorBadge {
                background: #e2e8f0;
                color: #0f172a;
                border-radius: 12px;
                padding: 6px 12px;
                font-weight: 700;
            }
            QLabel#stereoBadge {
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
                min-width: 84px;
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
            QLabel#zoneObject {
                color: #334155;
                font-size: 10px;
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
        self.worker.detections_ready.connect(self.update_detections)
        self.worker.log_ready.connect(self._append_log)
        self.worker.camera_status.connect(self.set_camera_status)
        self.worker.stereo_status.connect(self.set_stereo_status)
        self.worker.detector_status.connect(self.set_detector_status)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.set_camera_status("Starting...")
        self.set_stereo_status("Stereo starting...")
        self.set_detector_status("YOLO starting...")
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

    @Slot(object)
    def update_detections(self, scene: SceneResult):
        for zone_key, card in self.zone_cards.items():
            candidates = [
                detection
                for detection in scene.detections
                if zone_key in detection.zones
            ]
            if not candidates:
                card.update_detection(None)
                continue

            with_depth = [item for item in candidates if item.depth_m is not None]
            nearest = (
                min(with_depth, key=lambda item: item.depth_m)
                if with_depth
                else max(candidates, key=lambda item: item.confidence)
            )
            card.update_detection(nearest)

    @Slot(str)
    def set_detector_status(self, status: str):
        normalized = status.lower()
        if "ready" in normalized and "cpu" not in normalized:
            display_status = "YOLO · CUDA FP16"
        elif "ready" in normalized and "cpu" in normalized:
            display_status = "YOLO · CPU"
        elif "loading" in normalized or "starting" in normalized:
            display_status = "YOLO · Loading"
        elif "error" in normalized or "disabled:" in normalized:
            display_status = "YOLO · Error"
        elif "disabled" in normalized:
            display_status = "YOLO · Disabled"
        else:
            display_status = "YOLO · Stopped"

        self.detector_badge.setText(display_status)
        self.detector_badge.setToolTip(status)
        if "ready" in normalized and "cpu" not in normalized:
            colors = "background:#dcfce7; color:#166534;"
        elif "cpu" in normalized or "loading" in normalized or "starting" in normalized:
            colors = "background:#fef3c7; color:#92400e;"
        elif "error" in normalized or "disabled:" in normalized:
            colors = "background:#fee2e2; color:#b91c1c;"
        else:
            colors = "background:#e2e8f0; color:#0f172a;"
        self.detector_badge.setStyleSheet(
            "QLabel#detectorBadge { border-radius:12px; padding:6px 12px; "
            f"font-weight:700; {colors} }}"
        )

    @Slot(str)
    def set_stereo_status(self, status: str):
        normalized = status.lower()
        if "ready" in normalized and "vpi" in normalized:
            display_status = "Stereo · VPI CUDA"
            colors = "background:#dcfce7; color:#166534;"
        elif "ready" in normalized or "fallback" in normalized:
            display_status = "Stereo · SGBM CPU"
            colors = "background:#fef3c7; color:#92400e;"
        elif "loading" in normalized or "starting" in normalized:
            display_status = "Stereo · Loading"
            colors = "background:#dbeafe; color:#1d4ed8;"
        elif "error" in normalized or "failed" in normalized:
            display_status = "Stereo · Error"
            colors = "background:#fee2e2; color:#b91c1c;"
        else:
            display_status = "Stereo · Stopped"
            colors = "background:#e2e8f0; color:#0f172a;"
        self.stereo_badge.setText(display_status)
        self.stereo_badge.setToolTip(status)
        self.stereo_badge.setStyleSheet(
            "QLabel#stereoBadge { border-radius:12px; padding:6px 12px; "
            f"font-weight:700; {colors} }}"
        )

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
        self.set_stereo_status("Stereo stopped")
        self.set_detector_status("YOLO stopped")

    def closeEvent(self, event):
        self.stop_stream()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(10000)
        super().closeEvent(event)
