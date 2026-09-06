"""Responsive PySide6 interface matching the approved dashboard draft."""

from __future__ import annotations

import time
from typing import Dict, Optional

from PySide6.QtCore import QThread, Slot
from PySide6.QtGui import QCloseEvent, QGuiApplication, QImage
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import (
    WINDOW_DEFAULT_HEIGHT,
    WINDOW_DEFAULT_WIDTH,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    ZONE_KEYS,
)
from stream_worker import RuntimeOptions, StreamWorker
from ui_components import StatusBadge, VideoPanel, ZoneCard


class MainWindow(QMainWindow):
    def __init__(self, options: RuntimeOptions):
        super().__init__()
        self.options = options
        self.thread: Optional[QThread] = None
        self.worker: Optional[StreamWorker] = None
        self.setWindowTitle(
            "Low-Cost Binocular Object-Aware Distance-to-Haptic Navigation System"
        )
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self._build_ui()
        self._apply_styles()
        self._set_initial_geometry()

    def _set_initial_geometry(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)
            return
        available = screen.availableGeometry()
        width = min(WINDOW_DEFAULT_WIDTH, max(WINDOW_MIN_WIDTH, available.width() - 40))
        height = min(WINDOW_DEFAULT_HEIGHT, max(WINDOW_MIN_HEIGHT, available.height() - 40))
        self.resize(width, height)
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(62)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 8, 14, 8)
        header_layout.setSpacing(8)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("Object-Aware Distance-to-Haptic System")
        title.setObjectName("mainTitle")
        subtitle = QLabel(
            "Dual IMX219 stereo depth · YOLO11n detection · three-zone feedback"
        )
        subtitle.setObjectName("mainSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box, 1)

        self.camera_badge = StatusBadge("Camera")
        self.stereo_badge = StatusBadge("CUDA")
        self.detector_badge = StatusBadge("YOLO")
        self.haptic_badge = StatusBadge("Haptic")
        header_layout.addWidget(self.camera_badge)
        header_layout.addWidget(self.stereo_badge)
        header_layout.addWidget(self.detector_badge)
        header_layout.addWidget(self.haptic_badge)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.haptic_button = QPushButton("Enable Haptics")
        self.emergency_button = QPushButton("EMERGENCY STOP")
        self.emergency_button.setObjectName("emergencyButton")
        self.stop_button.setEnabled(False)
        self.haptic_button.setEnabled(False)
        self.emergency_button.setEnabled(False)
        header_layout.addWidget(self.start_button)
        header_layout.addWidget(self.stop_button)
        header_layout.addWidget(self.haptic_button)
        header_layout.addWidget(self.emergency_button)
        root.addWidget(header)

        content = QHBoxLayout()
        content.setSpacing(10)
        root.addLayout(content, 1)

        left_column = QVBoxLayout()
        left_column.setSpacing(10)
        self.camera_panel = VideoPanel("STREAM VIDEO", "Waiting for IMX219 stream...")
        left_column.addWidget(self.camera_panel, 77)

        cards = QHBoxLayout()
        cards.setSpacing(10)
        self.zone_cards: Dict[str, ZoneCard] = {}
        for key in ZONE_KEYS:
            card = ZoneCard(key)
            self.zone_cards[key] = card
            cards.addWidget(card, 1)
        left_column.addLayout(cards, 23)
        content.addLayout(left_column, 65)

        right_column = QVBoxLayout()
        right_column.setSpacing(10)
        self.depth_panel = VideoPanel("DEPTH HEATMAP", "Waiting for CUDA depth...")
        right_column.addWidget(self.depth_panel, 1)

        diagnostics_panel = QFrame()
        diagnostics_panel.setObjectName("panel")
        diagnostics_layout = QVBoxLayout(diagnostics_panel)
        diagnostics_layout.setContentsMargins(10, 8, 10, 10)
        diagnostics_title = QLabel("SYSTEM DIAGNOSTICS")
        diagnostics_title.setObjectName("panelTitle")
        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setObjectName("diagnosticsBox")
        self.diagnostics.setReadOnly(True)
        diagnostics_layout.addWidget(diagnostics_title)
        diagnostics_layout.addWidget(self.diagnostics, 1)
        right_column.addWidget(diagnostics_panel, 1)

        log_panel = QFrame()
        log_panel.setObjectName("panel")
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(10, 8, 10, 10)
        log_title = QLabel("LOG")
        log_title.setObjectName("panelTitle")
        self.log = QPlainTextEdit()
        self.log.setObjectName("logBox")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(250)
        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log, 1)
        right_column.addWidget(log_panel, 1)
        content.addLayout(right_column, 35)

        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.haptic_button.clicked.connect(self.toggle_haptics)
        self.emergency_button.clicked.connect(self.emergency_stop)
        self._append_log("UI ready; haptics are disabled")

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background:#eef2f7; color:#172033; font-family:Arial; font-size:12px; }
            QFrame#header, QFrame#panel, QFrame#zoneCard {
                background:white; border:1px solid #cbd5e1; border-radius:10px;
            }
            QLabel#mainTitle { font-size:18px; font-weight:700; }
            QLabel#mainSubtitle { color:#64748b; font-size:10px; }
            QLabel#panelTitle { color:#334155; font-size:11px; font-weight:700; }
            QLabel#videoFrame { background:#020617; color:#94a3b8; border-radius:7px; }
            QLabel#zoneName { color:#64748b; font-size:12px; font-weight:700; }
            QLabel#zoneValue { color:#0f172a; font-size:30px; font-weight:700; }
            QPushButton {
                background:#2563eb; color:white; border:none; border-radius:8px;
                padding:8px 11px; font-weight:700;
            }
            QPushButton:hover { background:#1d4ed8; }
            QPushButton:disabled { background:#94a3b8; color:#e2e8f0; }
            QPushButton#emergencyButton { background:#dc2626; }
            QPushButton#emergencyButton:hover { background:#b91c1c; }
            QPlainTextEdit#diagnosticsBox {
                background:#0f172a; color:#e2e8f0; border:none; border-radius:6px;
                font-family:Monospace; font-size:10px; padding:5px;
            }
            QPlainTextEdit#logBox {
                background:#020617; color:#86efac; border:none; border-radius:6px;
                font-family:Monospace; font-size:10px; padding:5px;
            }
            """
        )

    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {message}")
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot()
    def start_stream(self) -> None:
        if self.thread is not None:
            return
        self.thread = QThread(self)
        self.worker = StreamWorker(self.options)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.camera_ready.connect(self.camera_panel.set_image)
        self.worker.depth_ready.connect(self.depth_panel.set_image)
        self.worker.zones_ready.connect(self.update_zones)
        self.worker.diagnostics_ready.connect(self.update_diagnostics)
        self.worker.log_ready.connect(self._append_log)
        self.worker.camera_status.connect(self.camera_badge.set_status)
        self.worker.stereo_status.connect(self.stereo_badge.set_status)
        self.worker.detector_status.connect(self.detector_badge.set_status)
        self.worker.haptic_status.connect(self.haptic_badge.set_status)
        self.worker.haptics_enabled.connect(self.set_haptics_enabled)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.haptic_button.setEnabled(True)
        self.emergency_button.setEnabled(True)
        self._append_log("Starting stereo stream")

    @Slot()
    def stop_stream(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
        self.stop_button.setEnabled(False)
        self.haptic_button.setEnabled(False)
        self.emergency_button.setEnabled(False)

    @Slot()
    def toggle_haptics(self) -> None:
        if self.worker is not None:
            self.worker.request_haptics_toggle()

    @Slot()
    def emergency_stop(self) -> None:
        if self.worker is not None:
            self.worker.request_emergency_stop()
        self._append_log("Emergency stop requested")

    @Slot(bool)
    def set_haptics_enabled(self, enabled: bool) -> None:
        self.haptic_button.setText("Disable Haptics" if enabled else "Enable Haptics")

    @Slot(object)
    def update_zones(self, measurements) -> None:
        for measurement in measurements:
            key = "center" if measurement.name.lower() == "centre" else measurement.name.lower()
            if key in self.zone_cards:
                self.zone_cards[key].set_distance(measurement.distance_m)

    @Slot(object)
    def update_diagnostics(self, lines) -> None:
        self.diagnostics.setPlainText("\n".join(lines))

    @Slot(str)
    def on_worker_error(self, message: str) -> None:
        self._append_log(f"Worker error: {message}")

    @Slot()
    def on_thread_finished(self) -> None:
        self.thread = None
        self.worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.haptic_button.setEnabled(False)
        self.haptic_button.setText("Enable Haptics")
        self.emergency_button.setEnabled(False)
        self.detector_badge.set_status("Stopped")
        for card in self.zone_cards.values():
            card.set_distance(None)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_stream()
        thread = self.thread
        if thread is not None:
            thread.wait(10000)
        event.accept()
