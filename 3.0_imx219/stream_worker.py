"""Qt worker joining the tested stereo core to UI and haptic outputs."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

import stereo_core as core
from actuators import ActuatorManager
from config import (
    DEPTH_RESULT_MAX_AGE_SECONDS,
    DISPLAY_LABELS,
    ZONE_KEYS,
)
from data_models import MotorPattern, PATTERN_OFF
from haptic_policy import pattern_from_depth
from zone_reducer import reduce_nine_to_three


@dataclass(frozen=True)
class RuntimeOptions:
    calibration: Path
    vpi_profile: Path
    backend: str = "vpi-cuda"
    left_id: int = 0
    right_id: int = 1
    max_depth_m: float = 5.0


class StreamWorker(QObject):
    camera_ready = Signal(QImage)
    depth_ready = Signal(QImage)
    zones_ready = Signal(object)
    diagnostics_ready = Signal(object)
    log_ready = Signal(str)
    camera_status = Signal(str)
    stereo_status = Signal(str)
    haptic_status = Signal(str)
    haptics_enabled = Signal(bool)
    finished = Signal()
    error = Signal(str)

    def __init__(self, options: RuntimeOptions):
        super().__init__()
        self.options = options
        self._stop_requested = threading.Event()
        self._commands: queue.SimpleQueue[tuple[str, Optional[str]]] = (
            queue.SimpleQueue()
        )
        self._actuator_lock = threading.Lock()
        self._actuators: Optional[ActuatorManager] = None
        self._haptics_are_enabled = False
        self._last_pattern_names: Dict[str, Optional[str]] = {
            key: None for key in ZONE_KEYS
        }
        self.capture: Optional[core.SynchronizedStereoCapture] = None
        self.depth_worker: Optional[core.AsyncDepthProcessor] = None
        self.left_camera = None
        self.right_camera = None

    def request_stop(self) -> None:
        self._stop_requested.set()
        self.request_emergency_stop()
        capture = self.capture
        if capture is not None:
            capture.stop()

    def request_haptics_toggle(self) -> None:
        self._commands.put(("disable" if self._haptics_are_enabled else "enable", None))

    def request_emergency_stop(self) -> None:
        # update_patterns only changes thread-safe controller commands; it does
        # not perform I2C work in the UI thread. This makes emergency stop
        # immediate even if camera capture is temporarily stalled.
        with self._actuator_lock:
            manager = self._actuators
        if manager is not None:
            manager.update_patterns({key: PATTERN_OFF for key in ZONE_KEYS})
        self._commands.put(("emergency", None))

    def _actuator_fault(self, message: str) -> None:
        self._commands.put(("fault", message))

    def _start_actuators(self) -> None:
        if self._haptics_are_enabled:
            return
        self.haptic_status.emit("Starting")
        manager = ActuatorManager(self._actuator_fault)
        try:
            manager.start()
        except Exception as error:
            self.haptic_status.emit("Error")
            self.haptics_enabled.emit(False)
            self.log_ready.emit(f"Haptic initialization failed: {error}")
            return
        with self._actuator_lock:
            self._actuators = manager
        self._haptics_are_enabled = True
        self._last_pattern_names = {key: None for key in ZONE_KEYS}
        self.haptics_enabled.emit(True)
        self.haptic_status.emit("Enabled")
        self.log_ready.emit("Haptics enabled | Left=SC2 Centre=SC3 Right=SC4")

    def _stop_actuators(self, status: str, message: str) -> None:
        self._haptics_are_enabled = False
        with self._actuator_lock:
            manager = self._actuators
            self._actuators = None
        if manager is not None:
            try:
                manager.stop()
            except Exception as error:
                self.log_ready.emit(f"Haptic shutdown error: {error}")
        self._last_pattern_names = {key: None for key in ZONE_KEYS}
        self.haptics_enabled.emit(False)
        self.haptic_status.emit(status)
        self.log_ready.emit(message)

    def _process_commands(self) -> None:
        while True:
            try:
                action, detail = self._commands.get_nowait()
            except queue.Empty:
                return
            if action == "enable":
                self._start_actuators()
            elif action == "disable":
                self._stop_actuators("Disabled", "Haptics disabled; all motors off")
            elif action == "emergency":
                self._stop_actuators(
                    "Emergency stop", "EMERGENCY STOP; all motors off"
                )
            elif action == "fault":
                self._stop_actuators(
                    "Error", f"Haptic fault; all motors off: {detail}"
                )

    @staticmethod
    def _measurement_key(measurement: core.DepthMeasurement) -> str:
        return "center" if measurement.name.lower() == "centre" else measurement.name.lower()

    @classmethod
    def patterns_for_measurements(
        cls, measurements: tuple[core.DepthMeasurement, ...]
    ) -> Dict[str, MotorPattern]:
        patterns = {key: PATTERN_OFF for key in ZONE_KEYS}
        for measurement in measurements:
            key = cls._measurement_key(measurement)
            if key in patterns:
                patterns[key] = pattern_from_depth(measurement.distance_m)
        return patterns

    def _apply_haptics(
        self, measurements: tuple[core.DepthMeasurement, ...], reason: str
    ) -> None:
        if not self._haptics_are_enabled:
            return
        patterns = self.patterns_for_measurements(measurements)
        with self._actuator_lock:
            manager = self._actuators
        if manager is None:
            return
        manager.update_patterns(patterns)
        by_key = {self._measurement_key(item): item for item in measurements}
        for key, pattern in patterns.items():
            if self._last_pattern_names[key] == pattern.name:
                continue
            measurement = by_key.get(key)
            distance = (
                "invalid"
                if measurement is None or measurement.distance_m is None
                else f"{measurement.distance_m:.2f} m"
            )
            self.log_ready.emit(
                f"{DISPLAY_LABELS[key]} haptic -> {pattern.name} | "
                f"distance={distance} | {reason}"
            )
            self._last_pattern_names[key] = pattern.name

    @staticmethod
    def _to_qimage(frame: core.np.ndarray) -> QImage:
        rgb = core.cv2.cvtColor(frame, core.cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        return QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()

    @staticmethod
    def _preview(
        frame: core.np.ndarray,
        measurements: tuple[core.DepthMeasurement, ...],
        footer: str,
    ) -> core.np.ndarray:
        annotated = frame.copy()
        core.draw_measurements(annotated, measurements)
        preview = core.cv2.resize(
            annotated,
            (core.DISPLAY_PANEL_WIDTH, core.DISPLAY_PANEL_HEIGHT),
            interpolation=core.cv2.INTER_AREA,
        )
        footer_overlay = preview.copy()
        core.cv2.rectangle(
            footer_overlay,
            (0, core.DISPLAY_PANEL_HEIGHT - 34),
            (core.DISPLAY_PANEL_WIDTH, core.DISPLAY_PANEL_HEIGHT),
            (0, 0, 0),
            -1,
        )
        core.cv2.addWeighted(
            footer_overlay,
            0.30,
            preview,
            0.70,
            0.0,
            preview,
        )
        core.cv2.putText(
            preview,
            footer,
            (12, core.DISPLAY_PANEL_HEIGHT - 11),
            core.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            core.cv2.LINE_AA,
        )
        return preview

    def _diagnostics(
        self,
        result: core.DepthResult,
        measurements: tuple[core.DepthMeasurement, ...],
    ) -> tuple[str, ...]:
        assert self.capture is not None
        left_fps, right_fps, skew_ms, dropped_left, dropped_right = (
            self.capture.synchronization_stats()
        )
        zones = " | ".join(
            f"{item.name[0]}:"
            + ("--" if item.distance_m is None else f"{item.distance_m:.2f}m")
            for item in measurements
        )
        return (
            f"Camera FPS L/R: {left_fps:.1f} / {right_fps:.1f}",
            f"Pair skew: {skew_ms:.2f} ms",
            f"Buffer drops L/R: {dropped_left} / {dropped_right}",
            f"Depth FPS: {result.depth_fps:.1f}",
            f"Valid depth: {result.valid_percentage:.1f}%",
            f"Zones {zones}",
            f"Stereo: {self.depth_worker.backend_name}",
            "Zone backend: 9 medians -> 3 nearest column medians",
        ) + result.diagnostic_lines + self.depth_worker.status_lines() + (
            f"Haptics: {'enabled' if self._haptics_are_enabled else 'disabled'}",
        )

    @Slot()
    def run(self) -> None:
        self._stop_requested.clear()
        last_capture_sequence = 0
        last_depth_sequence = -1
        last_depth_received = time.monotonic()
        stale_reported = False
        latest_result: Optional[core.DepthResult] = None
        nine_measurements = core.empty_measurements(
            (core.EXPECTED_HEIGHT, core.EXPECTED_WIDTH)
        )
        measurements = reduce_nine_to_three(nine_measurements)

        try:
            self.camera_status.emit("Starting")
            self.stereo_status.emit("Loading")
            self.haptic_status.emit("Disabled")
            self.haptics_enabled.emit(False)
            core.check_gstreamer()
            calibration = core.load_calibration(self.options.calibration)
            maps = core.convert_rectification_maps(calibration)
            settings = core.default_vpi_settings()
            profile_source = "built-in defaults"
            if self.options.backend == "vpi-cuda":
                settings, profile_source = core.load_vpi_profile(
                    calibration, self.options.vpi_profile
                )

            self.depth_worker = core.AsyncDepthProcessor(
                self.options.backend,
                calibration,
                maps,
                self.options.max_depth_m,
                settings,
                profile_source,
            )
            self.depth_worker.start()
            self.stereo_status.emit(
                "VPI CUDA ready"
                if self.options.backend == "vpi-cuda"
                else "OpenCV CPU ready"
            )
            self.log_ready.emit(
                f"Stereo ready | {self.depth_worker.backend_name} | "
                f"calibration={calibration.path.name}"
            )

            self.left_camera, self.right_camera = core.open_cameras(
                self.options.left_id, self.options.right_id, calibration
            )
            self.capture = core.SynchronizedStereoCapture(
                self.left_camera, self.right_camera
            )
            self.capture.start()
            self.camera_status.emit("Connected")
            self.log_ready.emit(
                f"IMX219 cameras connected | left={self.options.left_id} | "
                f"right={self.options.right_id}"
            )

            while not self._stop_requested.is_set():
                self._process_commands()
                stereo_frame = self.capture.get_latest(last_capture_sequence)
                last_capture_sequence = stereo_frame.sequence
                core.validate_frame_resolution(
                    stereo_frame.left, stereo_frame.right, calibration
                )
                current_left = core.rectify_left_frame(stereo_frame.left, maps)
                self.depth_worker.submit(
                    stereo_frame.sequence, current_left, stereo_frame.right
                )

                result = self.depth_worker.latest_result()
                if result is not None and result.sequence > last_depth_sequence:
                    latest_result = result
                    last_depth_sequence = result.sequence
                    last_depth_received = time.monotonic()
                    stale_reported = False
                    nine_measurements = result.measurements
                    measurements = reduce_nine_to_three(nine_measurements)
                    self._apply_haptics(measurements, "fresh zone value")
                    self.zones_ready.emit(measurements)
                    self.depth_ready.emit(
                        self._to_qimage(
                            self._preview(
                                result.depth_view,
                                measurements,
                                "DEPTH: NEAR RED | FAR BLUE",
                            )
                        )
                    )
                    self.diagnostics_ready.emit(
                        self._diagnostics(result, measurements)
                    )

                if (
                    latest_result is not None
                    and time.monotonic() - last_depth_received
                    > DEPTH_RESULT_MAX_AGE_SECONDS
                    and not stale_reported
                ):
                    nine_measurements = core.empty_measurements(
                        (calibration.height, calibration.width)
                    )
                    measurements = reduce_nine_to_three(nine_measurements)
                    self._apply_haptics(measurements, "stale-depth safety stop")
                    self.zones_ready.emit(measurements)
                    self.log_ready.emit("Depth result stale; all zone values invalid")
                    stale_reported = True

                self.camera_ready.emit(
                    self._to_qimage(
                        self._preview(
                            current_left, measurements, "RECTIFIED LEFT"
                        )
                    )
                )

            self.log_ready.emit("Stream stop requested")
        except Exception as error:
            if not self._stop_requested.is_set():
                self.camera_status.emit("Error")
                self.stereo_status.emit("Error")
                self.error.emit(str(error))
                self.log_ready.emit(f"ERROR: {error}")
        finally:
            self._stop_actuators("Disabled", "All haptic motors safely stopped")
            if self.capture is not None:
                self.capture.stop()
                self.capture = None
            if self.depth_worker is not None:
                self.depth_worker.stop()
                self.depth_worker = None
            if self.left_camera is not None:
                self.left_camera.release()
                self.left_camera = None
            if self.right_camera is not None:
                self.right_camera.release()
                self.right_camera = None
            self.camera_status.emit("Stopped")
            self.stereo_status.emit("Stopped")
            self.finished.emit()
