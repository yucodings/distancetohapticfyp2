"""Qt worker orchestrating stereo capture, depth, YOLO, and haptics."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from camera_backend import check_gstreamer, cv2, np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

from actuators import ActuatorManager
from calibration import (
    Calibration,
    RectificationMaps,
    convert_rectification_maps,
    load_calibration,
    rectify_left,
)
from config import (
    CALIBRATION_PATH,
    CAMERA_PAIR_TIMEOUT_SECONDS,
    DEPTH_RESULT_MAX_AGE_MS,
    ENABLE_ACTUATORS,
    ENABLE_YOLO,
    HEIGHT,
    LEFT_SENSOR_ID,
    LOG_EVERY_N_RESULTS,
    RIGHT_SENSOR_ID,
    STEREO_BACKEND,
    WIDTH,
    YOLO_FRAME_INTERVAL,
    YOLO_RESULT_MAX_AGE_MS,
    ZONE_KEYS,
)
from data_models import DetectionResult, FramePacket, MotorPattern, SceneResult, ZoneResult
from hazard_policy import pattern_from_depth
from stereo_capture import (
    SynchronizedStereoCapture,
    open_cameras,
    validate_frame_resolution,
)
from stereo_depth import AsyncDepthProcessor, DepthResult
from zone_depth import compute_stereo_depth_zones, empty_stereo_zones

if TYPE_CHECKING:
    from inference_worker import LatestFrameInferenceWorker


class StreamWorker(QObject):
    color_ready = Signal(QImage)
    depth_ready = Signal(QImage)
    zones_ready = Signal(object)
    detections_ready = Signal(object)
    log_ready = Signal(str)
    camera_status = Signal(str)
    stereo_status = Signal(str)
    detector_status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self.left_camera = None
        self.right_camera = None
        self.capture: Optional[SynchronizedStereoCapture] = None
        self.depth_worker: Optional[AsyncDepthProcessor] = None
        self.inference_worker: Optional["LatestFrameInferenceWorker"] = None
        self.actuators: Optional[ActuatorManager] = None
        self.calibration: Optional[Calibration] = None
        self.maps: Optional[RectificationMaps] = None
        self.latest_depth: Optional[DepthResult] = None
        self.latest_scene: Optional[SceneResult] = None
        self.last_pattern_names = {zone_name: None for zone_name in ZONE_KEYS}

    def stop(self) -> None:
        self._running = False

    def _log(self, message: str) -> None:
        self.log_ready.emit(message)

    def _stereo_update(self, message: str) -> None:
        self.stereo_status.emit(message)
        self._log(message)

    def _detector_update(self, message: str) -> None:
        self.detector_status.emit(message)
        self._log(message)

    @staticmethod
    def blend_zone(
        frame: np.ndarray,
        rect: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        alpha: float,
    ) -> None:
        x1, y1, x2, y2 = rect
        roi = frame[y1:y2, x1:x2]
        if roi.size:
            overlay = np.full_like(roi, color)
            cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)

    @staticmethod
    def draw_detection_overlays(
        frame: np.ndarray, detections: List[DetectionResult]
    ) -> None:
        color_map = {
            "clear": (160, 220, 160),
            "green": (110, 220, 150),
            "orange": (0, 190, 255),
            "red": (80, 90, 255),
            "invalid": (220, 220, 220),
        }
        height, width = frame.shape[:2]
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = color_map.get(detection.tone, color_map["invalid"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            distance = (
                "-- m"
                if detection.depth_m is None
                else f"{detection.depth_m:.2f} m"
            )
            label = (
                f"YOLO {detection.label} {detection.confidence:.0%} | {distance}"
            )
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
            )
            text_x = max(0, min(x1, width - text_width - 8))
            text_y = y1 - 7 if y1 - text_height - 10 >= 0 else y1 + text_height + 10
            cv2.rectangle(
                frame,
                (text_x, max(0, text_y - text_height - 5)),
                (
                    min(width - 1, text_x + text_width + 7),
                    min(height - 1, text_y + baseline + 3),
                ),
                (20, 20, 20),
                -1,
            )
            cv2.putText(
                frame,
                label,
                (text_x + 3, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

    def draw_camera_overlay(
        self,
        frame: np.ndarray,
        zones: Dict[str, ZoneResult],
        detections: List[DetectionResult],
    ) -> np.ndarray:
        output = frame.copy()
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
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                border_map.get(zone.tone, (255, 255, 255)),
                2,
            )
            label = (
                f"STEREO {zone.display_name}: --"
                if zone.depth_m is None
                else f"STEREO {zone.display_name}: {zone.depth_m:.2f} m"
            )
            cv2.rectangle(output, (x1 + 6, 6), (min(x2 - 6, x1 + 245), 38), (20, 20, 20), -1)
            cv2.putText(
                output,
                label,
                (x1 + 12, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if zone.nearest_point is not None:
                cv2.circle(output, zone.nearest_point, 7, (255, 255, 255), 2)
                cv2.circle(output, zone.nearest_point, 4, (255, 0, 0), -1)

        self.draw_detection_overlays(output, detections)
        cv2.rectangle(
            output,
            (8, output.shape[0] - 38),
            (260, output.shape[0] - 8),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            output,
            "Blue dot = supported stereo surface",
            (16, output.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return output

    @staticmethod
    def to_qimage_bgr(frame: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        return QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()

    def _patterns_for_zones(
        self, zones: Dict[str, ZoneResult]
    ) -> Dict[str, MotorPattern]:
        return {
            zone_name: pattern_from_depth(zones[zone_name].depth_m)
            for zone_name in ZONE_KEYS
        }

    def _apply_patterns(
        self,
        zones: Dict[str, ZoneResult],
        reason: str,
    ) -> None:
        patterns = self._patterns_for_zones(zones)
        if self.actuators is not None:
            self.actuators.update_patterns(patterns)
        for zone_name, pattern in patterns.items():
            if self.last_pattern_names[zone_name] == pattern.name:
                continue
            depth = zones[zone_name].depth_m
            distance = "invalid" if depth is None else f"{depth:.3f} m"
            self._log(
                f"STEREO-DEPTH {zone_name.upper()} actuator -> {pattern.name} | "
                f"depth={distance} | strength=0x{pattern.level:02X} | {reason}"
            )
            self.last_pattern_names[zone_name] = pattern.name

    @Slot()
    def run(self) -> None:
        self._running = True
        last_capture_sequence = 0
        last_depth_sequence = -1
        last_scene_frame = -1
        result_count = 0
        zones = empty_stereo_zones((HEIGHT, WIDTH), "Waiting for stereo depth")
        depth_stale = True

        try:
            self.camera_status.emit("Starting...")
            self.stereo_status.emit("Loading calibration")
            check_gstreamer()
            self.calibration = load_calibration(CALIBRATION_PATH)
            self.maps = convert_rectification_maps(self.calibration)
            self._log(
                "Calibration ready | "
                f"{self.calibration.width}x{self.calibration.height} | "
                f"mode {self.calibration.sensor_mode} | "
                f"baseline {self.calibration.baseline_m * 1000.0:.2f} mm"
            )

            self.depth_worker = AsyncDepthProcessor(
                STEREO_BACKEND,
                self.calibration,
                self.maps,
                self._stereo_update,
            )
            self.depth_worker.start()

            self.left_camera, self.right_camera = open_cameras(
                LEFT_SENSOR_ID, RIGHT_SENSOR_ID, self.calibration
            )
            self.capture = SynchronizedStereoCapture(
                self.left_camera, self.right_camera
            )
            self.capture.start()
            self.camera_status.emit("Connected")
            self._log(
                f"IMX219 pair connected | left={LEFT_SENSOR_ID} | "
                f"right={RIGHT_SENSOR_ID}"
            )

            if ENABLE_ACTUATORS:
                self.actuators = ActuatorManager(self._log)
                self.actuators.start()
                self._log("Actuators ready | Left=SC2 Center=SC3 Right=SC4")
            else:
                self._log("Actuators disabled by configuration")

            if ENABLE_YOLO:
                try:
                    from inference_worker import LatestFrameInferenceWorker

                    self.inference_worker = LatestFrameInferenceWorker(
                        self._detector_update
                    )
                    self.inference_worker.start()
                except Exception as error:
                    self.detector_status.emit(f"YOLO disabled: {error}")
                    self._log(
                        f"YOLO unavailable; stereo haptics continue: {error}"
                    )
            else:
                self.detector_status.emit("YOLO disabled")

            while self._running:
                assert self.capture is not None
                assert self.depth_worker is not None
                assert self.calibration is not None
                assert self.maps is not None

                stereo_frame = self.capture.get_latest(
                    last_capture_sequence, CAMERA_PAIR_TIMEOUT_SECONDS
                )
                last_capture_sequence = stereo_frame.sequence
                validate_frame_resolution(
                    stereo_frame.left, stereo_frame.right, self.calibration
                )
                current_left = rectify_left(stereo_frame.left, self.maps)
                self.depth_worker.submit(
                    stereo_frame.sequence,
                    stereo_frame.captured_at * 1000.0,
                    current_left,
                    stereo_frame.right,
                )

                new_depth = self.depth_worker.latest_after(last_depth_sequence)
                if new_depth is not None:
                    self.latest_depth = new_depth
                    last_depth_sequence = new_depth.sequence
                    zones = compute_stereo_depth_zones(
                        new_depth.depth_map, new_depth.valid_depth_mask
                    )
                    depth_stale = False
                    result_count += 1
                    self._apply_patterns(zones, "fresh stereo depth")

                    if (
                        self.inference_worker is not None
                        and new_depth.sequence % YOLO_FRAME_INTERVAL == 0
                    ):
                        self.inference_worker.submit(
                            FramePacket(
                                frame_id=new_depth.sequence,
                                captured_at_ms=new_depth.captured_at_ms,
                                color_image=new_depth.left_rectified,
                                depth_in_meters=new_depth.depth_map,
                            )
                        )

                now_ms = time.monotonic() * 1000.0
                if (
                    self.latest_depth is not None
                    and now_ms - self.latest_depth.captured_at_ms
                    > DEPTH_RESULT_MAX_AGE_MS
                    and not depth_stale
                ):
                    zones = empty_stereo_zones(
                        (HEIGHT, WIDTH), "Stereo depth is stale"
                    )
                    self._apply_patterns(zones, "stale-depth safety stop")
                    depth_stale = True

                if self.inference_worker is not None:
                    scene = self.inference_worker.latest_after(last_scene_frame)
                    if scene is not None:
                        self.latest_scene = scene
                        last_scene_frame = scene.frame_id
                        self.detections_ready.emit(scene)

                detections: List[DetectionResult] = []
                if (
                    self.latest_scene is not None
                    and now_ms - self.latest_scene.captured_at_ms
                    <= YOLO_RESULT_MAX_AGE_MS
                ):
                    detections = self.latest_scene.detections

                display_frame = (
                    self.latest_depth.left_rectified
                    if self.latest_depth is not None
                    else current_left
                )
                overlay = self.draw_camera_overlay(display_frame, zones, detections)
                self.color_ready.emit(self.to_qimage_bgr(overlay))
                self.zones_ready.emit(zones)
                if self.latest_depth is not None:
                    self.depth_ready.emit(
                        self.to_qimage_bgr(self.latest_depth.depth_view)
                    )

                if new_depth is not None and result_count % LOG_EVERY_N_RESULTS == 0:
                    left_fps, right_fps, skew_ms, dropped_left, dropped_right = (
                        self.capture.stats()
                    )
                    self._log(
                        f"Stereo stats | camera L/R={left_fps:.1f}/{right_fps:.1f} "
                        f"FPS | depth={new_depth.depth_fps:.1f} FPS | "
                        f"pair skew={skew_ms:.2f} ms | valid="
                        f"{new_depth.valid_percentage:.1f}% | drops="
                        f"{dropped_left}/{dropped_right}"
                    )
                    for line in (
                        new_depth.diagnostic_lines
                        + self.depth_worker.status_lines()
                    ):
                        self._log(line)

        except Exception as error:
            self.camera_status.emit("Error")
            self.stereo_status.emit("Error")
            self.error.emit(str(error))
            self._log(f"ERROR: {error}")
        finally:
            try:
                if self.inference_worker is not None:
                    self.inference_worker.stop()
            except Exception:
                pass
            self.inference_worker = None
            try:
                if self.actuators is not None:
                    self.actuators.stop()
                    self._log("Actuators safely stopped")
            except Exception as error:
                self._log(f"Actuator shutdown error: {error}")
            self.actuators = None
            try:
                if self.capture is not None:
                    self.capture.stop()
            except Exception:
                pass
            self.capture = None
            try:
                if self.depth_worker is not None:
                    self.depth_worker.stop()
            except Exception:
                pass
            self.depth_worker = None
            if self.left_camera is not None:
                self.left_camera.release()
                self.left_camera = None
            if self.right_camera is not None:
                self.right_camera.release()
                self.right_camera = None
            self.camera_status.emit("Stopped")
            self.stereo_status.emit("Stopped")
            self.detector_status.emit("YOLO stopped")
            self.finished.emit()
