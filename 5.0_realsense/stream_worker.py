import time
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

# The Jetson has both a pip RealSense binding and the system binding used by the
# working sudo camera setup. Select only the system RealSense package without
# putting all system packages ahead of the matched PyTorch/Ultralytics stack.
_SYSTEM_SITE_PACKAGES = Path("/usr/lib/python3/dist-packages")
if _SYSTEM_SITE_PACKAGES.is_dir():
    sys.path.insert(0, str(_SYSTEM_SITE_PACKAGES))
try:
    import pyrealsense2 as rs
finally:
    if sys.path and sys.path[0] == str(_SYSTEM_SITE_PACKAGES):
        sys.path.pop(0)

from actuators import ActuatorManager
from config import (
    ENABLE_ACTUATORS,
    ENABLE_YOLO,
    FPS,
    HEIGHT,
    LOG_EVERY_N_FRAMES,
    REALSENSE_RETRY_DELAY_SECONDS,
    REALSENSE_START_RETRIES,
    WIDTH,
    YOLO_FRAME_INTERVAL,
    YOLO_RESULT_MAX_AGE_MS,
    ZONE_KEYS,
)
from data_models import (
    DetectionResult,
    FramePacket,
    MotorPattern,
    SceneResult,
    ZoneResult,
)
from hazard_policy import pattern_from_depth
from raw_depth import compute_raw_depth_zones

if TYPE_CHECKING:
    from inference_worker import LatestFrameInferenceWorker


class StreamWorker(QObject):
    color_ready = Signal(QImage)
    depth_ready = Signal(QImage)
    zones_ready = Signal(object)
    detections_ready = Signal(object)
    log_ready = Signal(str)
    camera_status = Signal(str)
    detector_status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self.pipeline = None
        self.actuators: Optional[ActuatorManager] = None
        self.inference_worker: Optional["LatestFrameInferenceWorker"] = None
        self.latest_scene: Optional[SceneResult] = None
        self.last_scene_frame_id = -1
        self.last_pattern_names = {zone_name: None for zone_name in ZONE_KEYS}
        self.last_detection_signature = None

    def stop(self):
        self._running = False

    def _log(self, message: str):
        self.log_ready.emit(message)

    def _detector_update(self, message: str):
        self.detector_status.emit(message)
        self._log(message)

    def _start_realsense(self):
        last_error: Optional[Exception] = None

        for attempt in range(1, REALSENSE_START_RETRIES + 1):
            if not self._running:
                raise RuntimeError("Camera start cancelled")

            pipeline = rs.pipeline()
            stream_config = rs.config()
            stream_config.enable_stream(
                rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS
            )
            stream_config.enable_stream(
                rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS
            )

            try:
                profile = pipeline.start(stream_config)
                self.pipeline = pipeline
                return profile
            except Exception as error:
                last_error = error
                self._log(
                    f"RealSense connection attempt {attempt}/"
                    f"{REALSENSE_START_RETRIES} failed: {error}"
                )
                if attempt < REALSENSE_START_RETRIES:
                    self.camera_status.emit(
                        f"Retrying camera ({attempt + 1}/{REALSENSE_START_RETRIES})"
                    )
                    time.sleep(REALSENSE_RETRY_DELAY_SECONDS)

        raise RuntimeError(
            f"Unable to start RealSense after {REALSENSE_START_RETRIES} attempts: "
            f"{last_error}"
        )

    @staticmethod
    def pattern_from_depth(depth_m: Optional[float]) -> MotorPattern:
        return pattern_from_depth(depth_m)

    def compute_raw_depth_zones(
        self, depth_in_meters: np.ndarray
    ) -> Dict[str, ZoneResult]:
        return compute_raw_depth_zones(depth_in_meters)

    @staticmethod
    def blend_zone(frame: np.ndarray, rect: Tuple[int, int, int, int], color: Tuple[int, int, int], alpha: float):
        x1, y1, x2, y2 = rect
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return
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

        frame_height, frame_width = frame.shape[:2]
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = color_map.get(detection.tone, color_map["invalid"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            distance = "-- m" if detection.depth_m is None else f"{detection.depth_m:.2f} m"
            label = f"YOLO {detection.label} {detection.confidence:.0%} | {distance}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
            )
            text_x = max(0, min(x1, frame_width - text_width - 8))
            text_y = y1 - 7 if y1 - text_height - 10 >= 0 else y1 + text_height + 10
            box_top = max(0, text_y - text_height - 5)
            box_bottom = min(frame_height - 1, text_y + baseline + 3)
            cv2.rectangle(
                frame,
                (text_x, box_top),
                (min(frame_width - 1, text_x + text_width + 7), box_bottom),
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
        color_frame: np.ndarray,
        zones: Dict[str, ZoneResult],
        detections: Optional[List[DetectionResult]] = None,
    ) -> np.ndarray:
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

            label = (
                f"RAW {zone.display_name}: --"
                if zone.depth_m is None
                else f"RAW {zone.display_name}: {zone.depth_m:.2f} m"
            )
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
                point_x, point_y = zone.nearest_point
                cv2.circle(output, (point_x, point_y), 7, (255, 255, 255), 2)
                cv2.circle(output, (point_x, point_y), 4, (255, 0, 0), -1)

        self.draw_detection_overlays(output, detections or [])

        cv2.rectangle(output, (8, output.shape[0] - 38), (205, output.shape[0] - 8), (20, 20, 20), -1)
        cv2.putText(output, "Blue dot = nearest raw depth", (16, output.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
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
        align = rs.align(rs.stream.color)

        try:
            self.camera_status.emit("Connecting...")
            self._log("Starting RealSense pipeline")
            profile = self._start_realsense()

            if ENABLE_ACTUATORS:
                self.actuators = ActuatorManager(self._log)
                self.actuators.start()
                self._log("Actuators ready | TCA9548A channels SC2/SC3/SC4")
            else:
                self._log("Actuators disabled")

            for _ in range(10):
                self.pipeline.wait_for_frames()

            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            self.camera_status.emit("Connected")
            self._log(f"Camera connected | depth scale = {depth_scale:.6f} m/unit")

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
                        f"YOLO unavailable; raw-depth haptics continue: {error}"
                    )
            else:
                self.detector_status.emit("YOLO disabled")
                self._log("YOLO disabled by configuration")

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
                depth_in_meters = depth_image.astype(np.float32) * np.float32(depth_scale)
                zones = self.compute_raw_depth_zones(depth_in_meters)

                captured_at_ms = time.monotonic() * 1000.0
                if (
                    self.inference_worker is not None
                    and frame_counter % YOLO_FRAME_INTERVAL == 0
                ):
                    self.inference_worker.submit(
                        FramePacket(
                            frame_id=frame_counter,
                            captured_at_ms=captured_at_ms,
                            color_image=color_image,
                            depth_in_meters=depth_in_meters,
                        )
                    )

                if self.inference_worker is not None:
                    scene = self.inference_worker.latest_after(self.last_scene_frame_id)
                    if scene is not None:
                        self.latest_scene = scene
                        self.last_scene_frame_id = scene.frame_id
                        self.detections_ready.emit(scene)

                detections: List[DetectionResult] = []
                if (
                    self.latest_scene is not None
                    and captured_at_ms - self.latest_scene.captured_at_ms
                    <= YOLO_RESULT_MAX_AGE_MS
                ):
                    detections = self.latest_scene.detections

                if ENABLE_ACTUATORS:
                    patterns = {
                        zone_name: self.pattern_from_depth(zone.depth_m)
                        for zone_name, zone in zones.items()
                    }
                    if self.actuators is not None:
                        self.actuators.update_patterns(patterns)

                    for zone_name, pattern in patterns.items():
                        if self.last_pattern_names[zone_name] != pattern.name:
                            depth_m = zones[zone_name].depth_m
                            raw_depth = (
                                "no valid raw depth"
                                if depth_m is None
                                else f"raw nearest={depth_m:.3f} m"
                            )
                            envelope = (
                                "off"
                                if pattern.level == 0
                                else "continuous"
                                if pattern.on_time == pattern.off_time == 0.0
                                else f"{pattern.on_time:.2f}s on/"
                                f"{pattern.off_time:.2f}s off"
                            )
                            self._log(
                                f"RAW-DEPTH {zone_name.upper()} actuator -> "
                                f"{pattern.name} | {raw_depth} | "
                                f"strength=0x{pattern.level:02X} | {envelope}"
                            )
                            self.last_pattern_names[zone_name] = pattern.name

                overlay_frame = self.draw_camera_overlay(color_image, zones, detections)
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
                    self._log("Nearest raw-depth zones computed")

                    active_alerts = [z for z in zones.values() if z.depth_m is not None and z.depth_m <= 2.0]
                    if active_alerts:
                        nearest = min(active_alerts, key=lambda z: z.depth_m)
                        self._log(f"Alert triggered: {nearest.display_name} | {nearest.alert_message.split('] ', 1)[1]}")
                    else:
                        self._log("Alert triggered: none (clear path)")

                    if self.latest_scene is not None:
                        valid_detections = [
                            detection
                            for detection in self.latest_scene.detections
                            if detection.depth_m is not None
                        ]
                        signature = tuple(
                            sorted(
                                (d.label, tuple(d.zones))
                                for d in self.latest_scene.detections
                            )
                        )
                        if signature != self.last_detection_signature:
                            if valid_detections:
                                nearest_object = min(
                                    valid_detections, key=lambda item: item.depth_m
                                )
                                self._log(
                                    "YOLO: "
                                    f"{len(self.latest_scene.detections)} object(s) | "
                                    f"nearest {nearest_object.label} "
                                    f"at {nearest_object.depth_m:.2f} m | "
                                    f"{self.latest_scene.inference_ms:.1f} ms"
                                )
                            elif self.latest_scene.detections:
                                self._log(
                                    f"YOLO: {len(self.latest_scene.detections)} object(s), "
                                    "no valid object depth"
                                )
                            else:
                                self._log("YOLO: no recognized objects")
                            self.last_detection_signature = signature

        except Exception as exc:
            self.camera_status.emit("Error")
            self.error.emit(str(exc))
            self._log(f"ERROR: {exc}")
        finally:
            try:
                if self.inference_worker is not None:
                    self.inference_worker.stop()
                    self.inference_worker = None
            except Exception:
                pass
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
