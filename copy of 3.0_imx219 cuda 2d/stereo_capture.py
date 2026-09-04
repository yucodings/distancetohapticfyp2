"""Low-latency capture and software pairing for two CSI IMX219 sensors."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from calibration import Calibration
from camera_backend import cv2, np
from config import (
    CAMERA_FRAME_BUFFER_SIZE,
    MAX_CONSECUTIVE_CAPTURE_FAILURES,
)


@dataclass(frozen=True)
class CameraFrame:
    sequence: int
    arrived_at: float
    image: np.ndarray


@dataclass(frozen=True)
class StereoFrame:
    sequence: int
    captured_at: float
    pair_skew_ms: float
    left: np.ndarray
    right: np.ndarray


def gstreamer_pipeline(sensor_id: int, calibration: Calibration) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} "
        f"sensor-mode={calibration.sensor_mode} ! "
        f"video/x-raw(memory:NVMM), width=(int){calibration.width}, "
        f"height=(int){calibration.height}, format=(string)NV12, "
        f"framerate=(fraction){calibration.capture_fps}/1 ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "nvvidconv flip-method=0 ! "
        f"video/x-raw, width=(int){calibration.width}, "
        f"height=(int){calibration.height}, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def open_camera(sensor_id: int, calibration: Calibration) -> cv2.VideoCapture:
    pipeline = gstreamer_pipeline(sensor_id, calibration)
    camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open IMX219 sensor-id={sensor_id}. Pipeline: {pipeline}"
        )
    return camera


def open_cameras(
    left_sensor_id: int,
    right_sensor_id: int,
    calibration: Calibration,
) -> tuple[cv2.VideoCapture, cv2.VideoCapture]:
    if left_sensor_id == right_sensor_id:
        raise ValueError("Left and right sensor IDs must differ")
    left: Optional[cv2.VideoCapture] = None
    try:
        left = open_camera(left_sensor_id, calibration)
        right = open_camera(right_sensor_id, calibration)
        return left, right
    except Exception:
        if left is not None:
            left.release()
        raise


def validate_frame_resolution(
    left: np.ndarray,
    right: np.ndarray,
    calibration: Calibration,
) -> None:
    expected = (calibration.height, calibration.width)
    if left.shape[:2] != expected or right.shape[:2] != expected:
        raise RuntimeError(
            f"Calibration expects {expected[1]}x{expected[0]}; received left "
            f"{left.shape[1]}x{left.shape[0]} and right "
            f"{right.shape[1]}x{right.shape[0]}. Frames will not be resized."
        )


class SynchronizedStereoCapture:
    """Pair frames by closest host arrival timestamps.

    IMX219 exposure timestamps are not available through these OpenCV Argus
    pipelines, so this reduces software skew but does not claim hardware sync.
    """

    def __init__(
        self,
        left_camera: cv2.VideoCapture,
        right_camera: cv2.VideoCapture,
    ):
        self.left_camera = left_camera
        self.right_camera = right_camera
        self._condition = threading.Condition()
        self._running = False
        self._start_barrier: Optional[threading.Barrier] = None
        self._threads: list[threading.Thread] = []
        self._left_buffer: deque[CameraFrame] = deque(
            maxlen=CAMERA_FRAME_BUFFER_SIZE
        )
        self._right_buffer: deque[CameraFrame] = deque(
            maxlen=CAMERA_FRAME_BUFFER_SIZE
        )
        self._error: Optional[str] = None
        self._camera_sequences = {"left": 0, "right": 0}
        self._camera_fps = {"left": 0.0, "right": 0.0}
        self._dropped = {"left": 0, "right": 0}
        self._pair_sequence = 0
        self._last_pair_skew_ms = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_barrier = threading.Barrier(3)
        self._threads = [
            threading.Thread(
                target=self._reader,
                args=("left", self.left_camera, self._left_buffer),
                name="imx219-left-capture",
                daemon=True,
            ),
            threading.Thread(
                target=self._reader,
                args=("right", self.right_camera, self._right_buffer),
                name="imx219-right-capture",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()
        try:
            self._start_barrier.wait(timeout=2.0)
        except threading.BrokenBarrierError as error:
            self.stop()
            raise RuntimeError("Could not start both IMX219 readers") from error

    def _reader(
        self,
        side: str,
        camera: cv2.VideoCapture,
        buffer: deque[CameraFrame],
    ) -> None:
        previous_time: Optional[float] = None
        failures = 0
        try:
            assert self._start_barrier is not None
            self._start_barrier.wait(timeout=2.0)
            while self._running:
                grabbed = camera.grab()
                arrived_at = time.monotonic()
                retrieved, image = camera.retrieve()
                if not grabbed or not retrieved or image is None:
                    failures += 1
                    if failures < MAX_CONSECUTIVE_CAPTURE_FAILURES:
                        continue
                    raise RuntimeError(
                        f"{failures} consecutive {side} capture failures"
                    )

                failures = 0
                if previous_time is not None:
                    instantaneous = 1.0 / max(arrived_at - previous_time, 1e-6)
                    current = self._camera_fps[side]
                    self._camera_fps[side] = (
                        instantaneous
                        if current == 0.0
                        else 0.90 * current + 0.10 * instantaneous
                    )
                previous_time = arrived_at

                with self._condition:
                    self._camera_sequences[side] += 1
                    if len(buffer) == buffer.maxlen:
                        self._dropped[side] += 1
                    buffer.append(
                        CameraFrame(
                            self._camera_sequences[side], arrived_at, image
                        )
                    )
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = f"{side} capture failed: {error}"
                self._running = False
                self._condition.notify_all()

    def get_latest(self, after_sequence: int, timeout: float) -> StereoFrame:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError(self._error)
                if not self._running:
                    raise RuntimeError("Stereo capture stopped")
                if self._left_buffer and self._right_buffer:
                    left_index, right_index = min(
                        (
                            (left_index, right_index)
                            for left_index in range(len(self._left_buffer))
                            for right_index in range(len(self._right_buffer))
                        ),
                        key=lambda pair: (
                            abs(
                                self._left_buffer[pair[0]].arrived_at
                                - self._right_buffer[pair[1]].arrived_at
                            ),
                            -max(
                                self._left_buffer[pair[0]].arrived_at,
                                self._right_buffer[pair[1]].arrived_at,
                            ),
                        ),
                    )
                    for _ in range(left_index):
                        self._left_buffer.popleft()
                        self._dropped["left"] += 1
                    for _ in range(right_index):
                        self._right_buffer.popleft()
                        self._dropped["right"] += 1

                    left = self._left_buffer.popleft()
                    right = self._right_buffer.popleft()
                    self._pair_sequence += 1
                    self._last_pair_skew_ms = abs(
                        left.arrived_at - right.arrived_at
                    ) * 1000.0
                    if self._pair_sequence > after_sequence:
                        return StereoFrame(
                            self._pair_sequence,
                            (left.arrived_at + right.arrived_at) / 2.0,
                            self._last_pair_skew_ms,
                            left.image,
                            right.image,
                        )
                    continue

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError("Timed out waiting for an IMX219 frame pair")
                self._condition.wait(remaining)

    def stats(self) -> tuple[float, float, float, int, int]:
        with self._condition:
            return (
                self._camera_fps["left"],
                self._camera_fps["right"],
                self._last_pair_skew_ms,
                self._dropped["left"],
                self._dropped["right"],
            )

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._start_barrier is not None:
            try:
                self._start_barrier.abort()
            except threading.BrokenBarrierError:
                pass
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()
