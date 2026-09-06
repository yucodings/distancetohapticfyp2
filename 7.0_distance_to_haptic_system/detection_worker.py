"""Newest-frame-only YOLO worker with non-blocking shared-GPU access."""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from config import DETECTION_MAX_CONSECUTIVE_ERRORS
from data_models import DetectionScene
from gpu_scheduler import GpuScheduler
from tensorrt_detector import TensorRTDetector


StatusCallback = Callable[[str], None]
DetectorFactory = Callable[[Path, float, float, int], TensorRTDetector]


@dataclass(frozen=True)
class DetectionInput:
    sequence: int
    captured_at: float
    frame: np.ndarray


class LatestFrameDetectionWorker:
    """Run inference at a fixed maximum rate without delaying camera capture."""

    def __init__(
        self,
        model_path: Path,
        scheduler: GpuScheduler,
        detection_fps: float,
        confidence: float,
        iou: float,
        max_results: int,
        status_callback: Optional[StatusCallback] = None,
        detector_factory: DetectorFactory = TensorRTDetector,
    ) -> None:
        if detection_fps <= 0.0:
            raise ValueError("Detection FPS must be positive")
        self.model_path = Path(model_path)
        self.scheduler = scheduler
        self.period = 1.0 / detection_fps
        self.confidence = confidence
        self.iou = iou
        self.max_results = max_results
        self.status_callback = status_callback
        self.detector_factory = detector_factory
        self._condition = threading.Condition()
        self._pending: Optional[DetectionInput] = None
        self._latest: Optional[DetectionScene] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._error: Optional[str] = None
        self._next_submission = 0.0
        self._replaced = 0
        self._rate_limited = 0
        self._gpu_busy_skips = 0
        self._smoothed_fps = 0.0

    def _status(self, message: str) -> None:
        if self.status_callback is not None:
            self.status_callback(message)

    def start(self, timeout: float = 60.0) -> bool:
        with self._condition:
            if self._running:
                return self._error is None
            self._running = True
        self._thread = threading.Thread(
            target=self._run, name="yolo-tensorrt-worker", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            self._status("Error: TensorRT initialization timed out")
            self.stop()
            return False
        with self._condition:
            return self._error is None

    def submit(self, sequence: int, captured_at: float, frame: np.ndarray) -> bool:
        now = time.monotonic()
        with self._condition:
            if not self._running or self._error is not None:
                return False
            if now < self._next_submission:
                self._rate_limited += 1
                return False
            self._next_submission = now + self.period
            if self._pending is not None:
                self._replaced += 1
            self._pending = DetectionInput(
                sequence=sequence,
                captured_at=captured_at,
                frame=frame.copy(),
            )
            self._condition.notify_all()
            return True

    def latest_result(self) -> Optional[DetectionScene]:
        with self._condition:
            return self._latest

    def has_failed(self) -> bool:
        with self._condition:
            return self._error is not None

    def status_lines(self) -> tuple[str, ...]:
        with self._condition:
            latest = self._latest
            error = self._error
            replaced = self._replaced
            rate_limited = self._rate_limited
            gpu_busy = self._gpu_busy_skips
            fps = self._smoothed_fps
        if error is not None:
            return (f"Detector: disabled ({error})",)
        if latest is None:
            return (
                "Detector: YOLO11n TensorRT FP16 | waiting",
                f"Detection dropped rate/GPU/replaced: {rate_limited}/{gpu_busy}/{replaced}",
            )
        return (
            f"Detector: YOLO11n TensorRT FP16 | {fps:.1f} FPS",
            f"Detection latency/objects: {latest.inference_ms:.1f} ms / {len(latest.detections)}",
            f"Detection dropped rate/GPU/replaced: {rate_limited}/{gpu_busy}/{replaced}",
        )

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._pending = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def _wait_for_input(self) -> Optional[DetectionInput]:
        with self._condition:
            while self._running and self._pending is None:
                self._condition.wait()
            if not self._running:
                return None
            item = self._pending
            self._pending = None
            return item

    def _fail(self, error: Exception) -> None:
        message = f"{type(error).__name__}: {error}"
        with self._condition:
            self._error = message
            self._running = False
            self._pending = None
            self._condition.notify_all()
        self._status(f"Error: {message}")

    def _run(self) -> None:
        detector: Optional[TensorRTDetector] = None
        previous_completion: Optional[float] = None
        try:
            self._status("Loading")
            detector = self.detector_factory(
                self.model_path,
                self.confidence,
                self.iou,
                self.max_results,
            )
            # Startup happens before camera capture. A blocking warmup confirms
            # that VPI and TensorRT allocations coexist before navigation runs.
            with self.scheduler.detection_job(blocking=True) as acquired:
                assert acquired
                detector.warmup()
            self._status(
                f"Ready | {detector.device_label} | "
                f"{len(detector.metadata.names)} classes"
            )
            self._ready.set()

            consecutive_errors = 0
            while True:
                item = self._wait_for_input()
                if item is None:
                    return
                with self.scheduler.detection_job(blocking=False) as acquired:
                    if not acquired:
                        with self._condition:
                            self._gpu_busy_skips += 1
                        continue
                    started = time.perf_counter()
                    try:
                        detections = detector.detect(item.frame)
                    except Exception as error:
                        consecutive_errors += 1
                        self._status(f"Inference warning: {error}")
                        if consecutive_errors >= DETECTION_MAX_CONSECUTIVE_ERRORS:
                            raise RuntimeError(
                                "detector stopped after repeated inference errors"
                            ) from error
                        continue
                    inference_ms = (time.perf_counter() - started) * 1000.0

                completed = time.monotonic()
                if previous_completion is not None and completed > previous_completion:
                    instantaneous = 1.0 / (completed - previous_completion)
                    with self._condition:
                        self._smoothed_fps = (
                            instantaneous
                            if self._smoothed_fps == 0.0
                            else 0.85 * self._smoothed_fps + 0.15 * instantaneous
                        )
                previous_completion = completed
                consecutive_errors = 0
                with self._condition:
                    self._latest = DetectionScene(
                        source_sequence=item.sequence,
                        captured_at=item.captured_at,
                        completed_at=completed,
                        detections=detections,
                        inference_ms=inference_ms,
                        detector_fps=self._smoothed_fps,
                        device=detector.device_label,
                    )
        except Exception as error:
            self._fail(error)
        finally:
            self._ready.set()
            if detector is not None:
                close = getattr(detector, "close", None)
                if callable(close):
                    close()
            detector = None
            gc.collect()
