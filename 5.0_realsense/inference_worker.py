import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

from config import YOLO_ALLOW_CPU_FALLBACK, YOLO_DEVICE, YOLO_MODEL_PATH
from data_models import FramePacket, SceneResult
from depth_fusion import fuse_detections_with_depth

if TYPE_CHECKING:
    from detector import YoloDetector


StatusCallback = Callable[[str], None]


class LatestFrameInferenceWorker:
    """Runs YOLO on a replaceable one-frame mailbox to prevent latency buildup."""

    def __init__(self, status_callback: Optional[StatusCallback] = None):
        self._status_callback = status_callback
        self._condition = threading.Condition()
        self._result_lock = threading.Lock()
        self._pending: Optional[FramePacket] = None
        self._latest_result: Optional[SceneResult] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _status(self, message: str):
        if self._status_callback is not None:
            self._status_callback(message)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="yolo-inference", daemon=True)
        self._thread.start()

    def submit(self, packet: FramePacket):
        with self._condition:
            if not self._running:
                return
            # The arrays reference RealSense frame memory, so keep independent copies.
            self._pending = FramePacket(
                frame_id=packet.frame_id,
                captured_at_ms=packet.captured_at_ms,
                color_image=packet.color_image.copy(),
                depth_in_meters=packet.depth_in_meters.copy(),
            )
            self._condition.notify()

    def latest_after(self, frame_id: int) -> Optional[SceneResult]:
        with self._result_lock:
            if self._latest_result is None or self._latest_result.frame_id <= frame_id:
                return None
            return self._latest_result

    def stop(self):
        with self._condition:
            self._running = False
            self._pending = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _create_detector(self) -> Optional["YoloDetector"]:
        # Import PyTorch/Ultralytics inside this background thread. Raw-depth
        # haptics therefore keep updating while YOLO imports and warms up.
        try:
            from detector import YoloDetector

            detector = YoloDetector(YOLO_MODEL_PATH, YOLO_DEVICE)
            detector.warmup()
            self._status(f"YOLO ready | {detector.device_label} | FP16")
            return detector
        except Exception as cuda_error:
            if not YOLO_ALLOW_CPU_FALLBACK:
                self._status(f"YOLO disabled: {cuda_error}")
                return None

            self._status(f"CUDA unavailable ({cuda_error}); loading YOLO on CPU")
            try:
                detector = YoloDetector(YOLO_MODEL_PATH, "cpu")
                detector.warmup()
                self._status("YOLO ready | CPU fallback")
                return detector
            except Exception as cpu_error:
                self._status(f"YOLO disabled: {cpu_error}")
                return None

    def _wait_for_packet(self) -> Optional[FramePacket]:
        with self._condition:
            while self._running and self._pending is None:
                self._condition.wait()
            if not self._running:
                return None
            packet = self._pending
            self._pending = None
            return packet

    def _run(self):
        self._status("Loading YOLO model...")
        detector = self._create_detector()
        if detector is None:
            self._running = False
            return

        consecutive_errors = 0
        while self._running:
            packet = self._wait_for_packet()
            if packet is None:
                break

            started = time.perf_counter()
            try:
                raw_detections = detector.detect(packet.color_image)
                detections = fuse_detections_with_depth(
                    raw_detections, packet.depth_in_meters
                )
                completed_at_ms = time.monotonic() * 1000.0
                scene = SceneResult(
                    frame_id=packet.frame_id,
                    captured_at_ms=packet.captured_at_ms,
                    completed_at_ms=completed_at_ms,
                    detections=detections,
                    inference_ms=(time.perf_counter() - started) * 1000.0,
                    device=detector.device_label,
                )
                with self._result_lock:
                    self._latest_result = scene
                consecutive_errors = 0
            except Exception as error:
                consecutive_errors += 1
                if consecutive_errors == 1 or consecutive_errors % 30 == 0:
                    self._status(f"YOLO inference error: {error}")

        self._status("YOLO stopped")
