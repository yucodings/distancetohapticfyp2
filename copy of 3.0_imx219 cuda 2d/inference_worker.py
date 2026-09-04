"""Latest-frame YOLO client using an isolated NumPy/Ultralytics process."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from camera_backend import np
from data_models import DetectionResult, FramePacket, SceneResult
from depth_fusion import fuse_detections_with_depth


StatusCallback = Callable[[str], None]
PROTOCOL_PREFIX = "@@YOLO@@"


class LatestFrameInferenceWorker:
    """Keep only the newest frame while an isolated YOLO process is busy."""

    def __init__(self, status_callback: Optional[StatusCallback] = None):
        self._status_callback = status_callback
        self._condition = threading.Condition()
        self._result_lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._pending: Optional[FramePacket] = None
        self._latest_result: Optional[SceneResult] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None

    def _status(self, message: str) -> None:
        if self._status_callback is not None:
            self._status_callback(message)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="yolo-process-client", daemon=True
        )
        self._thread.start()

    def submit(self, packet: FramePacket) -> None:
        with self._condition:
            if not self._running:
                return
            self._pending = FramePacket(
                frame_id=packet.frame_id,
                captured_at_ms=packet.captured_at_ms,
                color_image=packet.color_image.copy(),
                depth_in_meters=packet.depth_in_meters.copy(),
            )
            self._condition.notify_all()

    def latest_after(self, frame_id: int) -> Optional[SceneResult]:
        with self._result_lock:
            if self._latest_result is None or self._latest_result.frame_id <= frame_id:
                return None
            return self._latest_result

    def _wait_for_packet(self) -> Optional[FramePacket]:
        with self._condition:
            while self._running and self._pending is None:
                self._condition.wait()
            if not self._running:
                return None
            packet = self._pending
            self._pending = None
            return packet

    def _launch_process(self) -> subprocess.Popen:
        worker_path = Path(__file__).resolve().with_name("yolo_process.py")
        environment = os.environ.copy()
        environment.setdefault("MPLBACKEND", "Agg")
        process = subprocess.Popen(
            [sys.executable, str(worker_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            bufsize=0,
        )
        with self._process_lock:
            self._process = process
        return process

    def _read_message(self, process: subprocess.Popen) -> Optional[dict]:
        assert process.stdout is not None
        while self._running:
            line = process.stdout.readline()
            if not line:
                return None
            output = line.decode("utf-8", errors="replace").rstrip()
            if output.startswith(PROTOCOL_PREFIX):
                return json.loads(output[len(PROTOCOL_PREFIX) :])
            if output:
                self._status(f"YOLO: {output}")
        return None

    @staticmethod
    def _send_packet(process: subprocess.Popen, packet: FramePacket) -> None:
        assert process.stdin is not None
        image = np.ascontiguousarray(packet.color_image, dtype=np.uint8)
        payload = image.tobytes()
        header = json.dumps(
            {
                "frame_id": packet.frame_id,
                "captured_at_ms": packet.captured_at_ms,
                "shape": list(image.shape),
                "payload_size": len(payload),
            }
        ).encode("utf-8")
        process.stdin.write(struct.pack("!I", len(header)))
        process.stdin.write(header)
        process.stdin.write(payload)
        process.stdin.flush()

    def _run(self) -> None:
        process: Optional[subprocess.Popen] = None
        try:
            self._status("Loading YOLO model in isolated process...")
            process = self._launch_process()
            startup = self._read_message(process)
            if startup is None:
                raise RuntimeError(
                    f"YOLO process exited during startup (code {process.poll()})"
                )
            if startup.get("type") != "ready":
                raise RuntimeError(startup.get("message", "YOLO startup failed"))
            self._status(f"YOLO ready | {startup['device']} | isolated process")

            while self._running:
                packet = self._wait_for_packet()
                if packet is None:
                    return
                started = time.perf_counter()
                self._send_packet(process, packet)
                message = self._read_message(process)
                if message is None:
                    raise RuntimeError(
                        f"YOLO process stopped unexpectedly (code {process.poll()})"
                    )
                if message.get("type") == "error":
                    self._status(f"YOLO inference error: {message['message']}")
                    continue
                if message.get("type") != "result":
                    continue

                raw_detections = [
                    DetectionResult(
                        class_id=int(item["class_id"]),
                        label=str(item["label"]),
                        confidence=float(item["confidence"]),
                        bbox=tuple(int(value) for value in item["bbox"]),
                    )
                    for item in message["detections"]
                ]
                detections = fuse_detections_with_depth(
                    raw_detections, packet.depth_in_meters
                )
                result = SceneResult(
                    frame_id=packet.frame_id,
                    captured_at_ms=packet.captured_at_ms,
                    completed_at_ms=time.monotonic() * 1000.0,
                    detections=detections,
                    inference_ms=float(
                        message.get(
                            "inference_ms",
                            (time.perf_counter() - started) * 1000.0,
                        )
                    ),
                    device=str(message.get("device", "unknown")),
                )
                with self._result_lock:
                    self._latest_result = result
        except Exception as error:
            if self._running:
                self._status(f"YOLO disabled: {error}")
        finally:
            self._running = False
            if process is not None:
                try:
                    if process.stdin is not None:
                        process.stdin.close()
                except Exception:
                    pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                with self._process_lock:
                    self._process = None

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._pending = None
            self._condition.notify_all()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
