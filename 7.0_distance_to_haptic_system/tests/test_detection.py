import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_models import Detection
from config import DETECTOR_MODEL_PATH
from detection_overlay import draw_detections
from detection_worker import LatestFrameDetectionWorker
from gpu_scheduler import GpuScheduler
from tensorrt_detector import (
    LetterboxTransform,
    decode_yolo_output,
    letterbox_bgr,
    read_engine_metadata,
)


class EngineMetadataTests(unittest.TestCase):
    def test_deployment_engine_is_the_expected_static_fp16_model(self):
        metadata = read_engine_metadata(DETECTOR_MODEL_PATH)

        self.assertEqual(metadata.task, "detect")
        self.assertEqual(metadata.batch, 1)
        self.assertEqual(metadata.image_size, (640, 640))
        self.assertTrue(metadata.half)
        self.assertFalse(metadata.nms)
        self.assertEqual(len(metadata.names), 27)
        self.assertIn("person", metadata.names)
        self.assertIn("stairs", metadata.names)
        self.assertIn("wet_floor_sign", metadata.names)

    def test_direct_runtime_does_not_import_ultralytics_or_torch(self):
        self.assertNotIn("ultralytics", sys.modules)
        self.assertNotIn("torch", sys.modules)


class TensorRTPostprocessingTests(unittest.TestCase):
    def test_letterbox_has_expected_shape_range_and_transform(self):
        frame = np.full((720, 1280, 3), 255, dtype=np.uint8)

        prepared, transform = letterbox_bgr(frame, (640, 640))

        self.assertEqual(prepared.shape, (1, 3, 640, 640))
        self.assertEqual(prepared.dtype, np.float32)
        self.assertAlmostEqual(float(prepared.max()), 1.0)
        self.assertAlmostEqual(transform.ratio, 0.5)
        self.assertEqual(transform.pad_x, 0.0)
        self.assertEqual(transform.pad_y, 140.0)

    def test_decode_maps_boxes_back_and_applies_class_aware_nms(self):
        names = ("chair", "person")
        output = np.zeros((1, 4 + len(names), 4), dtype=np.float32)
        # Two overlapping class-1 boxes: keep only the stronger first box.
        output[0, :4, 0] = (320, 320, 100, 200)
        output[0, 4 + 1, 0] = 0.90
        output[0, :4, 1] = (322, 322, 100, 200)
        output[0, 4 + 1, 1] = 0.70
        # An overlapping class-0 box remains because NMS is class-aware.
        output[0, :4, 2] = (320, 320, 100, 200)
        output[0, 4, 2] = 0.80
        # Below threshold.
        output[0, :4, 3] = (100, 100, 20, 20)
        output[0, 4, 3] = 0.20
        transform = LetterboxTransform(0.5, 0.0, 140.0, 1280, 720)

        detections = decode_yolo_output(
            output, transform, names, 0.45, 0.5, 30
        )

        self.assertEqual([item.label for item in detections], ["person", "chair"])
        self.assertEqual(detections[0].bbox, (540, 160, 740, 560))
        self.assertAlmostEqual(detections[0].confidence, 0.90, places=5)


class DetectionOverlayTests(unittest.TestCase):
    def test_overlay_changes_only_a_copy_when_caller_copies_first(self):
        original = np.zeros((100, 160, 3), dtype=np.uint8)
        annotated = original.copy()
        detection = Detection(3, "person", 0.91, (20, 30, 100, 90))

        returned = draw_detections(annotated, (detection,))

        self.assertIs(returned, annotated)
        self.assertFalse(np.any(original))
        self.assertTrue(np.any(annotated))

    def test_invalid_box_is_ignored(self):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        invalid = Detection(0, "chair", 0.9, (60, 40, 20, 10))
        draw_detections(frame, (invalid,))
        self.assertFalse(np.any(frame))


class _FakeMetadata:
    names = ("person",)


class _FakeDetector:
    metadata = _FakeMetadata()
    device_label = "fake TensorRT"

    def __init__(self, _path, _confidence, _iou, _max_results):
        self.warmed = False

    def warmup(self):
        self.warmed = True

    def detect(self, _frame):
        return (Detection(0, "person", 0.8, (1, 2, 10, 20)),)


class DetectionWorkerTests(unittest.TestCase):
    def test_worker_keeps_latest_result_without_haptic_data(self):
        worker = LatestFrameDetectionWorker(
            DETECTOR_MODEL_PATH,
            GpuScheduler(),
            detection_fps=100.0,
            confidence=0.45,
            iou=0.5,
            max_results=30,
            detector_factory=_FakeDetector,
        )
        self.assertTrue(worker.start(timeout=1.0))
        try:
            frame = np.zeros((72, 128, 3), dtype=np.uint8)
            self.assertTrue(worker.submit(7, time.monotonic(), frame))
            deadline = time.monotonic() + 1.0
            result = None
            while time.monotonic() < deadline:
                result = worker.latest_result()
                if result is not None:
                    break
                time.sleep(0.005)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.source_sequence, 7)
            self.assertEqual(result.detections[0].label, "person")
            self.assertFalse(hasattr(result, "distance_m"))
            self.assertFalse(hasattr(result, "motor_pattern"))
        finally:
            worker.stop()

    def test_detection_never_waits_behind_active_depth_job(self):
        scheduler = GpuScheduler()
        entered = threading.Event()
        release = threading.Event()

        def hold_depth():
            with scheduler.depth_job():
                entered.set()
                release.wait(timeout=1.0)

        thread = threading.Thread(target=hold_depth)
        thread.start()
        self.assertTrue(entered.wait(timeout=1.0))
        try:
            with scheduler.detection_job(blocking=False) as acquired:
                self.assertFalse(acquired)
        finally:
            release.set()
            thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
