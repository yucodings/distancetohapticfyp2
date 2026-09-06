import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_models import Detection
from detection_stabilizer import DetectionStabilizer, bbox_iou


def detected(
    class_id=3,
    label="person",
    confidence=0.8,
    bbox=(10, 20, 110, 220),
):
    return Detection(class_id, label, confidence, bbox)


def stabilizer():
    return DetectionStabilizer(
        max_misses=3,
        hold_seconds=1.5,
        match_iou=0.30,
        smoothing_alpha=0.65,
    )


class DetectionStabilizerTests(unittest.TestCase):
    def test_two_missed_results_keep_box_and_third_removes_it(self):
        tracker = stabilizer()
        self.assertEqual(tracker.update((detected(),), 10.0), (detected(),))
        self.assertEqual(tracker.update((), 10.4), (detected(),))
        self.assertEqual(tracker.update((), 10.8), (detected(),))
        self.assertEqual(tracker.update((), 11.2), ())

    def test_hard_age_limit_expires_box_when_gpu_submissions_are_skipped(self):
        tracker = stabilizer()
        tracker.update((detected(),), 20.0)
        self.assertEqual(tracker.visible(21.49), (detected(),))
        self.assertEqual(tracker.visible(21.50), ())

    def test_same_class_overlapping_box_is_smoothed_and_resets_misses(self):
        tracker = stabilizer()
        tracker.update((detected(),), 1.0)
        tracker.update((), 1.2)
        moved = detected(confidence=0.9, bbox=(20, 30, 120, 230))

        visible = tracker.update((moved,), 1.4)

        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0].bbox, (16, 26, 116, 226))
        self.assertEqual(visible[0].confidence, 0.9)
        self.assertEqual(tracker.update((), 1.6), visible)

    def test_different_classes_are_never_merged(self):
        tracker = stabilizer()
        tracker.update((detected(),), 1.0)
        chair = detected(class_id=0, label="chair", bbox=(12, 22, 112, 222))

        visible = tracker.update((chair,), 1.2)

        self.assertEqual([item.label for item in visible], ["person", "chair"])

    def test_iou_rejects_invalid_boxes(self):
        self.assertEqual(bbox_iou((4, 4, 2, 2), (0, 0, 10, 10)), 0.0)


if __name__ == "__main__":
    unittest.main()
