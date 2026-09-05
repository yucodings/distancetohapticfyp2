from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import test_depth_realsense_3frame as depth_test

np = depth_test.np


class ThreeZoneRealSenseTests(unittest.TestCase):
    def test_rois_are_ordered_non_overlapping_and_in_bounds(self):
        regions = depth_test.three_rois((720, 1280))
        self.assertEqual([name for name, _box in regions], ["Left", "Centre", "Right"])
        previous_x2 = 0
        for _name, (x1, y1, x2, y2) in regions:
            self.assertGreaterEqual(x1, previous_x2)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 1280)
            self.assertLessEqual(y2, 720)
            self.assertEqual(x2 - x1, round(1280 * depth_test.ROI_WIDTH_FRACTION))
            self.assertEqual(y2 - y1, round(720 * depth_test.ROI_HEIGHT_FRACTION))
            previous_x2 = x2

    def test_each_zone_has_independent_median(self):
        depth = np.zeros((100, 300), dtype=np.float32)
        expected = (1.0, 2.0, 3.0)
        for (_name, (x1, y1, x2, y2)), value in zip(
            depth_test.three_rois(depth.shape), expected
        ):
            depth[y1:y2, x1:x2] = value
        measurements, _valid = depth_test.measure_three_depths(
            depth, 0.1, 4.0, min_samples=1
        )
        self.assertEqual(
            tuple(item.distance_m for item in measurements), expected
        )

    def test_invalid_zone_does_not_disable_other_zones(self):
        depth = np.full((100, 300), 2.0, dtype=np.float32)
        _name, (x1, y1, x2, y2) = depth_test.three_rois(depth.shape)[0]
        depth[y1:y2, x1:x2] = 0.0
        measurements, _valid = depth_test.measure_three_depths(
            depth, 0.1, 4.0, min_samples=10
        )
        self.assertIsNone(measurements[0].distance_m)
        self.assertEqual(measurements[1].distance_m, 2.0)
        self.assertEqual(measurements[2].distance_m, 2.0)

    def test_sample_threshold_is_per_zone(self):
        depth = np.zeros((20, 60), dtype=np.float32)
        regions = depth_test.three_rois(depth.shape)
        for index, (_name, (x1, y1, _x2, _y2)) in enumerate(regions):
            depth[y1, x1 : x1 + index + 1] = 1.0
        measurements, _valid = depth_test.measure_three_depths(
            depth, 0.1, 4.0, min_samples=2
        )
        self.assertIsNone(measurements[0].distance_m)
        self.assertIsNotNone(measurements[1].distance_m)
        self.assertIsNotNone(measurements[2].distance_m)

    def test_display_has_two_panels_and_separate_status_row(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        display = depth_test.make_primary_display(image, image, True)
        self.assertEqual(
            display.shape,
            (
                depth_test.DISPLAY_PANEL_HEIGHT + depth_test.STATUS_PANEL_HEIGHT,
                depth_test.DISPLAY_PANEL_WIDTH * 2,
                3,
            ),
        )

    def test_save_contains_all_three_measurements(self):
        image = np.zeros((20, 60, 3), dtype=np.uint8)
        depth = np.ones((20, 60), dtype=np.float32)
        measurements, valid = depth_test.measure_three_depths(
            depth, 0.1, 4.0, min_samples=1
        )
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(depth_test, "RESULTS_DIR", Path(temporary)):
                output = depth_test.save_evidence(
                    image,
                    image,
                    depth,
                    valid,
                    measurements,
                    0.001,
                    0.1,
                    4.0,
                    "test-serial",
                )
            expected = {
                "annotated_colour.png",
                "depth_heatmap.png",
                "depth_metres_float32.npy",
                "valid_mask.npy",
                "three_zone_measurements.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            report = json.loads(
                (output / "three_zone_measurements.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["name"] for item in report["measurements"]],
                ["Left", "Centre", "Right"],
            )


if __name__ == "__main__":
    unittest.main()

