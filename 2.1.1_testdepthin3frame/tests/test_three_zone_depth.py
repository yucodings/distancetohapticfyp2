from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import test_depth_imx219_3frame as depth_test

np = depth_test.np


class ThreeZoneDepthTests(unittest.TestCase):
    def test_three_rois_are_ordered_non_overlapping_and_in_bounds(self):
        height, width = 720, 1280
        regions = depth_test.three_rois((height, width))

        self.assertEqual([name for name, _ in regions], ["Left", "Centre", "Right"])
        self.assertEqual(len(regions), 3)
        previous_x2 = 0
        for _name, (x1, y1, x2, y2) in regions:
            self.assertGreaterEqual(x1, previous_x2)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, width)
            self.assertLessEqual(y2, height)
            self.assertEqual(x2 - x1, round(width * depth_test.ROI_WIDTH_FRACTION))
            self.assertEqual(y2 - y1, round(height * depth_test.ROI_HEIGHT_FRACTION))
            previous_x2 = x2

    def test_each_zone_has_an_independent_median(self):
        depth = np.zeros((100, 300), dtype=np.float32)
        valid = np.ones(depth.shape, dtype=bool)
        expected = (1.0, 2.0, 3.0)
        for (_name, (x1, y1, x2, y2)), value in zip(
            depth_test.three_rois(depth.shape), expected
        ):
            depth[y1:y2, x1:x2] = value

        measurements = depth_test.measure_three_depths(depth, valid, min_samples=1)

        self.assertEqual(
            tuple(measurement.distance_m for measurement in measurements),
            expected,
        )

    def test_invalid_left_zone_does_not_invalidate_other_zones(self):
        depth = np.full((100, 300), 2.0, dtype=np.float32)
        valid = np.ones(depth.shape, dtype=bool)
        _name, (x1, y1, x2, y2) = depth_test.three_rois(depth.shape)[0]
        valid[y1:y2, x1:x2] = False

        measurements = depth_test.measure_three_depths(depth, valid, min_samples=10)

        self.assertIsNone(measurements[0].distance_m)
        self.assertEqual(measurements[0].sample_count, 0)
        self.assertEqual(measurements[1].distance_m, 2.0)
        self.assertEqual(measurements[2].distance_m, 2.0)

    def test_sample_threshold_is_applied_per_zone(self):
        depth = np.ones((20, 60), dtype=np.float32)
        valid = np.zeros(depth.shape, dtype=bool)
        regions = depth_test.three_rois(depth.shape)
        for index, (_name, (x1, y1, x2, y2)) in enumerate(regions):
            self.assertGreaterEqual(x2 - x1, index + 1)
            self.assertGreater(y2, y1)
            valid[y1, x1 : x1 + index + 1] = True

        measurements = depth_test.measure_three_depths(depth, valid, min_samples=2)

        self.assertIsNone(measurements[0].distance_m)
        self.assertIsNotNone(measurements[1].distance_m)
        self.assertIsNotNone(measurements[2].distance_m)

    def test_status_panel_does_not_cover_the_three_zone_images(self):
        source = np.zeros((720, 1280, 3), dtype=np.uint8)
        display = depth_test.make_primary_display(source, source)
        self.assertEqual(
            display.shape,
            (
                depth_test.DISPLAY_PANEL_HEIGHT + depth_test.STATUS_PANEL_HEIGHT,
                depth_test.DISPLAY_PANEL_WIDTH * 2,
                3,
            ),
        )
        image_row_before = display[: depth_test.DISPLAY_PANEL_HEIGHT].copy()
        measurements = depth_test.empty_measurements((720, 1280))

        depth_test.draw_status(
            display,
            30.0,
            30.0,
            30.0,
            15.0,
            50.0,
            False,
            "test-backend",
            1.0,
            0,
            0,
            measurements,
            (),
        )

        self.assertTrue(
            np.array_equal(
                display[: depth_test.DISPLAY_PANEL_HEIGHT],
                image_row_before,
            )
        )


if __name__ == "__main__":
    unittest.main()
