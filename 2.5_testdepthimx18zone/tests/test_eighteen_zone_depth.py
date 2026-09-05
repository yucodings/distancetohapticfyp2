from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import test_depth_imx219_18zone as depth_test

np = depth_test.np


class EighteenZoneDepthTests(unittest.TestCase):
    def test_eighteen_zones_cover_full_usable_area_without_gaps(self):
        height, width = 720, 1280
        regions = depth_test.eighteen_zones((height, width))

        self.assertEqual(
            [name for name, _ in regions],
            [
                "Upper 1", "Upper 2", "Upper 3",
                "Upper 4", "Upper 5", "Upper 6",
                "Middle 1", "Middle 2", "Middle 3",
                "Middle 4", "Middle 5", "Middle 6",
                "Lower 1", "Lower 2", "Lower 3",
                "Lower 4", "Lower 5", "Lower 6",
            ],
        )
        self.assertEqual(
            [box for _name, box in regions],
            [
                (0, 72, 213, 264),
                (213, 72, 426, 264),
                (426, 72, 640, 264),
                (640, 72, 853, 264),
                (853, 72, 1066, 264),
                (1066, 72, 1280, 264),
                (0, 264, 213, 456),
                (213, 264, 426, 456),
                (426, 264, 640, 456),
                (640, 264, 853, 456),
                (853, 264, 1066, 456),
                (1066, 264, 1280, 456),
                (0, 456, 213, 648),
                (213, 456, 426, 648),
                (426, 456, 640, 648),
                (640, 456, 853, 648),
                (853, 456, 1066, 648),
                (1066, 456, 1280, 648),
            ],
        )
        total_area = sum(
            (x2 - x1) * (y2 - y1) for _name, (x1, y1, x2, y2) in regions
        )
        self.assertEqual(total_area, width * (648 - 72))

    def test_smallest_supported_image_has_eighteen_nonempty_zones(self):
        regions = depth_test.eighteen_zones((3, 6))
        self.assertEqual(
            [box for _name, box in regions],
            [
                (0, 0, 1, 1),
                (1, 0, 2, 1),
                (2, 0, 3, 1),
                (3, 0, 4, 1),
                (4, 0, 5, 1),
                (5, 0, 6, 1),
                (0, 1, 1, 2), (1, 1, 2, 2), (2, 1, 3, 2),
                (3, 1, 4, 2), (4, 1, 5, 2), (5, 1, 6, 2),
                (0, 2, 1, 3), (1, 2, 2, 3), (2, 2, 3, 3),
                (3, 2, 4, 3), (4, 2, 5, 3), (5, 2, 6, 3),
            ],
        )
        with self.assertRaises(ValueError):
            depth_test.eighteen_zones((10, 5))
        with self.assertRaises(ValueError):
            depth_test.eighteen_zones((2, 10))

    def test_each_zone_has_an_independent_median(self):
        depth = np.zeros((100, 600), dtype=np.float32)
        valid = np.ones(depth.shape, dtype=bool)
        expected = tuple(float(value) for value in range(1, 19))
        for (_name, (x1, y1, x2, y2)), value in zip(
            depth_test.eighteen_zones(depth.shape), expected
        ):
            depth[y1:y2, x1:x2] = value

        measurements = depth_test.measure_eighteen_depths(depth, valid, min_samples=1)

        self.assertEqual(
            tuple(measurement.distance_m for measurement in measurements),
            expected,
        )

    def test_invalid_left_zone_does_not_invalidate_other_zones(self):
        depth = np.full((100, 600), 2.0, dtype=np.float32)
        valid = np.ones(depth.shape, dtype=bool)
        _name, (x1, y1, x2, y2) = depth_test.eighteen_zones(depth.shape)[0]
        valid[y1:y2, x1:x2] = False

        measurements = depth_test.measure_eighteen_depths(depth, valid, min_samples=10)

        self.assertIsNone(measurements[0].distance_m)
        self.assertEqual(measurements[0].sample_count, 0)
        self.assertTrue(
            all(measurement.distance_m == 2.0 for measurement in measurements[1:])
        )

    def test_sample_threshold_is_applied_per_zone(self):
        depth = np.ones((15, 120), dtype=np.float32)
        valid = np.zeros(depth.shape, dtype=bool)
        regions = depth_test.eighteen_zones(depth.shape)
        for index, (_name, (x1, y1, x2, y2)) in enumerate(regions):
            self.assertGreaterEqual(x2 - x1, index + 1)
            self.assertGreater(y2, y1)
            valid[y1, x1 : x1 + index + 1] = True

        measurements = depth_test.measure_eighteen_depths(depth, valid, min_samples=2)

        self.assertIsNone(measurements[0].distance_m)
        self.assertTrue(
            all(measurement.distance_m is not None for measurement in measurements[1:])
        )

    def test_valid_percentage_uses_only_the_zone_area(self):
        depth = np.ones((10, 60), dtype=np.float32)
        valid = np.zeros(depth.shape, dtype=bool)
        _name, (x1, y1, x2, y2) = depth_test.eighteen_zones(depth.shape)[1]
        zone_area = (x2 - x1) * (y2 - y1)
        valid[y1:y2, x1 : x1 + (x2 - x1) // 2] = True

        measurement = depth_test.measure_eighteen_depths(
            depth, valid, min_samples=1
        )[1]

        self.assertEqual(measurement.sample_count, zone_area // 2)
        self.assertAlmostEqual(measurement.valid_percentage, 50.0)

    def test_zone_below_one_point_six_percent_is_rejected_before_median(self):
        depth = np.ones((10, 25), dtype=np.float32)
        valid = np.zeros(depth.shape, dtype=bool)
        valid.flat[:3] = True

        with patch.object(depth_test.np, "median") as median:
            measurement = depth_test.measure_zone_depth(
                "Test", (0, 0, 25, 10), depth, valid, min_samples=1
            )

        self.assertEqual(measurement.sample_count, 3)
        self.assertAlmostEqual(measurement.valid_percentage, 1.2)
        self.assertIsNone(measurement.distance_m)
        median.assert_not_called()

    def test_zone_at_one_point_six_percent_keeps_its_median(self):
        depth = np.full((10, 25), 2.5, dtype=np.float32)
        valid = np.zeros(depth.shape, dtype=bool)
        valid.flat[:4] = True

        measurement = depth_test.measure_zone_depth(
            "Test", (0, 0, 25, 10), depth, valid, min_samples=1
        )

        self.assertAlmostEqual(measurement.valid_percentage, 1.6)
        self.assertEqual(measurement.distance_m, 2.5)

    def test_zone_overlay_changes_pixels_inside_all_eighteen_zones(self):
        image = np.zeros((100, 600, 3), dtype=np.uint8)
        measurements = depth_test.empty_measurements(image.shape[:2])
        depth_test.draw_measurements(image, measurements)

        for _name, (x1, y1, x2, y2) in depth_test.eighteen_zones(image.shape[:2]):
            self.assertTrue(np.any(image[y1:y2, x1:x2] > 0))

    def test_status_panel_does_not_cover_the_eighteen_zone_images(self):
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
