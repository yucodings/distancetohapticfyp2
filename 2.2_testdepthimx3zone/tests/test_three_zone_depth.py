from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import test_depth_imx219_3zone as depth_test

np = depth_test.np


class ThreeZoneDepthTests(unittest.TestCase):
    def test_three_zones_cover_full_usable_width_without_gaps(self):
        height, width = 720, 1280
        regions = depth_test.three_zones((height, width))

        self.assertEqual([name for name, _ in regions], ["Left", "Centre", "Right"])
        self.assertEqual(
            [box for _name, box in regions],
            [(0, 72, 426, 648), (426, 72, 853, 648), (853, 72, 1280, 648)],
        )
        self.assertEqual(regions[0][1][0], 0)
        self.assertEqual(regions[0][1][2], regions[1][1][0])
        self.assertEqual(regions[1][1][2], regions[2][1][0])
        self.assertEqual(regions[2][1][2], width)

    def test_smallest_supported_image_has_three_nonempty_zones(self):
        regions = depth_test.three_zones((1, 3))
        self.assertEqual(
            [box for _name, box in regions],
            [(0, 0, 1, 1), (1, 0, 2, 1), (2, 0, 3, 1)],
        )
        with self.assertRaises(ValueError):
            depth_test.three_zones((10, 2))

    def test_each_zone_has_an_independent_median(self):
        depth = np.zeros((100, 300), dtype=np.float32)
        valid = np.ones(depth.shape, dtype=bool)
        expected = (1.0, 2.0, 3.0)
        for (_name, (x1, y1, x2, y2)), value in zip(
            depth_test.three_zones(depth.shape), expected
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
        _name, (x1, y1, x2, y2) = depth_test.three_zones(depth.shape)[0]
        valid[y1:y2, x1:x2] = False

        measurements = depth_test.measure_three_depths(depth, valid, min_samples=10)

        self.assertIsNone(measurements[0].distance_m)
        self.assertEqual(measurements[0].sample_count, 0)
        self.assertEqual(measurements[1].distance_m, 2.0)
        self.assertEqual(measurements[2].distance_m, 2.0)

    def test_sample_threshold_is_applied_per_zone(self):
        depth = np.ones((20, 60), dtype=np.float32)
        valid = np.zeros(depth.shape, dtype=bool)
        regions = depth_test.three_zones(depth.shape)
        for index, (_name, (x1, y1, x2, y2)) in enumerate(regions):
            self.assertGreaterEqual(x2 - x1, index + 1)
            self.assertGreater(y2, y1)
            valid[y1, x1 : x1 + index + 1] = True

        measurements = depth_test.measure_three_depths(depth, valid, min_samples=2)

        self.assertIsNone(measurements[0].distance_m)
        self.assertIsNotNone(measurements[1].distance_m)
        self.assertIsNotNone(measurements[2].distance_m)

    def test_valid_percentage_uses_only_the_zone_area(self):
        depth = np.ones((10, 30), dtype=np.float32)
        valid = np.zeros(depth.shape, dtype=bool)
        _name, (x1, y1, x2, y2) = depth_test.three_zones(depth.shape)[1]
        zone_area = (x2 - x1) * (y2 - y1)
        valid[y1:y2, x1 : x1 + (x2 - x1) // 2] = True

        measurement = depth_test.measure_three_depths(
            depth, valid, min_samples=1
        )[1]

        self.assertEqual(measurement.sample_count, zone_area // 2)
        self.assertAlmostEqual(measurement.valid_percentage, 50.0)

    def test_zone_overlay_changes_pixels_inside_all_three_zones(self):
        image = np.zeros((100, 300, 3), dtype=np.uint8)
        measurements = depth_test.empty_measurements(image.shape[:2])
        depth_test.draw_measurements(image, measurements)

        for _name, (x1, y1, x2, y2) in depth_test.three_zones(image.shape[:2]):
            sample = image[(y1 + y2) // 2, (x1 + x2) // 2]
            self.assertTrue(np.any(sample > 0))

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
