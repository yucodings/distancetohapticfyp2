import inspect
import unittest

import numpy as np

from data_models import (
    PATTERN_CONTINUOUS,
    PATTERN_MEDIUM,
    PATTERN_OFF,
    PATTERN_SLOW,
)
from hazard_policy import pattern_from_depth
from raw_depth import compute_raw_depth_zones


class RawDepthZoneTests(unittest.TestCase):
    def test_each_vertical_third_uses_its_nearest_raw_pixel(self):
        frame = np.zeros((4, 9), dtype=np.float32)
        frame[2, 1] = 1.8
        frame[0, 2] = 2.4
        frame[3, 4] = 0.8
        frame[1, 7] = 0.3

        zones = compute_raw_depth_zones(frame)

        self.assertAlmostEqual(zones["left"].depth_m, 1.8, places=6)
        self.assertEqual(zones["left"].nearest_point, (1, 2))
        self.assertEqual(zones["left"].rect, (0, 0, 3, 4))
        self.assertAlmostEqual(zones["center"].depth_m, 0.8, places=6)
        self.assertEqual(zones["center"].nearest_point, (4, 3))
        self.assertEqual(zones["center"].rect, (3, 0, 6, 4))
        self.assertAlmostEqual(zones["right"].depth_m, 0.3, places=6)
        self.assertEqual(zones["right"].nearest_point, (7, 1))
        self.assertEqual(zones["right"].rect, (6, 0, 9, 4))

    def test_valid_depth_limits_are_strict(self):
        frame = np.array(
            [
                [0.0, 0.1, np.nan, 3.0, np.inf, -1.0, 0.1, 3.0, 0.0],
                [0.1, 3.0, 0.0, 0.1, 3.0, 0.0, np.nan, np.inf, -2.0],
            ],
            dtype=np.float32,
        )

        zones = compute_raw_depth_zones(frame)

        for zone in zones.values():
            self.assertIsNone(zone.depth_m)
            self.assertIsNone(zone.nearest_point)
            self.assertEqual(zone.tone, "invalid")

    def test_values_just_inside_raw_limits_are_valid(self):
        frame = np.zeros((1, 9), dtype=np.float32)
        frame[0, 0] = np.nextafter(np.float32(0.1), np.float32(np.inf))
        frame[0, 3] = np.nextafter(np.float32(3.0), np.float32(0.0))
        frame[0, 6] = 2.5

        zones = compute_raw_depth_zones(frame)

        self.assertIsNotNone(zones["left"].depth_m)
        self.assertLess(zones["left"].depth_m, 0.101)
        self.assertIsNotNone(zones["center"].depth_m)
        self.assertLess(zones["center"].depth_m, 3.0)
        self.assertAlmostEqual(zones["right"].depth_m, 2.5)

    def test_zone_extractor_has_no_detection_input(self):
        parameters = inspect.signature(compute_raw_depth_zones).parameters
        self.assertEqual(tuple(parameters), ("depth_in_meters",))


class HapticThresholdTests(unittest.TestCase):
    def test_exact_boundaries(self):
        cases = [
            (None, PATTERN_OFF),
            (2.000001, PATTERN_OFF),
            (2.0, PATTERN_SLOW),
            (1.5, PATTERN_SLOW),
            (1.499999, PATTERN_MEDIUM),
            (0.5, PATTERN_MEDIUM),
            (0.499999, PATTERN_CONTINUOUS),
            (0.1, PATTERN_CONTINUOUS),
        ]
        for depth_m, expected in cases:
            with self.subTest(depth_m=depth_m):
                self.assertEqual(pattern_from_depth(depth_m), expected)

    def test_exact_strengths_and_envelopes(self):
        self.assertEqual(
            (PATTERN_OFF.level, PATTERN_OFF.on_time, PATTERN_OFF.off_time),
            (0x00, 0.0, 0.0),
        )
        self.assertEqual(
            (PATTERN_SLOW.level, PATTERN_SLOW.on_time, PATTERN_SLOW.off_time),
            (0x30, 0.50, 0.50),
        )
        self.assertEqual(
            (PATTERN_MEDIUM.level, PATTERN_MEDIUM.on_time, PATTERN_MEDIUM.off_time),
            (0x60, 0.08, 0.12),
        )
        self.assertEqual(
            (
                PATTERN_CONTINUOUS.level,
                PATTERN_CONTINUOUS.on_time,
                PATTERN_CONTINUOUS.off_time,
            ),
            (0x7F, 0.0, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
