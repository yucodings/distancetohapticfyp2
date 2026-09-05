import unittest
from unittest.mock import patch

import stereo_core as core


np = core.np


class StereoZoneTests(unittest.TestCase):
    def test_geometry_matches_the_2_3_eighteen_zone_backend(self):
        self.assertEqual(
            core.eighteen_zones((720, 1280)),
            (
                ("Upper 1", (0, 72, 213, 264)),
                ("Upper 2", (213, 72, 426, 264)),
                ("Upper 3", (426, 72, 640, 264)),
                ("Upper 4", (640, 72, 853, 264)),
                ("Upper 5", (853, 72, 1066, 264)),
                ("Upper 6", (1066, 72, 1280, 264)),
                ("Middle 1", (0, 264, 213, 456)),
                ("Middle 2", (213, 264, 426, 456)),
                ("Middle 3", (426, 264, 640, 456)),
                ("Middle 4", (640, 264, 853, 456)),
                ("Middle 5", (853, 264, 1066, 456)),
                ("Middle 6", (1066, 264, 1280, 456)),
                ("Lower 1", (0, 456, 213, 648)),
                ("Lower 2", (213, 456, 426, 648)),
                ("Lower 3", (426, 456, 640, 648)),
                ("Lower 4", (640, 456, 853, 648)),
                ("Lower 5", (853, 456, 1066, 648)),
                ("Lower 6", (1066, 456, 1280, 648)),
            ),
        )

    def test_zone_values_are_eighteen_independent_medians(self):
        depth = np.zeros((100, 600), dtype=np.float32)
        valid = np.ones(depth.shape, dtype=bool)
        expected = tuple(value / 10.0 for value in range(1, 19))
        for (_name, (x1, y1, x2, y2)), value in zip(
            core.eighteen_zones(depth.shape), expected
        ):
            depth[y1:y2, x1:x2] = value

        measurements = core.measure_eighteen_depths(
            depth, valid, min_samples=1
        )

        for measurement, expected_value in zip(measurements, expected):
            self.assertAlmostEqual(
                measurement.distance_m, expected_value, places=5
            )

    def test_invalid_zone_does_not_change_other_zone_values(self):
        depth = np.full((100, 600), 1.0, dtype=np.float32)
        valid = np.ones(depth.shape, dtype=bool)
        _name, (x1, y1, x2, y2) = core.eighteen_zones(depth.shape)[1]
        valid[y1:y2, x1:x2] = False

        measurements = core.measure_eighteen_depths(
            depth, valid, min_samples=1
        )

        self.assertEqual(measurements[0].distance_m, 1.0)
        self.assertIsNone(measurements[1].distance_m)
        self.assertTrue(
            all(item.distance_m == 1.0 for item in measurements[2:])
        )

    def test_below_one_point_six_percent_skips_median(self):
        depth = np.ones((10, 25), dtype=np.float32)
        valid = np.zeros(depth.shape, dtype=bool)
        valid.flat[:3] = True

        with patch.object(core.np, "median") as median:
            measurement = core.measure_zone_depth(
                "Test", (0, 0, 25, 10), depth, valid, min_samples=1
            )

        self.assertIsNone(measurement.distance_m)
        self.assertAlmostEqual(measurement.valid_percentage, 1.2)
        median.assert_not_called()


if __name__ == "__main__":
    unittest.main()
