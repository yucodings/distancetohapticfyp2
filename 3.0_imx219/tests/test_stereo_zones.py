import unittest

import stereo_core as core


np = core.np


class StereoZoneTests(unittest.TestCase):
    def test_geometry_matches_2_1_1_1(self):
        self.assertEqual(
            core.three_zones((720, 1280)),
            (
                ("Left", (0, 72, 426, 648)),
                ("Centre", (426, 72, 853, 648)),
                ("Right", (853, 72, 1280, 648)),
            ),
        )

    def test_zone_values_are_independent_medians(self):
        depth = np.zeros((100, 300), dtype=np.float32)
        valid = np.ones(depth.shape, dtype=bool)
        for (_name, (x1, y1, x2, y2)), value in zip(
            core.three_zones(depth.shape), (0.4, 1.0, 1.8)
        ):
            depth[y1:y2, x1:x2] = value

        measurements = core.measure_three_depths(depth, valid, min_samples=1)

        for measurement, expected in zip(measurements, (0.4, 1.0, 1.8)):
            self.assertAlmostEqual(measurement.distance_m, expected, places=5)

    def test_invalid_zone_does_not_change_other_zone_values(self):
        depth = np.full((100, 300), 1.0, dtype=np.float32)
        valid = np.ones(depth.shape, dtype=bool)
        _name, (x1, y1, x2, y2) = core.three_zones(depth.shape)[1]
        valid[y1:y2, x1:x2] = False

        measurements = core.measure_three_depths(depth, valid, min_samples=1)

        self.assertEqual(measurements[0].distance_m, 1.0)
        self.assertIsNone(measurements[1].distance_m)
        self.assertEqual(measurements[2].distance_m, 1.0)


if __name__ == "__main__":
    unittest.main()
