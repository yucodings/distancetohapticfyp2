import inspect
import unittest
from pathlib import Path

from camera_backend import np
from data_models import (
    DetectionResult,
    PATTERN_CONTINUOUS,
    PATTERN_MEDIUM,
    PATTERN_OFF,
    PATTERN_SLOW,
)
from hazard_policy import pattern_from_depth
from zone_depth import compute_stereo_depth_zones, empty_stereo_zones


class StereoZoneTests(unittest.TestCase):
    def test_each_zone_selects_nearest_supported_surface(self):
        depth = np.full((60, 150), np.nan, dtype=np.float32)
        depth[15:21, 15:21] = 1.8
        depth[22:28, 68:74] = 0.8
        depth[30:36, 120:126] = 0.35
        valid = np.isfinite(depth)

        zones = compute_stereo_depth_zones(depth, valid)

        self.assertAlmostEqual(zones["left"].depth_m, 1.8, places=5)
        self.assertAlmostEqual(zones["center"].depth_m, 0.8, places=5)
        self.assertAlmostEqual(zones["right"].depth_m, 0.35, places=5)
        self.assertEqual(zones["left"].rect, (0, 0, 50, 60))
        self.assertEqual(zones["center"].rect, (50, 0, 100, 60))
        self.assertEqual(zones["right"].rect, (100, 0, 150, 60))
        for zone in zones.values():
            self.assertIsNotNone(zone.nearest_point)

    def test_isolated_near_speckle_does_not_override_supported_surface(self):
        depth = np.full((60, 150), np.nan, dtype=np.float32)
        depth[15, 15] = 0.2
        depth[25:31, 20:26] = 1.7
        valid = np.isfinite(depth)

        zones = compute_stereo_depth_zones(depth, valid)

        self.assertAlmostEqual(zones["left"].depth_m, 1.7, places=5)

    def test_small_connected_false_near_cluster_does_not_override_surface(self):
        depth = np.full((60, 150), np.nan, dtype=np.float32)
        depth[20:22, 15:22] = 0.3
        depth[30:36, 25:31] = 1.2
        valid = np.isfinite(depth)

        zones = compute_stereo_depth_zones(depth, valid)

        self.assertAlmostEqual(zones["left"].depth_m, 1.2, places=5)

    def test_border_surface_is_excluded(self):
        depth = np.full((60, 150), np.nan, dtype=np.float32)
        depth[0:6, 20:26] = 0.3
        depth[25:31, 20:26] = 1.4
        valid = np.isfinite(depth)

        zones = compute_stereo_depth_zones(depth, valid)

        self.assertAlmostEqual(zones["left"].depth_m, 1.4, places=5)

    def test_latest_vpi_result_has_no_false_continuous_alert(self):
        result_dir = (
            Path(__file__).resolve().parents[2]
            / "1.1_depth_visualization"
            / "results"
            / "vpi_auto_2026-09-04_15-27-41_727249"
        )
        depth_path = result_dir / "best_depth_metres_float32.npy"
        valid_path = result_dir / "best_valid_mask.npy"
        if not depth_path.is_file() or not valid_path.is_file():
            self.skipTest("latest saved VPI result is not present")
        depth = np.load(depth_path)
        valid = np.load(valid_path).astype(bool)

        zones = compute_stereo_depth_zones(depth, valid)

        for result in zones.values():
            if result.depth_m is not None:
                self.assertGreaterEqual(result.depth_m, 0.5)
                self.assertNotEqual(result.tone, "red")

    def test_no_supported_depth_means_off(self):
        zones = empty_stereo_zones((6, 9), "stale")
        for zone in zones.values():
            self.assertIsNone(zone.depth_m)
            self.assertEqual(pattern_from_depth(zone.depth_m), PATTERN_OFF)

    def test_zone_api_cannot_accept_yolo_results(self):
        parameters = tuple(inspect.signature(compute_stereo_depth_zones).parameters)
        self.assertEqual(parameters, ("depth_map", "valid_depth_mask"))
        detection = DetectionResult(0, "person", 0.9, (0, 0, 5, 5), 0.2)
        with self.assertRaises((AttributeError, ValueError)):
            compute_stereo_depth_zones(detection, detection)


class HapticPolicyTests(unittest.TestCase):
    def test_exact_threshold_boundaries(self):
        cases = (
            (None, PATTERN_OFF),
            (2.000001, PATTERN_OFF),
            (2.0, PATTERN_SLOW),
            (1.5, PATTERN_SLOW),
            (1.499999, PATTERN_MEDIUM),
            (0.5, PATTERN_MEDIUM),
            (0.499999, PATTERN_CONTINUOUS),
        )
        for depth, expected in cases:
            with self.subTest(depth=depth):
                self.assertEqual(pattern_from_depth(depth), expected)

    def test_exact_envelopes(self):
        self.assertEqual((PATTERN_SLOW.level, PATTERN_SLOW.on_time, PATTERN_SLOW.off_time), (0x30, 0.50, 0.50))
        self.assertEqual((PATTERN_MEDIUM.level, PATTERN_MEDIUM.on_time, PATTERN_MEDIUM.off_time), (0x60, 0.08, 0.12))
        self.assertEqual((PATTERN_CONTINUOUS.level, PATTERN_CONTINUOUS.on_time, PATTERN_CONTINUOUS.off_time), (0x7F, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
