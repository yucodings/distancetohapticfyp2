import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import test_depth_imx219_3zone as depth_test


class TunedProfileTests(unittest.TestCase):
    def test_default_profile_matches_latest_vpi_easy_mode(self):
        self.assertEqual(depth_test.DEFAULT_STEREO_BACKEND, "vpi-cuda")
        calibration = depth_test.load_calibration(
            depth_test.DEFAULT_CALIBRATION_PATH
        )
        settings, source = depth_test.load_vpi_profile(
            calibration, depth_test.DEFAULT_VPI_PROFILE_PATH
        )
        self.assertEqual(settings.confidence_threshold, 32767)
        self.assertEqual(settings.p1, 8)
        self.assertEqual(settings.p2, 96)
        self.assertEqual(settings.uniqueness, 0.8)
        self.assertFalse(settings.include_diagonals)
        self.assertEqual(settings.disparity_safety_margin_px, 8.0)
        self.assertIn("vpi_tuned_profile.json", source)

    def test_opencv_comparison_profile_is_still_available(self):
        self.assertEqual(depth_test.SGBM_BLOCK_SIZE, 11)
        self.assertEqual(depth_test.SGBM_NUM_DISPARITIES, 160)
        matcher = depth_test.create_stereo_matcher()
        self.assertEqual(matcher.getBlockSize(), 11)
        self.assertEqual(matcher.getNumDisparities(), 160)

    def test_latest_saved_vpi_three_zones_can_be_measured(self):
        result_dir = (
            WORKSPACE_DIR
            / "1.1_depth_visualization"
            / "results"
            / "vpi_auto_2026-09-04_15-27-41_727249"
        )
        depth_path = result_dir / "best_depth_metres_float32.npy"
        valid_path = result_dir / "best_valid_mask.npy"
        if not depth_path.is_file() or not valid_path.is_file():
            self.skipTest("saved 1 m tuning pair is not present")
        depth = depth_test.np.load(depth_path)
        valid = depth_test.np.load(valid_path).astype(bool)
        measurements = depth_test.measure_three_depths(depth, valid)
        self.assertEqual(
            [measurement.name for measurement in measurements],
            ["Left", "Centre", "Right"],
        )
        centre = measurements[1]
        self.assertIsNotNone(centre.distance_m)
        self.assertGreater(centre.sample_count, 100)
        self.assertGreater(centre.distance_m, 0.90)
        self.assertLess(centre.distance_m, 1.10)


if __name__ == "__main__":
    unittest.main()
