import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import test_depth_imx219 as depth_test


class TunedProfileTests(unittest.TestCase):
    def test_default_profile_matches_one_metre_autotune(self):
        self.assertEqual(depth_test.DEFAULT_STEREO_BACKEND, "opencv")
        self.assertEqual(depth_test.SGBM_BLOCK_SIZE, 11)
        self.assertEqual(depth_test.SGBM_NUM_DISPARITIES, 160)
        matcher = depth_test.create_stereo_matcher()
        self.assertEqual(matcher.getBlockSize(), 11)
        self.assertEqual(matcher.getNumDisparities(), 160)

    def test_saved_one_metre_pair_remains_near_one_metre(self):
        result_dir = (
            WORKSPACE_DIR
            / "1.1_depth_visualization"
            / "results"
            / "auto_2026-09-03_23-33-19_088158"
        )
        left_path = result_dir / "best_left_rectified.png"
        right_path = result_dir / "best_right_rectified.png"
        if not left_path.is_file() or not right_path.is_file():
            self.skipTest("saved 1 m tuning pair is not present")

        left = depth_test.cv2.imread(str(left_path))
        right = depth_test.cv2.imread(str(right_path))
        calibration = depth_test.load_calibration(
            depth_test.DEFAULT_CALIBRATION_PATH
        )
        engine = depth_test.OpenCvStereoEngine()
        disparity, confidence = engine.compute(left, right)
        depth, valid = depth_test.calculate_depth(
            disparity,
            calibration.q_matrix,
            depth_test.MAX_DEPTH_M,
            confidence,
            engine.minimum_disparity,
            engine.maximum_disparity,
        )
        distance, _, sample_count = depth_test.measure_centre_depth(depth, valid)
        self.assertIsNotNone(distance)
        self.assertGreater(sample_count, 100)
        self.assertGreater(distance, 0.90)
        self.assertLess(distance, 1.10)


if __name__ == "__main__":
    unittest.main()
