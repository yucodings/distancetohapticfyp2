from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tuner_core import (
    RoiStatistics,
    SgbmSettings,
    VpiSettings,
    automatic_tuning_score,
    compute_sgbm,
    cv2,
    depth_from_disparity,
    expected_disparity,
    load_calibration,
    make_alignment_overlay,
    np,
    roi_statistics,
    safety_tuning_score,
    settings_dict,
    vpi_autotune_candidates,
    vpi_disparity_masks,
    zone_safety_statistics,
)


def simple_q(focal_px: float = 100.0, baseline_m: float = 0.1) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, focal_px],
            [0.0, 0.0, 1.0 / baseline_m, 0.0],
        ],
        dtype=np.float64,
    )


class SettingsTests(unittest.TestCase):
    def test_sgbm_requires_odd_block_and_multiple_of_16(self):
        SgbmSettings(block_size=5, num_disparities=160).validate()
        with self.assertRaises(ValueError):
            SgbmSettings(block_size=4).validate()
        with self.assertRaises(ValueError):
            SgbmSettings(num_disparities=150).validate()

    def test_vpi_validates_range_and_penalties(self):
        VpiSettings(max_disparity=160, p1=3, p2=48).validate()
        with self.assertRaises(ValueError):
            VpiSettings(max_disparity=257).validate()
        with self.assertRaises(ValueError):
            VpiSettings(p1=49, p2=48).validate()
        with self.assertRaises(ValueError):
            VpiSettings(disparity_safety_margin_px=256).validate()

    def test_vpi_preset_explains_fixed_window(self):
        saved = settings_dict("vpi-cuda", VpiSettings())
        self.assertIn("fixed 9x7", saved["window_note"])

    def test_vpi_autotune_sweeps_every_native_quality_control(self):
        candidates = vpi_autotune_candidates()
        self.assertEqual(len(candidates), 96)
        self.assertEqual({item.p1 for item in candidates}, {1, 3, 5, 8})
        self.assertEqual({item.p2 for item in candidates}, {32, 48, 64, 96})
        self.assertEqual(
            {item.uniqueness for item in candidates}, {0.8, 0.9, 0.95}
        )
        self.assertEqual(
            {item.confidence_threshold for item in candidates},
            {4096, 8192, 16384, 32767},
        )
        self.assertEqual(
            {item.include_diagonals for item in candidates}, {False, True}
        )
        self.assertEqual({item.max_disparity for item in candidates}, {256})
        self.assertEqual(
            {item.disparity_safety_margin_px for item in candidates}, {8.0}
        )


class GeometryTests(unittest.TestCase):
    def test_vpi_disparity_margin_rejects_values_near_256(self):
        disparity = np.array(
            [[247.0, 248.0, 255.96875, 256.0, np.inf]], dtype=np.float32
        )
        safe, near_limit = vpi_disparity_masks(disparity, 0.0, 256.0, 8.0)
        np.testing.assert_array_equal(
            safe, [[True, False, False, False, False]]
        )
        np.testing.assert_array_equal(
            near_limit, [[False, True, True, False, False]]
        )

    def test_expected_disparity(self):
        q = simple_q()
        self.assertAlmostEqual(expected_disparity(q, 0.5), 20.0)
        self.assertAlmostEqual(expected_disparity(q, 1.0), 10.0)
        self.assertAlmostEqual(expected_disparity(q, 2.0), 5.0)

    def test_metric_depth_and_mask(self):
        disparity = np.array([[20.0, 10.0, 5.0, -1.0]], dtype=np.float32)
        accepted = np.array([[True, True, True, True]])
        depth, valid = depth_from_disparity(
            disparity, simple_q(), accepted, min_depth_m=0.1, max_depth_m=3.0
        )
        np.testing.assert_allclose(depth[0, :3], [0.5, 1.0, 2.0])
        self.assertFalse(valid[0, 3])
        self.assertTrue(np.isnan(depth[0, 3]))

    def test_roi_uses_robust_median_and_known_error(self):
        depth = np.ones((7, 7), dtype=np.float32)
        depth[3, 3] = 0.2
        valid = np.ones_like(depth, dtype=bool)
        stats = roi_statistics(depth, valid, (3, 3), 5, known_distance_m=1.1)
        self.assertAlmostEqual(stats.median_m, 1.0)
        self.assertAlmostEqual(stats.mad_m, 0.0)
        self.assertAlmostEqual(stats.error_m, -0.1)
        self.assertEqual(stats.valid_count, 25)

    def test_alignment_overlay_shape(self):
        left = np.zeros((40, 60, 3), dtype=np.uint8)
        right = left.copy()
        overlay = make_alignment_overlay(left, right)
        self.assertEqual(overlay.shape, left.shape)

    def test_automatic_score_prefers_accuracy_and_rejects_sparse_roi(self):
        accurate = roi_statistics(
            np.ones((9, 9), dtype=np.float32),
            np.ones((9, 9), dtype=bool),
            (4, 4),
            7,
            known_distance_m=1.0,
        )
        inaccurate_depth = np.full((9, 9), 1.3, dtype=np.float32)
        inaccurate = roi_statistics(
            inaccurate_depth,
            np.ones((9, 9), dtype=bool),
            (4, 4),
            7,
            known_distance_m=1.0,
        )
        self.assertLess(
            automatic_tuning_score(accurate, 60.0, 200.0, 1.0),
            automatic_tuning_score(inaccurate, 90.0, 100.0, 1.0),
        )
        sparse = roi_statistics(
            np.ones((9, 9), dtype=np.float32),
            np.eye(9, dtype=bool),
            (4, 4),
            7,
            known_distance_m=1.0,
        )
        self.assertIsNone(automatic_tuning_score(sparse, 10.0, 50.0, 1.0))

    def test_zone_safety_detects_supported_false_near_surface_in_each_third(self):
        depth = np.ones((30, 90), dtype=np.float32)
        valid = np.ones_like(depth, dtype=bool)
        disparity = np.full_like(depth, 10.0)
        confidence = np.ones_like(valid)
        for zone_index in range(3):
            x1 = zone_index * 30 + 8
            depth[8:18, x1 : x1 + 10] = 0.4
        zones = zone_safety_statistics(
            depth, valid, disparity, confidence, 256.0, 8.0, 1.0
        )
        self.assertEqual([zone.zone for zone in zones], ["left", "center", "right"])
        for zone in zones:
            self.assertGreater(zone.supported_unexpected_near_percentage, 3.0)
            self.assertAlmostEqual(zone.nearest_supported_m, 0.4)

    def test_multiframe_safety_score_penalizes_temporal_noise_and_rejects_near(self):
        def roi(median: float) -> RoiStatistics:
            return RoiStatistics(100, 100, 100.0, median, 0.01, median - 1.0)

        depth = np.ones((30, 90), dtype=np.float32)
        valid = np.ones_like(depth, dtype=bool)
        disparity = np.full_like(depth, 10.0)
        confidence = np.ones_like(valid)
        clean_zones = zone_safety_statistics(
            depth, valid, disparity, confidence, 256.0, 8.0, 1.0
        )
        stable = safety_tuning_score(
            [roi(1.0)] * 5, [clean_zones] * 5, [80.0] * 5, [50.0] * 5, 1.0
        )
        noisy = safety_tuning_score(
            [roi(value) for value in (0.8, 0.9, 1.0, 1.1, 1.2)],
            [clean_zones] * 5,
            [80.0] * 5,
            [50.0] * 5,
            1.0,
        )
        self.assertIsNotNone(stable.score)
        self.assertIsNotNone(noisy.score)
        self.assertLess(stable.score, noisy.score)

        unsafe_depth = depth.copy()
        unsafe_depth[5:20, 5:20] = 0.3
        unsafe_zones = zone_safety_statistics(
            unsafe_depth, valid, disparity, confidence, 256.0, 8.0, 1.0
        )
        unsafe = safety_tuning_score(
            [roi(1.0)] * 5, [unsafe_zones] * 5, [80.0] * 5, [50.0] * 5, 1.0
        )
        self.assertIsNone(unsafe.score)
        self.assertIn("unexpected-near", unsafe.rejection_reason)


class MatcherTests(unittest.TestCase):
    def test_sgbm_recovers_synthetic_shift(self):
        generator = np.random.RandomState(12)
        height, width, shift = 120, 320, 16
        texture = generator.randint(0, 256, (height, width), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0)
        left_gray = texture
        right_gray = np.zeros_like(left_gray)
        right_gray[:, :-shift] = left_gray[:, shift:]
        left = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
        disparity, valid = compute_sgbm(
            left,
            right,
            SgbmSettings(num_disparities=64, block_size=5, speckle_window_size=0),
        )
        interior = valid[:, 96:-32]
        values = disparity[:, 96:-32][interior]
        self.assertGreater(values.size, 500)
        self.assertAlmostEqual(float(np.median(values)), shift, delta=0.75)


class CalibrationTests(unittest.TestCase):
    def test_loads_minimal_calibration_and_hashes_it(self):
        height, width = 4, 6
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.npz"
            np.savez(
                path,
                image_width=width,
                image_height=height,
                sensor_mode=4,
                capture_fps=30,
                baseline_m=0.06,
                stereo_rms=0.5,
                Q=simple_q(),
                left_map1=xx,
                left_map2=yy,
                right_map1=xx,
                right_map2=yy,
            )
            calibration = load_calibration(path)
        self.assertEqual(calibration.width, width)
        self.assertEqual(calibration.height, height)
        self.assertEqual(len(calibration.sha256), 64)


if __name__ == "__main__":
    unittest.main()
