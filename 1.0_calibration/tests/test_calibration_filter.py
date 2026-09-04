from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import calibration_filter as quality


def sample(name: str, size: float = 20.0) -> quality.PairSample:
    corners = quality.np.array(
        [[[0.0, 0.0]], [[size, 0.0]], [[size, size]], [[0.0, size]]],
        dtype=quality.np.float32,
    )
    return quality.PairSample(
        filename=name,
        object_points=quality.np.zeros((4, 3), dtype=quality.np.float32),
        left_corners=corners,
        right_corners=corners.copy(),
        left_path=Path("left") / name,
        right_path=Path("right") / name,
    )


def metrics(name: str, rectified_p95: float = 0.5) -> dict[str, float | str]:
    return {
        "filename": name,
        "left_board_area_percent": 4.0,
        "right_board_area_percent": 4.0,
        "left_reprojection_rms_px": 0.4,
        "right_reprojection_rms_px": 0.4,
        "epipolar_p95_px": 0.6,
        "rectified_vertical_p95_px": rectified_p95,
        "rectified_vertical_max_px": rectified_p95,
    }


class CalibrationFilterTests(unittest.TestCase):
    def test_board_area_is_reported_as_image_percentage(self):
        corners = quality.np.array(
            [[[0.0, 0.0]], [[20.0, 0.0]], [[20.0, 10.0]], [[0.0, 10.0]]]
        )
        self.assertAlmostEqual(
            quality.board_area_percent(corners, (100, 100)), 2.0
        )

    def test_metric_violations_explain_the_failed_limit(self):
        reasons, score = quality.metric_violations(
            metrics("bad.png", rectified_p95=2.0),
            quality.QualityThresholds(
                max_rectified_vertical_p95_px=1.0
            ),
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("rectified vertical P95", reasons[0])
        self.assertAlmostEqual(score, 2.0)

    def test_filter_recalibrates_after_rejecting_bad_geometry(self):
        samples = [sample("a.png"), sample("b.png"), sample("c.png")]
        first = [metrics("a.png", 2.0), metrics("b.png"), metrics("c.png")]
        second = [metrics("b.png"), metrics("c.png")]
        calibration = {
            "left_rms": 0.4,
            "right_rms": 0.4,
            "stereo_rms": 0.5,
            "baseline_m": 0.06,
        }
        with (
            patch.object(quality, "calibrate_once", return_value=calibration),
            patch.object(quality, "rectify_once", return_value={}),
            patch.object(quality, "measure_pairs", side_effect=[first, second]),
        ):
            result = quality.filter_calibration_pairs(
                samples,
                (100, 100),
                min_pairs=2,
                thresholds=quality.QualityThresholds(
                    min_board_area_percent=1.0,
                    max_rectified_vertical_p95_px=1.0,
                ),
            )
        self.assertEqual([item.filename for item in result.accepted], ["b.png", "c.png"])
        self.assertEqual(result.rejected[0]["filename"], "a.png")
        self.assertEqual(len(result.history), 2)

    def test_tiny_board_is_rejected_before_calibration(self):
        samples = [sample("tiny.png", 2.0), sample("good.png", 20.0)]
        calibration = {
            "left_rms": 0.4,
            "right_rms": 0.4,
            "stereo_rms": 0.5,
            "baseline_m": 0.06,
        }
        with (
            patch.object(quality, "calibrate_once", return_value=calibration),
            patch.object(quality, "rectify_once", return_value={}),
            patch.object(
                quality, "measure_pairs", return_value=[metrics("good.png")]
            ),
        ):
            result = quality.filter_calibration_pairs(
                samples,
                (100, 100),
                min_pairs=1,
                thresholds=quality.QualityThresholds(
                    min_board_area_percent=1.0
                ),
            )
        self.assertEqual(result.rejected[0]["stage"], "board-size prefilter")


if __name__ == "__main__":
    unittest.main()
