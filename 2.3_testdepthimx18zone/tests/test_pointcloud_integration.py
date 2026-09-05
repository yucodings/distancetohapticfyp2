from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import test_depth_imx219_18zone as depth_test

np = depth_test.np


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


def synthetic_result() -> depth_test.DepthResult:
    disparity = np.full((8, 12), 10.0, dtype=np.float32)
    valid = np.ones(disparity.shape, dtype=bool)
    depth = np.ones(disparity.shape, dtype=np.float32)
    left = np.full((8, 12, 3), [30, 20, 10], dtype=np.uint8)
    return depth_test.DepthResult(
        sequence=7,
        disparity=disparity,
        depth_map=depth,
        valid_depth_mask=valid,
        left_rectified=left,
        depth_view=left.copy(),
        measurements=tuple(
            depth_test.DepthMeasurement(
                name,
                1.0,
                box,
                (box[2] - box[0]) * (box[3] - box[1]),
                100.0,
            )
            for name, box in depth_test.eighteen_zones(disparity.shape)
        ),
        valid_percentage=100.0,
        depth_fps=15.0,
        diagnostic_lines=(),
    )


class PointCloudIntegrationTests(unittest.TestCase):
    def test_depth_result_reconstructs_xyz(self):
        result = synthetic_result()
        with patch.object(depth_test, "POINT_CLOUD_STRIDE", 1):
            cloud = depth_test.point_cloud_from_depth_result(result, simple_q())
        self.assertEqual(cloud.count, 96)
        self.assertAlmostEqual(float(np.median(cloud.points[:, 2])), 1.0)

    def test_save_writes_complete_evidence_bundle(self):
        result = synthetic_result()
        with patch.object(depth_test, "POINT_CLOUD_STRIDE", 1):
            cloud = depth_test.point_cloud_from_depth_result(result, simple_q())
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(depth_test, "RESULTS_18ZONE_DIR", Path(temporary)):
                output = depth_test.save_3d_evidence(result, cloud)
            expected = {
                "left_rectified.png",
                "depth_heatmap.png",
                "orthographic_3d.png",
                "disparity_float32.npy",
                "depth_metres_float32.npy",
                "valid_mask.npy",
                "point_cloud.ply",
                "eighteen_zone_measurements.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)

    def test_saved_report_contains_all_eighteen_zones(self):
        result = synthetic_result()
        with patch.object(depth_test, "POINT_CLOUD_STRIDE", 1):
            cloud = depth_test.point_cloud_from_depth_result(result, simple_q())
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(depth_test, "RESULTS_18ZONE_DIR", Path(temporary)):
                output = depth_test.save_3d_evidence(result, cloud)
            import json

            report = json.loads(
                (output / "eighteen_zone_measurements.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["name"] for item in report["measurements"]],
                [
                    "Upper 1", "Upper 2", "Upper 3",
                    "Upper 4", "Upper 5", "Upper 6",
                    "Middle 1", "Middle 2", "Middle 3",
                    "Middle 4", "Middle 5", "Middle 6",
                    "Lower 1", "Lower 2", "Lower 3",
                    "Lower 4", "Lower 5", "Lower 6",
                ],
            )
            self.assertEqual(report["layout"]["type"], "eighteen_zone_grid")
            self.assertEqual(report["layout"]["columns"], 6)
            self.assertEqual(report["layout"]["rows"], 3)
            self.assertTrue(report["layout"]["full_width"])
            self.assertEqual(report["layout"]["top_fraction"], 0.1)
            self.assertEqual(report["layout"]["bottom_fraction"], 0.9)
            self.assertEqual(report["layout"]["minimum_valid_percentage"], 1.6)


if __name__ == "__main__":
    unittest.main()
