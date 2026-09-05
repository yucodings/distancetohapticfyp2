from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import test_depth_imx219_3frame as depth_test

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
        measurements=(
            depth_test.DepthMeasurement("Left", 1.0, (0, 2, 3, 6), 12, 100.0),
            depth_test.DepthMeasurement("Centre", 1.0, (4, 2, 7, 6), 12, 100.0),
            depth_test.DepthMeasurement("Right", 1.0, (8, 2, 11, 6), 12, 100.0),
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
            with patch.object(depth_test, "RESULTS_3D_DIR", Path(temporary)):
                output = depth_test.save_3d_evidence(result, cloud)
            expected = {
                "left_rectified.png",
                "depth_heatmap.png",
                "orthographic_3d.png",
                "disparity_float32.npy",
                "depth_metres_float32.npy",
                "valid_mask.npy",
                "point_cloud.ply",
                "three_zone_measurements.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)

    def test_saved_report_contains_all_three_zones(self):
        result = synthetic_result()
        with patch.object(depth_test, "POINT_CLOUD_STRIDE", 1):
            cloud = depth_test.point_cloud_from_depth_result(result, simple_q())
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(depth_test, "RESULTS_3D_DIR", Path(temporary)):
                output = depth_test.save_3d_evidence(result, cloud)
            import json

            report = json.loads(
                (output / "three_zone_measurements.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["name"] for item in report["measurements"]],
                ["Left", "Centre", "Right"],
            )


if __name__ == "__main__":
    unittest.main()
