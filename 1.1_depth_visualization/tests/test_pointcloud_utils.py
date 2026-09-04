from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pointcloud_utils import (
    SampledPointCloud,
    disparity_to_xyz,
    make_orthographic_view,
    sample_point_cloud,
    save_binary_ply,
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


class PointCloudTests(unittest.TestCase):
    def test_disparity_reprojects_to_metric_xyz_and_masks_invalid(self):
        disparity = np.full((2, 3), 10.0, dtype=np.float32)
        valid = np.ones(disparity.shape, dtype=bool)
        valid[0, 0] = False

        xyz, xyz_valid = disparity_to_xyz(disparity, simple_q(), valid)

        np.testing.assert_allclose(xyz[1, 2], [0.02, 0.01, 1.0], atol=1e-6)
        self.assertFalse(xyz_valid[0, 0])
        self.assertTrue(np.all(np.isnan(xyz[0, 0])))

    def test_sampling_converts_bgr_to_rgb(self):
        xyz = np.zeros((2, 2, 3), dtype=np.float32)
        xyz[:, :, 2] = 1.0
        colour = np.zeros((2, 2, 3), dtype=np.uint8)
        colour[0, 0] = [3, 2, 1]
        valid = np.zeros((2, 2), dtype=bool)
        valid[0, 0] = True

        cloud = sample_point_cloud(xyz, colour, valid, stride=1)

        self.assertEqual(cloud.count, 1)
        np.testing.assert_array_equal(cloud.colors_rgb[0], [1, 2, 3])

    def test_projection_and_binary_ply_are_created(self):
        cloud = SampledPointCloud(
            points=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            colors_rgb=np.array([[255, 20, 10]], dtype=np.uint8),
        )
        view = make_orthographic_view(cloud, width=640, height=360)
        self.assertEqual(view.shape, (360, 640, 3))

        with tempfile.TemporaryDirectory() as temporary:
            path = save_binary_ply(Path(temporary) / "cloud.ply", cloud)
            payload = path.read_bytes()
        self.assertIn(b"element vertex 1\n", payload[:256])
        self.assertIn(b"format binary_little_endian 1.0\n", payload[:256])


if __name__ == "__main__":
    unittest.main()
