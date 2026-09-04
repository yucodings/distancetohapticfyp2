import unittest
import hashlib
import inspect
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from calibration import load_calibration
from camera_backend import np
from config import CALIBRATION_PATH
import stereo_depth
from stereo_depth import (
    AsyncDepthProcessor,
    calculate_depth,
    create_vpi_engine,
    load_vpi_runtime_settings,
    make_depth_view,
    vpi_disparity_masks,
)


def simple_q() -> np.ndarray:
    q = np.zeros((4, 4), dtype=np.float64)
    q[2, 3] = 10.0
    q[3, 2] = 1.0
    return q


class StereoDepthTests(unittest.TestCase):
    def test_vpi_cuda_is_the_only_stereo_engine(self):
        self.assertFalse(hasattr(stereo_depth, "OpenCvStereoEngine"))
        self.assertEqual(
            tuple(inspect.signature(AsyncDepthProcessor).parameters),
            ("calibration", "maps", "status_callback"),
        )
        self.assertTrue(callable(create_vpi_engine))

    def test_vpi_initialization_failure_is_fatal_without_cpu_fallback(self):
        processor = AsyncDepthProcessor(None, None)
        with patch(
            "stereo_depth.create_vpi_engine",
            side_effect=RuntimeError("synthetic VPI failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "synthetic VPI failure"
            ):
                processor.start(timeout=1.0)
        processor.stop()

    def test_tuned_vpi_profile_is_calibration_locked(self):
        calibration = load_calibration(CALIBRATION_PATH)
        digest = hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest()
        profile = {
            "schema_version": 2,
            "backend": "vpi-cuda",
            "calibration_sha256": digest,
            "settings": {
                "min_disparity": 0,
                "max_disparity": 256,
                "confidence_threshold": 16384,
                "p1": 5,
                "p2": 64,
                "uniqueness": 0.9,
                "include_diagonals": True,
                "disparity_safety_margin_px": 8.0,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            settings, source = load_vpi_runtime_settings(calibration, path)
            self.assertEqual(settings.confidence_threshold, 16384)
            self.assertTrue(settings.include_diagonals)
            self.assertEqual(settings.disparity_safety_margin_px, 8.0)
            self.assertIn("tuned profile", source)

            profile["calibration_sha256"] = "0" * 64
            path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "calibration hash"):
                load_vpi_runtime_settings(calibration, path)

    def test_schema_one_vpi_profile_uses_safe_vpi_defaults(self):
        calibration = load_calibration(CALIBRATION_PATH)
        profile = {
            "schema_version": 1,
            "backend": "vpi-cuda",
            "calibration_sha256": hashlib.sha256(
                CALIBRATION_PATH.read_bytes()
            ).hexdigest(),
            "settings": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            settings, source = load_vpi_runtime_settings(calibration, path)
            self.assertEqual(settings.confidence_threshold, 8192)
            self.assertEqual(settings.uniqueness, 0.9)
            self.assertEqual(settings.disparity_safety_margin_px, 8.0)
            self.assertIn("legacy schema-1 profile ignored", source)

    def test_vpi_disparity_margin_rejects_last_eight_pixels(self):
        disparity = np.array(
            [[247.0, 248.0, 255.0, 256.0, np.inf]], dtype=np.float32
        )
        safe, near_limit, sentinel = vpi_disparity_masks(
            disparity, 0.0, 256.0, 8.0
        )
        np.testing.assert_array_equal(safe, [[True, False, False, False, False]])
        np.testing.assert_array_equal(
            near_limit, [[False, True, True, False, False]]
        )
        np.testing.assert_array_equal(
            sentinel, [[False, False, False, True, False]]
        )

    def test_disparity_converts_to_metric_z(self):
        disparity = np.array([[10.0, 5.0, 4.0]], dtype=np.float32)
        depth, valid = calculate_depth(disparity, simple_q())
        np.testing.assert_allclose(depth, [[1.0, 2.0, 2.5]], rtol=1e-6)
        self.assertTrue(np.all(valid))

    def test_invalid_disparity_depth_and_confidence_are_rejected(self):
        disparity = np.array(
            [[0.0, -1.0, 256.0, np.inf, 2.0, 100.0]], dtype=np.float32
        )
        confidence = np.array(
            [[True, True, True, True, True, False]], dtype=bool
        )
        depth, valid = calculate_depth(disparity, simple_q(), confidence)
        self.assertFalse(np.any(valid))
        self.assertTrue(np.all(np.isnan(depth)))

    def test_depth_colormap_keeps_invalid_pixels_black(self):
        depth = np.array([[1.0, np.nan]], dtype=np.float32)
        valid = np.array([[True, False]])
        view = make_depth_view(depth, valid)
        self.assertEqual(view.shape, (1, 2, 3))
        self.assertTrue(np.all(view[0, 1] == 0))

    def test_noncanonical_q_is_rejected(self):
        q = simple_q()
        q[2, 0] = 1.0
        with self.assertRaisesRegex(RuntimeError, "not canonical"):
            calculate_depth(np.ones((2, 2), dtype=np.float32), q)

if __name__ == "__main__":
    unittest.main()
