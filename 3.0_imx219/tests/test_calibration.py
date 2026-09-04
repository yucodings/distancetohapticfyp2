import hashlib
import unittest
from pathlib import Path

from calibration import convert_rectification_maps, load_calibration
from config import CALIBRATION_PATH, HEIGHT, SENSOR_MODE, WIDTH, YOLO_MODEL_PATH
from stereo_capture import gstreamer_pipeline


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LATEST_CALIBRATION_SOURCE = (
    WORKSPACE_ROOT
    / "1.0_calibration"
    / "images"
    / "2026-09-04_14-09-56_446423_recalibrated"
    / "stereo_calibration.npz"
)


class CalibrationAssetTests(unittest.TestCase):
    def test_copied_assets_match_approved_sources(self):
        calibration_hash = hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest()
        source_hash = hashlib.sha256(
            LATEST_CALIBRATION_SOURCE.read_bytes()
        ).hexdigest()
        model_hash = hashlib.sha256(YOLO_MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(calibration_hash, source_hash)
        self.assertEqual(calibration_hash, "8198efd156491578afda3b11a9f2fe9c44d150663e6347676830a14dde4cff2e")
        self.assertEqual(
            model_hash,
            "6e27f313f27ec785ced2b12f9edaba3547bb26ea6aded49f2026f9fa37a9e6a9",
        )

    def test_calibration_matches_runtime_capture(self):
        calibration = load_calibration(CALIBRATION_PATH)
        self.assertEqual((calibration.width, calibration.height), (WIDTH, HEIGHT))
        self.assertEqual(calibration.sensor_mode, SENSOR_MODE)
        self.assertAlmostEqual(calibration.baseline_m, 0.0602910216, places=8)
        maps = convert_rectification_maps(calibration)
        self.assertEqual(maps.left_map1.shape[:2], (HEIGHT, WIDTH))
        self.assertEqual(maps.right_map1.shape[:2], (HEIGHT, WIDTH))

    def test_gstreamer_pipeline_uses_calibrated_mode(self):
        calibration = load_calibration(CALIBRATION_PATH)
        pipeline = gstreamer_pipeline(0, calibration)
        self.assertIn("sensor-id=0", pipeline)
        self.assertIn("sensor-mode=4", pipeline)
        self.assertIn("width=(int)1280", pipeline)
        self.assertIn("height=(int)720", pipeline)
        self.assertIn("framerate=(fraction)30/1", pipeline)
        self.assertIn("drop=true", pipeline)


if __name__ == "__main__":
    unittest.main()
