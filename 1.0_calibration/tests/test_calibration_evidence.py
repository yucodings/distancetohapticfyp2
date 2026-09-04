import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import mycalibration as calibration


class CalibrationEvidenceTests(unittest.TestCase):
    def test_session_creates_all_evidence_directories(self):
        original_root = calibration.IMAGES_ROOT
        with tempfile.TemporaryDirectory() as temporary:
            calibration.IMAGES_ROOT = Path(temporary)
            try:
                folders = calibration.create_image_folders()
            finally:
                calibration.IMAGES_ROOT = original_root
            for directory in (
                folders.left_raw,
                folders.right_raw,
                folders.corners_left,
                folders.corners_right,
                folders.rejected_left,
                folders.rejected_right,
                folders.quality_rejected_left,
                folders.quality_rejected_right,
                folders.rectified,
            ):
                self.assertTrue(directory.is_dir())

    def test_annotation_keeps_raw_input_unchanged(self):
        image = calibration.np.zeros((80, 120, 3), dtype=calibration.np.uint8)
        original = image.copy()
        annotated = calibration.annotated_checkerboard(
            image, False, None, "REJECTED TEST"
        )
        self.assertEqual(annotated.shape, image.shape)
        self.assertTrue(calibration.np.array_equal(image, original))
        self.assertFalse(calibration.np.array_equal(annotated, original))

    def test_reprojection_error_is_zero_for_exact_projection(self):
        objects = calibration.make_object_points()
        camera_matrix = calibration.np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        distortion = calibration.np.zeros(5)
        rvec = calibration.np.zeros((3, 1))
        tvec = calibration.np.array([[0.0], [0.0], [1.0]])
        image_points, _ = calibration.cv2.projectPoints(
            objects, rvec, tvec, camera_matrix, distortion
        )
        errors = calibration.per_view_reprojection_errors(
            [objects],
            [image_points],
            [rvec],
            [tvec],
            camera_matrix,
            distortion,
        )
        self.assertAlmostEqual(errors[0], 0.0, places=6)

    def test_epipolar_error_detects_vertical_offset(self):
        fundamental = calibration.np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )
        left = calibration.np.array([[[10.0, 20.0]], [[30.0, 40.0]]])
        right = left.copy()
        self.assertTrue(
            calibration.np.allclose(
                calibration.epipolar_errors(left, right, fundamental), 0.0
            )
        )
        right[:, 0, 1] += 2.0
        self.assertTrue(
            calibration.np.allclose(
                calibration.epipolar_errors(left, right, fundamental), 2.0
            )
        )


if __name__ == "__main__":
    unittest.main()
