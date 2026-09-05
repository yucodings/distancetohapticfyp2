import sys
import unittest
from argparse import Namespace
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import test_depth_imx219_3frame as depth_test


class RuntimeSettingsTests(unittest.TestCase):
    def test_defaults_are_owned_by_2_1_1(self):
        settings = depth_test.default_vpi_settings()
        self.assertEqual(settings.max_disparity, 160)
        self.assertEqual(settings.confidence_threshold, 8192)
        self.assertEqual(settings.p1, 3)
        self.assertEqual(settings.p2, 48)
        self.assertEqual(settings.uniqueness, 0.4)
        self.assertFalse(settings.include_diagonals)

    def test_command_line_settings_need_no_profile(self):
        args = Namespace(
            vpi_max_disparity=128,
            vpi_confidence_threshold=4096,
            vpi_uniqueness=-1.0,
        )
        settings = depth_test.vpi_settings_from_args(args)
        self.assertEqual(settings.max_disparity, 128)
        self.assertEqual(settings.confidence_threshold, 4096)
        self.assertEqual(settings.uniqueness, -1.0)

    def test_live_adjustments_match_2_1(self):
        settings = depth_test.default_vpi_settings()
        self.assertEqual(
            depth_test.adjust_vpi_settings(settings, "more_disparity").max_disparity,
            176,
        )
        self.assertEqual(
            depth_test.adjust_vpi_settings(
                settings, "less_confidence"
            ).confidence_threshold,
            4096,
        )
        self.assertEqual(
            depth_test.adjust_vpi_settings(settings, "less_uniqueness").uniqueness,
            0.2,
        )

    def test_invalid_runtime_values_are_rejected(self):
        args = Namespace(
            vpi_max_disparity=100,
            vpi_confidence_threshold=8192,
            vpi_uniqueness=0.4,
        )
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            depth_test.vpi_settings_from_args(args)


if __name__ == "__main__":
    unittest.main()

