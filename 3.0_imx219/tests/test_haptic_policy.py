import unittest

from config import HAPTIC_LEVEL
from data_models import HazardBand, PATTERN_OFF
from haptic_policy import (
    PATTERN_FAST,
    PATTERN_SLOW,
    PATTERN_URGENT,
    band_from_depth,
    pattern_from_depth,
)


class HapticPolicyTests(unittest.TestCase):
    def test_invalid_and_clear_values_are_off(self):
        self.assertEqual(band_from_depth(None), HazardBand.CLEAR)
        self.assertEqual(pattern_from_depth(None), PATTERN_OFF)
        self.assertEqual(pattern_from_depth(2.0001), PATTERN_OFF)

    def test_exact_boundaries_follow_the_approved_ranges(self):
        self.assertEqual(pattern_from_depth(2.0), PATTERN_SLOW)
        self.assertEqual(pattern_from_depth(1.5), PATTERN_SLOW)
        self.assertEqual(pattern_from_depth(1.4999), PATTERN_FAST)
        self.assertEqual(pattern_from_depth(0.5), PATTERN_FAST)
        self.assertEqual(pattern_from_depth(0.4999), PATTERN_URGENT)

    def test_slow_pattern_is_point_two_seconds_every_one_point_five(self):
        self.assertAlmostEqual(PATTERN_SLOW.on_time, 0.20)
        self.assertAlmostEqual(PATTERN_SLOW.off_time, 1.30)

    def test_fast_pattern_is_point_two_seconds_every_point_eight(self):
        self.assertAlmostEqual(PATTERN_FAST.on_time, 0.20)
        self.assertAlmostEqual(PATTERN_FAST.off_time, 0.60)

    def test_urgent_pattern_has_short_perception_break(self):
        self.assertAlmostEqual(PATTERN_URGENT.on_time, 1.00)
        self.assertAlmostEqual(PATTERN_URGENT.off_time, 0.10)

    def test_all_active_patterns_use_one_fixed_strength(self):
        self.assertEqual(
            {PATTERN_SLOW.level, PATTERN_FAST.level, PATTERN_URGENT.level},
            {HAPTIC_LEVEL},
        )


if __name__ == "__main__":
    unittest.main()

