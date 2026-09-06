import unittest

import stereo_core as core
from data_models import PATTERN_OFF
from haptic_policy import PATTERN_FAST, PATTERN_SLOW, PATTERN_URGENT
from stream_worker import StreamWorker


def measurement(name, distance):
    return core.DepthMeasurement(name, distance, (0, 0, 10, 10), 100, 100.0)


class ZoneForwardingTests(unittest.TestCase):
    def test_three_values_map_to_the_matching_output_without_reprocessing(self):
        measurements = (
            measurement("Left", 1.8),
            measurement("Centre", 1.0),
            measurement("Right", 0.3),
        )

        patterns = StreamWorker.patterns_for_measurements(measurements)

        self.assertEqual(patterns["left"], PATTERN_SLOW)
        self.assertEqual(patterns["center"], PATTERN_FAST)
        self.assertEqual(patterns["right"], PATTERN_URGENT)

    def test_invalid_zone_is_independently_off(self):
        measurements = (
            measurement("Left", None),
            measurement("Centre", 0.8),
            measurement("Right", 3.0),
        )

        patterns = StreamWorker.patterns_for_measurements(measurements)

        self.assertEqual(patterns["left"], PATTERN_OFF)
        self.assertEqual(patterns["center"], PATTERN_FAST)
        self.assertEqual(patterns["right"], PATTERN_OFF)


if __name__ == "__main__":
    unittest.main()

