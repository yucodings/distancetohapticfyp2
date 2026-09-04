from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from stereo_depth_tuner import CapturedPair, collect_live_tuning_pairs
from tuner_core import np


class FakeCapture:
    def __init__(self):
        self.sequence = 0

    def get(self):
        self.sequence += 1
        frame = np.full((2, 3, 3), self.sequence, dtype=np.uint8)
        return CapturedPair(self.sequence, 0.25, frame, frame.copy())


class LiveEasyModeTests(unittest.TestCase):
    def test_collects_five_distinct_live_pairs(self):
        pairs = collect_live_tuning_pairs(FakeCapture())
        self.assertEqual(len(pairs), 5)
        self.assertEqual([pair.sequence for pair in pairs], [1, 2, 3, 4, 5])
        self.assertEqual(
            [int(pair.left[0, 0, 0]) for pair in pairs], [1, 2, 3, 4, 5]
        )

    def test_requires_at_least_three_frames(self):
        with self.assertRaises(ValueError):
            collect_live_tuning_pairs(FakeCapture(), count=2)


if __name__ == "__main__":
    unittest.main()
