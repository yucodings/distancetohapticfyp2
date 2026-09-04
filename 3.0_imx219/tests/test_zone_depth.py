import inspect
import unittest
from pathlib import Path

from camera_backend import np
from data_models import (
    DetectionResult,
    PATTERN_CONTINUOUS,
    PATTERN_MEDIUM,
    PATTERN_OFF,
    PATTERN_SLOW,
)
from hazard_policy import pattern_from_depth, pattern_from_depth_hysteresis
from zone_depth import (
    WinnerSwitchTracker,
    compute_stereo_depth_grid,
    empty_depth_grid,
    empty_stereo_zones,
)


HEIGHT = 120
WIDTH = 300


def blank_depth():
    return np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)


def fill_cell(depth, row, column, value):
    y1, y2 = HEIGHT * row // 3, HEIGHT * (row + 1) // 3
    x1, x2 = WIDTH * column // 3, WIDTH * (column + 1) // 3
    depth[y1:y2, x1:x2] = value


def cells_for_values(values):
    depth = blank_depth()
    for (row, column), value in values.items():
        fill_cell(depth, row, column, value)
    _zones, cells = compute_stereo_depth_grid(depth, np.isfinite(depth))
    return cells


class StereoGridTests(unittest.TestCase):
    def test_grid_has_exact_nine_cell_boundaries(self):
        depth = np.ones((HEIGHT, WIDTH), dtype=np.float32)
        _zones, cells = compute_stereo_depth_grid(depth, np.ones_like(depth, bool))

        self.assertEqual(len(cells), 9)
        self.assertEqual(cells["left_top"].rect, (0, 0, 100, 40))
        self.assertEqual(cells["center_middle"].rect, (100, 40, 200, 80))
        self.assertEqual(cells["right_bottom"].rect, (200, 80, 300, 120))
        self.assertEqual(cells["center_middle"].sample_rect, (120, 48, 180, 72))

    def test_cell_uses_only_its_centered_test_style_roi(self):
        depth = blank_depth()
        fill_cell(depth, 1, 1, 0.3)
        depth[48:72, 120:180] = 1.4

        zones, _cells = compute_stereo_depth_grid(depth, np.isfinite(depth))

        self.assertAlmostEqual(zones["center"].depth_m, 1.4)

    def test_each_column_selects_nearest_cell_median(self):
        depth = blank_depth()
        values = (
            (1.8, 2.2, 1.6),
            (0.9, 1.2, 1.1),
            (1.4, 0.7, 0.8),
        )
        for row in range(3):
            for column in range(3):
                fill_cell(depth, row, column, values[row][column])

        zones, cells = compute_stereo_depth_grid(depth, np.isfinite(depth))

        self.assertAlmostEqual(zones["left"].depth_m, 0.9)
        self.assertAlmostEqual(zones["center"].depth_m, 0.7)
        self.assertAlmostEqual(zones["right"].depth_m, 0.8)
        self.assertEqual(zones["left"].source_cell, "LM")
        self.assertEqual(zones["center"].source_cell, "CB")
        self.assertEqual(zones["right"].source_cell, "RB")
        self.assertEqual(
            {cell.display_name for cell in cells.values() if cell.selected},
            {"LM", "CB", "RB"},
        )

    def test_cell_median_ignores_isolated_near_pixel(self):
        depth = blank_depth()
        fill_cell(depth, 1, 0, 1.7)
        depth[50, 30] = 0.2

        zones, _cells = compute_stereo_depth_grid(depth, np.isfinite(depth))

        self.assertAlmostEqual(zones["left"].depth_m, 1.7)

    def test_bottom_row_is_used_normally(self):
        depth = blank_depth()
        fill_cell(depth, 0, 2, 1.8)
        fill_cell(depth, 1, 2, 1.2)
        fill_cell(depth, 2, 2, 0.6)

        zones, _cells = compute_stereo_depth_grid(depth, np.isfinite(depth))

        self.assertAlmostEqual(zones["right"].depth_m, 0.6)
        self.assertEqual(zones["right"].source_cell, "RB")

    def test_cell_below_minimum_pixel_count_is_invalid(self):
        depth = blank_depth()
        depth[48, 120:180] = 1.0
        depth[49, 120:159] = 1.0

        zones, cells = compute_stereo_depth_grid(depth, np.isfinite(depth))

        self.assertEqual(cells["center_middle"].valid_count, 99)
        self.assertIsNone(cells["center_middle"].depth_m)
        self.assertIsNone(zones["center"].depth_m)

    def test_cell_accepts_exactly_one_hundred_valid_pixels(self):
        depth = blank_depth()
        depth[48, 120:180] = 1.1
        depth[49, 120:160] = 1.1

        zones, cells = compute_stereo_depth_grid(depth, np.isfinite(depth))

        self.assertEqual(cells["center_middle"].valid_count, 100)
        self.assertAlmostEqual(zones["center"].depth_m, 1.1)

    def test_invalid_cells_are_ignored_and_invalid_column_turns_off(self):
        depth = blank_depth()
        fill_cell(depth, 1, 0, 1.3)

        zones, _cells = compute_stereo_depth_grid(depth, np.isfinite(depth))

        self.assertAlmostEqual(zones["left"].depth_m, 1.3)
        self.assertIsNone(zones["center"].depth_m)
        self.assertIsNone(zones["right"].depth_m)
        self.assertEqual(pattern_from_depth(zones["right"].depth_m), PATTERN_OFF)

    def test_latest_vpi_result_has_nine_medians_and_no_false_continuous_alert(self):
        result_dir = (
            Path(__file__).resolve().parents[2]
            / "1.1_depth_visualization"
            / "results"
            / "vpi_auto_2026-09-04_15-27-41_727249"
        )
        depth_path = result_dir / "best_depth_metres_float32.npy"
        valid_path = result_dir / "best_valid_mask.npy"
        if not depth_path.is_file() or not valid_path.is_file():
            self.skipTest("latest saved VPI result is not present")
        depth = np.load(depth_path)
        valid = np.load(valid_path).astype(bool)

        zones, cells = compute_stereo_depth_grid(depth, valid)

        self.assertEqual(len(cells), 9)
        for result in zones.values():
            if result.depth_m is not None:
                self.assertGreaterEqual(result.depth_m, 0.5)
                self.assertNotEqual(result.tone, "red")

    def test_empty_helpers_have_three_zones_and_nine_cells(self):
        zones = empty_stereo_zones((HEIGHT, WIDTH), "stale")
        cells = empty_depth_grid((HEIGHT, WIDTH))
        self.assertEqual(len(zones), 3)
        self.assertEqual(len(cells), 9)
        self.assertTrue(all(result.depth_m is None for result in zones.values()))
        self.assertTrue(all(result.depth_m is None for result in cells.values()))

    def test_grid_api_cannot_accept_yolo_results(self):
        parameters = tuple(inspect.signature(compute_stereo_depth_grid).parameters)
        self.assertEqual(parameters, ("depth_map", "valid_depth_mask"))
        detection = DetectionResult(0, "person", 0.9, (0, 0, 5, 5), 0.2)
        with self.assertRaises((AttributeError, ValueError)):
            compute_stereo_depth_grid(detection, detection)

class WinnerSwitchTrackerTests(unittest.TestCase):
    image_shape = (HEIGHT, WIDTH)

    @staticmethod
    def uniform_cells(depth_m):
        return cells_for_values(
            {(row, column): depth_m for row in range(3) for column in range(3)}
        )

    def test_first_valid_result_is_immediate(self):
        tracker = WinnerSwitchTracker()
        zones, _cells = tracker.update(self.uniform_cells(1.0), self.image_shape)
        for result in zones.values():
            self.assertAlmostEqual(result.depth_m, 1.0, places=5)

    def test_same_cell_depth_change_uses_current_frame_immediately(self):
        tracker = WinnerSwitchTracker()
        tracker.update(self.uniform_cells(1.0), self.image_shape)
        zones, _cells = tracker.update(self.uniform_cells(0.7), self.image_shape)
        for result in zones.values():
            self.assertAlmostEqual(result.depth_m, 0.7, places=5)

    def test_new_nearest_cell_must_win_twice(self):
        tracker = WinnerSwitchTracker()
        initial = {(0, 0): 1.0, (1, 0): 2.0}
        zones, _cells = tracker.update(cells_for_values(initial), self.image_shape)
        self.assertEqual(zones["left"].source_cell, "LT")

        changed = {(0, 0): 1.0, (1, 0): 0.7}
        zones, _cells = tracker.update(cells_for_values(changed), self.image_shape)
        self.assertEqual(zones["left"].source_cell, "LT")
        zones, _cells = tracker.update(cells_for_values(changed), self.image_shape)
        self.assertEqual(zones["left"].source_cell, "LM")

    def test_invalid_winner_does_not_reuse_old_depth(self):
        tracker = WinnerSwitchTracker()
        tracker.update(
            cells_for_values({(0, 0): 1.0, (1, 0): 2.0}),
            self.image_shape,
        )
        replacement = cells_for_values({(1, 0): 0.8})
        first, _cells = tracker.update(replacement, self.image_shape)
        self.assertIsNone(first["left"].depth_m)
        second, _cells = tracker.update(replacement, self.image_shape)
        self.assertAlmostEqual(second["left"].depth_m, 0.8)
        self.assertEqual(second["left"].source_cell, "LM")

    def test_missing_current_depth_is_invalid_immediately(self):
        tracker = WinnerSwitchTracker()
        tracker.update(self.uniform_cells(1.0), self.image_shape)
        blank_cells = cells_for_values({})
        zones, _cells = tracker.update(blank_cells, self.image_shape)
        self.assertTrue(all(result.depth_m is None for result in zones.values()))

    def test_reset_allows_a_new_first_winner_immediately(self):
        tracker = WinnerSwitchTracker()
        tracker.update(self.uniform_cells(0.4), self.image_shape)
        tracker.reset()
        zones, _cells = tracker.update(
            self.uniform_cells(1.2), self.image_shape
        )
        for result in zones.values():
            self.assertAlmostEqual(result.depth_m, 1.2, places=5)


class HapticPolicyTests(unittest.TestCase):
    def test_exact_threshold_boundaries(self):
        cases = (
            (None, PATTERN_OFF),
            (2.000001, PATTERN_OFF),
            (2.0, PATTERN_SLOW),
            (1.5, PATTERN_SLOW),
            (1.499999, PATTERN_MEDIUM),
            (0.5, PATTERN_MEDIUM),
            (0.499999, PATTERN_CONTINUOUS),
        )
        for depth, expected in cases:
            with self.subTest(depth=depth):
                self.assertEqual(pattern_from_depth(depth), expected)

    def test_exact_envelopes(self):
        self.assertEqual(
            (PATTERN_SLOW.level, PATTERN_SLOW.on_time, PATTERN_SLOW.off_time),
            (0x30, 0.50, 0.50),
        )
        self.assertEqual(
            (PATTERN_MEDIUM.level, PATTERN_MEDIUM.on_time, PATTERN_MEDIUM.off_time),
            (0x60, 0.08, 0.12),
        )
        self.assertEqual(
            (
                PATTERN_CONTINUOUS.level,
                PATTERN_CONTINUOUS.on_time,
                PATTERN_CONTINUOUS.off_time,
            ),
            (0x7F, 0.0, 0.0),
        )

    def test_hysteresis_prevents_threshold_chatter(self):
        self.assertEqual(
            pattern_from_depth_hysteresis(0.55, PATTERN_CONTINUOUS),
            PATTERN_CONTINUOUS,
        )
        self.assertEqual(
            pattern_from_depth_hysteresis(0.49, PATTERN_MEDIUM), PATTERN_MEDIUM
        )
        self.assertEqual(
            pattern_from_depth_hysteresis(1.49, PATTERN_SLOW), PATTERN_SLOW
        )
        self.assertEqual(
            pattern_from_depth_hysteresis(1.95, PATTERN_OFF), PATTERN_OFF
        )

    def test_hysteresis_changes_after_margin_and_invalid_stops_immediately(self):
        self.assertEqual(
            pattern_from_depth_hysteresis(0.40, PATTERN_MEDIUM),
            PATTERN_CONTINUOUS,
        )
        self.assertEqual(
            pattern_from_depth_hysteresis(0.60, PATTERN_CONTINUOUS),
            PATTERN_MEDIUM,
        )
        self.assertEqual(
            pattern_from_depth_hysteresis(1.40, PATTERN_SLOW), PATTERN_MEDIUM
        )
        self.assertEqual(
            pattern_from_depth_hysteresis(1.90, PATTERN_OFF), PATTERN_SLOW
        )
        self.assertEqual(
            pattern_from_depth_hysteresis(None, PATTERN_CONTINUOUS), PATTERN_OFF
        )


if __name__ == "__main__":
    unittest.main()
