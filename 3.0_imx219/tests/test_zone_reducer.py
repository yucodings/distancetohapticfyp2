import unittest
from unittest.mock import patch

import stereo_core as core
from haptic_policy import PATTERN_FAST, PATTERN_URGENT
from stream_worker import StreamWorker
from zone_reducer import reduce_eighteen_to_three


def measurements_for(values, shape=(720, 1280), valid_fraction=0.5):
    measurements = []
    for (name, box), value in zip(core.eighteen_zones(shape), values):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        samples = int(area * valid_fraction)
        measurements.append(
            core.DepthMeasurement(
                name=name,
                distance_m=value,
                box=box,
                sample_count=samples,
                valid_percentage=100.0 * samples / area,
            )
        )
    return tuple(measurements)


class ZoneReducerTests(unittest.TestCase):
    def test_nearest_of_six_valid_medians_is_selected_per_output_column(self):
        eighteen = measurements_for(
            (
                1.8, 1.6, 2.4, 1.0, 0.9, 1.2,
                0.4, 1.1, 2.2, 1.7, 1.5, 0.3,
                0.9, 0.7, 1.9, 1.4, 0.8, 1.0,
            )
        )

        three = reduce_eighteen_to_three(eighteen)

        self.assertEqual([item.name for item in three], ["Left", "Centre", "Right"])
        self.assertEqual(
            tuple(item.distance_m for item in three), (0.4, 1.0, 0.3)
        )

    def test_invalid_members_are_ignored_and_all_invalid_returns_none(self):
        values = [None] * 18
        values[4] = 1.1
        values[13] = 0.7

        three = reduce_eighteen_to_three(measurements_for(values))

        self.assertEqual(
            tuple(item.distance_m for item in three), (0.7, None, 1.1)
        )

    def test_six_boxes_are_merged_into_one_display_column(self):
        three = reduce_eighteen_to_three(measurements_for((1.0,) * 18))

        self.assertEqual(
            tuple(item.box for item in three),
            (
                (0, 72, 426, 648),
                (426, 72, 853, 648),
                (853, 72, 1280, 648),
            ),
        )

    def test_combined_valid_percentage_uses_all_six_cell_areas(self):
        eighteen = list(measurements_for((1.0,) * 18, valid_fraction=0.0))
        first = eighteen[0]
        first_area = (first.box[2] - first.box[0]) * (
            first.box[3] - first.box[1]
        )
        eighteen[0] = core.DepthMeasurement(
            first.name, first.distance_m, first.box, first_area, 100.0
        )
        left_indices = (0, 1, 6, 7, 12, 13)
        combined_area = sum(
            (eighteen[index].box[2] - eighteen[index].box[0])
            * (eighteen[index].box[3] - eighteen[index].box[1])
            for index in left_indices
        )

        left = reduce_eighteen_to_three(eighteen)[0]

        self.assertEqual(left.sample_count, first_area)
        self.assertAlmostEqual(
            left.valid_percentage, 100.0 * first_area / combined_area
        )

    def test_missing_or_misaligned_backend_cells_are_rejected(self):
        eighteen = measurements_for((1.0,) * 18)
        with self.assertRaisesRegex(ValueError, "exactly eighteen"):
            reduce_eighteen_to_three(eighteen[:-1])

        broken = list(eighteen)
        lower_two = broken[13]
        broken[13] = core.DepthMeasurement(
            lower_two.name,
            lower_two.distance_m,
            (
                lower_two.box[0] + 1,
                lower_two.box[1],
                lower_two.box[2],
                lower_two.box[3],
            ),
            lower_two.sample_count,
            lower_two.valid_percentage,
        )
        with self.assertRaisesRegex(ValueError, "contiguous 2x3 block"):
            reduce_eighteen_to_three(broken)

    def test_only_three_reduced_values_reach_haptic_policy(self):
        values = [2.5] * 18
        values[6] = 1.0
        values[8] = 0.9
        values[10] = 0.3
        three = reduce_eighteen_to_three(measurements_for(values))

        patterns = StreamWorker.patterns_for_measurements(three)

        self.assertEqual(patterns["left"], PATTERN_FAST)
        self.assertEqual(patterns["center"], PATTERN_FAST)
        self.assertEqual(patterns["right"], PATTERN_URGENT)
        self.assertEqual(len(patterns), 3)

    def test_display_renderer_receives_only_three_column_measurements(self):
        three = reduce_eighteen_to_three(measurements_for((1.0,) * 18))
        image = core.np.zeros((720, 1280, 3), dtype=core.np.uint8)

        with patch.object(core, "draw_measurements") as draw:
            preview = StreamWorker._preview(image, three, "RECTIFIED LEFT")

        self.assertEqual(len(draw.call_args.args[1]), 3)
        self.assertEqual(
            [item.name for item in draw.call_args.args[1]],
            ["Left", "Centre", "Right"],
        )
        self.assertEqual(
            preview.shape,
            (core.DISPLAY_PANEL_HEIGHT, core.DISPLAY_PANEL_WIDTH, 3),
        )

    def test_preview_footer_uses_thirty_percent_black_overlay(self):
        image = core.np.full((720, 1280, 3), 100, dtype=core.np.uint8)

        with patch.object(core, "draw_measurements"):
            preview = StreamWorker._preview(image, (), "")

        self.assertTrue(
            core.np.all(preview[core.DISPLAY_PANEL_HEIGHT - 35, -1] == 100)
        )
        self.assertTrue(
            core.np.all(preview[core.DISPLAY_PANEL_HEIGHT - 10, -1] == 70)
        )


if __name__ == "__main__":
    unittest.main()
