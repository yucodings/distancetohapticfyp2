import unittest
from unittest.mock import patch

import stereo_core as core
from haptic_policy import PATTERN_FAST, PATTERN_URGENT
from stream_worker import StreamWorker
from zone_reducer import reduce_nine_to_three


def measurements_for(values, shape=(720, 1280), valid_fraction=0.5):
    measurements = []
    for (name, box), value in zip(core.nine_zones(shape), values):
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
    def test_nearest_valid_median_is_selected_in_each_column(self):
        nine = measurements_for(
            (1.8, 1.0, 0.3, 0.4, 2.2, 0.9, 0.9, 1.7, 0.8)
        )

        three = reduce_nine_to_three(nine)

        self.assertEqual([item.name for item in three], ["Left", "Centre", "Right"])
        self.assertEqual(
            tuple(item.distance_m for item in three), (0.4, 1.0, 0.3)
        )

    def test_invalid_members_are_ignored_and_all_invalid_returns_none(self):
        nine = measurements_for(
            (None, None, 1.1, None, None, None, 0.7, None, None)
        )

        three = reduce_nine_to_three(nine)

        self.assertEqual(
            tuple(item.distance_m for item in three), (0.7, None, 1.1)
        )

    def test_three_boxes_are_merged_into_one_display_column(self):
        three = reduce_nine_to_three(measurements_for((1.0,) * 9))

        self.assertEqual(
            tuple(item.box for item in three),
            (
                (0, 72, 426, 648),
                (426, 72, 853, 648),
                (853, 72, 1280, 648),
            ),
        )

    def test_combined_valid_percentage_uses_all_three_zone_areas(self):
        nine = list(measurements_for((1.0,) * 9, valid_fraction=0.0))
        upper, middle, lower = nine[0], nine[3], nine[6]
        upper_area = (upper.box[2] - upper.box[0]) * (upper.box[3] - upper.box[1])
        middle_area = (middle.box[2] - middle.box[0]) * (
            middle.box[3] - middle.box[1]
        )
        lower_area = (lower.box[2] - lower.box[0]) * (lower.box[3] - lower.box[1])
        nine[0] = core.DepthMeasurement(
            upper.name, upper.distance_m, upper.box, upper_area, 100.0
        )

        left = reduce_nine_to_three(nine)[0]

        self.assertEqual(left.sample_count, upper_area)
        self.assertAlmostEqual(
            left.valid_percentage,
            100.0 * upper_area / (upper_area + middle_area + lower_area),
        )

    def test_missing_or_misaligned_backend_zones_are_rejected(self):
        nine = measurements_for((1.0,) * 9)
        with self.assertRaisesRegex(ValueError, "exactly nine"):
            reduce_nine_to_three(nine[:-1])

        broken = list(nine)
        lower = broken[6]
        broken[6] = core.DepthMeasurement(
            lower.name,
            lower.distance_m,
            (lower.box[0] + 1, lower.box[1], lower.box[2], lower.box[3]),
            lower.sample_count,
            lower.valid_percentage,
        )
        with self.assertRaisesRegex(ValueError, "not one contiguous column"):
            reduce_nine_to_three(broken)

    def test_only_three_reduced_values_reach_haptic_policy(self):
        nine = measurements_for(
            (1.8, 1.0, 0.3, 1.4, 2.2, 0.9, 0.9, 1.7, 0.8)
        )
        three = reduce_nine_to_three(nine)

        patterns = StreamWorker.patterns_for_measurements(three)

        self.assertEqual(patterns["left"], PATTERN_FAST)
        self.assertEqual(patterns["center"], PATTERN_FAST)
        self.assertEqual(patterns["right"], PATTERN_URGENT)
        self.assertEqual(len(patterns), 3)

    def test_display_renderer_receives_only_three_column_measurements(self):
        nine = measurements_for(
            (1.8, 1.0, 0.3, 1.4, 2.2, 0.9, 0.9, 1.7, 0.8)
        )
        three = reduce_nine_to_three(nine)
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
