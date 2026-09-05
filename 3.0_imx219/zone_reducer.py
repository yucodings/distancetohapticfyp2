"""Collapse 18 backend medians into three nearest column medians."""

from __future__ import annotations

from typing import Iterable

from stereo_core import DepthMeasurement


COLUMN_GROUPS = (
    (
        "Left",
        ("Upper 1", "Upper 2", "Middle 1", "Middle 2", "Lower 1", "Lower 2"),
    ),
    (
        "Centre",
        ("Upper 3", "Upper 4", "Middle 3", "Middle 4", "Lower 3", "Lower 4"),
    ),
    (
        "Right",
        ("Upper 5", "Upper 6", "Middle 5", "Middle 6", "Lower 5", "Lower 6"),
    ),
)


def _zone_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid zone box: {box}")
    return (x2 - x1) * (y2 - y1)


def _merge_column(
    output_name: str,
    cells: tuple[DepthMeasurement, ...],
) -> DepthMeasurement:
    if len(cells) != 6:
        raise ValueError(f"{output_name} must contain exactly six backend cells")

    rows = (cells[0:2], cells[2:4], cells[4:6])
    for left_cell, right_cell in rows:
        lx1, ly1, lx2, ly2 = left_cell.box
        rx1, ry1, rx2, ry2 = right_cell.box
        if (ly1, ly2) != (ry1, ry2) or lx2 != rx1:
            raise ValueError(
                f"{output_name} backend cells are not a contiguous 2x3 block"
            )

    row_boxes = tuple(
        (left.box[0], left.box[1], right.box[2], right.box[3])
        for left, right in rows
    )
    upper_box, middle_box, lower_box = row_boxes
    if not (
        (upper_box[0], upper_box[2])
        == (middle_box[0], middle_box[2])
        == (lower_box[0], lower_box[2])
        and upper_box[3] == middle_box[1]
        and middle_box[3] == lower_box[1]
    ):
        raise ValueError(
            f"{output_name} backend cells are not a contiguous 2x3 block"
        )

    distance_m = min(
        (cell.distance_m for cell in cells if cell.distance_m is not None),
        default=None,
    )
    sample_count = sum(cell.sample_count for cell in cells)
    combined_area = sum(_zone_area(cell.box) for cell in cells)
    if not 0 <= sample_count <= combined_area:
        raise ValueError(
            f"{output_name} valid sample count {sample_count} exceeds "
            f"combined zone area {combined_area}"
        )

    return DepthMeasurement(
        name=output_name,
        distance_m=distance_m,
        box=(upper_box[0], upper_box[1], upper_box[2], lower_box[3]),
        sample_count=sample_count,
        valid_percentage=100.0 * sample_count / combined_area,
    )


def reduce_eighteen_to_three(
    measurements: Iterable[DepthMeasurement],
) -> tuple[DepthMeasurement, DepthMeasurement, DepthMeasurement]:
    """Return Left/Centre/Right from the nearest valid median in each 2x3 block."""
    items = tuple(measurements)
    by_name = {item.name: item for item in items}
    expected = {
        source_name
        for _output_name, source_names in COLUMN_GROUPS
        for source_name in source_names
    }
    if len(items) != len(expected) or set(by_name) != expected:
        missing = sorted(expected - set(by_name))
        extra = sorted(set(by_name) - expected)
        raise ValueError(
            "Expected exactly eighteen named backend measurements; "
            f"missing={missing}, extra={extra}, received={len(items)}"
        )

    return tuple(
        _merge_column(
            output_name,
            tuple(by_name[source_name] for source_name in source_names),
        )
        for output_name, source_names in COLUMN_GROUPS
    )
