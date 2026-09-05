"""Collapse six backend medians into three nearest column medians."""

from __future__ import annotations

from typing import Iterable

from stereo_core import DepthMeasurement


COLUMN_PAIRS = (
    ("Left", "Upper Left", "Lower Left"),
    ("Centre", "Upper Centre", "Lower Centre"),
    ("Right", "Upper Right", "Lower Right"),
)


def _zone_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid zone box: {box}")
    return (x2 - x1) * (y2 - y1)


def _merge_column(
    output_name: str,
    upper: DepthMeasurement,
    lower: DepthMeasurement,
) -> DepthMeasurement:
    ux1, uy1, ux2, uy2 = upper.box
    lx1, ly1, lx2, ly2 = lower.box
    if (ux1, ux2) != (lx1, lx2) or uy2 != ly1:
        raise ValueError(
            f"{output_name} upper/lower zones are not one contiguous column: "
            f"{upper.box} and {lower.box}"
        )

    distances = (
        value
        for value in (upper.distance_m, lower.distance_m)
        if value is not None
    )
    distance_m = min(distances, default=None)
    sample_count = upper.sample_count + lower.sample_count
    combined_area = _zone_area(upper.box) + _zone_area(lower.box)
    if not 0 <= sample_count <= combined_area:
        raise ValueError(
            f"{output_name} valid sample count {sample_count} exceeds "
            f"combined zone area {combined_area}"
        )
    return DepthMeasurement(
        name=output_name,
        distance_m=distance_m,
        box=(ux1, uy1, ux2, ly2),
        sample_count=sample_count,
        valid_percentage=100.0 * sample_count / combined_area,
    )


def reduce_six_to_three(
    measurements: Iterable[DepthMeasurement],
) -> tuple[DepthMeasurement, DepthMeasurement, DepthMeasurement]:
    """Return Left/Centre/Right using the nearest valid median in each pair."""
    items = tuple(measurements)
    by_name = {item.name: item for item in items}
    expected = {
        source_name
        for _output_name, upper_name, lower_name in COLUMN_PAIRS
        for source_name in (upper_name, lower_name)
    }
    if len(items) != len(expected) or set(by_name) != expected:
        missing = sorted(expected - set(by_name))
        extra = sorted(set(by_name) - expected)
        raise ValueError(
            "Expected exactly six named backend measurements; "
            f"missing={missing}, extra={extra}, received={len(items)}"
        )

    return tuple(
        _merge_column(output_name, by_name[upper_name], by_name[lower_name])
        for output_name, upper_name, lower_name in COLUMN_PAIRS
    )

