"""Stable 3x3 median-grid distances from confidence-filtered stereo depth."""

from __future__ import annotations

from typing import Dict, Optional

from camera_backend import np
from config import (
    DEPTH_GRID_COLUMNS,
    DEPTH_GRID_ROWS,
    DISPLAY_LABELS,
    GRID_CELL_MIN_VALID_PIXELS,
    GRID_CELL_ROI_HEIGHT_FRACTION,
    GRID_CELL_ROI_WIDTH_FRACTION,
    GRID_WINNER_CONFIRM_FRAMES,
    MAX_VALID_DEPTH,
    MIN_VALID_DEPTH,
    ZONE_BORDER_MARGIN_X,
    ZONE_BORDER_MARGIN_Y,
    ZONE_KEYS,
)
from data_models import GridCellResult, ZoneResult
from hazard_policy import alert_message, tone_from_depth


ROW_NAMES = ("top", "middle", "bottom")
ROW_LABELS = ("T", "M", "B")
COLUMN_LABELS = ("L", "C", "R")


def _validate_grid_configuration() -> None:
    if DEPTH_GRID_ROWS != 3 or DEPTH_GRID_COLUMNS != 3:
        raise ValueError("The actuator depth grid must remain 3x3")
    if GRID_CELL_MIN_VALID_PIXELS < 1:
        raise ValueError("GRID_CELL_MIN_VALID_PIXELS must be positive")
    for name, fraction in (
        ("GRID_CELL_ROI_WIDTH_FRACTION", GRID_CELL_ROI_WIDTH_FRACTION),
        ("GRID_CELL_ROI_HEIGHT_FRACTION", GRID_CELL_ROI_HEIGHT_FRACTION),
    ):
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"{name} must be greater than 0 and at most 1")
    if GRID_WINNER_CONFIRM_FRAMES < 1:
        raise ValueError("GRID_WINNER_CONFIRM_FRAMES must be positive")


def _cell_rect(
    image_shape: tuple[int, int], row: int, column: int
) -> tuple[int, int, int, int]:
    height, width = image_shape
    return (
        width * column // DEPTH_GRID_COLUMNS,
        height * row // DEPTH_GRID_ROWS,
        width * (column + 1) // DEPTH_GRID_COLUMNS,
        height * (row + 1) // DEPTH_GRID_ROWS,
    )


def _centered_sample_rect(
    rect: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Use the same 20%-of-frame idea as test_depth_imx219 in each cell."""
    x1, y1, x2, y2 = rect
    cell_width = x2 - x1
    cell_height = y2 - y1
    sample_width = max(1, int(round(cell_width * GRID_CELL_ROI_WIDTH_FRACTION)))
    sample_height = max(1, int(round(cell_height * GRID_CELL_ROI_HEIGHT_FRACTION)))
    sample_x1 = x1 + (cell_width - sample_width) // 2
    sample_y1 = y1 + (cell_height - sample_height) // 2
    return (
        sample_x1,
        sample_y1,
        sample_x1 + sample_width,
        sample_y1 + sample_height,
    )


def _empty_cell(
    image_shape: tuple[int, int], row: int, column: int
) -> GridCellResult:
    zone_name = ZONE_KEYS[column]
    rect = _cell_rect(image_shape, row, column)
    sample_rect = _centered_sample_rect(rect)
    sx1, sy1, sx2, sy2 = sample_rect
    return GridCellResult(
        key=f"{zone_name}_{ROW_NAMES[row]}",
        display_name=f"{COLUMN_LABELS[column]}{ROW_LABELS[row]}",
        row=row,
        column=column,
        rect=rect,
        sample_rect=sample_rect,
        total_count=(sx2 - sx1) * (sy2 - sy1),
    )


def _measure_cell(
    depth_map: np.ndarray,
    valid: np.ndarray,
    row: int,
    column: int,
) -> GridCellResult:
    result = _empty_cell(depth_map.shape, row, column)
    x1, y1, x2, y2 = result.sample_rect
    sample_valid = valid[y1:y2, x1:x2]
    result.valid_count = int(np.count_nonzero(sample_valid))
    result.valid_percentage = (
        100.0 * result.valid_count / max(1, result.total_count)
    )
    if result.valid_count < GRID_CELL_MIN_VALID_PIXELS:
        return result

    sample_depth = depth_map[y1:y2, x1:x2]
    values = sample_depth[sample_valid]
    result.depth_m = float(np.median(values))
    coordinates = np.argwhere(sample_valid)
    representative_index = int(np.argmin(np.abs(values - result.depth_m)))
    local_y, local_x = coordinates[representative_index]
    result.representative_point = (x1 + int(local_x), y1 + int(local_y))
    return result


def _zone_from_cell(
    cell: Optional[GridCellResult],
    zone_name: str,
    image_shape: tuple[int, int],
    invalid_reason: str,
) -> ZoneResult:
    height, width = image_shape
    column = ZONE_KEYS.index(zone_name)
    result = ZoneResult(
        key=zone_name,
        display_name=DISPLAY_LABELS[zone_name],
        rect=(
            width * column // DEPTH_GRID_COLUMNS,
            0,
            width * (column + 1) // DEPTH_GRID_COLUMNS,
            height,
        ),
    )
    if cell is None or cell.depth_m is None:
        result.alert_message = f"[{result.display_name}] {invalid_reason}"
        return result
    result.depth_m = cell.depth_m
    result.nearest_point = cell.representative_point
    result.source_cell = cell.display_name
    result.alert_message = alert_message(result.depth_m, result.display_name)
    result.tone = tone_from_depth(result.depth_m)
    return result


def compute_stereo_depth_grid(
    depth_map: np.ndarray,
    valid_depth_mask: np.ndarray,
) -> tuple[Dict[str, ZoneResult], Dict[str, GridCellResult]]:
    """Measure centered cell medians and return raw nearest-column results."""
    _validate_grid_configuration()
    if depth_map.ndim != 2 or valid_depth_mask.shape != depth_map.shape:
        raise ValueError("Depth map and valid mask must be matching 2-D arrays")
    height, width = depth_map.shape
    if height < DEPTH_GRID_ROWS or width < DEPTH_GRID_COLUMNS:
        raise ValueError("Depth map is too small for a 3x3 grid")

    finite_valid = (
        valid_depth_mask
        & np.isfinite(depth_map)
        & (depth_map > MIN_VALID_DEPTH)
        & (depth_map < MAX_VALID_DEPTH)
    ).copy()
    margin_y = min(ZONE_BORDER_MARGIN_Y, max(0, (height - 3) // 2))
    margin_x = min(ZONE_BORDER_MARGIN_X, max(0, (width - 3) // 2))
    if margin_y:
        finite_valid[:margin_y, :] = False
        finite_valid[-margin_y:, :] = False
    if margin_x:
        finite_valid[:, :margin_x] = False
        finite_valid[:, -margin_x:] = False

    cells: Dict[str, GridCellResult] = {}
    for row in range(DEPTH_GRID_ROWS):
        for column in range(DEPTH_GRID_COLUMNS):
            cell = _measure_cell(depth_map, finite_valid, row, column)
            cells[cell.key] = cell

    zones: Dict[str, ZoneResult] = {}
    for column, zone_name in enumerate(ZONE_KEYS):
        candidates = [
            cell
            for cell in cells.values()
            if cell.column == column and cell.depth_m is not None
        ]
        selected = (
            min(candidates, key=lambda cell: float(cell.depth_m))
            if candidates
            else None
        )
        if selected is not None:
            selected.selected = True
        zones[zone_name] = _zone_from_cell(
            selected, zone_name, depth_map.shape, "No valid grid cell"
        )
    return zones, cells


class WinnerSwitchTracker:
    """Use current-frame cell medians and confirm only winner changes."""

    def __init__(self) -> None:
        _validate_grid_configuration()
        self._winner: Dict[str, Optional[str]] = {
            zone_name: None for zone_name in ZONE_KEYS
        }
        self._pending_winner: Dict[str, Optional[str]] = {
            zone_name: None for zone_name in ZONE_KEYS
        }
        self._pending_count = {zone_name: 0 for zone_name in ZONE_KEYS}

    def reset(self) -> None:
        for zone_name in ZONE_KEYS:
            self._winner[zone_name] = None
            self._pending_winner[zone_name] = None
            self._pending_count[zone_name] = 0

    def _select_winner(
        self,
        zone_name: str,
        candidates: list[GridCellResult],
    ) -> Optional[GridCellResult]:
        if not candidates:
            self._winner[zone_name] = None
            self._pending_winner[zone_name] = None
            self._pending_count[zone_name] = 0
            return None

        nearest = min(candidates, key=lambda cell: float(cell.depth_m))
        current_key = self._winner[zone_name]
        candidate_by_key = {cell.key: cell for cell in candidates}
        if current_key is None:
            pending_key = self._pending_winner[zone_name]
            if pending_key is None:
                # There is no previous winner at startup, so the first valid
                # current-frame result can be used without artificial delay.
                self._winner[zone_name] = nearest.key
            else:
                if pending_key == nearest.key:
                    self._pending_count[zone_name] += 1
                else:
                    self._pending_winner[zone_name] = nearest.key
                    self._pending_count[zone_name] = 1
                if self._pending_count[zone_name] >= GRID_WINNER_CONFIRM_FRAMES:
                    self._winner[zone_name] = nearest.key
                    self._pending_winner[zone_name] = None
                    self._pending_count[zone_name] = 0
        elif current_key not in candidate_by_key:
            # Never output historical depth from a cell that is invalid now.
            self._winner[zone_name] = None
            if self._pending_winner[zone_name] == nearest.key:
                self._pending_count[zone_name] += 1
            else:
                self._pending_winner[zone_name] = nearest.key
                self._pending_count[zone_name] = 1
            if self._pending_count[zone_name] >= GRID_WINNER_CONFIRM_FRAMES:
                self._winner[zone_name] = nearest.key
        elif nearest.key == current_key:
            self._pending_winner[zone_name] = None
            self._pending_count[zone_name] = 0
        else:
            if self._pending_winner[zone_name] == nearest.key:
                self._pending_count[zone_name] += 1
            else:
                self._pending_winner[zone_name] = nearest.key
                self._pending_count[zone_name] = 1
            if self._pending_count[zone_name] >= GRID_WINNER_CONFIRM_FRAMES:
                self._winner[zone_name] = nearest.key
                self._pending_winner[zone_name] = None
                self._pending_count[zone_name] = 0

        selected_key = self._winner[zone_name]
        return candidate_by_key.get(selected_key) if selected_key else None

    def update(
        self, raw_cells: Dict[str, GridCellResult], image_shape: tuple[int, int]
    ) -> tuple[Dict[str, ZoneResult], Dict[str, GridCellResult]]:
        expected_keys = set(empty_depth_grid(image_shape))
        if set(raw_cells) != expected_keys:
            raise ValueError("Winner tracker requires all nine grid cells")

        # The cell objects belong only to this new depth result. Clear the raw
        # selection flags so the overlay highlights confirmed winners only.
        cells = raw_cells
        for cell in cells.values():
            cell.selected = False
        zones: Dict[str, ZoneResult] = {}
        for column, zone_name in enumerate(ZONE_KEYS):
            candidates = [
                cell
                for cell in cells.values()
                if cell.column == column and cell.depth_m is not None
            ]
            selected = self._select_winner(zone_name, candidates)
            if selected is not None:
                selected.selected = True
            zones[zone_name] = _zone_from_cell(
                selected,
                zone_name,
                image_shape,
                "No valid confirmed grid depth",
            )
        return zones, cells


def empty_stereo_zones(
    image_shape: tuple[int, int], reason: str = "Stereo depth unavailable"
) -> Dict[str, ZoneResult]:
    return {
        zone_name: _zone_from_cell(None, zone_name, image_shape, reason)
        for zone_name in ZONE_KEYS
    }


def empty_depth_grid(image_shape: tuple[int, int]) -> Dict[str, GridCellResult]:
    return {
        cell.key: cell
        for row in range(DEPTH_GRID_ROWS)
        for column in range(DEPTH_GRID_COLUMNS)
        for cell in (_empty_cell(image_shape, row, column),)
    }
