"""Lightweight UI-only persistence for intermittent YOLO detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from data_models import Detection


@dataclass
class _Track:
    detection: Detection
    last_seen_at: float
    misses: int = 0


def bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    """Return intersection-over-union for two corner-format boxes."""
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection_width = max(0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    left_area = max(0, lx2 - lx1) * max(0, ly2 - ly1)
    right_area = max(0, rx2 - rx1) * max(0, ry2 - ry1)
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else float(intersection / union)


class DetectionStabilizer:
    """Persist same-class boxes briefly without running a heavy tracker.

    Results are updated only for completed detector submissions. A missing
    detection increments its track's miss count, while a skipped GPU
    submission leaves the track untouched until its hard time limit expires.
    """

    def __init__(
        self,
        *,
        max_misses: int,
        hold_seconds: float,
        match_iou: float,
        smoothing_alpha: float,
    ) -> None:
        if max_misses < 1:
            raise ValueError("max_misses must be at least 1")
        if hold_seconds <= 0:
            raise ValueError("hold_seconds must be positive")
        if not 0.0 <= match_iou <= 1.0:
            raise ValueError("match_iou must be between 0 and 1")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        self.max_misses = max_misses
        self.hold_seconds = hold_seconds
        self.match_iou = match_iou
        self.smoothing_alpha = smoothing_alpha
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        self._tracks.clear()

    def _smooth(self, previous: Detection, current: Detection) -> Detection:
        alpha = self.smoothing_alpha
        bbox = tuple(
            int(round(old_value * (1.0 - alpha) + new_value * alpha))
            for old_value, new_value in zip(previous.bbox, current.bbox)
        )
        return Detection(
            class_id=current.class_id,
            label=current.label,
            confidence=current.confidence,
            bbox=bbox,
        )

    def update(
        self,
        detections: Iterable[Detection],
        completed_at: float,
    ) -> tuple[Detection, ...]:
        """Consume one completed inference result and return visible boxes."""
        current = tuple(detections)
        unmatched_tracks = set(range(len(self._tracks)))
        matches: dict[int, int] = {}

        # Match confident detections first. Each existing track can be used
        # once, preventing two nearby same-class boxes from collapsing.
        detection_order = sorted(
            range(len(current)),
            key=lambda index: current[index].confidence,
            reverse=True,
        )
        for detection_index in detection_order:
            detection = current[detection_index]
            candidates = (
                (bbox_iou(self._tracks[index].detection.bbox, detection.bbox), index)
                for index in unmatched_tracks
                if self._tracks[index].detection.class_id == detection.class_id
            )
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_index >= 0 and best_iou >= self.match_iou:
                matches[detection_index] = best_index
                unmatched_tracks.remove(best_index)

        for detection_index, track_index in matches.items():
            track = self._tracks[track_index]
            track.detection = self._smooth(track.detection, current[detection_index])
            track.last_seen_at = completed_at
            track.misses = 0

        for track_index in unmatched_tracks:
            self._tracks[track_index].misses += 1

        for detection_index, detection in enumerate(current):
            if detection_index not in matches:
                self._tracks.append(_Track(detection, completed_at))

        return self.visible(completed_at)

    def visible(self, now: float) -> tuple[Detection, ...]:
        """Return current boxes after enforcing miss and hard-age limits."""
        self._tracks = [
            track
            for track in self._tracks
            if track.misses < self.max_misses
            and now - track.last_seen_at < self.hold_seconds
        ]
        return tuple(track.detection for track in self._tracks)
