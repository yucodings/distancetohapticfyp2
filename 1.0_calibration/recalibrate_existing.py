#!/usr/bin/env python3
"""Recalibrate from an existing 1.0 raw-image session, without cameras."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mycalibration as calibration_app
from calibration_filter import QualityThresholds, filter_calibration_pairs

cv2 = calibration_app.cv2


def newest_raw_session() -> Path:
    candidates = [
        path
        for path in calibration_app.IMAGES_ROOT.iterdir()
        if path.is_dir()
        and any((path / "left").glob("*.png"))
        and any((path / "right").glob("*.png"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No raw stereo session exists under {calibration_app.IMAGES_ROOT}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def session_folders(root: Path) -> calibration_app.SessionFolders:
    folders = calibration_app.SessionFolders(
        root=root,
        left_raw=root / "left",
        right_raw=root / "right",
        corners_left=root / "detected_corners" / "left",
        corners_right=root / "detected_corners" / "right",
        rejected_left=root / "rejected" / "left",
        rejected_right=root / "rejected" / "right",
        quality_rejected_left=root / "quality_rejected" / "left",
        quality_rejected_right=root / "quality_rejected" / "right",
        rectified=root / "rectified_validation",
    )
    for directory in (
        folders.corners_left,
        folders.corners_right,
        folders.rejected_left,
        folders.rejected_right,
        folders.quality_rejected_left,
        folders.quality_rejected_right,
        folders.rectified,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return folders


def output_folders(source: Path) -> calibration_app.SessionFolders:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f_recalibrated")
    root = calibration_app.IMAGES_ROOT / stamp
    folders = calibration_app.SessionFolders(
        root=root,
        left_raw=source / "left",
        right_raw=source / "right",
        corners_left=source / "detected_corners" / "left",
        corners_right=source / "detected_corners" / "right",
        rejected_left=source / "rejected" / "left",
        rejected_right=source / "rejected" / "right",
        quality_rejected_left=root / "quality_rejected" / "left",
        quality_rejected_right=root / "quality_rejected" / "right",
        rectified=root / "rectified_validation",
    )
    for directory in (
        folders.quality_rejected_left,
        folders.quality_rejected_right,
        folders.rectified,
    ):
        directory.mkdir(parents=True, exist_ok=False)
    return folders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        type=Path,
        help="existing session directory (default: newest session with raw pairs)",
    )
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--min-board-area-percent", type=float, default=1.2)
    parser.add_argument("--max-reprojection-rms-px", type=float, default=1.5)
    parser.add_argument("--max-epipolar-p95-px", type=float, default=3.0)
    parser.add_argument(
        "--max-rectified-vertical-p95-px", type=float, default=2.5
    )
    parser.add_argument("--max-rounds", type=int, default=4)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.min_pairs < calibration_app.MIN_VALID_PAIRS:
        raise ValueError(
            f"--min-pairs cannot be below {calibration_app.MIN_VALID_PAIRS}"
        )
    for name in (
        "min_board_area_percent",
        "max_reprojection_rms_px",
        "max_epipolar_p95_px",
        "max_rectified_vertical_p95_px",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be at least 1")


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    source = (
        args.session.expanduser().resolve()
        if args.session is not None
        else newest_raw_session().resolve()
    )
    if not (source / "left").is_dir() or not (source / "right").is_dir():
        raise FileNotFoundError(
            f"Session must contain left/ and right/ image folders: {source}"
        )

    print(f"Source session: {source}", flush=True)
    source_folders = session_folders(source)
    (
        object_points,
        left_points,
        right_points,
        image_size,
        accepted_paths,
        detection_records,
    ) = calibration_app.load_calibration_points(source_folders)
    samples = calibration_app.make_pair_samples(
        object_points, left_points, right_points, accepted_paths
    )
    thresholds = QualityThresholds(
        min_board_area_percent=args.min_board_area_percent,
        max_reprojection_rms_px=args.max_reprojection_rms_px,
        max_epipolar_p95_px=args.max_epipolar_p95_px,
        max_rectified_vertical_p95_px=(
            args.max_rectified_vertical_p95_px
        ),
        max_rounds=args.max_rounds,
    )
    print(
        f"Filtering {len(samples)} detected pairs; this can take several minutes...",
        flush=True,
    )
    filtered = filter_calibration_pairs(
        samples, image_size, args.min_pairs, thresholds
    )

    destination = output_folders(source)
    calibration_app.save_quality_rejection_evidence(
        destination, filtered.rejected, samples
    )
    calibration_app.apply_quality_rejections_to_records(
        detection_records, filtered.rejected
    )
    final_objects = [sample.object_points for sample in filtered.accepted]
    final_left = [sample.left_corners for sample in filtered.accepted]
    final_right = [sample.right_corners for sample in filtered.accepted]
    final_paths = [
        (sample.left_path, sample.right_path) for sample in filtered.accepted
    ]
    preview_path, rectified_metrics = calibration_app.save_rectified_preview(
        destination,
        final_paths,
        final_left,
        final_right,
        filtered.calibration,
        filtered.rectification,
    )
    npz_path = calibration_app.save_npz(
        destination.root,
        image_size,
        len(filtered.accepted),
        filtered.calibration,
        filtered.rectification,
    )
    report = calibration_app.build_calibration_report(
        destination,
        final_objects,
        final_left,
        final_right,
        final_paths,
        detection_records,
        filtered.calibration,
        rectified_metrics,
    )
    report["recalibration"] = {
        "source_session": str(source),
        "raw_images_modified": False,
        "thresholds": asdict(thresholds),
        "history": list(filtered.history),
        "rejected_pairs": list(filtered.rejected),
    }
    report_path = calibration_app.save_calibration_report(
        destination.root, report
    )

    print("\nRecalibration completed successfully.")
    print(f"Accepted:          {len(filtered.accepted)}")
    print(f"Auto-rejected:     {len(filtered.rejected)}")
    print(f"Left RMS:          {filtered.calibration['left_rms']:.6f} px")
    print(f"Right RMS:         {filtered.calibration['right_rms']:.6f} px")
    print(f"Stereo RMS:        {filtered.calibration['stereo_rms']:.6f} px")
    print(f"Baseline:          {filtered.calibration['baseline_m']:.6f} m")
    for rejection in filtered.rejected:
        print(f"Rejected {rejection['filename']}: {rejection['reason']}")
    print(f"Output session:    {destination.root}")
    print(f"Quality report:    {report_path}")
    print(f"Rectified preview: {preview_path}")
    print(f"Calibration NPZ:   {npz_path}")
    return destination.root


def main() -> int:
    try:
        run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
