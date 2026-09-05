"""Optional XYZ reconstruction, PLY export, and 3D diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class SampledPointCloud:
    points: np.ndarray
    colors_rgb: np.ndarray

    @property
    def count(self) -> int:
        return int(self.points.shape[0])


def disparity_to_xyz(
    disparity: np.ndarray,
    q_matrix: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject calibrated disparity and preserve only valid finite XYZ."""
    if disparity.ndim != 2:
        raise ValueError("Disparity must be a 2-D array")
    if q_matrix.shape != (4, 4):
        raise ValueError("Q matrix must be 4x4")
    if valid_mask.shape != disparity.shape:
        raise ValueError("Valid mask shape does not match disparity")
    xyz = cv2.reprojectImageTo3D(
        disparity.astype(np.float32, copy=False),
        q_matrix.astype(np.float64, copy=False),
        handleMissingValues=False,
        ddepth=cv2.CV_32F,
    )
    finite = valid_mask & np.all(np.isfinite(xyz), axis=2) & (xyz[:, :, 2] > 0)
    xyz[~finite] = np.nan
    return xyz, finite


def sample_point_cloud(
    xyz: np.ndarray,
    color_bgr: np.ndarray,
    valid_mask: np.ndarray,
    stride: int = 4,
    max_points: int = 100_000,
) -> SampledPointCloud:
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("XYZ must have shape HxWx3")
    if color_bgr.shape[:2] != xyz.shape[:2] or color_bgr.shape[2] != 3:
        raise ValueError("Color image shape does not match XYZ")
    if valid_mask.shape != xyz.shape[:2]:
        raise ValueError("Valid mask shape does not match XYZ")
    if stride < 1 or max_points < 1:
        raise ValueError("stride and max_points must be positive")

    sampled_xyz = xyz[::stride, ::stride]
    sampled_color = color_bgr[::stride, ::stride]
    sampled_valid = valid_mask[::stride, ::stride] & np.all(
        np.isfinite(sampled_xyz), axis=2
    )
    points = sampled_xyz[sampled_valid].astype(np.float32, copy=False)
    colors = sampled_color[sampled_valid][:, ::-1].astype(np.uint8, copy=False)
    if points.shape[0] > max_points:
        step = int(np.ceil(points.shape[0] / max_points))
        points = points[::step]
        colors = colors[::step]
    return SampledPointCloud(points.copy(), colors.copy())


def make_orthographic_view(
    cloud: SampledPointCloud,
    width: int = 960,
    height: int = 540,
    horizontal_limit_m: float = 2.0,
    vertical_limit_m: float = 1.5,
    min_depth_m: float = 0.1,
    max_depth_m: float = 3.0,
) -> np.ndarray:
    """Render top and front projections using OpenCV; no Open3D required."""
    if width < 320 or height < 240:
        raise ValueError("3D diagnostic view is too small")
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    half = width // 2
    cv2.line(canvas, (half, 0), (half, height), (80, 80, 80), 1)
    cv2.putText(
        canvas, "TOP: X / Z", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas, "FRONT: X / Y", (half + 14, 28), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2, cv2.LINE_AA
    )
    if cloud.count == 0:
        cv2.putText(
            canvas, "No valid 3D points", (width // 2 - 120, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA
        )
        return canvas

    points = cloud.points
    colors_bgr = cloud.colors_rgb[:, ::-1]
    within = (
        np.all(np.isfinite(points), axis=1)
        & (np.abs(points[:, 0]) <= horizontal_limit_m)
        & (points[:, 2] >= min_depth_m)
        & (points[:, 2] <= max_depth_m)
    )
    points = points[within]
    colors_bgr = colors_bgr[within]
    if points.size == 0:
        return canvas

    order = np.argsort(points[:, 2])[::-1]
    points = points[order]
    colors_bgr = colors_bgr[order]

    usable_height = height - 55
    top_x = np.clip(
        np.round((points[:, 0] / horizontal_limit_m + 1.0) * 0.5 * (half - 1)),
        0, half - 1
    ).astype(np.int32)
    top_y = np.clip(
        np.round(
            40
            + (points[:, 2] - min_depth_m)
            / (max_depth_m - min_depth_m)
            * (usable_height - 1)
        ),
        40, height - 16
    ).astype(np.int32)
    front_x = half + top_x
    front_y = np.clip(
        np.round(
            40
            + (points[:, 1] / vertical_limit_m + 1.0)
            * 0.5
            * (usable_height - 1)
        ),
        40, height - 16
    ).astype(np.int32)
    canvas[top_y, top_x] = colors_bgr
    canvas[front_y, front_x] = colors_bgr

    center_x = half // 2
    cv2.line(canvas, (center_x, 40), (center_x, height - 16), (80, 80, 80), 1)
    cv2.line(
        canvas,
        (half + center_x, 40),
        (half + center_x, height - 16),
        (80, 80, 80),
        1,
    )
    for distance in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        y = round(
            40
            + (distance - min_depth_m)
            / (max_depth_m - min_depth_m)
            * (usable_height - 1)
        )
        if 40 <= y < height:
            cv2.line(canvas, (0, y), (half - 1, y), (50, 50, 50), 1)
            cv2.putText(
                canvas, f"{distance:.1f}m", (4, min(height - 3, y + 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1,
                cv2.LINE_AA
            )
    cv2.putText(
        canvas, f"{cloud.count} sampled points", (14, height - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA
    )
    return canvas


def save_binary_ply(path: Path, cloud: SampledPointCloud) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.empty(
        cloud.count,
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    if cloud.count:
        vertices["x"], vertices["y"], vertices["z"] = cloud.points.T
        vertices["red"], vertices["green"], vertices["blue"] = cloud.colors_rgb.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {cloud.count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)
    return path


def open3d_available() -> bool:
    try:
        import open3d  # noqa: F401
    except ImportError:
        return False
    return True


def show_open3d_snapshot(
    cloud: SampledPointCloud,
    window_name: str = "IMX219 point cloud",
) -> None:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Open3D is not installed; use the built-in orthographic 3D view "
            "or open a saved PLY on another computer."
        ) from error
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        cloud.points.astype(np.float64, copy=False)
    )
    point_cloud.colors = o3d.utility.Vector3dVector(
        cloud.colors_rgb.astype(np.float64) / 255.0
    )
    o3d.visualization.draw_geometries([point_cloud], window_name=window_name)

