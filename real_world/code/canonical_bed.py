"""Per-trial 4-marker canonical bed frame (graph / policy metres).

Four symmetric top markers define origin and XY. Scale is never applied:
1 real metre = 1 model metre. ``marker_layout.json`` is not used here.

The fused ZED cloud is first expressed in the layout-bed frame via the
one-time ``T_bed_camera`` extrinsics. This module then recenters/rerotates
that metric cloud so Recover sees (0,0) at the detected bed center.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

TOP_MARKER_IDS = (0, 1, 2, 3)
# Physical bed posts: top ID and side ID share a corner.
CORNER_POST = {
    0: 0,
    10: 0,
    1: 1,
    11: 1,
    2: 2,
    12: 2,
    3: 3,
    13: 3,
}


class CanonicalBedError(ValueError):
    """Raised when four top markers cannot define a bed frame."""


@dataclass(frozen=True)
class CanonicalBedFrame:
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    centers_layout: dict[int, np.ndarray]
    detected_ids: tuple[int, ...]

    @property
    def rotation_layout_from_canonical(self) -> np.ndarray:
        """Columns are canonical axes expressed in layout-bed."""

        return np.column_stack((self.x_axis, self.y_axis, self.z_axis))

    def to_dict(self) -> dict:
        return {
            "frame": "canonical_bed",
            "metres": "identity",
            "origin_layout": self.origin.tolist(),
            "x_axis_layout": self.x_axis.tolist(),
            "y_axis_layout": self.y_axis.tolist(),
            "z_axis_layout": self.z_axis.tolist(),
            "centers_layout": {
                str(marker_id): np.asarray(center).tolist()
                for marker_id, center in sorted(self.centers_layout.items())
            },
            "detected_ids": [int(marker_id) for marker_id in self.detected_ids],
            "convention": {
                "origin": "centroid of top marker centers 0,1,2,3",
                "x_positive": "head-left to head-right (0->1, 3->2)",
                "y_positive": "head to foot (0->3, 1->2)",
                "scale": "none",
            },
        }


def physical_posts(marker_ids: Sequence[int]) -> frozenset[int]:
    """Map marker IDs to unique bed-corner posts."""

    posts = set()
    for marker_id in marker_ids:
        post = CORNER_POST.get(int(marker_id))
        if post is not None:
            posts.add(post)
    return frozenset(posts)


def sample_xyz_at_pixels(
    xyz: np.ndarray,
    pixels: np.ndarray,
    *,
    radius_px: int = 3,
) -> np.ndarray:
    """Median finite XYZ in a window around each pixel.

    ``xyz`` is HxWx3 in the camera metric frame. Invalid samples are NaN.
    """

    xyz = np.asarray(xyz, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise CanonicalBedError(f"xyz must be HxWx3, got {xyz.shape}")
    height, width = xyz.shape[:2]
    sampled = np.full((len(pixels), 3), np.nan, dtype=np.float64)
    radius = max(0, int(radius_px))
    for index, (u, v) in enumerate(pixels):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        col = int(round(float(u)))
        row = int(round(float(v)))
        c0 = max(0, col - radius)
        c1 = min(width, col + radius + 1)
        r0 = max(0, row - radius)
        r1 = min(height, row + radius + 1)
        window = xyz[r0:r1, c0:c1].reshape(-1, 3)
        finite = np.isfinite(window).all(axis=1)
        if not np.any(finite):
            continue
        sampled[index] = np.median(window[finite], axis=0)
    return sampled


def canonical_bed_frame(
    centers_layout: Mapping[int, Sequence[float]],
) -> CanonicalBedFrame:
    """Build origin and orthonormal axes from four top-marker centers."""

    missing = [marker_id for marker_id in TOP_MARKER_IDS if marker_id not in centers_layout]
    if missing:
        raise CanonicalBedError(
            "Need top marker centers for IDs 0,1,2,3; missing "
            f"{missing}"
        )
    p0 = np.asarray(centers_layout[0], dtype=np.float64).reshape(3)
    p1 = np.asarray(centers_layout[1], dtype=np.float64).reshape(3)
    p2 = np.asarray(centers_layout[2], dtype=np.float64).reshape(3)
    p3 = np.asarray(centers_layout[3], dtype=np.float64).reshape(3)
    for name, point in ("0", p0), ("1", p1), ("2", p2), ("3", p3):
        if not np.isfinite(point).all():
            raise CanonicalBedError(f"Top marker {name} center is not finite")

    origin = (p0 + p1 + p2 + p3) / 4.0
    vx = 0.5 * ((p1 - p0) + (p2 - p3))
    x_norm = float(np.linalg.norm(vx))
    if x_norm < 1e-6:
        raise CanonicalBedError("Degenerate +X from top markers 0,1,2,3")
    x_axis = vx / x_norm

    vy = 0.5 * ((p3 - p0) + (p2 - p1))
    vy = vy - float(np.dot(vy, x_axis)) * x_axis
    y_norm = float(np.linalg.norm(vy))
    if y_norm < 1e-6:
        raise CanonicalBedError("Degenerate +Y from top markers 0,1,2,3")
    y_axis = vy / y_norm

    z_axis = np.cross(x_axis, y_axis)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < 1e-6:
        raise CanonicalBedError("Degenerate +Z from top-marker axes")
    z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / float(np.linalg.norm(y_axis))

    centers = {
        0: p0.copy(),
        1: p1.copy(),
        2: p2.copy(),
        3: p3.copy(),
    }
    return CanonicalBedFrame(
        origin=origin,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        centers_layout=centers,
        detected_ids=TOP_MARKER_IDS,
    )


def layout_points_to_canonical(
    points_layout: np.ndarray,
    frame: CanonicalBedFrame,
) -> np.ndarray:
    points = np.asarray(points_layout, dtype=np.float64)
    flat = points.reshape(-1, points.shape[-1])
    xyz = flat[:, :3] - frame.origin[None, :]
    rotation = frame.rotation_layout_from_canonical
    canonical_xyz = xyz @ rotation
    if flat.shape[1] == 3:
        converted = canonical_xyz
    else:
        converted = flat.copy()
        converted[:, :3] = canonical_xyz
    return converted.reshape(points.shape)


def canonical_points_to_layout(
    points_canonical: np.ndarray,
    frame: CanonicalBedFrame,
) -> np.ndarray:
    points = np.asarray(points_canonical, dtype=np.float64)
    flat = points.reshape(-1, points.shape[-1])
    rotation = frame.rotation_layout_from_canonical
    layout_xyz = (rotation @ flat[:, :3].T).T + frame.origin[None, :]
    if flat.shape[1] == 3:
        converted = layout_xyz
    else:
        converted = flat.copy()
        converted[:, :3] = layout_xyz
    return converted.reshape(points.shape)


def layout_xy_to_canonical(
    xy_layout: np.ndarray,
    frame: CanonicalBedFrame,
    *,
    z: float = 0.0,
) -> np.ndarray:
    xy = np.asarray(xy_layout, dtype=np.float64).reshape(-1, 2)
    points = np.column_stack(
        [xy, np.full(len(xy), float(z), dtype=np.float64)]
    )
    return layout_points_to_canonical(points, frame)[:, :2].reshape(
        np.asarray(xy_layout).shape
    )


def canonical_xy_to_layout(
    xy_canonical: np.ndarray,
    frame: CanonicalBedFrame,
    *,
    z: float = 0.0,
) -> np.ndarray:
    xy = np.asarray(xy_canonical, dtype=np.float64).reshape(-1, 2)
    points = np.column_stack(
        [xy, np.full(len(xy), float(z), dtype=np.float64)]
    )
    return canonical_points_to_layout(points, frame)[:, :2].reshape(
        np.asarray(xy_canonical).shape
    )


def transform_action_canonical_to_layout(
    action_canonical: Sequence[float],
    frame: CanonicalBedFrame,
    *,
    z: float = 0.0,
) -> np.ndarray:
    action = np.asarray(action_canonical, dtype=np.float64).reshape(4)
    grasp, release = canonical_xy_to_layout(
        np.vstack([action[:2], action[2:]]),
        frame,
        z=z,
    )
    return np.concatenate([grasp, release])


def _transform_point(point: Sequence[float], transform: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return (matrix @ np.append(point, 1.0))[:3]


def canonical_frame_from_sim_origin(data: Mapping[str, object]) -> CanonicalBedFrame:
    """Build the trial frame from uncovered ArUco (``sim_origin_data.pkl``).

    Covered PCD frames can miss a top ID to glare or the blanket. Pose
    capture already solved 0–3; reuse those centers instead of re-detecting.
    """

    centers_cam = data.get("centers_m") or {}
    transform = data.get("bed_from_camera")
    if transform is None:
        raise CanonicalBedError("sim_origin_data.pkl has no bed_from_camera")
    centers_layout = {}
    for marker_id in TOP_MARKER_IDS:
        raw = centers_cam.get(marker_id, centers_cam.get(str(marker_id)))
        if raw is None:
            raise CanonicalBedError(
                "sim_origin_data.pkl is missing top marker "
                f"{marker_id}; detected={data.get('detected_ids')}"
            )
        centers_layout[int(marker_id)] = _transform_point(raw, transform)
    return canonical_bed_frame(centers_layout)


def write_canonical_from_sim_origin(pose_dir: Path) -> Path:
    """Write ``canonical_bed_frame.json`` next to ``sim_origin_data.pkl``."""

    pose_dir = Path(pose_dir)
    candidates = [
        pose_dir / "sim_origin_data.pkl",
        pose_dir.parent / "sim_origin_data.pkl",
    ]
    src = next((path for path in candidates if path.is_file()), None)
    if src is None:
        raise CanonicalBedError(
            f"No sim_origin_data.pkl under {pose_dir} or its parent"
        )
    from pickle_compat import load_pickle

    dest = src.parent / "canonical_bed_frame.json"
    save_canonical_frame(dest, canonical_frame_from_sim_origin(load_pickle(src)))
    return dest


def save_canonical_frame(path: Path, frame: CanonicalBedFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(frame.to_dict(), handle, indent=2)
        handle.write("\n")


def load_canonical_frame(path: Path) -> CanonicalBedFrame:
    path = Path(path)
    with path.open() as handle:
        payload = json.load(handle)
    centers = {
        int(marker_id): np.asarray(center, dtype=np.float64).reshape(3)
        for marker_id, center in (payload.get("centers_layout") or {}).items()
    }
    if len(centers) == 4:
        return canonical_bed_frame(centers)
    origin = np.asarray(payload["origin_layout"], dtype=np.float64).reshape(3)
    x_axis = np.asarray(payload["x_axis_layout"], dtype=np.float64).reshape(3)
    y_axis = np.asarray(payload["y_axis_layout"], dtype=np.float64).reshape(3)
    z_axis = np.asarray(payload["z_axis_layout"], dtype=np.float64).reshape(3)
    detected = tuple(int(marker_id) for marker_id in payload.get("detected_ids", []))
    return CanonicalBedFrame(
        origin=origin,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        centers_layout=centers,
        detected_ids=detected or TOP_MARKER_IDS,
    )


def resolve_canonical_path(pose_dir: Path) -> Optional[Path]:
    """Prefer the trial-root file so initial/intermediate/final share one frame."""

    pose_dir = Path(pose_dir)
    candidates = [
        pose_dir / "canonical_bed_frame.json",
        pose_dir.parent / "canonical_bed_frame.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def maybe_load_canonical_frame(pose_dir: Path) -> Optional[CanonicalBedFrame]:
    path = resolve_canonical_path(pose_dir)
    if path is None:
        return None
    return load_canonical_frame(path)


def canonicalize_body_xy(
    body_points: np.ndarray,
    frame: Optional[CanonicalBedFrame],
) -> np.ndarray:
    """Rotate/translate body XY into the canonical frame; leave extra columns."""

    if frame is None:
        return np.asarray(body_points, dtype=np.float64)
    body = np.asarray(body_points, dtype=np.float64).copy()
    if body.ndim != 2 or body.shape[1] < 2:
        raise CanonicalBedError(f"body_points must be Nx>=2, got {body.shape}")
    body[:, :2] = layout_xy_to_canonical(body[:, :2], frame)
    return body
