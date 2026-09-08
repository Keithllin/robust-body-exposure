"""Point-cloud merge policies for calibrated bed-frame captures."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def _select_points(
    cloud: o3d.geometry.PointCloud,
    keep: np.ndarray,
) -> o3d.geometry.PointCloud:
    indices = np.flatnonzero(keep).astype(np.int64)
    return cloud.select_by_index(indices.tolist())


def supported_by_3d(
    reference: o3d.geometry.PointCloud,
    candidate: o3d.geometry.PointCloud,
    radius_m: float,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Keep candidate points with a nearby reference point in bed XYZ.

    Candidate points are returned as the canonical geometry so reference
    points are used only for support and are not reintroduced into the merge.
    """

    if radius_m <= 0:
        raise ValueError("radius_m must be positive")

    reference_points = np.asarray(reference.points, dtype=np.float64)
    candidate_points = np.asarray(candidate.points, dtype=np.float64)
    if not len(reference_points) or not len(candidate_points):
        empty = _select_points(candidate, np.zeros(len(candidate_points), dtype=bool))
        return empty, {
            "num_reference_points": int(len(reference_points)),
            "num_candidate_points": int(len(candidate_points)),
            "num_supported_points": 0,
            "support_fraction": 0.0,
        }

    nearest = cKDTree(reference_points).query(
        candidate_points,
        k=1,
    )[0]
    keep = nearest <= radius_m
    supported = _select_points(candidate, keep)
    return supported, {
        "num_reference_points": int(len(reference_points)),
        "num_candidate_points": int(len(candidate_points)),
        "num_supported_points": int(np.count_nonzero(keep)),
        "support_fraction": float(np.mean(keep)),
        "metric": "bed_xyz",
        "radius_m": float(radius_m),
    }


def keep_largest_xy_component(
    cloud: o3d.geometry.PointCloud,
    connectivity_radius_m: float,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Remove disconnected bed-XY point components from a cloud."""

    if connectivity_radius_m <= 0:
        raise ValueError("connectivity_radius_m must be positive")
    points = np.asarray(cloud.points, dtype=np.float64)
    if not len(points):
        return cloud, {
            "num_input_points": 0,
            "num_components": 0,
            "num_largest_component_points": 0,
            "connectivity_radius_m": float(connectivity_radius_m),
        }

    neighbors = cKDTree(points[:, :2]).query_ball_tree(
        cKDTree(points[:, :2]),
        connectivity_radius_m,
    )
    unseen = set(range(len(points)))
    largest = []
    num_components = 0
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        component = [seed]
        while stack:
            index = stack.pop()
            for neighbor in neighbors[index]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        num_components += 1
        if len(component) > len(largest):
            largest = component

    filtered = cloud.select_by_index(largest)
    return filtered, {
        "num_input_points": int(len(points)),
        "num_components": int(num_components),
        "num_largest_component_points": int(len(largest)),
        "num_removed_points": int(len(points) - len(largest)),
        "connectivity_radius_m": float(connectivity_radius_m),
    }


def filter_ceiling_by_side_votes(
    ceiling: o3d.geometry.PointCloud,
    side_clouds: Sequence[o3d.geometry.PointCloud],
    xyz_radius_m: float,
    xy_radius_m: float,
    z_outlier_m: float = 0.03,
    column_m: float = 0.05,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Keep ceiling geometry; use side cameras only as a sparse-noise veto.

    Side cameras often see a fold wall or a slightly offset surface in the
    same XY column, so a raw XYZ mismatch is not enough to discard ceiling
    coverage.  A ceiling point is dropped only when all of these hold:

    * a side camera has an XY vote in the neighbourhood,
    * that vote does not match in 3D,
    * the point is a height outlier relative to other ceiling points in its
      XY column.

    Dense, internally consistent ceiling sheets are therefore preserved even
    when the side cameras disagree in Z.  Columns no side camera reached are
    left untouched.
    """

    if xyz_radius_m <= 0 or xy_radius_m <= 0:
        raise ValueError("xyz_radius_m and xy_radius_m must be positive")
    if z_outlier_m <= 0:
        raise ValueError("z_outlier_m must be positive")
    if column_m <= 0:
        raise ValueError("column_m must be positive")

    ceiling_points = np.asarray(ceiling.points, dtype=np.float64)
    side_blocks = [
        np.asarray(cloud.points, dtype=np.float64)
        for cloud in side_clouds
        if len(cloud.points)
    ]
    info = {
        "num_ceiling_points": int(len(ceiling_points)),
        "num_side_points": int(sum(len(block) for block in side_blocks)),
        "num_confirmed_points": 0,
        "num_unvoted_points": 0,
        "num_side_disagreements": 0,
        "num_z_outliers": 0,
        "num_rejected_points": 0,
        "num_kept_points": int(len(ceiling_points)),
        "xyz_radius_m": float(xyz_radius_m),
        "xy_radius_m": float(xy_radius_m),
        "z_outlier_m": float(z_outlier_m),
        "column_m": float(column_m),
        "metric": "ceiling_primary",
    }
    if not len(ceiling_points):
        return ceiling, info
    if not side_blocks:
        info["num_unvoted_points"] = int(len(ceiling_points))
        return ceiling, info

    side_points = np.concatenate(side_blocks, axis=0)
    distance_xyz = cKDTree(side_points).query(ceiling_points, k=1)[0]
    distance_xy = cKDTree(side_points[:, :2]).query(ceiling_points[:, :2], k=1)[0]
    confirmed = distance_xyz <= xyz_radius_m
    unvoted = distance_xy > xy_radius_m
    disagreement = ~confirmed & ~unvoted
    columns = np.floor(ceiling_points[:, :2] / column_m).astype(np.int64)
    _, inverse = np.unique(columns, axis=0, return_inverse=True)
    local_z = np.array(
        [
            np.median(ceiling_points[inverse == index, 2])
            for index in range(int(inverse.max()) + 1)
        ],
        dtype=np.float64,
    )
    z_outlier = np.abs(ceiling_points[:, 2] - local_z[inverse]) > z_outlier_m
    keep = confirmed | unvoted | ~z_outlier
    info["num_confirmed_points"] = int(np.count_nonzero(confirmed))
    info["num_unvoted_points"] = int(np.count_nonzero(unvoted & ~confirmed))
    info["num_side_disagreements"] = int(np.count_nonzero(disagreement))
    info["num_z_outliers"] = int(np.count_nonzero(z_outlier))
    info["num_rejected_points"] = int(np.count_nonzero(~keep))
    info["num_kept_points"] = int(np.count_nonzero(keep))
    return _select_points(ceiling, keep), info


def fill_uncovered_columns(
    merged: o3d.geometry.PointCloud,
    support: o3d.geometry.PointCloud,
    cell_m: float,
    dilation_cells: int,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Add support points in XY columns the merge does not cover at all.

    ``top_supported`` returns side-camera geometry only, so bed regions that no
    side camera can confirm are dropped even when the ceiling saw them
    cleanly.  The downstream Recover graph is 2D, so a missing column is pure
    information loss while an extra point inside an already-covered column only
    inflates per-column depth.  This fills the former without touching the
    latter.  ``cell_m`` should match the model voxel size and
    ``dilation_cells`` bounds how far a fill column may sit from existing
    coverage, so background blobs the support test rejected stay rejected.
    """

    if cell_m <= 0:
        raise ValueError("cell_m must be positive")
    if dilation_cells < 1:
        raise ValueError("dilation_cells must be >= 1")

    merged_points = np.asarray(merged.points, dtype=np.float64)
    support_points = np.asarray(support.points, dtype=np.float64)
    info = {
        "cell_m": float(cell_m),
        "dilation_cells": int(dilation_cells),
        "num_support_points": int(len(support_points)),
        "num_filled_points": 0,
        "num_filled_columns": 0,
        "num_columns_before": 0,
        "num_columns_after": 0,
    }
    if not len(merged_points) or not len(support_points):
        return merged, info

    def columns(points: np.ndarray) -> np.ndarray:
        return np.floor(points[:, :2] / cell_m).astype(np.int64)

    covered = set(map(tuple, columns(merged_points)))
    offsets = range(-dilation_cells, dilation_cells + 1)
    reachable = {
        (column[0] + dx, column[1] + dy)
        for column in covered
        for dx in offsets
        for dy in offsets
    }
    support_columns = columns(support_points)
    keep = np.array(
        [
            key not in covered and key in reachable
            for key in map(tuple, support_columns)
        ],
        dtype=bool,
    )
    info["num_columns_before"] = int(len(covered))
    if not keep.any():
        info["num_columns_after"] = int(len(covered))
        return merged, info

    filled = merged + _select_points(support, keep)
    info["num_filled_points"] = int(np.count_nonzero(keep))
    info["num_filled_columns"] = int(len(set(map(tuple, support_columns[keep]))))
    info["num_columns_after"] = int(
        len(set(map(tuple, columns(np.asarray(filled.points, dtype=np.float64)))))
    )
    return filled, info


def merge_clouds(
    clouds: Mapping[str, o3d.geometry.PointCloud],
    roles: Sequence[str],
    mode: str = "auto",
    support_role: str = "ceiling",
    radius_m: float = 0.03,
    xy_radius_m: float | None = None,
    fill_holes_from_support: bool = True,
    fill_cell_m: float = 0.05,
    fill_dilation_cells: int = 2,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Merge calibrated clouds using ceiling geometry or supported side clouds.

    ``auto`` applies ``ceiling_primary`` when the ceiling and at least one
    side camera are present, otherwise it falls back to a normal union.
    ``ceiling_primary`` keeps the ceiling cloud as canonical geometry and
    uses side cameras only to veto points whose XY neighbourhood was seen
    from the side at a disagreeing 3D location.  ``top_supported`` is the
    previous policy: side-camera geometry retained only when a nearby
    ceiling point exists, then optional hole-fill from the ceiling.
    """

    roles = tuple(roles)
    supported_modes = {"auto", "union", "top_supported", "ceiling_primary"}
    if mode not in supported_modes:
        raise ValueError(f"Unknown merge mode: {mode}")
    has_support_role = support_role in roles
    has_side = any(role.startswith("side_") for role in roles)
    needs_pair = mode in {"top_supported", "ceiling_primary"}
    if needs_pair and not has_support_role:
        raise ValueError(
            f"{mode} merge requires support role {support_role!r}"
        )
    if needs_pair and not has_side:
        raise ValueError(f"{mode} merge requires at least one side camera")
    if not has_support_role or not has_side:
        effective_mode = "union"
    elif mode == "auto":
        effective_mode = "ceiling_primary"
    else:
        effective_mode = mode

    xy_radius = float(radius_m if xy_radius_m is None else xy_radius_m)
    merged = o3d.geometry.PointCloud()
    support_info = {}
    hole_fill_info = None
    if effective_mode == "union":
        for role in roles:
            merged += clouds[role]
    elif effective_mode == "ceiling_primary":
        side_clouds = [
            clouds[role] for role in roles if role.startswith("side_")
        ]
        filtered, vote_info = filter_ceiling_by_side_votes(
            clouds[support_role],
            side_clouds,
            xyz_radius_m=radius_m,
            xy_radius_m=xy_radius,
            column_m=fill_cell_m,
        )
        filtered, component_info = keep_largest_xy_component(
            filtered,
            connectivity_radius_m=radius_m,
        )
        vote_info["component_filter"] = component_info
        vote_info["num_points_after_component"] = int(len(filtered.points))
        support_info[support_role] = vote_info
        merged = filtered
    else:
        reference = clouds[support_role]
        for role in roles:
            if not role.startswith("side_"):
                continue
            # First remove top-only points for this specific pair.  Then use
            # those pairwise-consensus top points to filter the side cloud.
            top_consensus, top_info = supported_by_3d(
                clouds[role],
                reference,
                radius_m,
            )
            supported, info = supported_by_3d(
                top_consensus,
                clouds[role],
                radius_m,
            )
            supported, component_info = keep_largest_xy_component(
                supported,
                connectivity_radius_m=radius_m,
            )
            info["top_consensus"] = top_info
            info["component_filter"] = component_info
            info["num_points_after_component"] = int(len(supported.points))
            support_info[role] = info
            merged += supported

    if not len(merged.points):
        raise RuntimeError(
            "The selected PCD merge produced zero points; "
            "increase the support radius or use --merge-mode union."
        )

    if effective_mode == "top_supported" and fill_holes_from_support:
        merged, hole_fill_info = fill_uncovered_columns(
            merged,
            clouds[support_role],
            cell_m=fill_cell_m,
            dilation_cells=fill_dilation_cells,
        )

    return merged, {
        "requested_mode": mode,
        "effective_mode": effective_mode,
        "support_role": (
            support_role
            if effective_mode in {"top_supported", "ceiling_primary"}
            else None
        ),
        "support_metric": (
            "bed_xyz"
            if effective_mode == "top_supported"
            else "ceiling_xy_veto"
            if effective_mode == "ceiling_primary"
            else None
        ),
        "support_radius_m": (
            float(radius_m)
            if effective_mode in {"top_supported", "ceiling_primary"}
            else None
        ),
        "xy_radius_m": (
            xy_radius if effective_mode == "ceiling_primary" else None
        ),
        "support": support_info,
        "hole_fill": hole_fill_info,
        "num_points_merged": int(len(merged.points)),
    }
