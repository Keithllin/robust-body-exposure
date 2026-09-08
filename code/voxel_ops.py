"""Voxel operators shared by simulation training and real-world runtime.

The voxel contract used by the voxel-aware models is:

* fixed XYZ cells;
* one representative original point per occupied cell;
* the representative is the point closest to that cell's XYZ barycenter.

The representative index is also applied to the target state so that the
initial and final nodes retain their correspondence.
"""

from typing import Tuple

import numpy as np


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")
    if len(array) == 0:
        raise ValueError(f"{name} must contain at least one point")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return array


def xyz_voxel_representatives(
    points: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return nearest-barycenter representatives and their source indices."""

    points = _validate_points(points, "points")
    voxel_size = float(voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError(
            f"voxel_size must be finite and positive, got {voxel_size}"
        )

    lower = np.min(points, axis=0)
    keys = np.floor((points - lower[None, :]) / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(
        keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    representative_indices = []
    for group_index in range(len(counts)):
        members = np.flatnonzero(inverse == group_index)
        barycenter = np.mean(points[members], axis=0)
        distances = np.linalg.norm(
            points[members] - barycenter[None, :],
            axis=1,
        )
        representative_indices.append(int(members[int(np.argmin(distances))]))

    representative_indices = np.asarray(representative_indices, dtype=np.int64)
    return points[representative_indices].copy(), representative_indices


def voxelize_xyz_state_pair(
    initial_points: np.ndarray,
    final_points: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Voxelize an initial/target pair using one shared membership map."""

    initial_points = _validate_points(initial_points, "initial_points")
    final_points = _validate_points(final_points, "final_points")
    if len(initial_points) != len(final_points):
        raise ValueError(
            "initial_points and final_points must have equal lengths, got "
            f"{len(initial_points)} and {len(final_points)}"
        )

    _, representative_indices = xyz_voxel_representatives(
        initial_points,
        voxel_size,
    )
    return (
        initial_points[representative_indices].copy(),
        final_points[representative_indices].copy(),
    )
