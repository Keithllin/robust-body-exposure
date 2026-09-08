"""Visible-layer sensor snap for a persistent predicted cloth graph.

``G_pred`` is the Uncover dynamics prior: the same N particles as the
initial voxel graph, after one predicted Uncover step.  The intermediate
ZED cloud is a partial surface observation, not a second complete graph.

This update snaps only the visible / top node of each predicted XY column
onto nearby sensor points.  Hidden fold layers keep their predicted
coordinates so stacking is not collapsed onto the observed surface.
"""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_XY_RADIUS_M = 0.04
DEFAULT_Z_RADIUS_M = 0.10
DEFAULT_COLUMN_VOXEL_M = 0.05


def _ensure_3d(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(f"Expected point state with shape (N, 2|3), got {points.shape}")
    if points.shape[1] == 3:
        return points.copy()
    return np.concatenate(
        [points, np.zeros((len(points), 1), dtype=points.dtype)],
        axis=1,
    )


def _copy_point_state(points: np.ndarray) -> np.ndarray:
    points = _ensure_3d(points)
    if len(points) == 0 or not np.all(np.isfinite(points)):
        raise ValueError("Point state must be non-empty and finite")
    return points.astype(np.float32, copy=True)


def _nearest_query(
    reference: np.ndarray,
    query: np.ndarray,
    k: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return distances and indices of the k nearest reference points."""

    reference = np.asarray(reference, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    if len(reference) == 0 or len(query) == 0:
        empty_d = np.empty((len(query), k), dtype=np.float32)
        empty_i = np.empty((len(query), k), dtype=np.int64)
        return empty_d, empty_i
    k = max(1, min(int(k), len(reference)))
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(reference).query(query, k=k)
    except (ImportError, TypeError):
        distances = []
        indices = []
        for start in range(0, len(query), 1024):
            block = query[start : start + 1024]
            pairwise = np.linalg.norm(
                block[:, None, :] - reference[None, :, :],
                axis=2,
            )
            nearest = np.argpartition(pairwise, kth=k - 1, axis=1)[:, :k]
            nearest_d = np.take_along_axis(pairwise, nearest, axis=1)
            order = np.argsort(nearest_d, axis=1)
            distances.append(np.take_along_axis(nearest_d, order, axis=1))
            indices.append(np.take_along_axis(nearest, order, axis=1))
        distances = np.concatenate(distances, axis=0)
        indices = np.concatenate(indices, axis=0)
    distances = np.asarray(distances, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    if k == 1:
        return distances.reshape(-1), indices.reshape(-1)
    if distances.ndim == 1:
        return distances.reshape(-1, 1), indices.reshape(-1, 1)
    return distances, indices


def top_layer_mask(
    predicted_state: np.ndarray,
    column_voxel_size: float = DEFAULT_COLUMN_VOXEL_M,
) -> np.ndarray:
    """True for the highest-Z node in each predicted XY column."""

    predicted = _copy_point_state(predicted_state)
    eligible = np.zeros(len(predicted), dtype=bool)
    if len(predicted) == 0:
        return eligible
    if column_voxel_size is None or column_voxel_size <= 0:
        eligible[:] = True
        return eligible
    keys = np.floor(predicted[:, :2] / float(column_voxel_size)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    for column in range(int(inverse.max()) + 1):
        members = np.flatnonzero(inverse == column)
        eligible[int(members[np.argmax(predicted[members, 2])])] = True
    return eligible


def attach_persistent_z(
    predicted_xy: np.ndarray,
    *,
    initial_xyz: np.ndarray | None = None,
    initial_xy: np.ndarray | None = None,
    g0_model: np.ndarray | None = None,
) -> np.ndarray:
    """Recover a per-node Z so snap can tell top vs hidden layers.

    Preference:
    1. Same-length ``initial_xyz`` (Uncover G0, index-aligned).
    2. Nearest G0 node to each ``initial_xy`` when voxel counts differ.
    3. Same-length ``g0_model``.
    4. Zeros.  With a column gate, zero Z still snaps only one node per cell.
    """

    predicted_xy = np.asarray(predicted_xy, dtype=np.float64).reshape(-1, 2)
    n = len(predicted_xy)
    if initial_xyz is not None:
        initial_xyz = np.asarray(initial_xyz, dtype=np.float64).reshape(-1, 3)
        if len(initial_xyz) == n:
            return initial_xyz[:, 2].copy()
    if (
        initial_xy is not None
        and g0_model is not None
        and len(g0_model) > 0
    ):
        initial_xy = np.asarray(initial_xy, dtype=np.float64).reshape(-1, 2)
        g0_model = np.asarray(g0_model, dtype=np.float64).reshape(-1, 3)
        if len(initial_xy) == n:
            _, idx = _nearest_query(g0_model[:, :2], initial_xy.astype(np.float32), k=1)
            return g0_model[np.asarray(idx, dtype=np.int64), 2].copy()
    if g0_model is not None:
        g0_model = np.asarray(g0_model, dtype=np.float64).reshape(-1, 3)
        if len(g0_model) == n:
            return g0_model[:, 2].copy()
    return np.zeros(n, dtype=np.float64)


def load_predicted_state_3d(
    npz_path,
    g0_model: np.ndarray | None = None,
) -> np.ndarray:
    """Build G_pred XYZ from an Uncover ``uncover_prediction.npz``."""

    loaded = np.load(npz_path, allow_pickle=False)
    if "predicted_model_xyz" in loaded.files:
        return _copy_point_state(loaded["predicted_model_xyz"])
    if "predicted_model_xy" not in loaded.files:
        raise KeyError(f"{npz_path} is missing predicted_model_xy")
    predicted_xy = np.asarray(loaded["predicted_model_xy"], dtype=np.float64).reshape(
        -1, 2
    )
    initial_xyz = (
        np.asarray(loaded["cloth_initial_xyz"], dtype=np.float64)
        if "cloth_initial_xyz" in loaded.files
        else None
    )
    initial_xy = (
        np.asarray(loaded["cloth_initial_xy"], dtype=np.float64)
        if "cloth_initial_xy" in loaded.files
        else None
    )
    z = attach_persistent_z(
        predicted_xy,
        initial_xyz=initial_xyz,
        initial_xy=initial_xy,
        g0_model=g0_model,
    )
    return _copy_point_state(np.column_stack([predicted_xy, z]))


def snap_visible_layer(
    predicted_state: np.ndarray,
    sensor_state: np.ndarray,
    xy_radius: float = DEFAULT_XY_RADIUS_M,
    z_radius: float = DEFAULT_Z_RADIUS_M,
    column_voxel_size: float = DEFAULT_COLUMN_VOXEL_M,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Correct only the visible top node of each predicted fold column.

    Each sensor point may claim at most one eligible predicted vertex:
    inside the XY/Z gates, and only among the topmost node of each XY
    column.  Matched vertices move to the averaged sensor location.
    Hidden / unmatched vertices stay exactly at the prediction.
    """

    predicted = _copy_point_state(predicted_state)
    sensor = _copy_point_state(sensor_state)
    eligible = top_layer_mask(predicted, column_voxel_size)
    corrected = predicted.copy()
    metadata: dict[str, Any] = {
        "xy_radius_m": float(xy_radius),
        "z_radius_m": float(z_radius),
        "column_voxel_size_m": (
            None if column_voxel_size is None else float(column_voxel_size)
        ),
        "num_pred_points": int(len(predicted)),
        "num_sensor_points": int(len(sensor)),
        "num_eligible_top": int(np.count_nonzero(eligible)),
        "num_visible": 0,
        "visible_ratio": 0.0,
        "mean_visible_snap_m": float("nan"),
        "mean_hidden_disp_m": 0.0,
        "num_hidden_unchanged": int(len(predicted)),
        "num_sensor_matched": 0,
        "num_sensor_unmatched": int(len(sensor)),
        "status": "unchanged_no_visible",
    }
    if (
        len(predicted) == 0
        or len(sensor) == 0
        or xy_radius <= 0
        or z_radius <= 0
        or not np.any(eligible)
    ):
        return corrected, metadata

    eligible_idx = np.flatnonzero(eligible)
    eligible_xy = predicted[eligible_idx, :2]
    num_neighbors = min(32, len(eligible_idx))
    distances, neighbor_local = _nearest_query(
        eligible_xy,
        sensor[:, :2],
        k=num_neighbors,
    )
    if num_neighbors == 1:
        distances = distances.reshape(-1, 1)
        neighbor_local = neighbor_local.reshape(-1, 1)

    snap_sum = np.zeros_like(predicted, dtype=np.float32)
    snap_count = np.zeros(len(predicted), dtype=np.int32)
    matched_sensor = np.zeros(len(sensor), dtype=bool)
    for sensor_idx in range(len(sensor)):
        candidate_distances = distances[sensor_idx]
        local_idx = neighbor_local[sensor_idx]
        valid = np.isfinite(candidate_distances) & (
            candidate_distances <= float(xy_radius)
        )
        local_idx = local_idx[valid]
        if len(local_idx) == 0:
            continue
        candidate_idx = eligible_idx[local_idx]
        z_compatible = (
            np.abs(predicted[candidate_idx, 2] - sensor[sensor_idx, 2])
            <= float(z_radius)
        )
        candidate_idx = candidate_idx[z_compatible]
        if len(candidate_idx) == 0:
            continue
        selected = int(candidate_idx[np.argmax(predicted[candidate_idx, 2])])
        snap_sum[selected] += sensor[sensor_idx] - predicted[selected]
        snap_count[selected] += 1
        matched_sensor[sensor_idx] = True

    visible_mask = snap_count > 0
    visible_idx = np.flatnonzero(visible_mask)
    if len(visible_idx) == 0:
        return corrected, metadata

    displacements = np.zeros_like(predicted, dtype=np.float32)
    displacements[visible_idx] = snap_sum[visible_idx] / snap_count[
        visible_idx, None
    ].astype(np.float32)
    corrected[visible_idx] = predicted[visible_idx] + displacements[visible_idx]

    hidden_disp = np.linalg.norm(corrected[~visible_mask] - predicted[~visible_mask], axis=1)
    metadata.update(
        {
            "num_visible": int(len(visible_idx)),
            "visible_ratio": float(len(visible_idx) / len(predicted)),
            "mean_visible_snap_m": float(
                np.mean(np.linalg.norm(displacements[visible_idx], axis=1))
            ),
            "mean_hidden_disp_m": float(np.mean(hidden_disp)) if len(hidden_disp) else 0.0,
            "num_hidden_unchanged": int(len(predicted) - len(visible_idx)),
            "num_sensor_matched": int(np.count_nonzero(matched_sensor)),
            "num_sensor_unmatched": int(
                len(sensor) - np.count_nonzero(matched_sensor)
            ),
            "status": "visible_only_corrected",
        }
    )
    return _copy_point_state(corrected), metadata
