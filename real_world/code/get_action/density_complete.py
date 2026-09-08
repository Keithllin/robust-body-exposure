"""Sensor-anchored local density completion for a 2D Recover graph.

Recover never sees Z or layer identity.  The folding cue it was trained on is
local XY multiplicity.  This module keeps G_sensor geometry and, only in
cells where sensor and pred both have cloth, adds a few nodes so the local
count moves toward pred.  New XY offsets are borrowed from the pred patch
and re-anchored on the sensor patch centroid.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np


DENSITY_MODES = ("density-1", "density-50", "density-full")
MIN_SEPARATION_M = 0.003


def _xy(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected (N, 2|3) points, got {points.shape}")
    return points[:, :2].copy()


def _as_xyz(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(f"Expected (N, 2|3) points, got {points.shape}")
    if points.shape[1] == 3:
        return points.copy()
    z = np.zeros((len(points), 1), dtype=np.float64)
    return np.concatenate([points, z], axis=1)


def cell_keys(xy: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor(np.asarray(xy, dtype=np.float64) / float(voxel_size)).astype(
        np.int64
    )


def _groups(xy: np.ndarray, voxel_size: float) -> dict[tuple[int, int], np.ndarray]:
    keys = cell_keys(xy, voxel_size)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, keys.tolist())):
        groups.setdefault((int(key[0]), int(key[1])), []).append(index)
    return {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()}


def target_count(n_sensor: int, n_pred: int, mode: str) -> int:
    if n_pred <= n_sensor:
        return int(n_sensor)
    gap = int(n_pred - n_sensor)
    if mode == "density-1":
        return int(n_sensor + min(1, gap))
    if mode == "density-50":
        return int(n_sensor + max(1, int(round(0.5 * gap)))) if gap else int(n_sensor)
    if mode == "density-full":
        return int(n_pred)
    raise ValueError(f"Unknown density mode: {mode}")


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    start = np.asarray(start, dtype=np.float64).reshape(2)
    end = np.asarray(end, dtype=np.float64).reshape(2)
    point = np.asarray(point, dtype=np.float64).reshape(2)
    span = end - start
    length = float(np.dot(span, span))
    if length <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, span) / length, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * span)))


def _place_offsets(
    sensor_xy: np.ndarray,
    pred_xy: np.ndarray,
    n_add: int,
    *,
    beta: float,
    min_separation_m: float,
) -> np.ndarray:
    if n_add <= 0 or len(pred_xy) == 0 or len(sensor_xy) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    sensor_c = np.mean(sensor_xy, axis=0)
    pred_c = np.mean(pred_xy, axis=0)
    offsets = pred_xy - pred_c[None, :]
    current = [np.asarray(point, dtype=np.float64) for point in sensor_xy]
    remaining = list(range(len(offsets)))
    added = []
    while remaining and len(added) < n_add:
        best_i = None
        best_d = -1.0
        for index in remaining:
            candidate = sensor_c + float(beta) * offsets[index]
            dist = min(float(np.linalg.norm(candidate - point)) for point in current)
            if dist > best_d:
                best_d = dist
                best_i = index
        remaining.remove(best_i)
        if best_d < float(min_separation_m):
            continue
        candidate = sensor_c + float(beta) * offsets[best_i]
        added.append(candidate)
        current.append(candidate)
    if not added:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(added, dtype=np.float64).reshape(-1, 2)


def complete_sensor_density(
    sensor_state: np.ndarray,
    pred_state: np.ndarray,
    *,
    mode: str,
    voxel_size: float = 0.05,
    beta: float = 1.0,
    min_separation_m: float = MIN_SEPARATION_M,
    trajectory: np.ndarray | None = None,
    trajectory_radius_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return G_sensor plus locally completed nodes and correction metadata."""

    if mode not in DENSITY_MODES:
        raise ValueError(f"mode must be one of {DENSITY_MODES}, got {mode!r}")
    sensor_xyz = _as_xyz(sensor_state)
    pred_xy = _xy(pred_state)
    sensor_xy = sensor_xyz[:, :2]
    sensor_groups = _groups(sensor_xy, voxel_size)
    pred_groups = _groups(pred_xy, voxel_size)
    shared = sorted(set(sensor_groups) & set(pred_groups))

    added_xy: list[np.ndarray] = []
    eligible = 0
    completed = 0
    skipped_traj = 0
    for key in shared:
        sensor_idx = sensor_groups[key]
        pred_idx = pred_groups[key]
        n_s = int(len(sensor_idx))
        n_p = int(len(pred_idx))
        if n_p <= n_s:
            continue
        eligible += 1
        sensor_cell = sensor_xy[sensor_idx]
        if trajectory is not None and trajectory_radius_m > 0:
            action = np.asarray(trajectory, dtype=np.float64).reshape(4)
            centroid = np.mean(sensor_cell, axis=0)
            if _segment_distance(centroid, action[:2], action[2:]) > trajectory_radius_m:
                skipped_traj += 1
                continue
        n_target = target_count(n_s, n_p, mode)
        n_add = max(0, n_target - n_s)
        if n_add <= 0:
            continue
        new_xy = _place_offsets(
            sensor_cell,
            pred_xy[pred_idx],
            n_add,
            beta=beta,
            min_separation_m=min_separation_m,
        )
        if len(new_xy) == 0:
            continue
        added_xy.append(new_xy)
        completed += 1

    if added_xy:
        stacked = np.concatenate(added_xy, axis=0)
        z = np.full((len(stacked), 1), float(np.median(sensor_xyz[:, 2])))
        added_xyz = np.concatenate([stacked, z], axis=1)
        completed_xyz = np.concatenate([sensor_xyz, added_xyz], axis=0)
    else:
        stacked = np.zeros((0, 2), dtype=np.float64)
        completed_xyz = sensor_xyz

    metadata = {
        "mode": mode,
        "voxel_size_m": float(voxel_size),
        "beta": float(beta),
        "min_separation_m": float(min_separation_m),
        "num_sensor": int(len(sensor_xyz)),
        "num_pred": int(len(pred_xy)),
        "num_completed": int(len(completed_xyz)),
        "num_added": int(len(stacked)),
        "num_shared_cells": int(len(shared)),
        "num_density_gap_cells": int(eligible),
        "num_cells_completed": int(completed),
        "num_cells_skipped_trajectory": int(skipped_traj),
        "trajectory_radius_m": float(trajectory_radius_m),
        "status": "density_completed" if len(stacked) else "unchanged",
    }
    return completed_xyz.astype(np.float64), metadata


def augment_release_density(
    sensor_state: np.ndarray,
    action_xy: np.ndarray,
    *,
    fraction: float,
    voxel_size: float = 0.05,
    radius_m: float = 0.16,
    min_separation_m: float = MIN_SEPARATION_M,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Heuristic density bump near an Uncover grasp→release segment.

    Used only as a no-pred mechanism test.  New nodes are existing sensor
    points jittered by a 1 cm offset away from the local centroid.
    """

    sensor_xyz = _as_xyz(sensor_state)
    sensor_xy = sensor_xyz[:, :2]
    action = np.asarray(action_xy, dtype=np.float64).reshape(4)
    groups = _groups(sensor_xy, voxel_size)
    added = []
    touched = 0
    for key, idx in groups.items():
        cell = sensor_xy[idx]
        centroid = np.mean(cell, axis=0)
        if _segment_distance(centroid, action[:2], action[2:]) > float(radius_m):
            continue
        n_add = max(1, int(round(len(idx) * float(fraction))))
        touched += 1
        offsets = cell - centroid[None, :]
        for i in range(n_add):
            source = offsets[i % len(offsets)]
            if float(np.linalg.norm(source)) < 1e-6:
                source = np.asarray([min_separation_m, 0.0], dtype=np.float64)
            direction = source / max(float(np.linalg.norm(source)), 1e-9)
            candidate = cell[i % len(cell)] + direction * float(min_separation_m)
            if min(np.linalg.norm(candidate - p) for p in cell) < min_separation_m * 0.5:
                continue
            added.append(candidate)
    if added:
        stacked = np.asarray(added, dtype=np.float64).reshape(-1, 2)
        z = np.full((len(stacked), 1), float(np.median(sensor_xyz[:, 2])))
        out = np.concatenate(
            [sensor_xyz, np.concatenate([stacked, z], axis=1)], axis=0
        )
    else:
        stacked = np.zeros((0, 2), dtype=np.float64)
        out = sensor_xyz
    metadata = {
        "mode": f"release+{fraction:.2f}",
        "num_sensor": int(len(sensor_xyz)),
        "num_completed": int(len(out)),
        "num_added": int(len(stacked)),
        "num_cells_touched": int(touched),
        "radius_m": float(radius_m),
        "status": "release_augmented" if len(stacked) else "unchanged",
    }
    return out.astype(np.float64), metadata


def pts_per_column(points: np.ndarray, voxel_size: float) -> tuple[int, int, float]:
    xy = _xy(points)
    if len(xy) == 0:
        return 0, 0, float("nan")
    keys = np.unique(cell_keys(xy, voxel_size), axis=0)
    n_cols = int(len(keys))
    return int(len(xy)), n_cols, float(len(xy) / n_cols) if n_cols else float("nan")
