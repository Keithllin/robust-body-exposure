"""Execution-frame blanket grasp checks.

These checks use the measured bed-frame PCD and the exact bed-frame pickle.
They deliberately do not use the Recover model's draped/mirrored graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_MAX_GRASP_DISTANCE_M = 0.020
DEFAULT_SNAP_INWARD_M = 0.015


def read_pcd_xyz(path: Path) -> np.ndarray:
    """Read XYZ from an ASCII or binary PCD without Open3D."""

    header_lines: list[str] = []
    data_format = None
    with Path(path).open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            header_lines.append(line.decode("ascii").strip())
            if line.upper().startswith(b"DATA"):
                parts = line.decode("ascii").strip().split()
                data_format = parts[1].lower() if len(parts) > 1 else None
                break
        payload = handle.read()

    if data_format not in {"ascii", "binary"}:
        raise RuntimeError(f"Unsupported PCD header: {path}")

    fields = sizes = types = counts = points_count = None
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        key = parts[0].upper()
        if key == "FIELDS":
            fields = parts[1:]
        elif key == "SIZE":
            sizes = [int(v) for v in parts[1:]]
        elif key == "TYPE":
            types = parts[1:]
        elif key == "COUNT":
            counts = [int(v) for v in parts[1:]]
        elif key == "POINTS":
            points_count = int(parts[1])
    if fields is None or sizes is None or types is None:
        raise RuntimeError(f"PCD header missing field metadata: {path}")
    if counts is None:
        counts = [1] * len(fields)
    if any(field not in fields for field in ("x", "y", "z")):
        raise RuntimeError(f"PCD missing XYZ: {path}")

    if data_format == "ascii":
        rows = np.fromstring(payload.decode("ascii"), sep=" ")
        width = sum(counts)
        if rows.size % width:
            raise RuntimeError(f"Malformed ASCII PCD payload: {path}")
        rows = rows.reshape(-1, width)
        columns = []
        offset = 0
        for count, field in zip(counts, fields):
            if field in {"x", "y", "z"}:
                columns.append(rows[:, offset])
            offset += count
        return np.column_stack(columns).astype(np.float64, copy=False)

    dtype_map = {
        ("F", 4): np.dtype("<f4"),
        ("F", 8): np.dtype("<f8"),
        ("U", 1): np.dtype("u1"),
        ("U", 2): np.dtype("<u2"),
        ("U", 4): np.dtype("<u4"),
        ("I", 1): np.dtype("i1"),
        ("I", 2): np.dtype("<i2"),
        ("I", 4): np.dtype("<i4"),
    }
    structured_fields = []
    for field, size, kind, count in zip(fields, sizes, types, counts):
        base = dtype_map.get((kind, size))
        if base is None:
            raise RuntimeError(f"Unsupported binary PCD field {field}: {kind}{size}")
        structured_fields.append(
            (field, base, (count,)) if count != 1 else (field, base)
        )
    structured = np.frombuffer(
        payload,
        dtype=np.dtype(structured_fields),
        count=points_count if points_count is not None else -1,
    )
    return np.column_stack(
        [structured[field].reshape(-1) for field in ("x", "y", "z")]
    ).astype(np.float64, copy=False)


def nearest_grasp_xy_distance(
    action_bed: Sequence[float], blanket_points_bed: np.ndarray
) -> float:
    """Distance from the exact executed grasp XY to measured blanket XY."""

    action = np.asarray(action_bed, dtype=np.float64).reshape(4)
    points = np.asarray(blanket_points_bed, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(points[:, :2]).all(axis=1)
    if not np.any(finite):
        raise ValueError("Blanket PCD has no finite XY points")
    return float(
        np.min(np.linalg.norm(points[finite, :2] - action[:2][None, :], axis=1))
    )


@dataclass(frozen=True)
class GraspSnapResult:
    action: np.ndarray
    planned_xy: tuple[float, float]
    nearest_xy: tuple[float, float]
    snapped: bool
    distance_before_m: float
    distance_after_m: float
    inward_m: float
    used_nearest_only: bool

    def to_dict(self) -> dict:
        return {
            "snapped": bool(self.snapped),
            "planned_xy": [float(self.planned_xy[0]), float(self.planned_xy[1])],
            "nearest_xy": [float(self.nearest_xy[0]), float(self.nearest_xy[1])],
            "executed_xy": [float(self.action[0]), float(self.action[1])],
            "distance_before_m": float(self.distance_before_m),
            "distance_after_m": float(self.distance_after_m),
            "inward_m": float(self.inward_m),
            "used_nearest_only": bool(self.used_nearest_only),
        }


def snap_grasp_inward(
    action_bed: Sequence[float],
    blanket_points_bed: np.ndarray,
    *,
    on_cloth_m: float = DEFAULT_MAX_GRASP_DISTANCE_M,
    inward_m: float = DEFAULT_SNAP_INWARD_M,
) -> GraspSnapResult:
    """If the grasp is off cloth, snap to the nearest point plus an inward margin.

    Direction is planned-grasp → nearest cloth, then past that point by
    ``inward_m`` (default 1.5 cm) so execution is not on the rim. Release
    is unchanged. If the inward step lands farther from cloth than the
    nearest point (thin strip), fall back to the nearest point only.
    """

    action = np.asarray(action_bed, dtype=np.float64).reshape(4).copy()
    planned_xy = (float(action[0]), float(action[1]))
    points = np.asarray(blanket_points_bed, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(points[:, :2]).all(axis=1)
    if not np.any(finite):
        raise ValueError("Blanket PCD has no finite XY points")
    xy = points[finite, :2]
    dists = np.linalg.norm(xy - action[:2][None, :], axis=1)
    nearest_i = int(np.argmin(dists))
    nearest = xy[nearest_i].astype(np.float64, copy=False)
    before = float(dists[nearest_i])
    if before <= float(on_cloth_m):
        return GraspSnapResult(
            action=action,
            planned_xy=planned_xy,
            nearest_xy=(float(nearest[0]), float(nearest[1])),
            snapped=False,
            distance_before_m=before,
            distance_after_m=before,
            inward_m=0.0,
            used_nearest_only=False,
        )
    delta = nearest - action[:2]
    norm = float(np.linalg.norm(delta))
    if norm < 1e-9:
        snapped_xy = nearest.copy()
        used_nearest_only = True
        applied_inward = 0.0
    else:
        direction = delta / norm
        snapped_xy = nearest + float(inward_m) * direction
        after_inward = float(
            np.min(np.linalg.norm(xy - snapped_xy[None, :], axis=1))
        )
        if after_inward > float(on_cloth_m):
            snapped_xy = nearest.copy()
            used_nearest_only = True
            applied_inward = 0.0
        else:
            used_nearest_only = False
            applied_inward = float(inward_m)
    action[:2] = snapped_xy
    after = float(np.min(np.linalg.norm(xy - action[:2][None, :], axis=1)))
    return GraspSnapResult(
        action=action,
        planned_xy=planned_xy,
        nearest_xy=(float(nearest[0]), float(nearest[1])),
        snapped=True,
        distance_before_m=before,
        distance_after_m=after,
        inward_m=applied_inward,
        used_nearest_only=used_nearest_only,
    )
