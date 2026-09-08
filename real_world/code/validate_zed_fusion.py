#!/usr/bin/env python3
"""Empty-bed / rigid-scene check that 3 ZED clouds agree in layout-bed.

Canonicalization does not prove side/ceiling cloth overlap. This script
compares per-role filtered PCDs (still layout-bed) with a point-to-plane
residual on a fitted mattress plane.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from recover_runtime import _read_pcd_xyz  # noqa: E402


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    return normal / np.linalg.norm(normal), float(np.dot(normal, centroid))


def _point_to_plane(points: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return pts @ normal - offset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--max-median-m", type=float, default=0.02)
    args = parser.parse_args()
    roles = ("ceiling", "side_left", "side_right")
    clouds = {}
    for role in roles:
        path = args.pose_dir / f"pcd_filtered_{role}.pcd"
        if path.is_file():
            clouds[role] = _read_pcd_xyz(path)
    if "ceiling" not in clouds:
        print("Need pcd_filtered_ceiling.pcd", file=sys.stderr)
        return 2
    normal, offset = _fit_plane(clouds["ceiling"])
    report = {
        "ceiling_plane_normal": normal.tolist(),
        "roles": {},
    }
    ok = True
    for role, cloud in clouds.items():
        residual = _point_to_plane(cloud, normal, offset)
        stats = {
            "n": int(len(residual)),
            "median_abs_m": float(np.median(np.abs(residual))),
            "p95_abs_m": float(np.percentile(np.abs(residual), 95)),
        }
        report["roles"][role] = stats
        if role != "ceiling" and stats["median_abs_m"] > args.max_median_m:
            ok = False
    print(json.dumps(report, indent=2))
    if not ok:
        print(
            f"FAIL: a side camera median plane residual exceeded {args.max_median_m} m",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
