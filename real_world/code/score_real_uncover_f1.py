#!/usr/bin/env python3
"""Score real-world Uncover F1 from initial + captured intermediate PCDs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parents[1]
GET_ACTION_DIR = CODE_DIR / "get_action"
for path in (
    CODE_DIR,
    GET_ACTION_DIR,
    REPO_ROOT / "code",
    REPO_ROOT / "assistive-gym-fem",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from canonical_bed import canonicalize_body_xy, maybe_load_canonical_frame  # noqa: E402
from cma_gnn_util import compute_fscore_uncover  # noqa: E402
from recover_runtime import ModelFrameAdapter, compute_graph_stats, read_pcd_points, _radius_edges  # noqa: E402
from score_real_recover_f1 import _covered_status, _load_body_points  # noqa: E402


def score_real_uncover_f1(
    *,
    pose_dir: Path,
    subject_dir: Path,
    tl_code: int,
    initial_pcd: Path,
    intermediate_pcd: Path,
    voxel_size: float = 0.05,
    rot_draping: bool = True,
    max_points: int = 0,
    manikin: bool = False,
    reuse_subject_body: bool = False,
    model_dir: Path | None = None,
) -> dict:
    adapter = ModelFrameAdapter()
    apply_draping = bool(rot_draping)
    vs = float(voxel_size)
    if model_dir is not None:
        try:
            from recover_runtime import load_recover_model

            _, config = load_recover_model(model_dir=model_dir, device="cpu")
            apply_draping = bool(config.rot_draping)
            if np.isfinite(config.voxel_size) and float(config.voxel_size) > 0:
                vs = float(config.voxel_size)
        except Exception:
            pass
    states = {}
    graph_panels = {}
    for name, path in (
        ("initial", initial_pcd),
        ("intermediate", intermediate_pcd),
    ):
        _, points_model = read_pcd_points(
            path,
            adapter,
            max_points=max_points,
            voxel_size=vs,
            apply_draping=apply_draping,
        )
        xy = points_model[:, :2].copy()
        states[name] = xy
        edge_index = _radius_edges(xy, 0.06)
        graph_panels[name] = compute_graph_stats(
            xy,
            edge_index,
            voxel_size=vs,
            max_points=max_points,
        )

    body_points = canonicalize_body_xy(
        _load_body_points(
            pose_dir,
            subject_dir,
            manikin,
            tl_code,
            allow_subject=reuse_subject_body,
        ),
        maybe_load_canonical_frame(pose_dir),
    )
    initial_status = _covered_status(body_points, states["initial"])
    intermediate_status = _covered_status(body_points, states["intermediate"])
    f1 = float(compute_fscore_uncover(initial_status, intermediate_status))
    if not math.isfinite(f1):
        f1 = float("nan")
    return {
        "real_uncover_f1": f1,
        "voxel_size": vs,
        "rot_draping": apply_draping,
        "num_nodes": {
            name: int(panel["num_nodes"]) for name, panel in graph_panels.items()
        },
        "graph_stats": graph_panels,
        "model_dir": None if model_dir is None else str(Path(model_dir).resolve()),
        "initial_pcd": str(Path(initial_pcd).resolve()),
        "intermediate_pcd": str(Path(intermediate_pcd).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--tl-code", type=int, required=True)
    parser.add_argument("--initial-pcd", type=Path, required=True)
    parser.add_argument("--intermediate-pcd", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--manikin", type=int, default=0)
    parser.add_argument("--reuse-subject-body", action="store_true")
    args = parser.parse_args()

    result = score_real_uncover_f1(
        pose_dir=args.pose_dir.resolve(),
        subject_dir=args.subject_dir.resolve(),
        tl_code=args.tl_code,
        initial_pcd=args.initial_pcd.resolve(),
        intermediate_pcd=args.intermediate_pcd.resolve(),
        voxel_size=args.voxel_size,
        max_points=args.max_points,
        manikin=bool(args.manikin),
        reuse_subject_body=bool(args.reuse_subject_body),
        model_dir=None if args.model_dir is None else args.model_dir.resolve(),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"real_uncover_f1={result['real_uncover_f1']:.4f}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
