#!/usr/bin/env python3
"""Score real-world Recover F1 from initial / intermediate / final PCDs."""

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

from pickle_compat import load_pickle  # noqa: E402
from canonical_bed import canonicalize_body_xy, maybe_load_canonical_frame  # noqa: E402
from trial_layout import ensure_trial_body_info, resolve_body_info  # noqa: E402
from recover_runtime import (  # noqa: E402
    ModelFrameAdapter,
    compute_graph_stats,
    load_recover_model,
    read_pcd_points,
    _radius_edges,
)
from cma_gnn_util import compute_fscore_recover  # noqa: E402


def _load_body_points(
    pose_dir: Path,
    subject_dir: Path,
    manikin: bool,
    target_limb_code: int,
    allow_subject: bool = False,
) -> np.ndarray:
    from assistive_gym.envs.bu_gnn_util import get_body_points_from_obs

    pose_path = (subject_dir if manikin else pose_dir) / "human_pose.pkl"
    body_path = (
        subject_dir / "body_info.pkl"
        if manikin
        else ensure_trial_body_info(
            pose_dir, subject_dir, allow_subject=allow_subject
        )
        or resolve_body_info(pose_dir, subject_dir, allow_subject=allow_subject)
    )
    human_pose = np.asarray(load_pickle(pose_path), dtype=np.float64)
    body_info = load_pickle(body_path)
    body_points = get_body_points_from_obs(
        human_pose=human_pose,
        target_limb_code=target_limb_code,
        body_info=body_info,
    )
    return np.asarray(body_points, dtype=np.float64)


def _covered_status(body_points: np.ndarray, cloth_xy: np.ndarray) -> np.ndarray:
    from assistive_gym.envs.bu_gnn_util import get_covered_status

    return np.asarray(get_covered_status(body_points, cloth_xy))


def score_real_recover_f1(
    *,
    pose_dir: Path,
    subject_dir: Path,
    tl_code: int,
    model_dir: Path,
    initial_pcd: Path,
    intermediate_pcd: Path,
    final_pcd: Path,
    max_points: int = 0,
    manikin: bool = False,
    reuse_subject_body: bool = False,
) -> dict:
    _, config = load_recover_model(model_dir=model_dir, device="cpu")
    adapter = ModelFrameAdapter()
    apply_draping = bool(config.rot_draping)
    states = {}
    graph_panels = {}
    for name, path in (
        ("initial", initial_pcd),
        ("intermediate", intermediate_pcd),
        ("final", final_pcd),
    ):
        _, points_model = read_pcd_points(
            path,
            adapter,
            max_points=max_points,
            voxel_size=config.voxel_size,
            apply_draping=apply_draping,
        )
        xy = points_model[:, :2].copy()
        states[name] = xy
        edge_index = _radius_edges(xy, config.edge_threshold)
        graph_panels[name] = compute_graph_stats(
            xy,
            edge_index,
            voxel_size=config.voxel_size,
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
    final_status = _covered_status(body_points, states["final"])
    f1 = float(
        compute_fscore_recover(
            initial_status,
            intermediate_status,
            final_status,
            False,
        )
    )
    if not math.isfinite(f1):
        f1 = float("nan")
    return {
        "real_recover_f1": f1,
        "voxel_size": (
            None if np.isnan(config.voxel_size) else float(config.voxel_size)
        ),
        "edge_threshold": float(config.edge_threshold),
        "num_nodes": {
            name: int(panel["num_nodes"]) for name, panel in graph_panels.items()
        },
        "graph_stats": graph_panels,
        "model_dir": str(Path(model_dir).resolve()),
        "checkpoint": str(config.checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--tl-code", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--initial-pcd", type=Path, required=True)
    parser.add_argument("--intermediate-pcd", type=Path, required=True)
    parser.add_argument("--final-pcd", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--manikin", type=int, default=0)
    parser.add_argument(
        "--reuse-subject-body",
        action="store_true",
        help="Copy subject_dir/body_info.pkl into this trial. Default is "
        "trial then this exp's poses/pose_n only.",
    )
    args = parser.parse_args()

    result = score_real_recover_f1(
        pose_dir=args.pose_dir.resolve(),
        subject_dir=args.subject_dir.resolve(),
        tl_code=args.tl_code,
        model_dir=args.model_dir.resolve(),
        initial_pcd=args.initial_pcd.resolve(),
        intermediate_pcd=args.intermediate_pcd.resolve(),
        final_pcd=args.final_pcd.resolve(),
        max_points=args.max_points,
        manikin=bool(args.manikin),
        reuse_subject_body=bool(args.reuse_subject_body),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"real_recover_f1={result['real_recover_f1']:.4f}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
