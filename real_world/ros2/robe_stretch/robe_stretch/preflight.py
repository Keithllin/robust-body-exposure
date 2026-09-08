#!/usr/bin/env python3
"""Reject unreachable BedPull goals. Never clip the policy action."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from .path_setup import add_workstation_code

CODE_DIR = add_workstation_code()

from execution_manifest import write_execution_manifest  # noqa: E402
from stretch_cartesian import StretchWorkspace  # noqa: E402
from stretch_reachability import check_pull_path  # noqa: E402


def _load_action(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        action = payload.get("scaled_action", payload.get("action_bed"))
        return np.asarray(action, dtype=np.float64).reshape(4)
    with path.open("rb") as handle:
        return np.asarray(pickle.load(handle), dtype=np.float64).reshape(4)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--trial-id", type=str, default="")
    parser.add_argument("--pull-z", type=float, default=0.40)
    parser.add_argument("--current-ee-base", type=float, nargs=3, default=[0.25, 0.0, 0.40])
    parser.add_argument("--current-wrist", type=float, default=0.25)
    parser.add_argument("--current-lift", type=float, default=0.40)
    parser.add_argument("--mirror-x", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.mirror_x:
        print(
            "WARN: --mirror-x records planner-side mirroring only; "
            "the executor never remirrors the pickle.",
            file=sys.stderr,
        )
    action = _load_action(args.action)
    workspace = StretchWorkspace()
    # After yaw align the executor works in base_link. For preflight on the
    # workstation we treat canonical XY as already aligned with base XY.
    result = check_pull_path(
        action[:2],
        action[2:],
        pull_z=args.pull_z,
        current_ee_base=np.asarray(args.current_ee_base, dtype=np.float64),
        current_wrist_extension=args.current_wrist,
        current_lift=args.current_lift,
        workspace=workspace,
    )
    print(json.dumps(result.to_dict(), indent=2))
    dest = args.manifest or (args.pose_dir / "execution_manifest.json")
    write_execution_manifest(
        dest,
        trial_id=args.trial_id or args.pose_dir.name,
        planned_action_bed=action.tolist(),
        mirror_x_applied=bool(args.mirror_x),
        reachability=result.to_dict(),
        canonical_frame_path=args.pose_dir / "canonical_bed_frame.json",
        marker_layout_path=CODE_DIR.parent / "calibration" / "marker_layout.json",
        zed_extrinsics_path=CODE_DIR.parent / "calibration" / "zed_extrinsics.json",
        extra={"pull_z": args.pull_z},
    )
    print(f"Wrote {dest}")
    return 0 if result.reachable else 2


if __name__ == "__main__":
    sys.exit(main())
