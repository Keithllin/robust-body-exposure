#!/usr/bin/env python3
"""Workstation preflight: whole-path reachability, never clip, write manifest."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from blanket_grasp import (  # noqa: E402
    DEFAULT_MAX_GRASP_DISTANCE_M,
    DEFAULT_SNAP_INWARD_M,
    nearest_grasp_xy_distance,
    read_pcd_xyz,
    snap_grasp_inward,
)
from execution_manifest import write_execution_manifest  # noqa: E402
from stretch_limits import (  # noqa: E402
    HARDWARE_ARM_MAX_M,
    PLANNER_ARM_MAX_M,
    load_cma_reach,
    planner_workspace,
    snapshot_path,
)
from stretch_reachability import check_bed_frame_pull  # noqa: E402


def _load_action(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        action = payload.get("scaled_action", payload.get("action_bed"))
        return np.asarray(action, dtype=np.float64).reshape(4)
    with path.open("rb") as handle:
        return np.asarray(pickle.load(handle), dtype=np.float64).reshape(4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--trial-id", type=str, default="")
    parser.add_argument("--pull-z", type=float, default=0.40)
    parser.add_argument(
        "--current-ee-base",
        type=float,
        nargs=3,
        default=(0.25, 0.0, 0.40),
    )
    parser.add_argument("--current-wrist", type=float, default=0.25)
    parser.add_argument("--current-lift", type=float, default=0.40)
    parser.add_argument("--mirror-x", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--blanket-pcd",
        type=Path,
        default=None,
        help="Measured PCD for hard execution-frame grasp-on-cloth rejection.",
    )
    parser.add_argument(
        "--max-grasp-cloth-distance-m",
        type=float,
        default=DEFAULT_MAX_GRASP_DISTANCE_M,
        help="On-cloth distance. Recover snaps instead of rejecting when "
        "--snap-off-cloth is set.",
    )
    parser.add_argument(
        "--snap-off-cloth",
        action="store_true",
        help="Recover: snap an off-cloth grasp to the nearest point plus "
        "an inward margin instead of rejecting.",
    )
    parser.add_argument(
        "--write-snapped",
        action="store_true",
        help="Overwrite --action with the snapped 4-vector.",
    )
    parser.add_argument(
        "--snap-inward-m",
        type=float,
        default=DEFAULT_SNAP_INWARD_M,
        help="Inward margin past the nearest cloth point (default 0.015 m).",
    )
    parser.add_argument(
        "--stretch-reach-snapshot",
        type=Path,
        default=None,
        help="Parking snapshot. Default: pose_dir/stretch_reach_snapshot.json",
    )
    parser.add_argument(
        "--arm-max-m",
        type=float,
        default=PLANNER_ARM_MAX_M,
        help="Planner EE/arm cap used with the snapshot (default 0.50).",
    )
    args = parser.parse_args()
    if args.mirror_x:
        print(
            "WARN: --mirror-x records planner-side mirroring only; "
            "the executor never remirrors the pickle.",
            file=sys.stderr,
        )
    action = _load_action(args.action)
    snap = args.stretch_reach_snapshot or snapshot_path(args.pose_dir)
    workspace = planner_workspace(args.arm_max_m)
    extra = {"pull_z": args.pull_z, "frame_id": "bed"}
    cloth_ok = True
    if args.blanket_pcd is not None:
        points = read_pcd_xyz(args.blanket_pcd)
        grasp_distance = nearest_grasp_xy_distance(action, points)
        threshold = float(args.max_grasp_cloth_distance_m)
        cloth_ok = grasp_distance <= threshold
        extra["grasp_cloth_check"] = {
            "ok": bool(cloth_ok),
            "snapped": False,
            "frame": "bed_execution",
            "blanket_pcd": str(args.blanket_pcd),
            "nearest_xy_distance_m": grasp_distance,
            "max_distance_m": threshold,
        }
        print(
            "Execution-frame grasp-on-cloth: "
            f"nearest={grasp_distance * 100.0:.1f} cm "
            f"limit={threshold * 100.0:.1f} cm "
            f"{'PASS' if cloth_ok else 'OFF_CLOTH'}"
        )
        if not cloth_ok and args.snap_off_cloth:
            grasp_snap = snap_grasp_inward(
                action,
                points,
                on_cloth_m=threshold,
                inward_m=float(args.snap_inward_m),
            )
            action = np.asarray(grasp_snap.action, dtype=np.float64)
            grasp_distance = grasp_snap.distance_after_m
            cloth_ok = grasp_distance <= threshold
            extra["grasp_cloth_check"].update(grasp_snap.to_dict())
            extra["grasp_cloth_check"]["ok"] = bool(cloth_ok)
            extra["grasp_cloth_check"]["nearest_xy_distance_m"] = grasp_distance
            print(
                "SNAP recover grasp onto cloth: "
                f"planned {grasp_snap.distance_before_m * 100.0:.1f} cm away → "
                f"nearest + {grasp_snap.inward_m * 100.0:.1f} cm inward "
                f"(after {grasp_snap.distance_after_m * 100.0:.1f} cm) "
                f"{'PASS' if cloth_ok else 'REJECT'}"
            )
            if args.write_snapped and grasp_snap.snapped:
                dest = Path(args.action)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as handle:
                    pickle.dump(action, handle, protocol=2)
                print(f"Wrote snapped action {dest}")
    if snap.is_file():
        reach_ctx = load_cma_reach(
            pose_dir=args.pose_dir,
            snapshot_path_or_none=snap,
            arm_max_m=float(args.arm_max_m),
        )
        result = reach_ctx.check(action)
        extra["stretch_reach_snapshot"] = str(snap)
        extra["arm_max_m"] = float(args.arm_max_m)
        extra["arm_hardware_max"] = HARDWARE_ARM_MAX_M
        extra["arm_planning_max"] = float(args.arm_max_m)
        extra["trajectory_max"] = result.max_wrist_extension
        extra["reserve"] = (
            None
            if result.max_wrist_extension is None
            else float(args.arm_max_m) - float(result.max_wrist_extension)
        )
        extra["geometry"] = "execution_wrist_down"
        extra["check"] = "execution_wrist_down"
        print(
            f"Preflight execution_wrist_down {snap} "
            f"planning_max={args.arm_max_m:.3f} "
            f"trajectory_max={result.max_wrist_extension}"
        )
    else:
        result = check_bed_frame_pull(
            action[:2],
            action[2:],
            workspace=workspace,
        )
        extra["check"] = "bed_frame_stroke"
        print("Preflight using bed-frame stroke (no parking snapshot)")
    print(json.dumps(result.to_dict(), indent=2))
    dest = args.manifest or (args.pose_dir / "execution_manifest.json")
    calib = CODE_DIR.parent / "calibration"
    write_execution_manifest(
        dest,
        trial_id=args.trial_id or args.pose_dir.name,
        planned_action_bed=action.tolist(),
        mirror_x_applied=bool(args.mirror_x),
        reachability=result.to_dict(),
        canonical_frame_path=args.pose_dir / "canonical_bed_frame.json",
        marker_layout_path=calib / "marker_layout.json",
        zed_extrinsics_path=calib / "zed_extrinsics.json",
        extra=extra,
    )
    print(f"Wrote {dest}")
    if not result.reachable:
        print("REJECT: do not clip; replan CMA bounds or park the robot.", file=sys.stderr)
        return 2
    if not cloth_ok:
        print(
            "REJECT: exact executed grasp is outside the measured blanket PCD. "
            "Do not send /bed_pull; replan.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
