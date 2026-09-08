#!/usr/bin/env python3
"""Write the base/layout snapshot for execution-wrist-down reachability.

Stores ``T_odom_layout`` / ``T_odom_base`` and reconstructs
``ee_base_arm0_wrist_down``. CMA and the executor share this file.
Needs Humble + stretch_driver TF. Does not move the robot — run
``prepare_wrist_down.py`` first so Uncover/Recover CMA start from the
execution gripper pose.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
import sys

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from session_paths import resolve_session_dir_once, session_paths
from stretch_cartesian import (
    PLANNING_SAFE_EXTENSION_M,
    StretchWorkspace,
    ee_base_from_tf,
    execution_ee_from_snapshot,
)
from stretch_limits import SNAPSHOT_NAME, PLANNER_ARM_MAX_M
from stretch_live_tf import tf_looks_zero_pose
from stretch_localize import t_odom_layout_for_motion, transform_to_mat


def _stamp_to_mat(msg) -> np.ndarray:
    t = msg.transform.translation
    q = msg.transform.rotation
    return transform_to_mat((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--save-uncover-home",
        action="store_true",
        help="Record this parking as Uncover start. Session home is written "
        "only if missing; Recover snapshots must not pass this flag.",
    )
    parser.add_argument(
        "--cloth-z",
        type=float,
        default=None,
        help="Layout-frame blanket Z. Default: median of initial/blanket_pcd.pcd",
    )
    parser.add_argument(
        "--planning-wrist-down-prepared",
        type=int,
        default=1,
        help="1 if PREPARE_FOR_PLANNING ran before this snapshot.",
    )
    parser.add_argument(
        "--planning-debug-override",
        type=int,
        default=0,
        help="1 if --skip-prepare-wrist-down was used (not a standard trial).",
    )
    args = parser.parse_args()
    pose = args.pose_dir.resolve()
    out = args.output or (pose / SNAPSHOT_NAME)

    import rclpy
    from rclpy.duration import Duration as RclDuration
    from sensor_msgs.msg import JointState
    from tf2_ros import Buffer, TransformListener

    session = session_paths(resolve_session_dir_once())
    origin = json.loads(session.origin_corrected.read_text())
    t_ol = t_odom_layout_for_motion(
        np.asarray(origin["T_odom_layout"], dtype=np.float64)
    )
    if args.cloth_z is not None:
        cloth_z = float(args.cloth_z)
    else:
        from overlay_action_on_ceiling import _load_blanket_points, _resolve_bed_z
        from trial_layout import resolve_initial_pcd

        pcd = resolve_initial_pcd(pose)
        cloth_z = _resolve_bed_z(None, pcd, _load_blanket_points(pcd))

    rclpy.init()
    node = rclpy.create_node("snapshot_stretch_reach")
    joints: dict[str, float] = {}
    jsp_joints: dict[str, float] = {}

    def _on_js(msg: JointState) -> None:
        joints.update(zip(msg.name, msg.position))

    def _on_jsp(msg: JointState) -> None:
        jsp_joints.update(zip(msg.name, msg.position))

    node.create_subscription(JointState, "/stretch/joint_states", _on_js, 10)
    node.create_subscription(JointState, "/joint_states", _on_jsp, 10)
    buf = Buffer(cache_time=RclDuration(seconds=30.0))
    TransformListener(buf, node)
    deadline = time.monotonic() + 20.0
    next_log = time.monotonic() + 2.0
    t_ob = t_ee = None
    last = None
    live_deadline = None

    def _lookup(parent: str, child: str):
        if not buf.can_transform(
            parent, child, rclpy.time.Time(), timeout=RclDuration(seconds=0.0)
        ):
            return None
        return _stamp_to_mat(
            buf.lookup_transform(parent, child, rclpy.time.Time())
        )

    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        try:
            if t_ob is None:
                t_ob = _lookup("odom", "base_link")
            t_ee_now = _lookup("base_link", "link_grasp_center")
            if t_ee_now is not None:
                t_ee = t_ee_now
            if t_ob is not None and t_ee is not None and joints:
                lift_now = float(joints.get("joint_lift", 0.0))
                ee_z = float(t_ee[2, 3])
                if tf_looks_zero_pose(ee_z_base=ee_z, joint_lift=lift_now):
                    break
                if live_deadline is None:
                    live_deadline = time.monotonic() + 6.0
                    print(
                        "TF EE still looks live (leftover broadcast). "
                        "Waiting 6s for RSP zero + JSP-lag "
                        "(executor no longer broadcasts live TF)..."
                    )
                elif time.monotonic() >= live_deadline:
                    break
        except Exception as exc:  # noqa: BLE001
            last = exc
        if time.monotonic() >= next_log:
            next_log = time.monotonic() + 2.0
            have = []
            if t_ob is not None:
                have.append("odom→base")
            if t_ee is not None:
                have.append("base→EE")
            if joints:
                have.append(f"joints({len(joints)})")
            print(
                "snapshot waiting: "
                + (", ".join(have) if have else "no TF/joints yet")
                + f" last={last}"
            )
    frames = ""
    try:
        frames = buf.all_frames_as_string()
    except Exception:  # noqa: BLE001
        frames = ""
    node.destroy_node()
    rclpy.shutdown()
    if t_ob is None or t_ee is None or not joints:
        hint = (
            "Need stretch_driver with broadcast_odom_tf:=True on DOMAIN 12. "
            "Workstation must see /tf (odom→base_link) and "
            "/stretch/joint_states. Executor alone is not enough."
        )
        odom_seen = "parent odom" in frames or "Frame odom " in frames
        print(f"TF frames (truncated):\n{frames[:1500] or '(none)'}")
        print(
            f"have odom→base={t_ob is not None} "
            f"base→EE={t_ee is not None} "
            f"stretch_joints={sorted(joints)[:8]} "
            f"odom_in_tree={odom_seen}"
        )
        raise SystemExit(f"TF snapshot failed: {last or 'timeout'}; {hint}")

    def _arm(src: dict[str, float]) -> float:
        if "wrist_extension" in src:
            return float(src["wrist_extension"])
        if "joint_arm" in src:
            return float(src["joint_arm"])
        return float(sum(src.get(f"joint_arm_l{i}", 0.0) for i in range(4)))

    wrist = _arm(joints)
    lift = float(joints.get("joint_lift", 0.0))
    pitch = float(joints.get("joint_wrist_pitch", 0.0))
    ee_tf = t_ee[:3, 3]
    live = not tf_looks_zero_pose(ee_z_base=float(ee_tf[2]), joint_lift=lift)
    ee_corr, d_lift, d_arm = ee_base_from_tf(
        ee_tf,
        lift_real=lift,
        arm_real=wrist,
        lift_jsp=float(jsp_joints.get("joint_lift", 0.0)),
        arm_jsp=_arm(jsp_joints) if jsp_joints else 0.0,
        workspace=StretchWorkspace(arm_max_m=PLANNER_ARM_MAX_M),
    )
    payload = {
        "T_odom_layout": t_ol.tolist(),
        "T_odom_base": t_ob.tolist(),
        "ee_base": ee_corr.tolist(),
        "ee_base_tf": ee_tf.tolist(),
        "wrist_extension": wrist,
        "joint_lift": lift,
        "joint_wrist_pitch": pitch,
        "jsp_joint_lift": float(jsp_joints.get("joint_lift", 0.0)),
        "jsp_wrist_extension": _arm(jsp_joints) if jsp_joints else 0.0,
        "tf_joint_delta_lift": float(d_lift),
        "tf_joint_delta_arm": float(d_arm),
        "tf_live": bool(live),
        "ee_source": "execution_wrist_down",
        "cloth_z_layout": float(cloth_z),
        "origin_path": str(session.origin_corrected),
        "pose_dir": str(pose),
        "planner_arm_max_m": PLANNER_ARM_MAX_M,
        "planning_wrist_down_prepared": bool(args.planning_wrist_down_prepared),
        "planning_debug_override": bool(args.planning_debug_override),
        "planning_safe_extension_m": PLANNING_SAFE_EXTENSION_M,
        "standard_experiment_config": bool(args.planning_wrist_down_prepared)
        and not bool(args.planning_debug_override),
    }
    payload["ee_base_arm0_wrist_down"] = execution_ee_from_snapshot(
        payload, StretchWorkspace(arm_max_m=PLANNER_ARM_MAX_M)
    ).tolist()
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    if args.save_uncover_home:
        from uncover_home import record_uncover_home

        wrote = record_uncover_home(out, pose, session.session_dir)
        print(f"uncover home → {wrote}")
    print(
        f"ee_arm0_wrist_down="
        f"{np.asarray(payload['ee_base_arm0_wrist_down']).round(3).tolist()} "
        f"ee_base={np.asarray(payload['ee_base']).round(3).tolist()} "
        f"ee_tf={np.asarray(ee_tf).round(3).tolist()} "
        f"tf_live={live} "
        f"d_lift={d_lift:.3f} d_arm={d_arm:.3f} "
        f"wrist={wrist:.3f} lift={lift:.3f} "
        f"cloth_z={cloth_z:.4f}"
    )
    if live:
        print(
            "TF EE is live; mapping joint_wrist_pitch="
            f"{pitch:.3f} rad to full wrist-down for the execution model. "
            "Do not treat mid-pull leftover pitch as already wrist-down."
        )
    else:
        print("EE source=jsp_lag (URDF-zero TF + lift/arm from /stretch/joint_states)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
