#!/usr/bin/env python3
"""PREPARE_FOR_PLANNING: open gripper, retract to planning-safe extension,
pitch wrist to −90°, settle. Then snapshot/CMA. No base drive, no /bed_pull.

  source real_world/ros2/source_humble.sh
  python3 real_world/code/prepare_wrist_down.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from stretch_cartesian import (  # noqa: E402
    PLANNING_GRIPPER_OPEN,
    PLANNING_SAFE_EXTENSION_M,
    default_execution_wrist,
    parking_wrist_down_failures,
    plan_wrist_down_steps,
    wrist_from_params,
)
from stretch_contact import traj_error_code  # noqa: E402
from stretch_hello_pose import encode_move_to_pose  # noqa: E402

# Same numbers as robe_stretch/config/executor.yaml gripper_down / retract.
_DEFAULT_WRIST = default_execution_wrist()
DEFAULT_YAW = float(_DEFAULT_WRIST.yaw_rad)
DEFAULT_PITCH = float(_DEFAULT_WRIST.pitch_rad)
DEFAULT_ROLL = float(_DEFAULT_WRIST.roll_rad)
DEFAULT_SETTLE_S = 0.5
DISABLE_EFFORT = 100.0
TRAJ_ACTION = "/stretch_controller/follow_joint_trajectory"


def _arm_from_joints(joints: dict[str, float]) -> float:
    if "wrist_extension" in joints:
        return float(joints["wrist_extension"])
    if "joint_arm" in joints:
        return float(joints["joint_arm"])
    return float(sum(joints.get(f"joint_arm_l{i}", 0.0) for i in range(4)))


def _traj_goal(pose: dict, *, contact_off: bool = False):
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    encoded = encode_move_to_pose(
        pose, custom_contact_thresholds=contact_off
    )
    point = JointTrajectoryPoint()
    point.positions = [float(v) for v in encoded["positions"]]
    if encoded["effort"] is not None:
        point.effort = [float(v) for v in encoded["effort"]]
    # Position-mode HelloNode: firmware trapezoid. Wait separately.
    point.time_from_start = Duration(sec=0, nanosec=0)
    goal = FollowJointTrajectory.Goal()
    goal.goal_time_tolerance = Duration(sec=1, nanosec=0)
    goal.trajectory = JointTrajectory()
    goal.trajectory.joint_names = list(encoded["joint_names"])
    goal.trajectory.points = [point]
    return goal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaw", type=float, default=DEFAULT_YAW)
    parser.add_argument("--pitch", type=float, default=DEFAULT_PITCH)
    parser.add_argument("--roll", type=float, default=DEFAULT_ROLL)
    parser.add_argument(
        "--planning-safe-extension-m",
        type=float,
        default=PLANNING_SAFE_EXTENSION_M,
        help="Retract to this parking tuck before wrist-down (not a heuristic).",
    )
    parser.add_argument(
        "--gripper-open",
        type=float,
        default=PLANNING_GRIPPER_OPEN,
        help="Release the gripper to this opening before retract/wrist-down.",
    )
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S)
    parser.add_argument(
        "--skip-if-already-down",
        action="store_true",
        default=True,
        help="No-op when pitch/yaw/roll/arm already match execution parking.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Send the trajectory even if already wrist-down.",
    )
    args = parser.parse_args()

    import rclpy
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger

    rclpy.init()
    node = rclpy.create_node("prepare_wrist_down")
    joints: dict[str, float] = {}

    def _on_js(msg: JointState) -> None:
        joints.update(zip(msg.name, msg.position))

    node.create_subscription(JointState, "/stretch/joint_states", _on_js, 10)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and not joints:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not joints:
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit("no /stretch/joint_states; is stretch_driver up on DOMAIN 12?")

    def _gripper_joint() -> str:
        if "stretch_gripper" in joints:
            return "stretch_gripper"
        return "joint_gripper_finger_left"

    pitch = float(joints.get("joint_wrist_pitch", 0.0))
    yaw = float(joints.get("joint_wrist_yaw", 0.0))
    roll = float(joints.get("joint_wrist_roll", 0.0))
    arm = _arm_from_joints(joints)
    safe = float(args.planning_safe_extension_m)
    print(
        f"current wrist pitch={pitch:.3f} yaw={yaw:.3f} roll={roll:.3f} "
        f"arm={arm:.3f} m  planning_safe_extension={safe:.3f} m"
    )

    switch = node.create_client(Trigger, "/stretch/switch_to_position_mode")
    if switch.wait_for_service(timeout_sec=2.0):
        fut = switch.call_async(Trigger.Request())
        wait = time.monotonic() + 3.0
        while time.monotonic() < wait and not fut.done():
            rclpy.spin_once(node, timeout_sec=0.1)

    client = ActionClient(node, FollowJointTrajectory, TRAJ_ACTION)
    if not client.wait_for_server(timeout_sec=8.0):
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(
            f"{TRAJ_ACTION} has no server — stretch_driver must be in "
            "position mode (broadcast_odom_tf:=True). Do not send /bed_pull."
        )

    def _spin_for(seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

    def _refresh() -> tuple[float, float, float, float]:
        _spin_for(0.4)
        return (
            float(joints.get("joint_wrist_pitch", 0.0)),
            float(joints.get("joint_wrist_yaw", 0.0)),
            float(joints.get("joint_wrist_roll", 0.0)),
            _arm_from_joints(joints),
        )

    def _send(pose: dict, *, duration_s: float, contact_off: bool = False):
        goal = _traj_goal(pose, contact_off=contact_off)
        send = client.send_goal_async(goal)
        wait = time.monotonic() + duration_s + 8.0
        while time.monotonic() < wait and not send.done():
            rclpy.spin_once(node, timeout_sec=0.1)
        handle = send.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f"FollowJointTrajectory rejected: {pose}")
        result_fut = handle.get_result_async()
        wait = time.monotonic() + duration_s + 8.0
        while time.monotonic() < wait and not result_fut.done():
            rclpy.spin_once(node, timeout_sec=0.1)
        wrapped = result_fut.result() if result_fut.done() else None
        result = getattr(wrapped, "result", None)
        code = traj_error_code(result)
        err = getattr(result, "error_string", "") or ""
        if code not in (0, None):
            print(f"  trajectory error_code={code} {err}".rstrip())
        return result

    print(f"release gripper {args.gripper_open:.3f}")
    _send({_gripper_joint(): float(args.gripper_open)}, duration_s=2.0)
    _spin_for(1.2)

    pose = wrist_from_params(
        yaw_rad=float(args.yaw),
        pitch_rad=float(args.pitch),
        roll_rad=float(args.roll),
        planning_safe_extension_m=safe,
    )
    steps = plan_wrist_down_steps(
        pitch_rad=pitch,
        yaw_rad=yaw,
        roll_rad=roll,
        wrist_extension_m=arm,
        pose=pose,
        skip_if_already_down=args.skip_if_already_down and not args.force,
    )
    if not steps:
        print("already execution wrist-down; snapshot/CMA can start")
        node.destroy_node()
        rclpy.shutdown()
        return 0

    for step in steps:
        joints = dict(step.joints)
        if step.contact_off and "joint_arm" in joints:
            joints["joint_arm"] = (float(joints["joint_arm"]), DISABLE_EFFORT)
        print(f"{step.name}: {step.reason or step.joints}")
        _send(
            joints,
            duration_s=step.duration_s,
            contact_off=step.contact_off,
        )
        wait_s = max(1.5, float(args.settle_s) if step.name == "wrist_down" else 2.0)
        if "joint_arm" in step.joints:
            wait_s = max(wait_s, abs(arm - float(step.joints["joint_arm"])) / 0.08)
        _spin_for(wait_s)
        pitch, yaw, roll, arm = _refresh()
        print(
            f"  after {step.name} pitch={pitch:.3f} yaw={yaw:.3f} "
            f"roll={roll:.3f} arm={arm:.3f} m"
        )
    print(
        f"after  wrist pitch={pitch:.3f} yaw={yaw:.3f} roll={roll:.3f} "
        f"arm={arm:.3f} m"
    )
    reasons = parking_wrist_down_failures(
        pitch_rad=pitch,
        yaw_rad=yaw,
        roll_rad=roll,
        wrist_extension_m=arm,
        pose=pose,
    )
    if reasons:
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(
            "wrist did not reach execution down (" + "; ".join(reasons) + ")"
        )
    node.destroy_node()
    rclpy.shutdown()
    print("execution wrist-down ready; snapshot/CMA can start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
