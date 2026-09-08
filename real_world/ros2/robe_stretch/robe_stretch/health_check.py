#!/usr/bin/env python3
"""Stretch 3 health checks: driver, D435i TF, optional 5 cm move, /runstop."""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration


class HealthCheck(Node):
    def __init__(self) -> None:
        super().__init__("robe_stretch_health")
        self.joint_msg = None
        self.create_subscription(
            JointState, "/stretch/joint_states", self._on_js, 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.runstop = self.create_client(SetBool, "/runstop")
        self.traj_mode = self.create_client(
            Trigger, "/stretch/switch_to_position_mode"
        )

    def _on_js(self, msg: JointState) -> None:
        self.joint_msg = msg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--move-5cm",
        action="store_true",
        help="Command a 5 cm translate_mobile_base (requires position mode)",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)
    rclpy.init()
    node = HealthCheck()
    deadline = time.monotonic() + args.timeout
    while node.joint_msg is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    ok = True
    if node.joint_msg is None:
        print("FAIL: no /stretch/joint_states (is stretch_driver up?)")
        ok = False
    else:
        names = list(node.joint_msg.name)
        print(f"OK: joint_states n={len(names)}")
        for required in ("joint_lift", "joint_head_pan"):
            if required not in names:
                print(f"FAIL: missing {required}")
                ok = False
    try:
        node.tf_buffer.lookup_transform(
            "base_link",
            "camera_color_optical_frame",
            rclpy.time.Time(),
            timeout=Duration(seconds=3.0),
        )
        print("OK: TF base_link → camera_color_optical_frame")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: D435i/head TF ({exc})")
        ok = False
    if node.runstop.wait_for_service(timeout_sec=2.0):
        print("OK: /runstop service present (hardware runstop is still primary)")
    else:
        print("FAIL: /runstop missing")
        ok = False
    if args.move_5cm:
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration as MsgDuration

        client = ActionClient(
            node, FollowJointTrajectory, "/stretch_controller/follow_joint_trajectory"
        )
        if not client.wait_for_server(timeout_sec=5.0):
            print("FAIL: FollowJointTrajectory not available")
            ok = False
        else:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = JointTrajectory()
            goal.trajectory.joint_names = ["translate_mobile_base"]
            point = JointTrajectoryPoint()
            point.positions = [0.05]
            point.time_from_start = MsgDuration(sec=3, nanosec=0)
            goal.trajectory.points = [point]
            send = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(node, send, timeout_sec=8.0)
            print("OK: sent 5 cm translate_mobile_base")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
