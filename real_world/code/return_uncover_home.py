#!/usr/bin/env python3
"""Drive back to Uncover start parking. translate_mobile_base only.

Each chunk waits until odom shows that increment finished, then
``/stop_the_robot``. Do not stack the next goal on a still-rolling base.
Source Humble first. Does not rotate, does not redo 136.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from session_paths import resolve_session_dir_once
from stretch_hello_pose import encode_move_to_pose
from stretch_localize import transform_to_mat
from uncover_home import (
    DONE_XY_M,
    LATERAL_WARN_M,
    YAW_LIMIT_RAD,
    along_bed_return,
    load_uncover_home,
    resolve_uncover_home,
)

CHUNK_M = 0.25
SPEED_M_S = 0.08
MAX_CHUNKS = 10
SETTLE_XY_M = 0.03


def _stamp_to_mat(msg) -> np.ndarray:
    t = msg.transform.translation
    q = msg.transform.rotation
    return transform_to_mat((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


def _lookup_base(node, buf):
    import rclpy
    from rclpy.duration import Duration as RclDuration

    deadline = time.monotonic() + 5.0
    last = None
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.15)
        try:
            return _stamp_to_mat(
                buf.lookup_transform(
                    "odom",
                    "base_link",
                    rclpy.time.Time(),
                    timeout=RclDuration(seconds=0.3),
                )
            )
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"TF odom→base_link failed: {last}")


def _stop_base(node, client, handle) -> None:
    import rclpy
    from std_srvs.srv import Trigger

    if handle is not None:
        try:
            cancel = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, cancel, timeout_sec=2.0)
        except Exception:  # noqa: BLE001
            pass
    stop = node.create_client(Trigger, "/stop_the_robot")
    if stop.wait_for_service(timeout_sec=1.5):
        fut = stop.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, fut, timeout_sec=2.0)
    time.sleep(0.4)


def _send_translate(node, d_base: float):
    from builtin_interfaces.msg import Duration as MsgDuration
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from std_srvs.srv import Trigger
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    import rclpy

    mode = node.create_client(Trigger, "/stretch/switch_to_position_mode")
    if mode.wait_for_service(timeout_sec=3.0):
        fut = mode.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
    client = ActionClient(
        node, FollowJointTrajectory, "/stretch_controller/follow_joint_trajectory"
    )
    if not client.wait_for_server(timeout_sec=8.0):
        raise RuntimeError("FollowJointTrajectory not available")
    encoded = encode_move_to_pose({"translate_mobile_base": float(d_base)})
    duration = float(max(3.0, min(16.0, abs(d_base) / SPEED_M_S)))
    point = JointTrajectoryPoint()
    point.positions = [float(v) for v in encoded["positions"]]
    sec = int(duration)
    nsec = int(round((duration - sec) * 1e9))
    point.time_from_start = MsgDuration(sec=sec, nanosec=max(0, nsec))
    goal = FollowJointTrajectory.Goal()
    goal.goal_time_tolerance = MsgDuration(sec=1, nanosec=0)
    goal.trajectory = JointTrajectory()
    goal.trajectory.joint_names = list(encoded["joint_names"])
    goal.trajectory.points = [point]
    send = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send, timeout_sec=6.0)
    handle = send.result()
    if handle is None or not handle.accepted:
        raise RuntimeError("translate_mobile_base goal rejected")
    return client, handle, duration


def _wait_chunk(node, buf, t_before: np.ndarray, step: float, duration: float) -> float:
    """Block until odom traveled ~step, or motion dies. Returns signed travel."""

    import rclpy

    x_axis = t_before[:2, 0]
    x_axis = x_axis / float(np.linalg.norm(x_axis))
    deadline = time.monotonic() + duration + 4.0
    last = 0.0
    still = 0
    moved = 0.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        t_now = _lookup_base(node, buf)
        moved = float((t_now[:2, 3] - t_before[:2, 3]) @ x_axis)
        if abs(moved - step) <= SETTLE_XY_M:
            time.sleep(0.25)
            t_now = _lookup_base(node, buf)
            return float((t_now[:2, 3] - t_before[:2, 3]) @ x_axis)
        if abs(moved - last) < 0.008:
            still += 1
        else:
            still = 0
        last = moved
        if still >= 10 and abs(moved) > 0.04:
            return moved
        time.sleep(0.1)
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--home", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    session = resolve_session_dir_once()
    home_path = args.home or resolve_uncover_home(args.pose_dir, session)
    if home_path is None:
        print("No uncover_home.json (session or trial). Skip return.", file=sys.stderr)
        return 2
    home = load_uncover_home(home_path)
    t_home = np.asarray(home["T_odom_base"], dtype=np.float64)

    import rclpy
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = rclpy.create_node("return_uncover_home")
    buf = Buffer()
    TransformListener(buf, node)
    client = handle = None
    try:
        t0 = _lookup_base(node, buf)
        first = along_bed_return(t0, t_home)
        start_sign = np.sign(first["translate_mobile_base"]) or 1.0
        travel_cap = float(first["xy_err_m"] + 0.08)
        traveled = 0.0
        print(
            f"home={home_path} start_xy_err={first['xy_err_m']:.3f} "
            f"cap={travel_cap:.3f} m (stop and reassess each chunk)"
        )
        for i in range(MAX_CHUNKS):
            t_now = _lookup_base(node, buf)
            cmd = along_bed_return(t_now, t_home)
            print(
                f"return[{i}] xy_err={cmd['xy_err_m']:.3f} "
                f"d_base={cmd['translate_mobile_base']:+.3f} "
                f"lateral={cmd['lateral_m']:.3f} "
                f"yaw={np.degrees(cmd['yaw_err_rad']):.1f} deg "
                f"traveled={traveled:.3f}"
            )
            if abs(cmd["yaw_err_rad"]) > YAW_LIMIT_RAD:
                _stop_base(node, client, handle)
                raise SystemExit(
                    f"REJECT return: yaw {np.degrees(cmd['yaw_err_rad']):.1f} deg. "
                    "Park parallel; do not rotate_mobile_base."
                )
            if cmd["lateral_m"] > LATERAL_WARN_M:
                print(
                    f"WARN: {cmd['lateral_m']*100:.1f} cm off the along-bed line."
                )
            if cmd["xy_err_m"] <= DONE_XY_M:
                _stop_base(node, client, handle)
                print(f"At Uncover home ({cmd['xy_err_m']*100:.1f} cm).")
                return 0
            if start_sign * cmd["translate_mobile_base"] < 0:
                _stop_base(node, client, handle)
                print(
                    f"Overshot home (sign flipped). Stop. "
                    f"xy_err={cmd['xy_err_m']:.3f} m"
                )
                return 0
            if traveled > travel_cap:
                _stop_base(node, client, handle)
                raise SystemExit(
                    f"STOP: traveled {traveled:.3f} m past cap {travel_cap:.3f}. "
                    "Hit runstop if it is still rolling."
                )
            step = float(
                np.clip(cmd["translate_mobile_base"], -CHUNK_M, CHUNK_M)
            )
            if abs(step) < 0.005:
                _stop_base(node, client, handle)
                print("Along-track leftover < 5 mm.")
                return 0
            if args.dry_run:
                print(f"dry-run would translate_mobile_base={step:+.3f} m")
                return 0
            print(f"translate_mobile_base={step:+.3f} m (wait for odom, then stop)")
            t_before = t_now
            client, handle, duration = _send_translate(node, step)
            moved = _wait_chunk(node, buf, t_before, step, duration)
            _stop_base(node, client, handle)
            handle = None
            traveled += abs(moved)
            print(f"  chunk odom moved {moved:+.3f} m (cmd {step:+.3f})")
            if abs(moved) < 0.02:
                leftover = along_bed_return(_lookup_base(node, buf), t_home)
                if leftover["xy_err_m"] <= max(DONE_XY_M, 0.05):
                    print(
                        f"At Uncover home ({leftover['xy_err_m']*100:.1f} cm); "
                        "last short translate did not move (driver deadband)."
                    )
                    return 0
                raise SystemExit(
                    "Base did not move this chunk. Not stacking another goal."
                )
        t_now = _lookup_base(node, buf)
        cmd = along_bed_return(t_now, t_home)
        _stop_base(node, client, handle)
        if cmd["xy_err_m"] > DONE_XY_M:
            raise SystemExit(
                f"RETURN incomplete xy_err={cmd['xy_err_m']:.3f} m. "
                "Do not start the next Uncover CMA."
            )
        print(f"Returned to Uncover home xy_err={cmd['xy_err_m']:.3f} m")
        return 0
    finally:
        try:
            _stop_base(node, client, handle)
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
