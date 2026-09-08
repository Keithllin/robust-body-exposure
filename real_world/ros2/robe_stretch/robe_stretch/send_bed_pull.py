#!/usr/bin/env python3
"""Workstation BedPull client: read scaled_action.pkl and send a ROS2 goal."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def _load_action(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        action = payload.get("scaled_action", payload.get("action_bed"))
        return np.asarray(action, dtype=np.float64).reshape(4)
    with path.open("rb") as handle:
        action = pickle.load(handle)
    return np.asarray(action, dtype=np.float64).reshape(4)


def send_bed_pull_goal(action, trial_id: str = "", frame_id: str = "bed"):
    """Send one /bed_pull goal. Returns (success, failure_stage, message)."""

    import rclpy
    from robe_stretch_interfaces.action import BedPull
    from rclpy.action import ActionClient

    rclpy.init()
    node = rclpy.create_node("bed_pull_client")
    client = ActionClient(node, BedPull, "bed_pull")
    try:
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("BedPull server not available")
        goal = BedPull.Goal()
        goal.grasp_x, goal.grasp_y, goal.release_x, goal.release_y = [
            float(v) for v in action
        ]
        goal.frame_id = frame_id
        goal.trial_id = trial_id
        send = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, send)
        handle = send.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("goal rejected")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        result = result_future.result().result
        return bool(result.success), str(result.failure_stage), str(result.message)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--trial-id", type=str, default="")
    parser.add_argument("--frame-id", type=str, default="bed")
    args = parser.parse_args(argv)
    if args.frame_id != "bed":
        print("frame_id must be bed", file=sys.stderr)
        return 2
    action = _load_action(args.action)
    try:
        from robe_stretch.path_setup import add_workstation_code

        add_workstation_code()
        from session_paths import resolve_session_dir_once, write_active_trial

        write_active_trial(resolve_session_dir_once(), args.action)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN active_trial.json not written: {exc}", file=sys.stderr)
    print(
        f"BedPull goal frame={args.frame_id} trial={args.trial_id} "
        f"action={action.tolist()} (canonical metres, never remirror)"
    )
    try:
        success, stage, message = send_bed_pull_goal(
            action, trial_id=args.trial_id, frame_id=args.frame_id
        )
    except ImportError as exc:
        print(
            f"ROS2 client not available ({exc}). Printed the goal only. "
            "Source the Humble workspace on the workstation to send it.",
            file=sys.stderr,
        )
        return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"success={success} stage={stage} {message}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
