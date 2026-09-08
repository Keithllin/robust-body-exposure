#!/usr/bin/env python3
"""Ceiling RGB: planned pull path plus live EE on the action overlay.

Orange arrow is the CMA grasp→release. Cyan polyline is the interpolated
pull. Green is current ``link_grasp_center`` projected at its real height
(same as the gripper in the photo — do not flatten to cloth Z). Purple is
the predicted pregrasp after base+arm.

Does not move the robot.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from canonical_bed import (  # noqa: E402
    canonical_points_to_layout,
    layout_points_to_canonical,
    load_canonical_frame,
    maybe_load_canonical_frame,
)
from overlay_action_on_ceiling import (  # noqa: E402
    _load_blanket_points,
    _load_sim_origin,
    _resolve_bed_z,
    bed_xyz_to_pixel,
    render_overlay,
)
from pickle_compat import load_pickle  # noqa: E402
from stretch_cartesian import interpolate_xy_line  # noqa: E402

ORANGE = (0, 165, 255)
CYAN = (255, 220, 0)
GREEN = (0, 220, 0)
PURPLE = (200, 0, 200)


def _project(sim: dict, points_layout: np.ndarray) -> np.ndarray:
    dist = sim.get("dist", None)
    distortion = (
        np.zeros(5, dtype=np.float64)
        if dist is None
        else np.asarray(dist, dtype=np.float64).reshape(-1)
    )
    return bed_xyz_to_pixel(
        points_layout,
        bed_from_camera=sim["bed_from_camera"],
        camera_matrix=sim["mtx"],
        distortion=distortion,
    )


def _draw_polyline(
    image: np.ndarray, pixels: np.ndarray, color: tuple[int, int, int]
) -> None:
    pts = np.round(pixels).astype(np.int32)
    if len(pts) < 2:
        return
    cv2.polylines(image, [pts], False, color, 3, cv2.LINE_AA)
    for xy in pts[1:-1: max(1, len(pts) // 8)]:
        cv2.circle(image, (int(xy[0]), int(xy[1])), 3, color, -1, cv2.LINE_AA)


def _draw_point(
    image: np.ndarray, uv: np.ndarray, color: tuple[int, int, int], label: str
) -> None:
    xy = (int(round(float(uv[0]))), int(round(float(uv[1]))))
    cv2.drawMarker(image, xy, color, cv2.MARKER_TILTED_CROSS, 26, 2)
    cv2.putText(
        image,
        label,
        (xy[0] + 12, xy[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _try_live_ee(
    pose_dir: Path,
    frame,
    sim: dict,
    image: np.ndarray,
    z_ref: float,
    grasp_layout: np.ndarray,
    release_layout: np.ndarray,
) -> dict:
    try:
        import rclpy
        from geometry_msgs.msg import TransformStamped
        from rclpy.duration import Duration as RclDuration
        from sensor_msgs.msg import JointState
        from tf2_ros import Buffer, TransformListener

        from marker_utils import invert_transform
        from session_paths import resolve_session_dir_once, session_paths
        from stretch_cartesian import StretchWorkspace, command_toward_target
        from stretch_localize import t_odom_layout_for_motion, transform_to_mat
    except Exception as exc:  # noqa: BLE001
        print(f"EE TF skipped: {exc}")
        return {"tf": False}

    def stamp_to_mat(msg: TransformStamped) -> np.ndarray:
        t = msg.transform.translation
        q = msg.transform.rotation
        return transform_to_mat((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

    try:
        session_dir = resolve_session_dir_once(None)
        origin_path = session_paths(session_dir).origin_corrected
        t_ol = t_odom_layout_for_motion(
            np.asarray(
                json.loads(origin_path.read_text())["T_odom_layout"],
                dtype=np.float64,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"EE TF skipped (session origin): {exc}")
        return {"tf": False}

    joints: dict[str, float] = {}

    def on_js(msg: JointState) -> None:
        joints.update(dict(zip(msg.name, msg.position)))

    rclpy.init()
    node = rclpy.create_node("overlay_ee_traj")
    node.create_subscription(JointState, "/stretch/joint_states", on_js, 10)
    # Latest-in-buffer, not workstation now(). lookup(..., Time(), timeout)
    # converts 0 to now(); Stretch TF stamps are ~0.7 s ahead, so every
    # retry becomes "extrapolation into the past" and green EE never draws.
    buf = Buffer(cache_time=RclDuration(seconds=30.0))
    TransformListener(buf, node)
    t_ob = t_be = t_og = None
    last = None
    deadline = time.monotonic() + 8.0

    def _latest(parent: str, child: str):
        if not buf.can_transform(
            parent, child, rclpy.time.Time(), timeout=RclDuration(seconds=0.0)
        ):
            return None
        return stamp_to_mat(buf.lookup_transform(parent, child, rclpy.time.Time()))

    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        try:
            got = _latest("odom", "base_link")
            if got is not None:
                t_ob = got
            got = _latest("base_link", "link_grasp_center")
            if got is not None:
                t_be = got
            got = _latest("odom", "link_grasp_center")
            if got is not None:
                t_og = got
            if t_ob is not None and t_be is not None and t_og is not None and joints:
                break
        except Exception as exc:  # noqa: BLE001
            last = exc
    node.destroy_node()
    rclpy.shutdown()
    if t_ob is None or t_be is None or t_og is None:
        print(f"EE TF skipped: {last}")
        return {"tf": False}

    ee_base = t_be[:3, 3].copy()
    # Real 3D grasp-center (odom→link), not cloth-Z. Flattening a 1 m lift
    # through a non-nadir ceiling camera throws the marker to the headboard.
    ee_layout = (invert_transform(t_ol) @ np.append(t_og[:3, 3], 1.0))[:3]
    ee_can = layout_points_to_canonical(ee_layout.reshape(1, 3), frame)[0]
    _draw_point(image, _project(sim, ee_layout.reshape(1, 3))[0], GREEN, "ee now")
    ee_on_cloth = ee_layout.copy()
    ee_on_cloth[2] = z_ref

    wrist = float(
        joints.get("wrist_extension", joints.get("joint_arm", 0.0))
    )
    if wrist == 0.0:
        wrist = float(sum(joints.get(f"joint_arm_l{i}", 0.0) for i in range(4)))
    lift = float(joints.get("joint_lift", ee_base[2]))
    p_g = invert_transform(t_ob) @ (t_ol @ np.append(grasp_layout, 1.0))
    target = p_g[:3].copy()
    target[2] = float(ee_base[2])
    cmd = command_toward_target(
        current_ee_base=ee_base,
        target_ee_base=target,
        current_wrist_extension=wrist,
        current_lift=lift,
        workspace=StretchWorkspace(),
    )
    pred = ee_base.copy()
    pred[0] += cmd.translate_mobile_base
    pred[1] += -(cmd.wrist_extension - wrist)
    pred_layout = (invert_transform(t_ol) @ (t_ob @ np.append(pred, 1.0)))[:3]
    pred_can = layout_points_to_canonical(pred_layout.reshape(1, 3), frame)[0]
    _draw_point(image, _project(sim, pred_layout.reshape(1, 3))[0], PURPLE, "pred grasp")

    approach = np.vstack([ee_on_cloth[:2], grasp_layout[:2]])
    approach_layout = np.column_stack(
        [approach, np.full((2,), z_ref, dtype=np.float64)]
    )
    _draw_polyline(image, _project(sim, approach_layout), GREEN)

    extra_path = np.vstack(
        [
            ee_layout,
            np.append(grasp_layout[:2], z_ref),
            np.append(release_layout[:2], z_ref),
        ]
    )
    return {
        "tf": True,
        "ee_project": "link_grasp_center_3d",
        "ee_canonical_xy": ee_can[:2].tolist(),
        "ee_canonical_z_m": float(ee_can[2]),
        "ee_layout_xyz": ee_layout.tolist(),
        "pred_canonical_xy": pred_can[:2].tolist(),
        "approach_waypoints": extra_path.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", type=str, default="uncover")
    parser.add_argument("--sim-origin", type=Path, default=None)
    parser.add_argument("--blanket-pcd", type=Path, default=None)
    args = parser.parse_args()
    pose = args.pose_dir
    action_path = args.action
    if not action_path.is_file():
        print(f"ERROR: missing action {action_path}", file=sys.stderr)
        return 2
    if not args.image.is_file():
        print(f"ERROR: missing ceiling RGB {args.image}", file=sys.stderr)
        return 2
    sim_path = args.sim_origin or (pose / "sim_origin_data.pkl")
    if not sim_path.is_file():
        print(f"ERROR: missing sim_origin {sim_path}", file=sys.stderr)
        return 2
    action = np.asarray(load_pickle(action_path), dtype=np.float64).reshape(4)
    frame = maybe_load_canonical_frame(pose) or load_canonical_frame(
        pose / "canonical_bed_frame.json"
    )
    sim = _load_sim_origin(sim_path)
    pcd = args.blanket_pcd
    if pcd is None:
        for candidate in (
            args.image.parent / "blanket_pcd.pcd",
            pose / "initial" / "blanket_pcd.pcd",
        ):
            if candidate.is_file():
                pcd = candidate
                break
    z_ref = _resolve_bed_z(None, pcd, points=_load_blanket_points(pcd) if pcd else None)
    grasp_layout = canonical_points_to_layout(
        np.array([[action[0], action[1], z_ref]], dtype=np.float64), frame
    )[0]
    release_layout = canonical_points_to_layout(
        np.array([[action[2], action[3], z_ref]], dtype=np.float64), frame
    )[0]
    waypoints = interpolate_xy_line(
        action[:2], action[2:], speed_m_s=0.05, dt_s=0.08
    )
    path_can = np.column_stack(
        [waypoints, np.full((len(waypoints),), z_ref, dtype=np.float64)]
    )
    path_layout = canonical_points_to_layout(path_can, frame)

    render_overlay(
        args.image,
        sim_path,
        [(args.label, action)],
        args.output,
        title=f"{args.label}  orange=CMA  cyan=pull path  green=ee  purple=pred",
        blanket_pcd=pcd,
        draw_pcd=True,
    )
    canvas = cv2.imread(str(args.output), cv2.IMREAD_COLOR)
    if canvas is None:
        print(f"ERROR: failed to re-read {args.output}", file=sys.stderr)
        return 1
    _draw_polyline(canvas, _project(sim, path_layout), CYAN)
    extra = _try_live_ee(
        pose, frame, sim, canvas, z_ref, grasp_layout, release_layout
    )
    cv2.putText(
        canvas,
        f"path {len(waypoints)} pts  z={z_ref:.3f}m  tf={'yes' if extra.get('tf') else 'no'}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(args.output), canvas):
        print(f"ERROR: failed to write {args.output}", file=sys.stderr)
        return 1
    sidecar = {
        "label": args.label,
        "z_ref_m": z_ref,
        "n_waypoints": int(len(waypoints)),
        "action": action.tolist(),
        "output": str(args.output),
        **extra,
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"Wrote EE/traj overlay: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
