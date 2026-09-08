#!/usr/bin/env python3
"""Overlay Stretch TF links on a ceiling RGB via Z-down sim_origin.

Uses raw ``T_odom_layout`` from ``stretch_origin.json`` (no ``ensure_layout_z_up``).
Does not write origin, recalibrate, or touch the executor.

Example:
  source real_world/ros2/source_humble.sh
  python3 real_world/code/overlay_stretch_tf_ceiling.py \\
    --trial-dir real_world/STUDY_DATA/subject_smoke/pose_1_TL13_1787929851
  # session/stretch defaults: raw origin, ceiling_now_ee.png, overlay PNG/JSON
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration as RclDuration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from canonical_bed import (  # noqa: E402
    canonical_points_to_layout,
    load_canonical_frame,
)
from marker_utils import detect_markers, invert_transform  # noqa: E402
from overlay_action_on_ceiling import bed_xyz_to_pixel, _load_sim_origin  # noqa: E402
from session_paths import (  # noqa: E402
    print_session_resolved,
    resolve_session_dir_once,
    session_paths,
)
from stretch_localize import transform_to_mat  # noqa: E402

# URDF fixed joint: link_aruco_top_wrist -> link_aruco_top_wrist_big
_T_WRIST_TO_BIG = np.eye(4, dtype=np.float64)
_T_WRIST_TO_BIG[2, 3] = -0.005

REAL_WORLD = CODE_DIR.parent
DEFAULT_TRIAL = (
    REAL_WORLD / "STUDY_DATA" / "subject_smoke" / "pose_1_TL13_1787929851"
)
DEFAULT_EXTRINSICS = REAL_WORLD / "calibration" / "zed_extrinsics.json"


def _stamp_to_mat(msg) -> np.ndarray:
    t = msg.transform.translation
    q = msg.transform.rotation
    return transform_to_mat((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


class TfNode(Node):
    def __init__(self) -> None:
        super().__init__("overlay_stretch_tf_ceiling")
        self._joints: dict[str, float] = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(JointState, "/stretch/joint_states", self._on_js, 10)

    def _on_js(self, msg: JointState) -> None:
        self._joints = dict(zip(msg.name, msg.position))

    def lookup(self, parent: str, child: str) -> np.ndarray:
        deadline = time.monotonic() + 8.0
        last = None
        while time.monotonic() < deadline:
            try:
                stamped = self.tf_buffer.lookup_transform(
                    parent, child, rclpy.time.Time(), timeout=RclDuration(seconds=1.0)
                )
                return _stamp_to_mat(stamped)
            except Exception as exc:  # noqa: BLE001
                last = exc
                rclpy.spin_once(self, timeout_sec=0.2)
        raise RuntimeError(f"TF {parent} -> {child} failed: {last}")


def _detect_136(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (center_uv, corners_uv) for DICT_6X6_250 id 136."""

    found: dict[int, np.ndarray] = {}
    for scale in (1.0, 2.0, 3.0):
        if scale == 1.0:
            frame = image_bgr
        else:
            frame = cv2.resize(
                image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        for mid, corners in detect_markers(frame, "DICT_6X6_250").items():
            if mid not in found:
                found[mid] = corners / scale
    if 136 not in found:
        raise RuntimeError("ArUco 136 not detected in ceiling image")
    corners = found[136]
    return corners.mean(axis=0), corners


def _pose_marker_camera(
    corners: np.ndarray,
    size_m: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    half = float(size_m) / 2.0
    obj = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    ok, rvec, tvec = cv2.solvePnP(
        obj,
        corners.reshape(-1, 1, 2).astype(np.float64),
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed for marker 136")
    rot, _ = cv2.Rodrigues(rvec)
    t_cam_marker = np.eye(4, dtype=np.float64)
    t_cam_marker[:3, :3] = rot
    t_cam_marker[:3, 3] = tvec.reshape(3)
    return t_cam_marker


def _delta(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = np.asarray(a, dtype=np.float64).reshape(3) - np.asarray(
        b, dtype=np.float64
    ).reshape(3)
    return {
        "dx": round(float(d[0]), 4),
        "dy": round(float(d[1]), 4),
        "dz": round(float(d[2]), 4),
        "norm_xy": round(float(np.linalg.norm(d[:2])), 4),
        "norm_xyz": round(float(np.linalg.norm(d)), 4),
    }


def _draw(
    image: np.ndarray,
    uv: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    *,
    marker: int = cv2.MARKER_CROSS,
) -> None:
    xy = (int(round(float(uv[0]))), int(round(float(uv[1]))))
    cv2.drawMarker(image, xy, color, markerType=marker, markerSize=28, thickness=2)
    cv2.circle(image, xy, 10, color, 2)
    cv2.putText(
        image,
        label,
        (xy[0] + 12, xy[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Session dir (default: resolve sessions/current once)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Ceiling RGB; default: session/stretch/ceiling_now_ee.png",
    )
    parser.add_argument(
        "--extrinsics",
        type=Path,
        default=None,
        help="zed_extrinsics.json; default: session/zed then calibration/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Annotated PNG; default: session/stretch/ceiling_stretch_tf_overlay.png",
    )
    parser.add_argument(
        "--origin",
        type=Path,
        default=None,
        help="stretch_origin JSON; default: session raw (never corrected)",
    )
    args = parser.parse_args()
    trial = args.trial_dir.resolve()
    session_dir = resolve_session_dir_once(args.session_dir)
    paths = session_paths(session_dir)
    if args.origin is not None:
        origin_path = args.origin.resolve()
    elif paths.origin_raw.is_file():
        origin_path = paths.origin_raw
    elif paths.origin_json.is_file():
        origin_path = paths.origin_json
    else:
        raise SystemExit(f"missing session raw origin under {paths.stretch_dir}")
    image_path = (args.image or paths.ceiling_now).resolve()
    out_path = (
        args.output or (paths.stretch_dir / "ceiling_stretch_tf_overlay.png")
    ).resolve()
    if args.extrinsics is not None:
        extrinsics_path = args.extrinsics.resolve()
    elif paths.zed_extrinsics.is_file():
        extrinsics_path = paths.zed_extrinsics
    else:
        extrinsics_path = DEFAULT_EXTRINSICS
    print_session_resolved(session_dir, origin_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read {image_path}")

    sim = _load_sim_origin(trial / "sim_origin_data.pkl")
    t_bed_cam_sim = np.asarray(sim["bed_from_camera"], dtype=np.float64)
    k_sim = np.asarray(sim["mtx"], dtype=np.float64)
    dist_sim = np.asarray(sim.get("dist", np.zeros(5)), dtype=np.float64).reshape(-1)

    # Ceiling 136 → bed (Z-down), same frame as sim_origin / zed_extrinsics.
    ext = json.loads(extrinsics_path.read_text())
    ceil = ext["cameras"]["ceiling"]
    k_zed = np.asarray(ceil["camera_matrix"], dtype=np.float64)
    dist_zed = np.asarray(ceil["distortion"], dtype=np.float64)
    t_bed_cam_zed = np.asarray(ceil["T_bed_camera_image"], dtype=np.float64)
    h, w = image.shape[:2]
    calib_wh = ceil.get("image_size_wh")
    if calib_wh is not None and (int(calib_wh[0]), int(calib_wh[1])) != (w, h):
        sx = w / float(calib_wh[0])
        sy = h / float(calib_wh[1])
        k_zed = k_zed.copy()
        k_zed[0, :] *= sx
        k_zed[1, :] *= sy

    det_uv, corners = _detect_136(image)
    t_cam_marker = _pose_marker_camera(corners, 0.1, k_zed, dist_zed)
    t_bed_marker = t_bed_cam_zed @ t_cam_marker
    p_ceil_136 = t_bed_marker[:3, 3].copy()

    origin = json.loads(origin_path.read_text())
    # RAW Z-down layout — do not ensure_layout_z_up for ceiling projection.
    t_ol = np.asarray(origin["T_odom_layout"], dtype=np.float64).reshape(4, 4)
    print(f"origin={origin_path}")
    if float(t_ol[2, 2]) >= 0.0:
        print(
            "NOTE: stretch_origin R[2,2]>=0; still using raw T_odom_layout "
            "(no Z-up flip) for this overlay."
        )

    frame = load_canonical_frame(trial / "canonical_bed_frame.json")
    action_path = trial / "uncover" / "uncover_scaled_action.pkl"
    with action_path.open("rb") as handle:
        action = np.asarray(pickle.load(handle), dtype=np.float64).reshape(4)
    # Policy pick in layout Z-down via canonical→layout (camera bed frame).
    pick_layout = canonical_points_to_layout(
        np.array([[action[0], action[1], 0.08]], dtype=np.float64), frame
    )[0]

    rclpy.init()
    node = TfNode()
    end = time.monotonic() + 4.0
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    t_odom_wrist = node.lookup("odom", "link_aruco_top_wrist")
    # wrist_big not published; apply URDF fixed offset only for this drawing.
    t_odom_wrist_big = t_odom_wrist @ _T_WRIST_TO_BIG
    t_odom_body = node.lookup("odom", "link_gripper_s3_body")
    t_odom_tip_l = node.lookup("odom", "link_gripper_fingertip_left")
    t_odom_tip_r = node.lookup("odom", "link_gripper_fingertip_right")
    t_odom_grasp = node.lookup("odom", "link_grasp_center")

    def odom_to_layout(t_odom_link: np.ndarray) -> np.ndarray:
        return (invert_transform(t_ol) @ np.append(t_odom_link[:3, 3], 1.0))[:3]

    points_layout = {
        "ceil_det_136": p_ceil_136,
        "tf_wrist_big": odom_to_layout(t_odom_wrist_big),
        "tf_gripper_body": odom_to_layout(t_odom_body),
        "tf_tip_L": odom_to_layout(t_odom_tip_l),
        "tf_tip_R": odom_to_layout(t_odom_tip_r),
        "tf_tip_mid": 0.5
        * (odom_to_layout(t_odom_tip_l) + odom_to_layout(t_odom_tip_r)),
        "tf_grasp_center": odom_to_layout(t_odom_grasp),
        "policy_pick": pick_layout,
    }

    def project(p: np.ndarray) -> np.ndarray:
        return bed_xyz_to_pixel(
            np.asarray(p, dtype=np.float64).reshape(1, 3),
            bed_from_camera=t_bed_cam_sim,
            camera_matrix=k_sim,
            distortion=dist_sim,
        )[0]

    pixels = {name: project(xyz) for name, xyz in points_layout.items()}
    # Detection pixel is image measurement, not a reprojected 3D point.
    pixels["ceil_det_136_uv"] = det_uv

    canvas = image.copy()
    cv2.polylines(
        canvas,
        [np.round(corners).astype(np.int32)],
        True,
        (0, 255, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    styles = [
        ("ceil_det_136_uv", "DET136", (0, 255, 0), cv2.MARKER_TILTED_CROSS),
        ("ceil_det_136", "ceil136_3D", (0, 255, 255), cv2.MARKER_CROSS),
        ("tf_wrist_big", "wrist_big", (0, 140, 255), cv2.MARKER_CROSS),
        ("tf_gripper_body", "grip_body", (255, 128, 0), cv2.MARKER_CROSS),
        ("tf_tip_L", "tip_L", (255, 0, 255), cv2.MARKER_CROSS),
        ("tf_tip_R", "tip_R", (200, 0, 200), cv2.MARKER_CROSS),
        ("tf_tip_mid", "tip_mid", (180, 0, 255), cv2.MARKER_DIAMOND),
        ("tf_grasp_center", "grasp_ctr", (255, 255, 0), cv2.MARKER_CROSS),
        ("policy_pick", "policy_pick", (0, 165, 255), cv2.MARKER_STAR),
    ]
    for key, label, color, marker in styles:
        _draw(canvas, pixels[key], color, label, marker=marker)

    for i, (_, label, color, _) in enumerate(styles):
        cv2.putText(
            canvas,
            label,
            (12, 26 + 20 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)

    yaw = float(node._joints.get("joint_wrist_yaw", float("nan")))
    pitch = float(node._joints.get("joint_wrist_pitch", float("nan")))
    wrist_ext = float(
        node._joints.get(
            "wrist_extension",
            sum(node._joints.get(f"joint_arm_l{i}", 0.0) for i in range(4)),
        )
    )

    report = {
        "image": str(image_path),
        "output": str(out_path),
        "origin": str(origin_path),
        "projection": "Z-down raw T_odom_layout + sim_origin bed_from_camera",
        "T_odom_layout_R22": round(float(t_ol[2, 2]), 4),
        "ensure_layout_z_up_applied": False,
        "joints": {
            "wrist_extension": round(wrist_ext, 4),
            "wrist_yaw_deg": round(float(np.degrees(yaw)), 2),
            "wrist_pitch_deg": round(float(np.degrees(pitch)), 2),
        },
        "points_layout_Zdown_m": {
            name: np.round(xyz, 4).tolist() for name, xyz in points_layout.items()
        },
        "pixels": {
            name: [round(float(uv[0]), 1), round(float(uv[1]), 1)]
            for name, uv in pixels.items()
        },
        "deltas_layout_Zdown_m": {
            "tf_wrist_big_minus_ceil136_3D": _delta(
                points_layout["tf_wrist_big"], points_layout["ceil_det_136"]
            ),
            "tf_gripper_body_minus_ceil136_3D": _delta(
                points_layout["tf_gripper_body"], points_layout["ceil_det_136"]
            ),
            "tf_tip_mid_minus_ceil136_3D": _delta(
                points_layout["tf_tip_mid"], points_layout["ceil_det_136"]
            ),
            "tf_grasp_center_minus_ceil136_3D": _delta(
                points_layout["tf_grasp_center"], points_layout["ceil_det_136"]
            ),
            "tf_grasp_center_minus_tf_tip_mid": _delta(
                points_layout["tf_grasp_center"], points_layout["tf_tip_mid"]
            ),
            "tf_tip_L_minus_tf_tip_R": _delta(
                points_layout["tf_tip_L"], points_layout["tf_tip_R"]
            ),
            "policy_pick_minus_tf_grasp_center": _delta(
                points_layout["policy_pick"], points_layout["tf_grasp_center"]
            ),
        },
        "pixel_err_DET136_vs_ceil136_3D_reproj": {
            "du": round(float(pixels["ceil_det_136"][0] - det_uv[0]), 1),
            "dv": round(float(pixels["ceil_det_136"][1] - det_uv[1]), 1),
            "norm_px": round(
                float(np.linalg.norm(pixels["ceil_det_136"] - det_uv)), 1
            ),
        },
        "pixel_err_DET136_vs_tf_wrist_big": {
            "du": round(float(pixels["tf_wrist_big"][0] - det_uv[0]), 1),
            "dv": round(float(pixels["tf_wrist_big"][1] - det_uv[1]), 1),
            "norm_px": round(
                float(np.linalg.norm(pixels["tf_wrist_big"] - det_uv)), 1
            ),
        },
    }

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {json_path}")

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
