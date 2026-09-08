#!/usr/bin/env python3
"""Live D435i bed-tag preview with head pan/tilt. Does not pull.

Must run on the Stretch desktop (DISPLAY). D435i pixels stay on the robot;
do not run this on RCHI. After save, copy stretch_origin.json / 
head_sweep_range.json to the trial dir. PnP uses DICT_5X5_100 bed posts.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as MsgDuration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclDuration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from . import bed_origin
from .camera_stream import ColorCamera
from .camera_tf import t_odom_camera_from_head_joints
from .head_sweep import resolve_pan_angles, sweep_head_views
from .hello_motion import follow_joint_trajectory_goal
from .path_setup import add_workstation_code, require_stretch_for_d435i
from .sample_origin import _stamp_to_mat

add_workstation_code()

from canonical_bed import physical_posts  # noqa: E402
from marker_utils import DEFAULT_LAYOUT_PATH, detect_markers, draw_detections  # noqa: E402
from stretch_localize import marker_pixel_sizes, mat_to_quat_xyzw  # noqa: E402

EXPECTED = (0, 1, 2, 3, 10, 11, 12, 13)
# Match Stretch URDF / stretch_driver, not a symmetric ±80° clip.
# Hardware pan is ~-234°..+112°; this robot's URDF is -3.9..1.5 rad.
PAN_LIMIT = (-3.9, 1.5)
TILT_LIMIT = (-1.53, 0.79)
# Official keyboard_teleop default is medium; head keys use 2× that.
# HeadPanCommandGroup.acceptable_joint_error is 0.15 rad (~8.6°), so
# 6° (small) is treated as already there and the neck does not move.
STEP = 2.0 * math.radians(6.0)
JOG_MIN_PERIOD_S = 0.22
START_KEYS = {ord("["), ord("{"), ord("1")}
END_KEYS = {ord("]"), ord("}"), ord("2"), ord(".")}


class PreviewOrigin(Node):
    def __init__(self) -> None:
        super().__init__("preview_origin")
        self.cam = ColorCamera(self)
        self.image = None
        self.camera_matrix = None
        self.distortion = np.zeros(5)
        self.pan = 0.0
        self.tilt = -0.6
        self.have_joints = False
        self._last_jog_t = 0.0
        self.running = True
        self.rotate_k = 1
        self.last_save_note = ""
        self.joint_state = JointState()
        self.robot_mode = "position"
        self._kb = None
        self._stretch_js = False
        self.declare_parameter("min_posts", 3)
        self.declare_parameter("localization.min_posts", 3)
        self.declare_parameter("localization.min_posts_per_view", 1)
        self.declare_parameter("localization.min_pixel_size", 20.0)
        self.declare_parameter("localization.settle_s", 2.0)
        self.declare_parameter("localization.dwell_s", 2.5)
        self.declare_parameter("localization.frames_per_view", 8)
        self.declare_parameter("localization.reprojection_rms_max_px", 12.0)
        self.declare_parameter("localization.translation_std_max_m", 0.08)
        self.declare_parameter("localization.rotation_max_deg", 2.0)
        self.declare_parameter("localization.origin_spread_max_m", 0.12)
        self.declare_parameter("localization.head_tilt", -0.15)
        self.declare_parameter("localization.pan_start", -1.05)
        self.declare_parameter("localization.pan_end", -2.59)
        self.declare_parameter("localization.n_stops", 2)
        self.declare_parameter("output_dir", "/tmp/robe")
        self.declare_parameter("layout_path", str(DEFAULT_LAYOUT_PATH))
        self.declare_parameter("frames.odom", "odom")
        self.declare_parameter("frames.base", "base_link")
        self.declare_parameter("frames.camera_optical", "camera_color_optical_frame")
        self.declare_parameter("frames.head_pan", "link_head_pan")
        self.declare_parameter("frames.head_tilt", "link_head_tilt")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.traj = ActionClient(
            self, FollowJointTrajectory, "/stretch_controller/follow_joint_trajectory"
        )
        self.trajectory_client = self.traj
        self.create_subscription(
            JointState, "/stretch/joint_states", self._on_stretch_joints, 1
        )
        self.create_subscription(JointState, "/joint_states", self._on_joints, 1)
        self.create_subscription(String, "/mode", self._on_mode, 10)
        self._pos_mode = self.create_client(Trigger, "/stretch/switch_to_position_mode")
        try:
            from stretch_core.keyboard import KBHit

            self._kb = KBHit()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"stretch_core.keyboard.KBHit unavailable: {exc}")
        tilt0 = float(self.get_parameter("localization.head_tilt").value)
        self.range_start = float(self.get_parameter("localization.pan_start").value)
        self.range_end = float(self.get_parameter("localization.pan_end").value)
        self.start_pose = {"recorded": False, "pan": self.range_start, "tilt": tilt0}
        self.end_pose = {"recorded": False, "pan": self.range_end, "tilt": tilt0}

    def _sync_cam(self) -> None:
        self.image = self.cam.bgr
        self.camera_matrix = self.cam.camera_matrix
        self.distortion = self.cam.distortion

    def _apply_joints(self, msg: JointState) -> None:
        names = list(msg.name)
        if "joint_head_pan" not in names or "joint_head_tilt" not in names:
            return
        self.joint_state = msg
        self.pan = float(msg.position[names.index("joint_head_pan")])
        self.tilt = float(msg.position[names.index("joint_head_tilt")])
        self.have_joints = True

    def _on_stretch_joints(self, msg: JointState) -> None:
        self._stretch_js = True
        self._apply_joints(msg)

    def _on_joints(self, msg: JointState) -> None:
        if self._stretch_js:
            return
        self._apply_joints(msg)

    def _on_mode(self, msg: String) -> None:
        self.robot_mode = str(msg.data)

    def restore_terminal(self) -> None:
        if self._kb is not None:
            try:
                self._kb.set_normal_term()
            except Exception:
                pass

    def switch_to_position_mode(self) -> None:
        if not self._pos_mode.wait_for_service(timeout_sec=2.0):
            self.last_save_note = "no /stretch/switch_to_position_mode"
            return
        self._pos_mode.call_async(Trigger.Request())

    def _head_command(self, key_char: str):
        """Preview pan signs: j = −pan, l = +pan (opposite official teleop)."""

        delta = STEP
        if key_char in "jJ":
            return {"joint": "joint_head_pan", "delta": -delta}
        if key_char in "lL":
            return {"joint": "joint_head_pan", "delta": delta}
        if key_char in "iI":
            return {"joint": "joint_head_tilt", "delta": delta}
        if key_char in "kK,":
            return {"joint": "joint_head_tilt", "delta": -delta}
        return None

    def send_command(self, command) -> None:
        """Copied from stretch_core.keyboard_teleop.KeyboardTeleopNode.send_command."""

        joint_state = self.joint_state
        if joint_state is None or command is None:
            return
        if time.monotonic() - self._last_jog_t < JOG_MIN_PERIOD_S:
            return
        if self.robot_mode != "position":
            self.last_save_note = f"mode={self.robot_mode}; need position"
            return
        if not joint_state.name:
            self.last_save_note = "no /stretch/joint_states yet"
            return
        joint_name = command["joint"]
        if joint_name not in list(joint_state.name):
            self.last_save_note = f"{joint_name} missing in joint_states"
            return
        joint_index = list(joint_state.name).index(joint_name)
        joint_value = float(joint_state.position[joint_index])
        new_value = joint_value + float(command["delta"])
        lo, hi = PAN_LIMIT if "pan" in joint_name else TILT_LIMIT
        new_value = float(np.clip(new_value, lo, hi))
        point = JointTrajectoryPoint()
        point.time_from_start = MsgDuration(sec=0, nanosec=0)
        trajectory_goal = FollowJointTrajectory.Goal()
        trajectory_goal.trajectory.joint_names = [joint_name]
        point.positions = [new_value]
        trajectory_goal.trajectory.points = [point]
        if not self.trajectory_client.server_is_ready():
            self.last_save_note = "trajectory server not ready"
            return
        send = self.trajectory_client.send_goal_async(trajectory_goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=1.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            self.last_save_note = f"teleop {joint_name} rejected"
            return
        result_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_fut, timeout_sec=4.0)
        wrapped = result_fut.result()
        result = wrapped.result if wrapped is not None else None
        code = getattr(result, "error_code", None)
        if code not in (0, None):
            self.last_save_note = f"teleop {joint_name} error {code}"
            return
        self._last_jog_t = time.monotonic()
        self.last_save_note = (
            f"teleop {joint_name} {math.degrees(command['delta']):+.1f} deg"
        )

    def poll_terminal_keys(self) -> None:
        """Official teleop reads the launch terminal via KBHit, not the GUI."""

        if self._kb is None or not self._kb.kbhit():
            return
        ch = self._kb.getch()
        if ch in ("q", "Q"):
            self.running = False
            return
        command = self._head_command(ch)
        if command is not None:
            self.send_command(command)
        elif ch in "12[]":
            dummy = self.cam.bgr
            if dummy is not None:
                self.handle_key(ord(ch), dummy)

    def _jog_head(self, joint_name: str, delta: float) -> None:
        self.send_command({"joint": joint_name, "delta": delta})

    def _send_head(self, *, wait: bool = False, duration_s: float = 2.5) -> None:
        """Absolute pan+tilt for the recorded sweep, not for key jogging."""

        if not self.have_joints:
            self.last_save_note = "head joints not ready; wait for /stretch/joint_states"
            return
        self.pan = float(np.clip(self.pan, PAN_LIMIT[0], PAN_LIMIT[1]))
        self.tilt = float(np.clip(self.tilt, TILT_LIMIT[0], TILT_LIMIT[1]))
        if not self.traj.wait_for_server(timeout_sec=0.1):
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = ["joint_head_pan", "joint_head_tilt"]
        point = JointTrajectoryPoint()
        point.positions = [self.pan, self.tilt]
        sec = int(duration_s)
        nsec = int(round((duration_s - sec) * 1e9))
        point.time_from_start = MsgDuration(sec=sec, nanosec=nsec)
        goal.trajectory.points = [point]
        send = self.traj.send_goal_async(goal)
        if not wait:
            return
        rclpy.spin_until_future_complete(self, send, timeout_sec=duration_s + 4.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            return
        rclpy.spin_until_future_complete(
            self, handle.get_result_async(), timeout_sec=duration_s + 4.0
        )

    def _spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _range_path(self) -> Path:
        out_dir = Path(str(self.get_parameter("output_dir").value))
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / "head_sweep_range.json"

    def _lookup_mat(self, parent: str, child: str) -> np.ndarray:
        stamped = self.tf_buffer.lookup_transform(
            parent, child, rclpy.time.Time(), timeout=RclDuration(seconds=0.5)
        )
        return _stamp_to_mat(stamped)

    def _t_odom_camera(self, pan: float) -> tuple[np.ndarray, dict]:
        odom = str(self.get_parameter("frames.odom").value)
        base = str(self.get_parameter("frames.base").value)
        camera = str(self.get_parameter("frames.camera_optical").value)
        t_odom_base = self._lookup_mat(odom, base)
        t_base_cam = self._lookup_mat(base, camera)
        t_base_pan = None
        t_base_tilt = None
        try:
            t_base_pan = self._lookup_mat(
                base, str(self.get_parameter("frames.head_pan").value)
            )
            t_base_tilt = self._lookup_mat(
                base, str(self.get_parameter("frames.head_tilt").value)
            )
        except Exception:  # noqa: BLE001
            pass
        t_odom_cam, info = t_odom_camera_from_head_joints(
            t_odom_base,
            t_base_cam,
            pan,
            joint_tilt=float(self.tilt),
            t_base_pan0=t_base_pan,
            t_base_tilt0=t_base_tilt,
        )
        if info["method"] != "tf_live":
            self.get_logger().warn(
                f"TF look {info['yaw_tf_rad']:+.2f} != joint pan "
                f"{info['joint_pan_rad']:+.2f}; {info['method']} "
                f"tilt={info['joint_tilt_rad']:+.2f}"
            )
        return t_odom_cam, info

    def _snapshot_camera(self) -> dict:
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.02)
        pose: dict = {
            "recorded": True,
            "pan": float(self.pan),
            "tilt": float(self.tilt),
            "parent": str(self.get_parameter("frames.odom").value),
            "child": str(self.get_parameter("frames.camera_optical").value),
        }
        try:
            mat, info = self._t_odom_camera(float(self.pan))
            xyz = mat[:3, 3]
            qx, qy, qz, qw = mat_to_quat_xyzw(mat[:3, :3])
            pose.update(
                {
                    "xyz_m": [float(v) for v in xyz],
                    "quat_xyzw": [qx, qy, qz, qw],
                    "T_odom_camera": mat.tolist(),
                    "tf_look_yaw_rad": info["yaw_tf_rad"],
                    "tf_look_fk_rad": info["look_fk_rad"],
                    "tf_method": info["method"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            pose["tf_error"] = str(exc)
        try:
            t_base_cam = self._lookup_mat(
                str(self.get_parameter("frames.base").value),
                pose["child"],
            )
            pose["T_base_camera"] = t_base_cam.tolist()
            pose["xyz_in_base_m"] = [float(v) for v in t_base_cam[:3, 3]]
        except Exception:  # noqa: BLE001
            pass
        return pose

    def _persist_range(self) -> Path:
        path = self._range_path()
        payload = {
            "start": self.start_pose,
            "end": self.end_pose,
            "pan_start": float(self.start_pose["pan"]),
            "pan_end": float(self.end_pose["pan"]),
            "n_stops": int(self.get_parameter("localization.n_stops").value),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def _capture_endpoint(self, which: str, image_bgr: np.ndarray) -> None:
        pose = self._snapshot_camera()
        if which == "start":
            self.start_pose = pose
            self.range_start = float(pose["pan"])
            stem = "head_range_start"
        else:
            self.end_pose = pose
            self.range_end = float(pose["pan"])
            stem = "head_range_end"
        path = self._persist_range()
        cv2.imwrite(str(path.parent / f"{stem}.png"), image_bgr)
        xyz = pose.get("xyz_m")
        xyz_txt = (
            "  xyz " + " ".join(f"{v:.3f}" for v in xyz) if xyz else "  TF missing"
        )
        method = pose.get("tf_method")
        yaw_txt = ""
        if method:
            yaw_txt = (
                f"  look={float(pose.get('tf_look_yaw_rad', 0.0)):+.2f}"
                f"→{float(pose.get('tf_look_fk_rad', 0.0)):+.2f} {method}"
            )
        self.last_save_note = (
            f"{which.upper()} pan={pose['pan']:+.3f} tilt={pose['tilt']:+.3f}"
            f"{xyz_txt}{yaw_txt}  -> {path.name}"
        )
        print(self.last_save_note, flush=True)
        self.get_logger().info(self.last_save_note)

    def _pose_hud(self, pose: dict, label: str) -> str:
        mark = "REC" if pose.get("recorded") else "yaml"
        xyz = pose.get("xyz_m")
        xyz_txt = "  xyz " + " ".join(f"{v:.2f}" for v in xyz) if xyz else ""
        return (
            f"{label} [{mark}] pan {float(pose['pan']):+.2f}  "
            f"tilt {float(pose['tilt']):+.2f}{xyz_txt}"
        )

    def _annotate(self, image_bgr: np.ndarray) -> np.ndarray:
        self._sync_cam()
        detections = detect_markers(image_bgr, bed_origin.DICTIONARY)
        vis = draw_detections(image_bgr, detections, expected_ids=EXPECTED)
        ids = sorted(int(k) for k in detections)
        posts = sorted(physical_posts(ids))
        sizes = marker_pixel_sizes(detections)
        min_posts = int(self.get_parameter("min_posts").value)
        ready = len(posts) >= min_posts
        rms_txt = ""
        if ready and self.camera_matrix is not None:
            pose, _, _, _ = bed_origin.pnp_layout_from_image(
                image_bgr,
                self.camera_matrix,
                self.distortion,
                layout_path=str(self.get_parameter("layout_path").value),
                min_posts=min_posts,
            )
            if pose is not None:
                rms_txt = f"  RMS {pose['reprojection_rms_px']:.1f}px"
        color = (0, 200, 0) if ready else (0, 180, 255)
        lines = [
            f"{'READY' if ready else 'AIM'}  posts {posts or '[]'}  ids {ids or '[]'}{rms_txt}",
            f"now pan {self.pan:+.2f}  tilt {self.tilt:+.2f}",
            self._pose_hud(self.start_pose, "start"),
            self._pose_hud(self.end_pose, "end  "),
            "TERMINAL: j/l/i/k head (keyboard_teleop)  1 start  2 end  w sweep  s save",
            "Need 3 bed posts (corners 0-3 / sides 10-13), not the dummy chest tag.",
        ]
        if self.last_save_note:
            lines.append(self.last_save_note)
        y = 28
        for line in lines:
            cv2.putText(
                vis, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
            )
            y += 28
        if sizes:
            size_line = "  ".join(f"{i}:{sizes[i]:.0f}px" for i in sorted(sizes))
            cv2.putText(
                vis,
                size_line,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
        return vis

    def _save(self, image_bgr: np.ndarray) -> None:
        out_dir = Path(str(self.get_parameter("output_dir").value))
        out_dir.mkdir(parents=True, exist_ok=True)
        detections = detect_markers(image_bgr, bed_origin.DICTIONARY)
        vis = draw_detections(image_bgr, detections, expected_ids=EXPECTED)
        cv2.imwrite(str(out_dir / "stretch_origin_tags.png"), vis)
        report = {
            "ok": False,
            "dictionary": "DICT_5X5_100",
            "all_detected_ids": sorted(int(k) for k in detections),
            "posts": sorted(int(v) for v in physical_posts(detections.keys())),
        }
        if self.camera_matrix is None:
            self.last_save_note = "SAVE failed: no camera_info"
            return
        try:
            t_odom_camera, _info = self._t_odom_camera(float(self.pan))
        except Exception as exc:  # noqa: BLE001
            self.last_save_note = f"SAVE failed: TF {exc}"
            (out_dir / "stretch_origin.json").write_text(json.dumps(report, indent=2) + "\n")
            return
        sample = bed_origin.sample_from_image(
            image_bgr,
            self.camera_matrix,
            self.distortion,
            t_odom_camera,
            layout_path=str(self.get_parameter("layout_path").value),
            min_posts=int(self.get_parameter("min_posts").value),
        )
        if sample is None:
            report["ok"] = False
            (out_dir / "stretch_origin.json").write_text(json.dumps(report, indent=2) + "\n")
            self.last_save_note = "SAVED view but PnP not ready (need >=3 posts)"
            return
        report.update(
            {
                "ok": True,
                "marker_ids": list(sample.marker_ids),
                "posts": sorted(sample.posts),
                "reprojection_rms_px": sample.reprojection_rms_px,
                "T_odom_layout": sample.t_odom_layout.tolist(),
            }
        )
        (out_dir / "stretch_origin.json").write_text(json.dumps(report, indent=2) + "\n")
        self.last_save_note = (
            f"SAVED ok  posts={report['posts']}  RMS={sample.reprojection_rms_px:.2f}px"
        )
        self.get_logger().info(self.last_save_note)

    def _save_fused(self, fused, views) -> None:
        out_dir = Path(str(self.get_parameter("output_dir").value))
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.cam.bgr is not None:
            image_bgr = self.cam.bgr
            detections = detect_markers(image_bgr, bed_origin.DICTIONARY)
            vis = draw_detections(image_bgr, detections, expected_ids=EXPECTED)
            cv2.imwrite(str(out_dir / "stretch_origin_tags.png"), vis)
        if float(fused.reprojection_rms_px) > 12.0:
            self.last_save_note = (
                f"SWEEP reject: fused RMS {fused.reprojection_rms_px:.1f}px > 12 "
                "(odom→camera TF not following head pan)"
            )
            self.get_logger().error(self.last_save_note)
            return
        origin = np.asarray(fused.t_odom_layout, dtype=np.float64)[:3, 3]
        report = {
            "ok": True,
            "method": "multiview_reprojection",
            "dictionary": "DICT_5X5_100",
            "marker_ids": list(fused.marker_ids),
            "posts": sorted(fused.posts),
            "reprojection_rms_px": fused.reprojection_rms_px,
            "T_odom_layout": np.asarray(fused.t_odom_layout).tolist(),
            "layout_origin_odom_m": origin.tolist(),
            "n_views": len(views),
            "views": [
                {
                    "posts": sorted(v.posts),
                    "marker_ids": list(v.marker_ids),
                    "reprojection_rms_px": v.reprojection_rms_px,
                }
                for v in views
            ],
        }
        path = out_dir / "stretch_origin.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        self.last_save_note = (
            f"SAVED fused posts={report['posts']}  "
            f"RMS={fused.reprojection_rms_px:.2f}px  {path}"
        )
        self.get_logger().info(self.last_save_note)

    def _grab_view_sample(self):
        prev = self.cam.stamp_ns()
        self._spin_for(float(self.get_parameter("localization.settle_s").value))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            ns = self.cam.stamp_ns()
            if ns is not None and ns != prev:
                break
        self._sync_cam()
        if self.cam.bgr is None or self.camera_matrix is None:
            return None
        try:
            t_odom_cam, info = self._t_odom_camera(float(self.pan))
        except Exception:  # noqa: BLE001
            return None
        self.get_logger().info(
            f"  cam TF pan={self.pan:+.2f} tilt={self.tilt:+.2f} "
            f"look={info['yaw_tf_rad']:+.2f}→{info['look_fk_rad']:+.2f} "
            f"{info['method']} xyz={np.round(info['xyz_odom_m'], 3).tolist()}"
        )
        image_bgr = self.cam.bgr
        min_posts = int(self.get_parameter("localization.min_posts_per_view").value)
        min_pixel_size = float(self.get_parameter("localization.min_pixel_size").value)
        rms_max = float(self.get_parameter("localization.reprojection_rms_max_px").value)
        pose, used_ids, posts, sizes = bed_origin.pnp_layout_from_image(
            image_bgr,
            self.camera_matrix,
            self.distortion,
            layout_path=str(self.get_parameter("layout_path").value),
            min_posts=min_posts,
            min_pixel_size=min_pixel_size,
        )
        rms = None if pose is None else float(pose["reprojection_rms_px"])
        sample = bed_origin.sample_from_image(
            image_bgr,
            self.camera_matrix,
            self.distortion,
            t_odom_cam,
            layout_path=str(self.get_parameter("layout_path").value),
            min_posts=min_posts,
            min_pixel_size=min_pixel_size,
            reprojection_rms_max_px=rms_max,
        )
        if sample is None:
            size_txt = " ".join(f"{i}:{sizes[i]:.0f}" for i in sorted(sizes))
            rms_txt = f"rms={rms:.2f}px" if rms is not None else "pnp=fail"
            self.get_logger().info(
                f"  saw ids={sorted(int(v) for v in used_ids) or []} "
                f"posts={sorted(posts) or []} {size_txt} {rms_txt} max={rms_max:.1f}"
            )
            out = Path(str(self.get_parameter("output_dir").value))
            out.mkdir(parents=True, exist_ok=True)
            vis = draw_detections(
                image_bgr,
                detect_markers(image_bgr, bed_origin.DICTIONARY),
                expected_ids=EXPECTED,
            )
            cv2.imwrite(str(out / f"sweep_fail_pan{self.pan:+.2f}.png"), vis)
        return sample

    def _interp_tilt(self, pan: float) -> float:
        p0 = float(self.start_pose["pan"])
        p1 = float(self.end_pose["pan"])
        t0 = float(self.start_pose["tilt"])
        t1 = float(self.end_pose["tilt"])
        if abs(p1 - p0) < 1e-9:
            return t0
        alpha = (float(pan) - p0) / (p1 - p0)
        return t0 + alpha * (t1 - t0)

    def _auto_sweep(self) -> None:
        dwell_s = float(self.get_parameter("localization.dwell_s").value)

        def look(pan: float) -> None:
            self.pan = float(pan)
            self.tilt = self._interp_tilt(pan)
            self.get_logger().info(
                f"dwell pan={self.pan:+.2f} tilt={self.tilt:+.2f} for {dwell_s:.1f}s"
            )
            self._send_head(wait=True, duration_s=2.5)
            self._spin_for(dwell_s)

        n_stops = int(self.get_parameter("localization.n_stops").value)
        if self.start_pose.get("recorded") and self.end_pose.get("recorded"):
            n_stops = max(n_stops, 2)
        pans = resolve_pan_angles(
            pan_start=self.range_start,
            pan_end=self.range_end,
            n_stops=n_stops,
        )
        print(
            f"SWEEP {n_stops} stops: "
            + " → ".join(f"{p:+.3f}" for p in pans),
            flush=True,
        )

        fused, views, reason = sweep_head_views(
            look=look,
            grab=self._grab_view_sample,
            pan_angles_rad=pans,
            frames_per_view=int(
                self.get_parameter("localization.frames_per_view").value
            ),
            min_posts=int(self.get_parameter("localization.min_posts").value),
            translation_std_max_m=float(
                self.get_parameter("localization.translation_std_max_m").value
            ),
            rotation_max_deg=float(
                self.get_parameter("localization.rotation_max_deg").value
            ),
            origin_spread_max_m=float(
                self.get_parameter("localization.origin_spread_max_m").value
            ),
            log=self.get_logger().info,
        )
        if fused is None:
            self.last_save_note = f"SWEEP fail: {reason}"
            return
        self._save_fused(fused, views)

    def handle_key(self, key: int, image_bgr: np.ndarray) -> None:
        if key in (ord("q"), 27):
            self.running = False
            return
        # Preview convention (opposite official teleop): j = −pan, l = +pan.
        if key in (ord("j"), ord("J")):
            self._jog_head("joint_head_pan", -STEP)
        elif key in (ord("l"), ord("L")):
            self._jog_head("joint_head_pan", STEP)
        elif key in (ord("i"), ord("I")):
            self._jog_head("joint_head_tilt", STEP)
        elif key in (ord("k"), ord("K"), ord(",")):
            self._jog_head("joint_head_tilt", -STEP)
        elif key == ord("r"):
            self.rotate_k = (self.rotate_k + 1) % 4
        elif key in START_KEYS:
            self._capture_endpoint("start", image_bgr)
        elif key in END_KEYS:
            self._capture_endpoint("end", image_bgr)
        elif key == ord("w"):
            self.last_save_note = (
                f"SWEEP {self.range_start:+.2f} → {self.range_end:+.2f}"
            )
            self._auto_sweep()
        elif key == ord("s"):
            self._save(image_bgr)
        else:
            self.last_save_note = (
                f"unhandled key {key}; click the preview window, then 1 or 2"
            )


def _cv2_highgui_ok() -> bool:
    try:
        cv2.namedWindow("_robe_probe", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("_robe_probe")
        return True
    except cv2.error:
        return False


def _bgr_to_tk_photo(image_bgr: np.ndarray, png_path: Path):
    import tkinter as tk

    png_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(png_path), image_bgr):
        raise RuntimeError(f"cv2.imwrite failed: {png_path}")
    return tk.PhotoImage(file=str(png_path))


def _run_tk_preview(node: PreviewOrigin) -> None:
    """Fallback when conda OpenCV is the headless build (no namedWindow)."""

    import tkinter as tk

    out_dir = Path(str(node.get_parameter("output_dir").value))
    png_path = out_dir / "preview_live.png"
    root = tk.Tk()
    root.title("stretch_origin_preview (Tk; conda OpenCV is headless)")
    root.geometry("960x720")
    status = tk.Label(
        root,
        text="Waiting for compressed D435i from Stretch…",
        font=("sans", 14),
        fg="#222",
        pady=24,
    )
    status.pack()
    label = tk.Label(root)
    label.pack()
    hint = tk.Label(
        root,
        text="Type j/l/i/k in the TERMINAL (official teleop). 1 start  2 end  w sweep  s save  q quit",
        font=("sans", 11),
    )
    hint.pack()
    deadline = time.monotonic() + 45.0
    last_bgr = {"img": None}
    last_wait_log = {"t": 0.0}

    def _tick() -> None:
        if not node.running or not rclpy.ok():
            root.destroy()
            return
        try:
            rclpy.spin_once(node, timeout_sec=0.02)
            node.poll_terminal_keys()
            node._sync_cam()
            if node.cam.bgr is None:
                now = time.monotonic()
                if now > deadline:
                    print(
                        "No D435i frames on this machine "
                        f"({node.cam.status_line()}). "
                        "Source source_humble.sh in this terminal, then retry. "
                        "WiFi cannot carry uncompressed image_raw; "
                        "this preview uses /camera/color/image_raw/compressed."
                    )
                    node.running = False
                    root.destroy()
                    return
                if now - last_wait_log["t"] >= 2.0:
                    last_wait_log["t"] = now
                    left = int(deadline - now)
                    print(
                        f"Still no camera image ({left}s left). "
                        f"{node.cam.status_line()}"
                    )
                    status.configure(
                        text=f"Waiting for compressed D435i… {left}s"
                    )
                root.after(30, _tick)
                return
            image_bgr = node.cam.bgr
            last_bgr["img"] = image_bgr
            vis = node._annotate(image_bgr)
            if node.rotate_k:
                vis = np.rot90(vis, k=node.rotate_k)
            h, w = vis.shape[:2]
            max_w = 960
            if w > max_w:
                scale = max_w / float(w)
                vis = cv2.resize(
                    vis,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            photo = _bgr_to_tk_photo(vis, png_path)
            status.pack_forget()
            label.configure(image=photo)
            label.image = photo
        except Exception as exc:
            print(f"preview tick failed: {exc}")
        root.after(40, _tick)

    def _on_key(event) -> None:
        image_bgr = last_bgr["img"]
        if image_bgr is None:
            return
        ch = event.keysym
        mapping = {
            "1": ord("1"),
            "bracketleft": ord("["),
            "2": ord("2"),
            "bracketright": ord("]"),
            "j": ord("j"),
            "l": ord("l"),
            "i": ord("i"),
            "k": ord("k"),
            "w": ord("w"),
            "s": ord("s"),
            "q": ord("q"),
            "Escape": 27,
        }
        key = mapping.get(ch)
        if key is None and len(event.char) == 1:
            key = ord(event.char)
        if key is not None:
            node.handle_key(key, image_bgr)
            if not node.running:
                root.destroy()

    root.bind("<Key>", _on_key)
    root.focus_force()
    root.after(30, _tick)
    root.mainloop()


def main(args=None) -> None:
    require_stretch_for_d435i()
    if not os.environ.get("DISPLAY"):
        print(
            "No DISPLAY. Run this on the Stretch desktop (AnyDesk), "
            "not a headless SSH session."
        )
        raise SystemExit(2)
    rclpy.init(args=args)
    node = PreviewOrigin()
    node.switch_to_position_mode()
    print(
        "Head keys go to THIS TERMINAL (same as ros2 run stretch_core keyboard_teleop):"
    )
    print("  i tilt up    , or k tilt down    j pan left    l pan right")
    print("  1/[ start   2/] end   w sweep   s save   q quit")
    print("You do not need to click the image window to move the head.")
    print(
        f"ROS_IP={os.environ.get('ROS_IP', '')}  "
        f"FASTRTPS={os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE', '')}"
    )
    humble = Path.home() / "robe" / "real_world" / "ros2" / "source_humble.sh"
    if not os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE") and humble.is_file():
        print(f"This terminal is missing DDS env; run: source {humble}")
    try:
        if _cv2_highgui_ok():
            cv2.namedWindow("stretch_origin_preview", cv2.WINDOW_NORMAL)
            deadline = time.monotonic() + 45.0
            while rclpy.ok() and node.running:
                rclpy.spin_once(node, timeout_sec=0.02)
                node.poll_terminal_keys()
                node._sync_cam()
                if node.cam.bgr is None:
                    if time.monotonic() > deadline:
                        print(
                            "No D435i frames. "
                            f"{node.cam.status_line()}. "
                            f"Source {humble} in this terminal."
                        )
                        raise SystemExit(2)
                    continue
                image_bgr = node.cam.bgr
                vis = node._annotate(image_bgr)
                if node.rotate_k:
                    vis = np.rot90(vis, k=node.rotate_k)
                cv2.imshow("stretch_origin_preview", vis)
                key = cv2.waitKey(1) & 0xFF
                if key not in (255, 0):
                    node.handle_key(key, image_bgr)
            cv2.destroyAllWindows()
        else:
            print(
                "OpenCV in robe-ros2 is headless; using a Tk window instead."
            )
            _run_tk_preview(node)
    finally:
        try:
            node.restore_terminal()
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
