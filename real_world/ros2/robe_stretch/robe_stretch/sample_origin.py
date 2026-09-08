#!/usr/bin/env python3
"""One-shot Stretch origin: head D435i layout PnP, no pull.

Prints marker IDs, physical posts, reprojection RMS, layout origin in odom,
and (if present) the trial canonical centroid mapped into odom. Writes
``/tmp/robe/stretch_origin.json`` and an annotated D435i image.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as MsgDuration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import TransformStamped
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclDuration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from . import bed_origin
from .camera_stream import ColorCamera
from .camera_tf import t_odom_camera_from_head_joints
from .head_sweep import resolve_pan_angles, sweep_head_views
from .path_setup import add_workstation_code, require_stretch_for_d435i

CODE_DIR = add_workstation_code()

from canonical_bed import maybe_load_canonical_frame  # noqa: E402
from marker_utils import DEFAULT_LAYOUT_PATH, detect_markers, draw_detections  # noqa: E402
from stretch_localize import (  # noqa: E402
    layout_canonical_disagreement_m,
    mat_to_quat_xyzw,
    transform_to_mat,
)


def _stamp_to_mat(msg: TransformStamped) -> np.ndarray:
    t = msg.transform.translation
    q = msg.transform.rotation
    return transform_to_mat((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


def _mat_to_stamp(mat: np.ndarray, parent: str, child: str, stamp) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(mat[0, 3])
    msg.transform.translation.y = float(mat[1, 3])
    msg.transform.translation.z = float(mat[2, 3])
    x, y, z, w = mat_to_quat_xyzw(mat[:3, :3])
    msg.transform.rotation.x = x
    msg.transform.rotation.y = y
    msg.transform.rotation.z = z
    msg.transform.rotation.w = w
    return msg


class SampleOrigin(Node):
    def __init__(self) -> None:
        super().__init__("sample_origin")
        self.bridge = None
        self.cam = ColorCamera(self)
        self.image = None
        self.camera_matrix = None
        self.distortion = np.zeros(5)
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
        self.declare_parameter("canonical_frame_path", "")
        self.declare_parameter("layout_path", str(DEFAULT_LAYOUT_PATH))
        self.declare_parameter("frames.odom", "odom")
        self.declare_parameter("frames.base", "base_link")
        self.declare_parameter("frames.layout", "layout")
        self.declare_parameter("frames.camera_optical", "camera_color_optical_frame")
        self.declare_parameter("frames.head_pan", "link_head_pan")
        self.declare_parameter("frames.head_tilt", "link_head_tilt")
        self.declare_parameter("head_tilt", -0.15)
        self.declare_parameter("look_first", False)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_pub = TransformBroadcaster(self)
        self.traj = ActionClient(
            self, FollowJointTrajectory, "/stretch_controller/follow_joint_trajectory"
        )
        self._cmd_pan = 0.0
        self._cmd_tilt = float(self.get_parameter("localization.head_tilt").value)
        self.create_subscription(
            JointState, "/stretch/joint_states", self._on_stretch_joints, 10
        )

    def _sync_cam(self) -> None:
        self.image = self.cam.bgr
        self.camera_matrix = self.cam.camera_matrix
        self.distortion = self.cam.distortion

    def _on_stretch_joints(self, msg: JointState) -> None:
        names = list(msg.name)
        if "joint_head_pan" in names:
            self._cmd_pan = float(msg.position[names.index("joint_head_pan")])
        if "joint_head_tilt" in names:
            self._cmd_tilt = float(msg.position[names.index("joint_head_tilt")])

    def _head_look(self, pan: float, tilt: float | None = None) -> None:
        if tilt is None:
            tilt = float(self.get_parameter("localization.head_tilt").value)
        self._cmd_pan = float(pan)
        self._cmd_tilt = float(tilt)
        if not self.traj.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn("No FollowJointTrajectory; leave the head aimed at the bed")
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = ["joint_head_pan", "joint_head_tilt"]
        point = JointTrajectoryPoint()
        point.positions = [float(pan), float(tilt)]
        point.time_from_start = MsgDuration(sec=2, nanosec=0)
        goal.trajectory.points = [point]
        send = self.traj.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=8.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            return
        rclpy.spin_until_future_complete(self, handle.get_result_async(), timeout_sec=6.0)
        dwell = float(self.get_parameter("localization.dwell_s").value)
        deadline = time.monotonic() + dwell
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _look_at_bed(self) -> None:
        if not bool(self.get_parameter("look_first").value):
            return
        self._head_look(0.0)

    def _image_stamp_ns(self):
        return self.cam.stamp_ns()

    def _grab_view_sample(self):
        prev = self._image_stamp_ns()
        settle = float(self.get_parameter("localization.settle_s").value)
        deadline = time.monotonic() + settle
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            ns = self._image_stamp_ns()
            if ns is not None and ns != prev:
                break
        self._sync_cam()
        if self.image is None or self.camera_matrix is None:
            return None
        odom = str(self.get_parameter("frames.odom").value)
        base = str(self.get_parameter("frames.base").value)
        camera = str(self.get_parameter("frames.camera_optical").value)
        try:
            t_odom_base = _stamp_to_mat(
                self.tf_buffer.lookup_transform(
                    odom, base, rclpy.time.Time(), timeout=RclDuration(seconds=1.0)
                )
            )
            t_base_cam = _stamp_to_mat(
                self.tf_buffer.lookup_transform(
                    base, camera, rclpy.time.Time(), timeout=RclDuration(seconds=1.0)
                )
            )
            t_base_pan = _stamp_to_mat(
                self.tf_buffer.lookup_transform(
                    base,
                    str(self.get_parameter("frames.head_pan").value),
                    rclpy.time.Time(),
                    timeout=RclDuration(seconds=0.5),
                )
            )
            t_base_tilt = _stamp_to_mat(
                self.tf_buffer.lookup_transform(
                    base,
                    str(self.get_parameter("frames.head_tilt").value),
                    rclpy.time.Time(),
                    timeout=RclDuration(seconds=0.5),
                )
            )
            t_odom_cam, info = t_odom_camera_from_head_joints(
                t_odom_base,
                t_base_cam,
                float(self._cmd_pan),
                joint_tilt=float(self._cmd_tilt),
                t_base_pan0=t_base_pan,
                t_base_tilt0=t_base_tilt,
            )
        except Exception:  # noqa: BLE001
            return None
        print(
            f"  cam TF pan={self._cmd_pan:+.2f} tilt={self._cmd_tilt:+.2f} "
            f"look={info['yaw_tf_rad']:+.2f}→{info['look_fk_rad']:+.2f} "
            f"{info['method']} xyz={np.round(info['xyz_odom_m'], 3).tolist()}",
            flush=True,
        )
        image_bgr = self.image
        return bed_origin.sample_from_image(
            image_bgr,
            self.camera_matrix,
            self.distortion,
            t_odom_cam,
            layout_path=str(self.get_parameter("layout_path").value),
            min_posts=int(self.get_parameter("localization.min_posts_per_view").value),
            min_pixel_size=float(
                self.get_parameter("localization.min_pixel_size").value
            ),
            reprojection_rms_max_px=float(
                self.get_parameter("localization.reprojection_rms_max_px").value
            ),
        )

    def wait_for_camera(self, timeout_s: float = 20.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            self._sync_cam()
            if self.image is not None and self.camera_matrix is not None:
                return True
        return False

    def sample(self) -> int:
        if not self.wait_for_camera():
            self.get_logger().error("No D435i image. Launch stretch_driver and d435i first.")
            return 2
        print(
            f"stretch joints pan={self._cmd_pan:+.3f} tilt={self._cmd_tilt:+.3f} "
            "(use these; /joint_states is often stuck at 0)",
            flush=True,
        )
        odom = str(self.get_parameter("frames.odom").value)
        out_dir = Path(str(self.get_parameter("output_dir").value))
        out_dir.mkdir(parents=True, exist_ok=True)
        default_tilt = float(self.get_parameter("localization.head_tilt").value)
        pan_start = float(self.get_parameter("localization.pan_start").value)
        pan_end = float(self.get_parameter("localization.pan_end").value)
        tilt_start = default_tilt
        tilt_end = default_tilt
        range_path = out_dir / "head_sweep_range.json"
        if range_path.is_file():
            try:
                recorded = json.loads(range_path.read_text())
            except json.JSONDecodeError:
                recorded = {}
            start = recorded.get("start") or {}
            end = recorded.get("end") or {}
            if start.get("recorded") and end.get("recorded"):
                pan_start = float(start["pan"])
                pan_end = float(end["pan"])
                tilt_start = float(start.get("tilt", default_tilt))
                tilt_end = float(end.get("tilt", default_tilt))
                print(f"Using recorded head range from {range_path}")
        n_stops = int(self.get_parameter("localization.n_stops").value)
        angles = resolve_pan_angles(
            pan_start=pan_start,
            pan_end=pan_end,
            n_stops=n_stops,
        )
        print("Pan sweep (rad):", [round(v, 3) for v in angles])
        print(
            "Tilt sweep (rad):",
            [
                round(
                    tilt_start
                    if abs(pan_end - pan_start) < 1e-9
                    else tilt_start
                    + (float(pan) - pan_start)
                    / (pan_end - pan_start)
                    * (tilt_end - tilt_start),
                    3,
                )
                for pan in angles
            ],
        )

        def look(pan: float) -> None:
            if abs(pan_end - pan_start) < 1e-9:
                tilt = tilt_start
            else:
                alpha = (float(pan) - pan_start) / (pan_end - pan_start)
                tilt = tilt_start + alpha * (tilt_end - tilt_start)
            self._head_look(float(pan), tilt)

        fused, views, reason = sweep_head_views(
            look=look,
            grab=self._grab_view_sample,
            pan_angles_rad=angles,
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
            log=print,
        )
        if self.cam.bgr is not None:
            image_bgr = self.cam.bgr
            detections = detect_markers(image_bgr, bed_origin.DICTIONARY)
            vis = draw_detections(
                image_bgr, detections, expected_ids=(0, 1, 2, 3, 10, 11, 12, 13)
            )
            cv2.imwrite(str(out_dir / "stretch_origin_tags.png"), vis)

        report = {
            "ok": fused is not None,
            "dictionary": "DICT_5X5_100",
            "reason": reason,
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
        print(reason)
        if fused is None:
            (out_dir / "stretch_origin.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"Wrote {out_dir / 'stretch_origin.json'}")
            print(f"Wrote {out_dir / 'stretch_origin_tags.png'}")
            print("ok=false: not syncing raw to the workstation.")
            return 2

        t_ol = fused.t_odom_layout
        layout_origin_odom = t_ol[:3, 3]
        report.update(
            {
                "marker_ids": list(fused.marker_ids),
                "posts": sorted(fused.posts),
                "reprojection_rms_px": fused.reprojection_rms_px,
                "T_odom_layout": t_ol.tolist(),
                "layout_origin_odom_m": layout_origin_odom.tolist(),
            }
        )
        print(f"Fused posts: {report['posts']}  worst RMS: {fused.reprojection_rms_px:.2f} px")
        if float(fused.reprojection_rms_px) > 12.0:
            report["ok"] = False
            report["reason"] = (
                f"fused RMS {fused.reprojection_rms_px:.1f}px > 12 "
                "(odom→camera TF not following head pan)"
            )
            (out_dir / "stretch_origin.json").write_text(json.dumps(report, indent=2) + "\n")
            print(report["reason"])
            print(f"Wrote {out_dir / 'stretch_origin.json'}  ok=false")
            return 2
        print(
            "Layout origin (marker_layout board, odom m): "
            f"{layout_origin_odom.round(4).tolist()}"
        )

        canon_path = str(self.get_parameter("canonical_frame_path").value)
        frame = None
        if canon_path:
            frame = maybe_load_canonical_frame(Path(canon_path).parent)
            if (Path(canon_path)).is_file():
                from canonical_bed import load_canonical_frame

                frame = load_canonical_frame(Path(canon_path))
        if frame is not None:
            o_h = np.append(frame.origin, 1.0)
            canonical_origin_odom = t_ol @ o_h
            disagree = layout_canonical_disagreement_m(frame.origin)
            report["canonical_origin_layout_m"] = frame.origin.tolist()
            report["canonical_origin_odom_m"] = canonical_origin_odom[:3].tolist()
            report["layout_vs_canonical_xy_m"] = disagree
            print(
                "Canonical centroid O in layout (m): "
                f"{np.asarray(frame.origin).round(4).tolist()}"
            )
            print(
                "Canonical origin in odom (m): "
                f"{canonical_origin_odom[:3].round(4).tolist()}"
            )
            print(f"|O_xy| layout vs board origin: {disagree:.3f} m")
            print("Actions execute relative to this canonical origin, not a second Stretch centroid.")
        else:
            print(
                "No canonical_bed_frame.json yet — this sample is T_odom_layout only. "
                "After ceiling capture, rsync that json and re-run to map pickle (O,R)."
            )

        (out_dir / "stretch_origin.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {out_dir / 'stretch_origin.json'}")
        print(f"Wrote {out_dir / 'stretch_origin_tags.png'}  (last sweep view)")
        try:
            add_workstation_code()
            from session_paths import sync_raw_to_workstation

            sync_raw_to_workstation(out_dir)
            print("Synced raw origin to workstation (corrected untouched).")
        except Exception as exc:  # noqa: BLE001
            print(f"WARN auto-sync raw to workstation failed: {exc}")
            print("Origin is on Stretch. On RCHI:")
            print("  bash real_world/sessions/pull_stretch_origin.sh")

        layout_name = str(self.get_parameter("frames.layout").value)
        for _ in range(60):
            msg = _mat_to_stamp(
                t_ol, odom, layout_name, self.get_clock().now().to_msg()
            )
            self.tf_pub.sendTransform(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
        print("Broadcast odom→layout for ~3s (RViz TF). Done. No pull.")
        return 0 if report["ok"] else 2


def main(args=None) -> None:
    require_stretch_for_d435i()
    rclpy.init(args=args)
    node = SampleOrigin()
    try:
        code = node.sample()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
