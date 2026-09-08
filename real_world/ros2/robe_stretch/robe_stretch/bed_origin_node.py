#!/usr/bin/env python3
"""Detect bed tags on the head D435i and repeat a frozen odom→layout TF.

Control must use the executor snapshot, not a live TF lookup of ``now()``.
This node samples on demand, then repeats the frozen transform for RViz.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from . import bed_origin
from .camera_stream import ColorCamera
from .path_setup import add_workstation_code, require_stretch_for_d435i

add_workstation_code()

from marker_utils import DEFAULT_LAYOUT_PATH  # noqa: E402
from stretch_localize import mat_to_quat_xyzw, transform_to_mat  # noqa: E402


def _stamp_to_mat(msg: TransformStamped) -> np.ndarray:
    t = msg.transform.translation
    q = msg.transform.rotation
    return transform_to_mat((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


def _mat_to_stamp(
    mat: np.ndarray, *, parent: str, child: str, stamp
) -> TransformStamped:
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


class BedOriginNode(Node):
    def __init__(self) -> None:
        super().__init__("bed_origin_node")
        self.cam = ColorCamera(self)
        self.camera_matrix = None
        self.distortion = np.zeros(5)
        self.latest_image = None
        self.frozen = None
        self.last_sample = None
        self.lock = threading.Lock()
        self.declare_parameter("min_posts", 3)
        self.declare_parameter("localization.min_pixel_size", 20.0)
        self.declare_parameter("localization.reprojection_rms_max_px", 12.0)
        self.declare_parameter("frames.odom", "odom")
        self.declare_parameter("frames.layout", "layout")
        self.declare_parameter(
            "frames.camera_optical", "camera_color_optical_frame"
        )
        self.declare_parameter("layout_path", str(DEFAULT_LAYOUT_PATH))
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Bool, "/robe_stretch/freeze_bed_tf", self._on_freeze, 1)
        self.create_subscription(
            TransformStamped,
            "/robe_stretch/frozen_layout_tf",
            self._on_external_freeze,
            1,
        )
        self.create_service(Trigger, "/robe_stretch/sample_layout", self._on_sample)
        self.create_service(Trigger, "/robe_stretch/freeze_layout", self._on_freeze_srv)
        self.sample_pub = self.create_publisher(
            TransformStamped, "/robe_stretch/layout_sample", 1
        )
        self.tf = TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self._repeat_frozen)

    def _sync_cam(self) -> None:
        with self.lock:
            self.latest_image = self.cam.bgr
            self.camera_matrix = self.cam.camera_matrix
            self.distortion = self.cam.distortion

    def _on_freeze(self, msg: Bool) -> None:
        if msg.data:
            self._freeze_last()

    def _on_freeze_srv(self, request, response):
        ok = self._freeze_last()
        response.success = bool(ok)
        response.message = "frozen" if ok else "no sample to freeze"
        return response

    def _on_external_freeze(self, msg: TransformStamped) -> None:
        with self.lock:
            self.frozen = msg
        self.get_logger().info("Accepted executor frozen odom→layout snapshot")

    def _lookup_t_odom_camera(self):
        odom = str(self.get_parameter("frames.odom").value)
        camera = str(self.get_parameter("frames.camera_optical").value)
        try:
            stamped = self.tf_buffer.lookup_transform(odom, camera, rclpy.time.Time())
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"TF {odom}→{camera} missing: {exc}")
            return None
        return _stamp_to_mat(stamped)

    def _on_sample(self, request, response):
        sample = self._sample_now()
        if sample is None:
            response.success = False
            response.message = "PnP failed (need ≥3 posts, DICT_5X5_100 layout)"
            return response
        response.success = True
        response.message = (
            f"posts={sorted(sample.posts)} ids={list(sample.marker_ids)} "
            f"rms={sample.reprojection_rms_px:.2f}px"
        )
        return response

    def _sample_now(self):
        self._sync_cam()
        with self.lock:
            image_bgr = self.latest_image
            camera_matrix = None if self.camera_matrix is None else self.camera_matrix.copy()
            distortion = self.distortion.copy()
        if image_bgr is None or camera_matrix is None:
            self.get_logger().warn("No D435i image/camera_info yet")
            return None
        t_odom_camera = self._lookup_t_odom_camera()
        if t_odom_camera is None:
            return None
        sample = bed_origin.sample_from_image(
            image_bgr,
            camera_matrix,
            distortion,
            t_odom_camera,
            layout_path=Path(str(self.get_parameter("layout_path").value)),
            min_posts=int(self.get_parameter("min_posts").value),
            min_pixel_size=float(
                self.get_parameter("localization.min_pixel_size").value
            ),
            reprojection_rms_max_px=float(
                self.get_parameter("localization.reprojection_rms_max_px").value
            ),
        )
        if sample is None:
            return None
        stamp = self.get_clock().now().to_msg()
        msg = _mat_to_stamp(
            sample.t_odom_layout,
            parent=str(self.get_parameter("frames.odom").value),
            child=str(self.get_parameter("frames.layout").value),
            stamp=stamp,
        )
        with self.lock:
            self.last_sample = sample
        self.sample_pub.publish(msg)
        self.get_logger().info(
            f"layout sample posts={sorted(sample.posts)} "
            f"rms={sample.reprojection_rms_px:.2f}px"
        )
        return sample

    def _freeze_last(self) -> bool:
        with self.lock:
            sample = self.last_sample
        if sample is None:
            sample = self._sample_now()
        if sample is None:
            return False
        stamp = self.get_clock().now().to_msg()
        frozen = _mat_to_stamp(
            sample.t_odom_layout,
            parent=str(self.get_parameter("frames.odom").value),
            child=str(self.get_parameter("frames.layout").value),
            stamp=stamp,
        )
        with self.lock:
            self.frozen = frozen
        self.get_logger().info("Frozen odom→layout snapshot (repeater only; executor owns control)")
        return True

    def set_frozen(self, transform: TransformStamped) -> None:
        with self.lock:
            self.frozen = transform

    def _repeat_frozen(self) -> None:
        with self.lock:
            if self.frozen is None:
                return
            stamped = TransformStamped()
            stamped.header = self.frozen.header
            stamped.child_frame_id = self.frozen.child_frame_id
            stamped.transform = self.frozen.transform
            stamped.header.stamp = self.get_clock().now().to_msg()
            self.tf.sendTransform(stamped)


def main(args=None) -> None:
    require_stretch_for_d435i()
    rclpy.init(args=args)
    node = BedOriginNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
