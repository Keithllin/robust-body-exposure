"""D435i color frames for RCHI over WiFi.

Uncompressed ``/camera/color/image_raw`` is 1280×720 and drops on this LAN.
Prefer ``.../compressed`` (same camera_info). Raw is only a fallback.
"""

from __future__ import annotations

import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

import cv2

# D435i publishes BEST_EFFORT; RELIABLE subscribers see zero frames.


class ColorCamera:
    def __init__(self, node: Node) -> None:
        self.node = node
        self.bgr: np.ndarray | None = None
        self.header = None
        self.source = ""
        self.comp_count = 0
        self.raw_count = 0
        self.camera_matrix = None
        self.distortion = np.zeros(5)
        node.create_subscription(
            CompressedImage,
            "/camera/color/image_raw/compressed",
            self._on_compressed,
            qos_profile_sensor_data,
        )
        node.create_subscription(
            Image, "/camera/color/image_raw", self._on_raw, qos_profile_sensor_data
        )
        node.create_subscription(
            CameraInfo, "/camera/color/camera_info", self._on_info, qos_profile_sensor_data
        )

    def stamp_ns(self) -> int | None:
        if self.header is None:
            return None
        return int(self.header.stamp.sec) * 1_000_000_000 + int(
            self.header.stamp.nanosec
        )

    def status_line(self) -> str:
        pubs_c = len(
            self.node.get_publishers_info_by_topic(
                "/camera/color/image_raw/compressed"
            )
        )
        pubs_r = len(
            self.node.get_publishers_info_by_topic("/camera/color/image_raw")
        )
        return (
            f"compressed {self.comp_count} frames ({pubs_c} pub)  "
            f"raw {self.raw_count} frames ({pubs_r} pub)  "
            f"source={self.source or 'none'}"
        )

    def _on_compressed(self, msg: CompressedImage) -> None:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        self.comp_count += 1
        self.bgr = bgr
        self.header = msg.header
        self.source = "compressed"

    def _on_raw(self, msg: Image) -> None:
        self.raw_count += 1
        if self.bgr is not None and self.source == "compressed":
            return
        try:
            from cv_bridge import CvBridge

            bgr = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        self.bgr = bgr
        self.header = msg.header
        self.source = "raw"

    def _on_info(self, msg: CameraInfo) -> None:
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.distortion = np.asarray(msg.d, dtype=np.float64).reshape(-1)
