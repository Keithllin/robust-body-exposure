"""Camera pose when robot_state_publisher is stuck at the URDF zero pose.

Hello Robot wires stretch_driver → /stretch/joint_states → joint_state_publisher
→ /joint_states → robot_state_publisher. On this robot JSP never applies the
driver source, so /joint_states stays at zeros and every head TF is the
zero-pose URDF (look ≈ +X, tilt 0, camera xyz glued).

Per-view PnP only needs intrinsics. Fusion multiplies by T_odom_camera.
A stale zero-pose TF makes opposite bed corners disagree by tens to hundreds
of pixels. Rebuild base→camera from the zero-pose link TFs plus the real
joint_head_pan / joint_head_tilt from /stretch/joint_states.
"""

from __future__ import annotations

import math

import numpy as np


def optical_look_yaw_xy(t_parent_cam: np.ndarray) -> float:
    """Yaw of camera optical +Z in the parent XY plane (ROS optical look)."""

    look = np.asarray(t_parent_cam, dtype=np.float64)[:3, 2]
    return float(math.atan2(look[1], look[0]))


def wrap_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def invert_transform(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    rot = mat[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ mat[:3, 3]
    return out


def _rz4(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return out


def t_base_camera_from_head_joints(
    t_base_pan0: np.ndarray,
    t_base_tilt0: np.ndarray,
    t_base_cam0: np.ndarray,
    pan: float,
    tilt: float,
) -> np.ndarray:
    """Apply pan then tilt about the URDF joint axes, starting from zero-pose TFs.

    joint_head_pan / joint_head_tilt are revolute about +Z of their child
    frames. Those frames in the stale TF are the q=0 poses.
    """

    t_pan0_tilt0 = invert_transform(t_base_pan0) @ np.asarray(t_base_tilt0)
    t_tilt0_cam = invert_transform(t_base_tilt0) @ np.asarray(t_base_cam0)
    return (
        np.asarray(t_base_pan0, dtype=np.float64)
        @ _rz4(float(pan))
        @ t_pan0_tilt0
        @ _rz4(float(tilt))
        @ t_tilt0_cam
    )


def tf_tracks_head_pan(t_base_cam: np.ndarray, joint_pan: float, tol: float = 0.2) -> bool:
    return abs(wrap_pi(optical_look_yaw_xy(t_base_cam) - float(joint_pan))) < tol


def t_odom_camera_from_head_joints(
    t_odom_base: np.ndarray,
    t_base_cam: np.ndarray,
    joint_pan: float,
    joint_tilt: float = 0.0,
    t_base_pan0: np.ndarray | None = None,
    t_base_tilt0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    look = optical_look_yaw_xy(t_base_cam)
    live = tf_tracks_head_pan(t_base_cam, joint_pan)
    if live or t_base_pan0 is None or t_base_tilt0 is None:
        aligned = np.asarray(t_base_cam, dtype=np.float64)
        method = "tf_live" if live else "tf_raw"
    else:
        aligned = t_base_camera_from_head_joints(
            t_base_pan0, t_base_tilt0, t_base_cam, joint_pan, joint_tilt
        )
        method = "fk_zero_tf"
    t_odom_cam = np.asarray(t_odom_base, dtype=np.float64) @ aligned
    info = {
        "method": method,
        "yaw_tf_rad": look,
        "joint_pan_rad": float(joint_pan),
        "joint_tilt_rad": float(joint_tilt),
        "look_fk_rad": optical_look_yaw_xy(aligned),
        "xyz_odom_m": [float(v) for v in t_odom_cam[:3, 3]],
        "xyz_base_m": [float(v) for v in aligned[:3, 3]],
    }
    return t_odom_cam, info
