"""Helpers for detecting URDF-zero Stretch TF.

Hello's RSP reads ``/joint_states``. On this robot JSP still publishes
zeros, so ``link_grasp_center`` stays at ~0.096 m. The executor does
**not** rebroadcast TF (that raced RSP and broke pregrasp). It adds
``(driver − JSP)`` lift/arm on the zero-pose EE, same as the 2026-08-29
working run. Snapshot uses ``tf_looks_zero_pose`` the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MovingJoint:
    name: str
    parent: str
    child: str
    kind: str  # prismatic | revolute
    axis: tuple[float, float, float]


# Stretch 3 / SE3 URDF (repo stretch/stretch/stretch.urdf). Axis is in the
# child/joint frame at q=0, so T(q) = T_tf_zero @ Joint(axis, q).
#
# Broadcast prismatic lift+arm only. Wrist/head stay at RSP q=0. Publishing
# revolute joints races Hello's RSP and can flip link_grasp_center by the
# 0.23 m URDF offset (~29 cm) between PREGRASP_PLAN and VERIFY.
MOVING_JOINTS: tuple[MovingJoint, ...] = (
    MovingJoint("joint_lift", "link_mast", "link_lift", "prismatic", (0.0, 0.0, 1.0)),
    MovingJoint("joint_arm_l3", "link_arm_l4", "link_arm_l3", "prismatic", (0.0, 0.0, 1.0)),
    MovingJoint("joint_arm_l2", "link_arm_l3", "link_arm_l2", "prismatic", (0.0, 0.0, 1.0)),
    MovingJoint("joint_arm_l1", "link_arm_l2", "link_arm_l1", "prismatic", (0.0, 0.0, 1.0)),
    MovingJoint("joint_arm_l0", "link_arm_l1", "link_arm_l0", "prismatic", (0.0, 0.0, 1.0)),
    MovingJoint("joint_wrist_yaw", "link_arm_l0", "link_wrist_yaw", "revolute", (0.0, 0.0, -1.0)),
    MovingJoint(
        "joint_wrist_pitch",
        "link_wrist_yaw_bottom",
        "link_wrist_pitch",
        "revolute",
        (0.0, 0.0, -1.0),
    ),
    MovingJoint(
        "joint_wrist_roll", "link_wrist_pitch", "link_wrist_roll", "revolute", (0.0, 0.0, 1.0)
    ),
    MovingJoint("joint_head_pan", "link_head", "link_head_pan", "revolute", (0.0, 0.0, 1.0)),
    MovingJoint(
        "joint_head_tilt", "link_head_pan", "link_head_tilt", "revolute", (0.0, 0.0, 1.0)
    ),
)
PRISMATIC_JOINTS: tuple[MovingJoint, ...] = tuple(
    j for j in MOVING_JOINTS if j.kind == "prismatic"
)

ZERO_POSE_EE_Z_M = 0.096
ZERO_POSE_EE_Z_MAX_M = 0.20


def tf_looks_zero_pose(*, ee_z_base: float, joint_lift: float) -> bool:
    """True when RSP EE height is URDF-zero while the lift is actually up."""

    return float(joint_lift) > 0.15 and float(ee_z_base) < ZERO_POSE_EE_Z_MAX_M


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )


def joint_motion_mat(kind: str, axis: tuple[float, float, float], q: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    axis_v = np.asarray(axis, dtype=np.float64).reshape(3)
    if kind == "prismatic":
        out[:3, 3] = axis_v * float(q)
        return out
    if kind != "revolute":
        raise ValueError(f"unknown joint kind: {kind}")
    out[:3, :3] = rotation_about_axis(axis_v, float(q))
    return out


def apply_joint_to_zero_tf(
    t_parent_child0: np.ndarray, spec: MovingJoint, q: float
) -> np.ndarray:
    t0 = np.asarray(t_parent_child0, dtype=np.float64).reshape(4, 4)
    return t0 @ joint_motion_mat(spec.kind, spec.axis, q)


def rotation_to_quat(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Return (x, y, z, w) from a 3x3 rotation matrix."""

    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)
