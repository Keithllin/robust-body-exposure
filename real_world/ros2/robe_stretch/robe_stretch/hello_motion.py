"""FollowJointTrajectory / streaming helpers matching HelloNode Humble."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .path_setup import add_workstation_code

add_workstation_code()

from stretch_hello_pose import encode_move_to_pose  # noqa: E402


# Stretch 3 Dex Wrist / SG3 — same indexing as hello_helpers.joint_qpos_conversion
STREAMING_IDX = {
    "ARM": 0,
    "LIFT": 1,
    "WRIST_YAW": 2,
    "WRIST_PITCH": 3,
    "WRIST_ROLL": 4,
    "HEAD_PAN": 5,
    "HEAD_TILT": 6,
    "GRIPPER": 7,
    "BASE_TRANSLATE": 8,
    "BASE_ROTATE": 9,
}
STREAMING_NUM_JOINTS = 10


def try_streaming_idx():
    try:
        from hello_helpers.joint_qpos_conversion import get_Idx

        idx = get_Idx("eoa_wrist_dw3_tool_sg3")
        return {
            "ARM": idx.ARM,
            "LIFT": idx.LIFT,
            "WRIST_YAW": idx.WRIST_YAW,
            "WRIST_PITCH": idx.WRIST_PITCH,
            "WRIST_ROLL": idx.WRIST_ROLL,
            "HEAD_PAN": idx.HEAD_PAN,
            "HEAD_TILT": idx.HEAD_TILT,
            "GRIPPER": idx.GRIPPER,
            "BASE_TRANSLATE": idx.BASE_TRANSLATE,
            "BASE_ROTATE": idx.BASE_ROTATE,
            "num_joints": idx.num_joints,
        }
    except Exception:  # noqa: BLE001
        return {**STREAMING_IDX, "num_joints": STREAMING_NUM_JOINTS}


def follow_joint_trajectory_goal(
    pose: Mapping[str, Any],
    *,
    custom_contact_thresholds: bool = False,
    duration_s: float = 0.0,
) -> FollowJointTrajectory.Goal:
    encoded = encode_move_to_pose(
        pose, custom_contact_thresholds=custom_contact_thresholds
    )
    point = JointTrajectoryPoint()
    point.positions = [float(v) for v in encoded["positions"]]
    if encoded["velocities"] is not None:
        point.velocities = [float(v) for v in encoded["velocities"]]
    if encoded["accelerations"] is not None:
        point.accelerations = [float(v) for v in encoded["accelerations"]]
    if encoded["effort"] is not None:
        point.effort = [float(v) for v in encoded["effort"]]
    sec = int(duration_s)
    nsec = int(round((duration_s - sec) * 1e9))
    # HelloNode position-mode uses time_from_start=0; firmware trapezoid.
    # A non-zero duration is for trajectory mode / explicit pacing only.
    point.time_from_start = Duration(sec=sec, nanosec=max(0, nsec))
    goal = FollowJointTrajectory.Goal()
    goal.goal_time_tolerance = Duration(sec=1, nanosec=0)
    goal.trajectory = JointTrajectory()
    goal.trajectory.joint_names = list(encoded["joint_names"])
    goal.trajectory.points = [point]
    return goal


def multi_dof_base_point(x: float, y: float, yaw: float):
    """Trajectory-mode base joint is named ``position``, not translate_mobile_base."""

    from geometry_msgs.msg import Transform, Twist
    from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint

    transform = Transform()
    transform.translation.x = float(x)
    transform.translation.y = float(y)
    transform.translation.z = 0.0
    # yaw about Z as quaternion
    transform.rotation.z = float(np_sin_half(yaw))
    transform.rotation.w = float(np_cos_half(yaw))
    point = MultiDOFJointTrajectoryPoint()
    point.transforms = [transform]
    point.velocities = [Twist()]
    return point


def np_sin_half(yaw: float) -> float:
    import math

    return math.sin(yaw / 2.0)


def np_cos_half(yaw: float) -> float:
    import math

    return math.cos(yaw / 2.0)


def fill_streaming_qpos(
    *,
    current: Optional[Sequence[float]] = None,
    arm: Optional[float] = None,
    lift: Optional[float] = None,
    gripper: Optional[float] = None,
    wrist_yaw: Optional[float] = None,
    wrist_pitch: Optional[float] = None,
    wrist_roll: Optional[float] = None,
    head_pan: Optional[float] = None,
    head_tilt: Optional[float] = None,
    base_translate: float = 0.0,
    base_rotate: float = 0.0,
) -> list[float]:
    """Absolute arm/lift/wrist; incremental base. Unspecified joints keep ``current``."""

    idx = try_streaming_idx()
    n = int(idx["num_joints"])
    qpos = [0.0] * n
    if current is not None and len(current) >= n:
        qpos = [float(v) for v in current[:n]]
    mapping = {
        "ARM": arm,
        "LIFT": lift,
        "GRIPPER": gripper,
        "WRIST_YAW": wrist_yaw,
        "WRIST_PITCH": wrist_pitch,
        "WRIST_ROLL": wrist_roll,
        "HEAD_PAN": head_pan,
        "HEAD_TILT": head_tilt,
    }
    for key, value in mapping.items():
        if value is not None:
            qpos[idx[key]] = float(value)
    qpos[idx["BASE_TRANSLATE"]] = float(base_translate)
    qpos[idx["BASE_ROTATE"]] = float(base_rotate)
    return qpos
