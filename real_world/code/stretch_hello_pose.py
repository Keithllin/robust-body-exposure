"""HelloNode.move_to_pose goal encoding — copied contract, not the driver.

Stretch Humble treats a 2-tuple as (position, velocity) unless
``custom_contact_thresholds=True``, in which case it is (position, effort).
If any joint has an effort, every joint in that goal must have one.
"""

from __future__ import annotations

import numbers
from typing import Any, Mapping


class HelloPoseError(ValueError):
    """Invalid move_to_pose dictionary."""


def parse_joint_goal(
    value: Any, *, custom_contact_thresholds: bool
) -> tuple[float, float | None, float | None, float | None]:
    """Return (position, velocity, acceleration, effort); missing fields are None."""

    if isinstance(value, str):
        raise HelloPoseError(f"Goal must be numeric, not str: {value}")
    if isinstance(value, numbers.Real):
        return float(value), None, None, None
    if not isinstance(value, (tuple, list)):
        raise HelloPoseError(f"Goal must be a scalar or tuple/list, got {type(value)}")
    if len(value) == 0 or len(value) > 4:
        raise HelloPoseError(f"Goal must be scalar or length 1..4, got {value}")
    position = float(value[0])
    if custom_contact_thresholds:
        if len(value) == 1:
            return position, None, None, None
        if len(value) == 2:
            return position, None, None, float(value[1])
        if len(value) == 3:
            raise HelloPoseError(
                "With custom_contact_thresholds=True, 3-tuples are ambiguous. "
                f"Use (pos, effort) or (pos, vel, acc, effort). Got: {value}"
            )
        return position, float(value[1]), float(value[2]), float(value[3])
    velocity = float(value[1]) if len(value) >= 2 else None
    acceleration = float(value[2]) if len(value) >= 3 else None
    effort = float(value[3]) if len(value) >= 4 else None
    return position, velocity, acceleration, effort


def encode_move_to_pose(
    pose: Mapping[str, Any], *, custom_contact_thresholds: bool = False
) -> dict:
    """Build the JointTrajectoryPoint field lists HelloNode would send."""

    if not pose:
        raise HelloPoseError("pose is empty")
    joint_names = list(pose.keys())
    parsed = [
        parse_joint_goal(pose[name], custom_contact_thresholds=custom_contact_thresholds)
        for name in joint_names
    ]
    any_effort = any(item[3] is not None for item in parsed)
    if any_effort and any(item[3] is None for item in parsed):
        raise HelloPoseError(
            "effort/contact-threshold provided for some joints but not all. "
            "Lift-only contact goals must contain only the lift joint."
        )
    any_vel = any(item[1] is not None for item in parsed)
    any_acc = any(item[2] is not None for item in parsed)
    point: dict = {
        "joint_names": joint_names,
        "positions": [item[0] for item in parsed],
        "velocities": None,
        "accelerations": None,
        "effort": None,
        "custom_contact_thresholds": bool(custom_contact_thresholds),
    }
    if any_vel:
        point["velocities"] = [item[1] if item[1] is not None else 0.0 for item in parsed]
    if any_acc:
        point["accelerations"] = [
            item[2] if item[2] is not None else 0.0 for item in parsed
        ]
    if any_effort:
        point["effort"] = [item[3] for item in parsed]
    return point
