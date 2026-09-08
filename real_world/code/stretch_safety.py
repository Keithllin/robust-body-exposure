"""Production safety gates for BedPull: geometry, controller, goal params.

Workstation preflight and the ROS executor share these checks so a leftover
tune parameter cannot silently clip or warn-and-go on hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from stretch_cartesian import (
    ExecutionWristPose,
    PLANNING_SAFE_EXTENSION_M,
    YAW_SWEEP_CLEARANCE_M,
    default_execution_wrist,
    parking_wrist_down_failures,
    validate_execution_wrist,
)
from stretch_grasp_stages import (
    ALLOWED_CONTROLLERS,
    FAULT_GOAL_REJECT,
    FAULT_LIVE_GEOMETRY,
    FAULT_WRIST_VERIFY,
    LIVE_GEOMETRY_POLICIES,
    normalize_recipe,
    normalize_stop_after,
    recipe_stop_after,
)
from stretch_limits import HARDWARE_ARM_MAX_M, LIVE_GEOMETRY_MATCH_M


DEFAULT_LIVE_GEOMETRY_POLICY = "reject"
PRODUCTION_CONTROLLER = "streaming"


class SafetyReject(RuntimeError):
    """Goal or live check failed. ``fault_code`` is the stable operator token."""

    def __init__(self, fault_code: str, detail: str):
        super().__init__(detail)
        self.fault_code = fault_code
        self.detail = detail


@dataclass(frozen=True)
class ParamRange:
    minimum: float
    maximum: float
    inclusive: bool = True


GOAL_PARAM_RANGES: dict[str, ParamRange] = {
    "pull.clearance_above_bed_m": ParamRange(0.10, 0.80),
    "pull.xy_speed_m_s": ParamRange(0.01, 0.20),
    "pull.dt_s": ParamRange(0.04, 0.25),
    "workspace.arm_min_m": ParamRange(0.0, 0.10),
    "workspace.arm_max_m": ParamRange(0.30, HARDWARE_ARM_MAX_M),
    "workspace.lift_min_m": ParamRange(0.0, 0.20),
    "workspace.lift_max_m": ParamRange(0.80, 1.20),
    "approach.retract_arm_m": ParamRange(0.0, 0.15),
    "approach.clear_above_cloth_m": ParamRange(0.04, 0.30),
    "approach.pregrasp_pass_m": ParamRange(0.005, 0.05),
    "approach.pregrasp_reject_m": ParamRange(0.010, 0.08),
    "contact.max_descent_m": ParamRange(0.10, 0.70),
    "contact.min_descent_m": ParamRange(0.0, 0.20),
    "contact.breakaway_m": ParamRange(0.0, 0.10),
    "contact.unload_drop_pct": ParamRange(5.0, 40.0),
    "contact.disable_effort": ParamRange(50.0, 120.0),
}

RESTART_PARAMS = (
    "workspace.arm_min_m",
    "workspace.arm_max_m",
    "workspace.lift_min_m",
    "workspace.lift_max_m",
    "workspace.base_forward_in_base",
    "workspace.arm_extends_in_base",
    "frames.odom",
    "frames.base",
    "frames.layout",
    "frames.grasp_center",
)


def normalize_live_geometry_policy(value: object) -> str:
    text = str(value or DEFAULT_LIVE_GEOMETRY_POLICY).strip().lower()
    if text not in LIVE_GEOMETRY_POLICIES:
        raise ValueError(
            f"live_geometry.policy={value!r} is not one of "
            f"{', '.join(LIVE_GEOMETRY_POLICIES)}"
        )
    return text


def normalize_controller(value: object, *, allow_unvalidated: bool = False) -> str:
    text = str(value or PRODUCTION_CONTROLLER).strip().lower()
    if text == "trajectory" and not allow_unvalidated:
        raise SafetyReject(
            FAULT_GOAL_REJECT,
            "pull.controller=trajectory is not validated. "
            "Keep streaming (default) or position.",
        )
    if text not in ALLOWED_CONTROLLERS and not (
        allow_unvalidated and text == "trajectory"
    ):
        raise SafetyReject(
            FAULT_GOAL_REJECT,
            f"pull.controller={text!r} is not allowed. "
            f"Use one of {', '.join(ALLOWED_CONTROLLERS)}.",
        )
    return text


def _in_range(name: str, value: object, spec: ParamRange) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{name}={value!r} is not a number"
    if spec.inclusive:
        ok = spec.minimum <= number <= spec.maximum
    else:
        ok = spec.minimum < number < spec.maximum
    if ok:
        return None
    return (
        f"{name}={number:g} outside "
        f"[{spec.minimum:g}, {spec.maximum:g}]"
    )


def validate_goal_params(
    params: Mapping[str, Any],
    *,
    production: bool = True,
    allow_unvalidated_controller: bool = False,
) -> list[str]:
    """Goal-accept checks. Empty list means the snapshot is safe to run."""

    errors: list[str] = []
    controller = params.get("pull.controller", PRODUCTION_CONTROLLER)
    try:
        normalize_controller(
            controller, allow_unvalidated=allow_unvalidated_controller
        )
    except SafetyReject as exc:
        errors.append(exc.detail)

    try:
        normalize_live_geometry_policy(params.get("live_geometry.policy"))
    except ValueError as exc:
        errors.append(str(exc))

    try:
        normalize_stop_after(params.get("pull.stop_after", ""))
    except ValueError as exc:
        errors.append(str(exc))

    try:
        recipe = normalize_recipe(params.get("pull.recipe", "full-pull"))
        stop = str(params.get("pull.stop_after") or "")
        want = recipe_stop_after(recipe)
        if stop and want and stop != want:
            errors.append(
                f"pull.recipe={recipe} wants stop_after={want or 'DONE'} "
                f"but snapshot has {stop}"
            )
    except ValueError as exc:
        errors.append(str(exc))

    if production:
        leftover = str(params.get("pull.stop_after") or "").strip()
        recipe = str(params.get("pull.recipe") or "full-pull").strip()
        if leftover and recipe in ("", "full-pull"):
            errors.append(
                f"production full-pull forbids leftover pull.stop_after={leftover!r}"
            )
        if bool(params.get("pull.stop_after_pregrasp")):
            errors.append("production forbids pull.stop_after_pregrasp leftover")

    for name, spec in GOAL_PARAM_RANGES.items():
        if name not in params:
            continue
        message = _in_range(name, params[name], spec)
        if message:
            errors.append(message)

    arm_max = params.get("workspace.arm_max_m")
    arm_min = params.get("workspace.arm_min_m")
    if arm_max is not None and arm_min is not None:
        if float(arm_max) <= float(arm_min):
            errors.append("workspace.arm_max_m must be > workspace.arm_min_m")

    pass_m = params.get("approach.pregrasp_pass_m")
    reject_m = params.get("approach.pregrasp_reject_m")
    if pass_m is not None and reject_m is not None:
        if float(reject_m) < float(pass_m):
            errors.append(
                "approach.pregrasp_reject_m must be >= approach.pregrasp_pass_m"
            )
    return errors


def assert_goal_params(
    params: Mapping[str, Any],
    *,
    production: bool = True,
    allow_unvalidated_controller: bool = False,
) -> None:
    errors = validate_goal_params(
        params,
        production=production,
        allow_unvalidated_controller=allow_unvalidated_controller,
    )
    if errors:
        raise SafetyReject(FAULT_GOAL_REJECT, "; ".join(errors))


def apply_live_geometry_gate(
    *,
    error_m: float,
    limit_m: float = LIVE_GEOMETRY_MATCH_M,
    policy: object = DEFAULT_LIVE_GEOMETRY_POLICY,
    label: str,
) -> str | None:
    """Reject or warn when live geometry drifted from the planning snapshot.

    Returns a warning string when policy=warn. Raises SafetyReject on reject.
    """

    policy_norm = normalize_live_geometry_policy(policy)
    if float(error_m) <= float(limit_m):
        return None
    detail = (
        f"{label} {float(error_m):.3f} m exceeds live-geometry limit "
        f"{float(limit_m):.3f} m"
    )
    if policy_norm == "reject":
        raise SafetyReject(FAULT_LIVE_GEOMETRY, f"REJECT: {detail}")
    return detail


def verify_execution_wrist_joints(
    *,
    pitch_rad: float,
    yaw_rad: float,
    roll_rad: float,
    wrist_extension_m: float | None = None,
    pose: ExecutionWristPose | None = None,
) -> list[str]:
    target = pose or default_execution_wrist()
    config_errors = validate_execution_wrist(target)
    if config_errors:
        return config_errors
    return parking_wrist_down_failures(
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad,
        roll_rad=roll_rad,
        wrist_extension_m=wrist_extension_m,
        pose=target,
    )


def assert_execution_wrist_joints(
    *,
    pitch_rad: float,
    yaw_rad: float,
    roll_rad: float,
    wrist_extension_m: float | None = None,
    pose: ExecutionWristPose | None = None,
    stage: str = "WRIST_DOWN",
) -> None:
    reasons = verify_execution_wrist_joints(
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad,
        roll_rad=roll_rad,
        wrist_extension_m=wrist_extension_m,
        pose=pose,
    )
    if reasons:
        raise SafetyReject(
            FAULT_WRIST_VERIFY,
            f"{stage} wrist verify failed ({'; '.join(reasons)})",
        )


def wrist_clearance_required_m(pose: ExecutionWristPose | None = None) -> float:
    target = pose or default_execution_wrist()
    return max(
        float(target.yaw_sweep_clearance_m),
        float(YAW_SWEEP_CLEARANCE_M),
        float(PLANNING_SAFE_EXTENSION_M),
    )
