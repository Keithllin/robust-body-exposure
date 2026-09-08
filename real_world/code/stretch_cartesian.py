"""Stretch planar decomposition matching Hello Robot hardware, not IK.

Park parallel to the bed long side (no auto yaw-align). Official FUNMAP:

* ``+translate_mobile_base`` along ``+X`` of ``base_link``
* ``+wrist_extension`` / ``joint_arm`` along ``-Y`` of ``base_link``

With that parking, bed +Y (head→foot) is base +X and bed +X (into the
bed) is base -Y when T is Z-up. Stretch PnP / overlay T is Z-down; keep
that raw T so layout +Y is not flipped. ``ensure_layout_z_up`` (Rx π)
reverses ``translate_mobile_base``. Targets are already in ``base_link``
after TF.

``translate_mobile_base`` is incremental; arm/lift are absolute.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StretchWorkspace:
    arm_min_m: float = 0.0
    arm_max_m: float = 0.52
    lift_min_m: float = 0.0
    lift_max_m: float = 1.10
    # Hello: drive forward is +X_base; telescoping arm is -Y_base.
    base_forward_in_base: tuple[float, float] = (1.0, 0.0)
    arm_extends_in_base: tuple[float, float] = (0.0, -1.0)


@dataclass(frozen=True)
class JointCommand:
    translate_mobile_base: float
    wrist_extension: float
    joint_lift: float


def command_toward_target(
    *,
    current_ee_base: np.ndarray,
    target_ee_base: np.ndarray,
    current_wrist_extension: float,
    current_lift: float,
    workspace: StretchWorkspace,
) -> JointCommand:
    """Split one planar step the way grasp_object.py does.

    ``translate_mobile_base`` = Δ along ``+X_base`` (not wrist).
    ``wrist_extension`` = current + Δ along ``-Y_base``.
    """

    current = np.asarray(current_ee_base, dtype=np.float64).reshape(3)
    target = np.asarray(target_ee_base, dtype=np.float64).reshape(3)
    delta = target - current
    base_axis = np.asarray(workspace.base_forward_in_base, dtype=np.float64)
    arm_axis = np.asarray(workspace.arm_extends_in_base, dtype=np.float64)
    d_base = float(delta[0] * base_axis[0] + delta[1] * base_axis[1])
    d_arm = float(delta[0] * arm_axis[0] + delta[1] * arm_axis[1])
    wrist = float(current_wrist_extension + d_arm)
    # target[2] is EE Z in the same frame as current EE, not joint_lift.
    if np.isfinite(target[2]) and np.isfinite(current[2]):
        lift = float(current_lift + (target[2] - current[2]))
    else:
        lift = float(current_lift)
    return JointCommand(
        translate_mobile_base=d_base,
        wrist_extension=wrist,
        joint_lift=lift,
    )


def correct_ee_base_for_jsp_lag(
    ee_tf_base: np.ndarray,
    *,
    lift_real: float,
    arm_real: float,
    lift_jsp: float = 0.0,
    arm_jsp: float = 0.0,
    workspace: StretchWorkspace | None = None,
    min_delta: float = 0.02,
) -> tuple[np.ndarray, float, float]:
    """Fix EE when robot_state_publisher is reading zeroed ``/joint_states``.

    ``stretch_driver`` publishes live lift/arm on ``/stretch/joint_states``.
    If JSP never applies that source list, TF EE stays at URDF zero pose.
    Add ``(driver − JSP)`` along lift Z and ``workspace.arm_extends_in_base``.
    """

    workspace = workspace or StretchWorkspace()
    p = np.asarray(ee_tf_base, dtype=np.float64).reshape(3).copy()
    d_lift = float(lift_real) - float(lift_jsp)
    d_arm = float(arm_real) - float(arm_jsp)
    if abs(d_lift) < float(min_delta) and abs(d_arm) < float(min_delta):
        return p, d_lift, d_arm
    arm = np.array(
        [
            float(workspace.arm_extends_in_base[0]),
            float(workspace.arm_extends_in_base[1]),
            0.0,
        ],
        dtype=np.float64,
    )
    return p + np.array([0.0, 0.0, d_lift]) + d_arm * arm, d_lift, d_arm


def ee_base_from_tf(
    ee_tf_base: np.ndarray,
    *,
    lift_real: float,
    arm_real: float,
    lift_jsp: float = 0.0,
    arm_jsp: float = 0.0,
    workspace: StretchWorkspace | None = None,
    min_delta: float = 0.02,
) -> tuple[np.ndarray, float, float]:
    """Use raw TF when it already tracks lift; only then apply JSP-lag.

    After a restart the last live TF can still show Z≈1 m. Adding
    driver−JSP on top double-counts the arm and PREGRASP_PLAN asks for
    ``joint_arm < 0``.
    """

    from stretch_live_tf import tf_looks_zero_pose

    p = np.asarray(ee_tf_base, dtype=np.float64).reshape(3)
    if not tf_looks_zero_pose(ee_z_base=float(p[2]), joint_lift=float(lift_real)):
        return p.copy(), 0.0, 0.0
    return correct_ee_base_for_jsp_lag(
        p,
        lift_real=lift_real,
        arm_real=arm_real,
        lift_jsp=lift_jsp,
        arm_jsp=arm_jsp,
        workspace=workspace,
        min_delta=min_delta,
    )


# URDF joint_grasp_center xyz="0 0 0.23" on link_gripper_s3_body.
GRASP_CENTER_ALONG_TOOL_Z_M = 0.23
# Stale-TF detector only (hardware down, JSP still 0). Not “already
# at execution wrist-down”: leftover Uncover pitch −1.22 still has ~8 cm XY.
WRIST_DOWN_PITCH_RAD = -1.0
FULL_WRIST_DOWN_PITCH_RAD = -0.5 * math.pi
EXECUTION_WRIST_DOWN = "wrist_down"
# Hardware matches executor.yaml gripper_down.pitch ≈ −1.57.
# Yaw +90° about link_arm_l0 Z (0,0,−1): initial grasp, reused for Recover.
EXECUTION_WRIST_YAW_RAD = 0.5 * math.pi
EXECUTION_WRIST_ROLL_RAD = 0.0
WRIST_DOWN_PITCH_TOL_RAD = 0.15
WRIST_DOWN_YAW_ROLL_TOL_RAD = 0.20
# Parking tuck before planning snapshot. Same number as executor
# approach.retract_arm_m; this is the named planning-safe pose, not a
# "more than ~5 cm" heuristic.
PLANNING_SAFE_EXTENSION_M = 0.05
PLANNING_GRIPPER_OPEN = 0.35
PREPARE_RETRACT_ARM_M = PLANNING_SAFE_EXTENSION_M
# Tucked + pitch-down + large yaw sweeps the gripper through the mast.
# Extend this far before commanding yaw to EXECUTION_WRIST_YAW_RAD, then retract.
YAW_SWEEP_CLEARANCE_M = 0.15


@dataclass(frozen=True)
class ExecutionWristPose:
    """Single source of truth for the execution start wrist.

    Shared by prepare, snapshot, preflight, and the executor. Yaw is +90°
    about link_arm_l0 Z; pitch is full down. Do not fork these numbers in
    YAML or CLI without going through ``wrist_from_params``.
    """

    yaw_rad: float = EXECUTION_WRIST_YAW_RAD
    pitch_rad: float = FULL_WRIST_DOWN_PITCH_RAD
    roll_rad: float = EXECUTION_WRIST_ROLL_RAD
    planning_safe_extension_m: float = PLANNING_SAFE_EXTENSION_M
    yaw_sweep_clearance_m: float = YAW_SWEEP_CLEARANCE_M
    gripper_open: float = PLANNING_GRIPPER_OPEN
    pitch_tol_rad: float = WRIST_DOWN_PITCH_TOL_RAD
    yaw_roll_tol_rad: float = WRIST_DOWN_YAW_ROLL_TOL_RAD


@dataclass(frozen=True)
class WristMotionStep:
    """One prepare / executor wrist command. ``contact_off`` uses disable effort."""

    name: str
    joints: dict
    duration_s: float
    contact_off: bool = False
    reason: str = ""


def default_execution_wrist() -> ExecutionWristPose:
    return ExecutionWristPose()


def wrist_from_params(
    *,
    yaw_rad: float | None = None,
    pitch_rad: float | None = None,
    roll_rad: float | None = None,
    planning_safe_extension_m: float | None = None,
    yaw_sweep_clearance_m: float | None = None,
    gripper_open: float | None = None,
    pitch_tol_rad: float | None = None,
    yaw_roll_tol_rad: float | None = None,
) -> ExecutionWristPose:
    """Build a wrist pose, filling unspecified fields from the hardware default."""

    base = default_execution_wrist()
    return ExecutionWristPose(
        yaw_rad=base.yaw_rad if yaw_rad is None else float(yaw_rad),
        pitch_rad=base.pitch_rad if pitch_rad is None else float(pitch_rad),
        roll_rad=base.roll_rad if roll_rad is None else float(roll_rad),
        planning_safe_extension_m=(
            base.planning_safe_extension_m
            if planning_safe_extension_m is None
            else float(planning_safe_extension_m)
        ),
        yaw_sweep_clearance_m=(
            base.yaw_sweep_clearance_m
            if yaw_sweep_clearance_m is None
            else float(yaw_sweep_clearance_m)
        ),
        gripper_open=base.gripper_open if gripper_open is None else float(gripper_open),
        pitch_tol_rad=base.pitch_tol_rad if pitch_tol_rad is None else float(pitch_tol_rad),
        yaw_roll_tol_rad=(
            base.yaw_roll_tol_rad if yaw_roll_tol_rad is None else float(yaw_roll_tol_rad)
        ),
    )


def validate_execution_wrist(pose: ExecutionWristPose) -> list[str]:
    """Reject wrist numbers that would sweep the gripper through the mast."""

    reasons: list[str] = []
    if abs(float(pose.yaw_rad)) > math.pi + 1e-6:
        reasons.append(f"yaw {pose.yaw_rad:.3f} outside ±π")
    if float(pose.pitch_rad) > FULL_WRIST_DOWN_PITCH_RAD + 0.35:
        reasons.append(
            f"pitch {pose.pitch_rad:.3f} is not wrist-down "
            f"(want ≈ {FULL_WRIST_DOWN_PITCH_RAD:.3f})"
        )
    if float(pose.planning_safe_extension_m) < 0.0:
        reasons.append("planning_safe_extension_m < 0")
    if float(pose.yaw_sweep_clearance_m) < float(pose.planning_safe_extension_m):
        reasons.append(
            "yaw_sweep_clearance_m must be >= planning_safe_extension_m "
            f"({pose.yaw_sweep_clearance_m:.3f} < {pose.planning_safe_extension_m:.3f})"
        )
    if float(pose.pitch_tol_rad) <= 0.0 or float(pose.yaw_roll_tol_rad) <= 0.0:
        reasons.append("wrist tolerances must be positive")
    return reasons


def parking_wrist_down_failures(
    *,
    pitch_rad: float,
    yaw_rad: float = 0.0,
    roll_rad: float = 0.0,
    wrist_extension_m: float | None = None,
    planning_safe_extension_m: float = PLANNING_SAFE_EXTENSION_M,
    retract_arm_m: float | None = None,
    pose: ExecutionWristPose | None = None,
) -> list[str]:
    """Why ``parking_is_execution_wrist_down`` is false (empty if already down)."""

    target = pose or default_execution_wrist()
    safe = (
        float(retract_arm_m)
        if retract_arm_m is not None
        else float(planning_safe_extension_m)
        if planning_safe_extension_m != PLANNING_SAFE_EXTENSION_M
        else float(target.planning_safe_extension_m)
    )
    reasons: list[str] = []
    pitch_limit = float(target.pitch_rad) + float(target.pitch_tol_rad)
    if float(pitch_rad) > pitch_limit:
        reasons.append(
            f"pitch {float(pitch_rad):.3f} want <= {pitch_limit:.3f}"
        )
    want_yaw = float(target.yaw_rad)
    want_roll = float(target.roll_rad)
    yaw_tol = float(target.yaw_roll_tol_rad)
    if abs(float(yaw_rad) - want_yaw) > yaw_tol:
        reasons.append(
            f"yaw {float(yaw_rad):.3f} want {want_yaw:.3f}±{yaw_tol:.2f}"
        )
    if abs(float(roll_rad) - want_roll) > yaw_tol:
        reasons.append(
            f"roll {float(roll_rad):.3f} want {want_roll:.3f}±{yaw_tol:.2f}"
        )
    if wrist_extension_m is not None and float(wrist_extension_m) > float(safe) + 0.01:
        reasons.append(
            f"arm {float(wrist_extension_m):.3f} want <= {float(safe) + 0.01:.3f}"
        )
    return reasons


def parking_is_execution_wrist_down(
    *,
    pitch_rad: float,
    yaw_rad: float = 0.0,
    roll_rad: float = 0.0,
    wrist_extension_m: float | None = None,
    planning_safe_extension_m: float = PLANNING_SAFE_EXTENSION_M,
    retract_arm_m: float | None = None,
    pose: ExecutionWristPose | None = None,
) -> bool:
    """True when the gripper is already the execution start pose.

    Mid-Uncover pitch (~−1.22) is not full down — tool still has ~8 cm XY.
    """

    return not parking_wrist_down_failures(
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad,
        roll_rad=roll_rad,
        wrist_extension_m=wrist_extension_m,
        planning_safe_extension_m=planning_safe_extension_m,
        retract_arm_m=retract_arm_m,
        pose=pose,
    )


def yaw_or_roll_needs_clearance(
    *,
    yaw_rad: float,
    roll_rad: float,
    pose: ExecutionWristPose | None = None,
) -> bool:
    """True when commanding yaw/roll would sweep the gripper through the mast."""

    target = pose or default_execution_wrist()
    tol = float(target.yaw_roll_tol_rad)
    return (
        abs(float(yaw_rad) - float(target.yaw_rad)) > tol
        or abs(float(roll_rad) - float(target.roll_rad)) > tol
    )


def plan_wrist_down_steps(
    *,
    pitch_rad: float,
    yaw_rad: float,
    roll_rad: float,
    wrist_extension_m: float,
    pose: ExecutionWristPose | None = None,
    skip_if_already_down: bool = False,
) -> list[WristMotionStep]:
    """Safe sequence: extend 15 cm, yaw/roll, full wrist-down, retract.

    Same order as ``prepare_wrist_down``. Executor must use this before
    commanding +90° yaw; a tucked arm plus large yaw hits the mast.
    """

    target = pose or default_execution_wrist()
    bad = validate_execution_wrist(target)
    if bad:
        raise ValueError("invalid execution wrist: " + "; ".join(bad))
    if skip_if_already_down and parking_is_execution_wrist_down(
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad,
        roll_rad=roll_rad,
        wrist_extension_m=wrist_extension_m,
        pose=target,
    ):
        return []

    steps: list[WristMotionStep] = []
    arm = float(wrist_extension_m)
    clearance = float(target.yaw_sweep_clearance_m)
    if yaw_or_roll_needs_clearance(yaw_rad=yaw_rad, roll_rad=roll_rad, pose=target):
        if arm < clearance - 0.01:
            steps.append(
                WristMotionStep(
                    name="yaw_sweep_clearance",
                    joints={"joint_arm": clearance},
                    duration_s=4.0,
                    contact_off=True,
                    reason=(
                        f"extend {arm:.3f}->{clearance:.3f} m before yaw "
                        f"{float(yaw_rad):.3f}->{target.yaw_rad:.3f}"
                    ),
                )
            )
            arm = clearance
        steps.append(
            WristMotionStep(
                name="wrist_yaw_roll",
                joints={
                    "joint_wrist_yaw": float(target.yaw_rad),
                    "joint_wrist_roll": float(target.roll_rad),
                },
                duration_s=6.0,
                reason="set execution yaw/roll with arm clear of the mast",
            )
        )
    steps.append(
        WristMotionStep(
            name="wrist_down",
            joints={
                "joint_wrist_yaw": float(target.yaw_rad),
                "joint_wrist_pitch": float(target.pitch_rad),
                "joint_wrist_roll": float(target.roll_rad),
            },
            duration_s=4.0,
            reason="full execution wrist-down",
        )
    )
    safe = float(target.planning_safe_extension_m)
    if arm > safe + 0.01:
        steps.append(
            WristMotionStep(
                name="retract_planning_safe",
                joints={"joint_arm": safe},
                duration_s=4.0,
                contact_off=True,
                reason=f"retract {arm:.3f}->{safe:.3f} m after wrist-down",
            )
        )
    return steps


def execution_wrist_as_dict(pose: ExecutionWristPose | None = None) -> dict:
    target = pose or default_execution_wrist()
    return {
        "yaw_rad": float(target.yaw_rad),
        "pitch_rad": float(target.pitch_rad),
        "roll_rad": float(target.roll_rad),
        "planning_safe_extension_m": float(target.planning_safe_extension_m),
        "yaw_sweep_clearance_m": float(target.yaw_sweep_clearance_m),
        "gripper_open": float(target.gripper_open),
    }


def arm_axis_base(workspace: StretchWorkspace | None = None) -> np.ndarray:
    workspace = workspace or StretchWorkspace()
    return np.array(
        [
            float(workspace.arm_extends_in_base[0]),
            float(workspace.arm_extends_in_base[1]),
            0.0,
        ],
        dtype=np.float64,
    )


def rewind_ee_to_arm0(
    ee_base: np.ndarray,
    wrist_extension: float,
    workspace: StretchWorkspace | None = None,
) -> np.ndarray:
    """``p(q) = p(0) + q * arm_axis`` → grasp-center at arm=0."""

    return (
        np.asarray(ee_base, dtype=np.float64).reshape(3)
        - float(wrist_extension) * arm_axis_base(workspace)
    )


def apply_execution_wrist_down(
    ee_arm0: np.ndarray,
    *,
    pitch_rad: float = 0.0,
    workspace: StretchWorkspace | None = None,
) -> np.ndarray:
    """Map observed grasp-center at ``pitch_rad`` to full wrist-down XY/Z.

    pitch=0: 0.23 m tool along −Y_base. pitch=−π/2: the same 0.23 m along
    −Z. A leftover Uncover pitch of −1.22 still has ~8 cm of tool XY;
    calling that “already down” made Recover fail the 2 cm live match.
    """

    p = np.asarray(ee_arm0, dtype=np.float64).reshape(3).copy()
    offset = GRASP_CENTER_ALONG_TOOL_Z_M
    arm = arm_axis_base(workspace)
    pitch = float(pitch_rad)
    tool_now = offset * math.cos(pitch) * arm + np.array(
        [0.0, 0.0, offset * math.sin(pitch)], dtype=np.float64
    )
    tool_down = np.array([0.0, 0.0, -offset], dtype=np.float64)
    return p - tool_now + tool_down


def execution_ee_from_snapshot(
    snapshot: dict,
    workspace: StretchWorkspace | None = None,
) -> np.ndarray:
    """Fixed execution grasp-center at arm=0, full wrist-down.

    Always recomputed from ``ee_base`` / arm / pitch. Stored
    ``ee_base_arm0_wrist_down`` is debug-only so a mid-pitch leftover
    cannot freeze the wrong model. Live TF uses the recorded pitch;
    stale zero-pose TF is pitch=0 geometry (JSP-lag).
    """

    ee0 = rewind_ee_to_arm0(
        snapshot["ee_base"],
        float(snapshot["wrist_extension"]),
        workspace,
    )
    if snapshot.get("tf_live"):
        pitch = float(snapshot.get("joint_wrist_pitch") or 0.0)
    else:
        pitch = 0.0
    return apply_execution_wrist_down(
        ee0, pitch_rad=pitch, workspace=workspace
    )


def wrist_tf_stale_at_zero(*, pitch_driver: float, pitch_jsp: float = 0.0) -> bool:
    """True when hardware is wrist-down but RSP/JSP still publish pitch=0."""

    return float(pitch_driver) < WRIST_DOWN_PITCH_RAD and abs(float(pitch_jsp)) < 0.35


def apply_wrist_down_grasp_offset(
    ee_base: np.ndarray,
    *,
    pitch_driver: float,
    pitch_jsp: float = 0.0,
    workspace: StretchWorkspace | None = None,
) -> np.ndarray:
    """Map wrist-q=0 TF grasp_center to physical wrist-down XY/Z.

    At URDF pitch=0 the 0.23 m tool offset lies along the arm (−Y_base).
    After ``WRIST_DOWN`` it lies along −Z. RSP still at q=0 reports the
    EE ~0.23 m too far into the bed and ~0.23 m too high — plan then
    retracts to arm=0 while VERIFY says short by ~11 cm.
    """

    p = np.asarray(ee_base, dtype=np.float64).reshape(3).copy()
    if not wrist_tf_stale_at_zero(pitch_driver=pitch_driver, pitch_jsp=pitch_jsp):
        return p
    workspace = workspace or StretchWorkspace()
    arm = np.array(
        [
            float(workspace.arm_extends_in_base[0]),
            float(workspace.arm_extends_in_base[1]),
            0.0,
        ],
        dtype=np.float64,
    )
    offset = GRASP_CENTER_ALONG_TOOL_Z_M
    return p - offset * arm + np.array([0.0, 0.0, -offset])


def grasp_xy_correction(
    current_grasp_xy: np.ndarray, target_grasp_xy: np.ndarray
) -> np.ndarray:
    """Bed-frame XY delta after wrist-down settle; executor re-queries TF then applies this."""

    current = np.asarray(current_grasp_xy, dtype=np.float64).reshape(2)
    target = np.asarray(target_grasp_xy, dtype=np.float64).reshape(2)
    return target - current


def interpolate_xy_line(
    grasp_xy: np.ndarray,
    release_xy: np.ndarray,
    *,
    speed_m_s: float,
    dt_s: float = 0.1,
) -> np.ndarray:
    """Constant-speed XY waypoints, including both endpoints."""

    start = np.asarray(grasp_xy, dtype=np.float64).reshape(2)
    end = np.asarray(release_xy, dtype=np.float64).reshape(2)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if speed_m_s <= 0:
        raise ValueError("pull speed must be positive")
    if length < 1e-6:
        return start[None, :]
    duration = length / float(speed_m_s)
    n_steps = max(1, int(np.ceil(duration / float(dt_s))))
    alphas = np.linspace(0.0, 1.0, n_steps + 1)
    return start[None, :] + alphas[:, None] * delta[None, :]


def path_length(points_xy: np.ndarray) -> float:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def trajectory_metrics(
    actual_xy: np.ndarray,
    grasp_xy: np.ndarray,
    release_xy: np.ndarray,
    *,
    actual_z: np.ndarray | None = None,
    pull_z: float | None = None,
    duration_s: float | None = None,
) -> dict:
    """Endpoint, cross-track, Z, and timing metrics for a dry linear pull."""

    actual = np.asarray(actual_xy, dtype=np.float64).reshape(-1, 2)
    grasp = np.asarray(grasp_xy, dtype=np.float64).reshape(2)
    release = np.asarray(release_xy, dtype=np.float64).reshape(2)
    line = release - grasp
    length = float(np.linalg.norm(line))
    if length < 1e-9:
        direction = np.array([1.0, 0.0])
    else:
        direction = line / length
    normal = np.array([-direction[1], direction[0]])
    rel = actual - grasp[None, :]
    along = rel @ direction
    cross = rel @ normal
    metrics = {
        "grasp_error_m": float(np.linalg.norm(actual[0] - grasp)),
        "release_error_m": float(np.linalg.norm(actual[-1] - release)),
        "cross_track_max_m": float(np.max(np.abs(cross))),
        "cross_track_rms_m": float(np.sqrt(np.mean(cross**2))),
        "path_length_m": path_length(actual),
        "requested_length_m": length,
        "along_min_m": float(np.min(along)),
        "along_max_m": float(np.max(along)),
    }
    if actual_z is not None and pull_z is not None:
        z = np.asarray(actual_z, dtype=np.float64).reshape(-1)
        metrics["z_error_max_m"] = float(np.max(np.abs(z - pull_z)))
        metrics["z_error_std_m"] = float(np.std(z))
    if duration_s is not None and duration_s > 0:
        metrics["duration_s"] = float(duration_s)
        metrics["mean_xy_speed_m_s"] = metrics["path_length_m"] / float(duration_s)
        if length > 0:
            metrics["requested_speed_m_s"] = length / float(duration_s)
    return metrics
