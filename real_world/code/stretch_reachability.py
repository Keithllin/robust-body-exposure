"""Whole-path reachability for a Stretch bedside grasp→release line.

Reject the trial if any interpolated point is outside the workspace.
Never clip or reshape the policy action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from stretch_cartesian import (
    StretchWorkspace,
    command_toward_target,
    execution_ee_from_snapshot,
    interpolate_xy_line,
)


@dataclass(frozen=True)
class ReachabilityResult:
    reachable: bool
    reason: str
    n_waypoints: int
    failing_index: int | None = None
    max_wrist_extension: float | None = None
    geometry: str = "execution_wrist_down"

    def to_dict(self) -> dict:
        return asdict(self)


def _in_range(value: float, lo: float, hi: float, slack: float = 1e-3) -> bool:
    """1 mm slack: IK can return -0.000 at the arm stop; do not clip the policy XY."""
    return float(lo) - slack <= float(value) <= float(hi) + slack


def check_pull_path(
    grasp_xy: np.ndarray,
    release_xy: np.ndarray,
    *,
    pull_z: float,
    current_ee_base: np.ndarray,
    current_wrist_extension: float,
    current_lift: float,
    workspace: StretchWorkspace,
    speed_m_s: float = 0.05,
    dt_s: float = 0.05,
    slack_m: float = 1e-3,
) -> ReachabilityResult:
    """Simulate the planned Stretch decomposition along the whole line.

    ``pull_z`` is EE height in ``base_link`` (same frame as
    ``current_ee_base``), not ``joint_lift``. Wrist-down grasp_center sits
    ~0.1 m above the lift carriage; EE Z of 1.19 m with lift at 1.10 is
    normal.     Lift limits apply only to ``cmd.joint_lift``. Negative
    ``wrist_extension`` (EE already past the grasp) is restricted to 0,
    not treated as unreachable. Over-max arm still rejects.
    """

    grasp = np.asarray(grasp_xy, dtype=np.float64).reshape(2)
    release = np.asarray(release_xy, dtype=np.float64).reshape(2)
    waypoints = interpolate_xy_line(grasp, release, speed_m_s=speed_m_s, dt_s=dt_s)
    if not _in_range(
        current_lift, workspace.lift_min_m, workspace.lift_max_m, slack=slack_m
    ):
        return ReachabilityResult(
            False,
            f"joint_lift {current_lift:.3f} m outside lift limits "
            f"[{workspace.lift_min_m}, {workspace.lift_max_m}]",
            len(waypoints),
            0,
        )

    ee = np.asarray(current_ee_base, dtype=np.float64).reshape(3).copy()
    wrist = float(current_wrist_extension)
    lift = float(current_lift)
    max_wrist = wrist
    # Approach the grasp first, then every pull sample.
    targets_xy = np.vstack([grasp[None, :], waypoints])
    for index, xy in enumerate(targets_xy):
        target = np.array([xy[0], xy[1], pull_z], dtype=np.float64)
        # targets here are in the same frame as current_ee_base after the
        # caller has mapped canonical -> base_link.
        cmd = command_toward_target(
            current_ee_base=ee,
            target_ee_base=target,
            current_wrist_extension=wrist,
            current_lift=lift,
            workspace=workspace,
        )
        raw_wrist = float(cmd.wrist_extension)
        if raw_wrist > workspace.arm_max_m + slack_m:
            return ReachabilityResult(
                False,
                f"waypoint {index} wrist_extension {raw_wrist:.3f} m "
                f"outside [{workspace.arm_min_m}, {workspace.arm_max_m}]",
                len(targets_xy),
                index,
                max(max_wrist, raw_wrist),
            )
        # Already past the target into the bed → restrict to 0, do not reject.
        # Policy XY is unchanged; only the hardware command is bounded.
        wrist_cmd = max(raw_wrist, float(workspace.arm_min_m))
        max_wrist = max(max_wrist, wrist_cmd)
        if not _in_range(
            cmd.joint_lift,
            workspace.lift_min_m,
            workspace.lift_max_m,
            slack=slack_m,
        ):
            return ReachabilityResult(
                False,
                f"waypoint {index} lift {cmd.joint_lift:.3f} m "
                f"outside [{workspace.lift_min_m}, {workspace.lift_max_m}]",
                len(targets_xy),
                index,
                max_wrist,
            )
        # Apply the increment so later waypoints see the updated pose.
        arm_axis = np.asarray(workspace.arm_extends_in_base, dtype=np.float64)
        base_axis = np.asarray(workspace.base_forward_in_base, dtype=np.float64)
        ee[0] += cmd.translate_mobile_base * base_axis[0] + (
            wrist_cmd - wrist
        ) * arm_axis[0]
        ee[1] += cmd.translate_mobile_base * base_axis[1] + (
            wrist_cmd - wrist
        ) * arm_axis[1]
        ee[2] = float(pull_z)
        wrist = wrist_cmd
        lift = cmd.joint_lift
    return ReachabilityResult(True, "ok", len(targets_xy), None, max_wrist)


def _invert_t(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    out = np.eye(4)
    rot = mat[:3, :3]
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ mat[:3, 3]
    return out


def layout_xyz_to_base(
    p_layout: np.ndarray,
    *,
    t_odom_layout: np.ndarray,
    t_odom_base: np.ndarray,
    ee_z_base: float | None = None,
) -> np.ndarray:
    p_h = np.append(np.asarray(p_layout, dtype=np.float64).reshape(3), 1.0)
    p_base = (_invert_t(t_odom_base) @ (t_odom_layout @ p_h))[:3].copy()
    if ee_z_base is not None:
        p_base[2] = float(ee_z_base)
    return p_base


def required_wrist_extension_xy(
    xy_canonical: np.ndarray,
    *,
    frame,
    snapshot: dict,
    workspace: StretchWorkspace | None = None,
    cloth_z: float | None = None,
) -> float:
    """``q_arm`` at one canonical XY using the execution wrist-down model.

    Same transform as ``check_bed_pull_reachability`` endpoints: cloth-Z
    canonical → layout → base, then ``command_toward_target`` from arm=0.
    """

    from canonical_bed import canonical_points_to_layout

    workspace = workspace or StretchWorkspace()
    xy = np.asarray(xy_canonical, dtype=np.float64).reshape(2)
    z = float(snapshot["cloth_z_layout"] if cloth_z is None else cloth_z)
    t_ol = np.asarray(snapshot["T_odom_layout"], dtype=np.float64)
    t_ob = np.asarray(snapshot["T_odom_base"], dtype=np.float64)
    ee0 = execution_ee_from_snapshot(snapshot, workspace)
    p_l = canonical_points_to_layout(
        np.array([[xy[0], xy[1], z]], dtype=np.float64), frame
    )[0]
    p_b = layout_xyz_to_base(
        p_l, t_odom_layout=t_ol, t_odom_base=t_ob, ee_z_base=float(ee0[2])
    )
    cmd = command_toward_target(
        current_ee_base=ee0,
        target_ee_base=np.array([p_b[0], p_b[1], ee0[2]], dtype=np.float64),
        current_wrist_extension=0.0,
        current_lift=float(snapshot.get("joint_lift") or 0.40),
        workspace=workspace,
    )
    return float(cmd.wrist_extension)


def check_bed_pull_reachability(
    action_canonical: np.ndarray,
    *,
    frame,
    snapshot: dict,
    workspace: StretchWorkspace | None = None,
    t_odom_layout: np.ndarray | None = None,
    t_odom_base: np.ndarray | None = None,
    cloth_z: float | None = None,
    speed_m_s: float = 0.05,
    dt_s: float = 0.08,
    slack_m: float = 1e-3,
) -> ReachabilityResult:
    """Whole-path reachability in the fixed execution wrist-down model.

    Does not use the robot's current parking EE. Reconstructs
    ``link_grasp_center`` at arm=0, wrist-down from the frozen
    ``T_odom_layout`` / ``T_odom_base`` snapshot, then simulates
    base +X / arm −Y along grasp→release. CMA calls this once after
    search (not per candidate). Workstation preflight and the executor
    use the same function.
    """

    from canonical_bed import canonical_points_to_layout

    workspace = workspace or StretchWorkspace()
    action = np.asarray(action_canonical, dtype=np.float64).reshape(4)
    z = float(snapshot["cloth_z_layout"] if cloth_z is None else cloth_z)
    t_ol = np.asarray(
        snapshot["T_odom_layout"] if t_odom_layout is None else t_odom_layout,
        dtype=np.float64,
    )
    t_ob = np.asarray(
        snapshot["T_odom_base"] if t_odom_base is None else t_odom_base,
        dtype=np.float64,
    )
    ee0 = execution_ee_from_snapshot(snapshot, workspace)
    grasp_l = canonical_points_to_layout(
        np.array([[action[0], action[1], z]], dtype=np.float64), frame
    )[0]
    release_l = canonical_points_to_layout(
        np.array([[action[2], action[3], z]], dtype=np.float64), frame
    )[0]
    grasp_b = layout_xyz_to_base(
        grasp_l, t_odom_layout=t_ol, t_odom_base=t_ob, ee_z_base=float(ee0[2])
    )
    release_b = layout_xyz_to_base(
        release_l, t_odom_layout=t_ol, t_odom_base=t_ob, ee_z_base=float(ee0[2])
    )
    lift0 = float(snapshot.get("joint_lift") or 0.40)
    result = check_pull_path(
        grasp_b[:2],
        release_b[:2],
        pull_z=float(ee0[2]),
        current_ee_base=ee0,
        current_wrist_extension=0.0,
        current_lift=lift0,
        workspace=workspace,
        speed_m_s=speed_m_s,
        dt_s=dt_s,
        slack_m=slack_m,
    )
    return ReachabilityResult(
        result.reachable,
        result.reason,
        result.n_waypoints,
        result.failing_index,
        result.max_wrist_extension,
        "execution_wrist_down",
    )


def check_canonical_action_from_snapshot(
    action_canonical: np.ndarray,
    *,
    frame,
    snapshot: dict,
    workspace: StretchWorkspace | None = None,
    cloth_z: float | None = None,
    speed_m_s: float = 0.05,
    dt_s: float = 0.08,
    slack_m: float = 1e-3,
) -> ReachabilityResult:
    """Alias for ``check_bed_pull_reachability`` (execution wrist-down)."""

    return check_bed_pull_reachability(
        action_canonical,
        frame=frame,
        snapshot=snapshot,
        workspace=workspace,
        cloth_z=cloth_z,
        speed_m_s=speed_m_s,
        dt_s=dt_s,
        slack_m=slack_m,
    )


def check_bed_frame_pull(
    grasp_xy: np.ndarray,
    release_xy: np.ndarray,
    *,
    workspace: StretchWorkspace,
    x_limits: tuple[float, float] = (-0.55, 0.55),
    y_limits: tuple[float, float] = (-1.10, 1.10),
) -> ReachabilityResult:
    """Reachability in canonical/bed metres, not base_link.

    Workstation preflight has no live TF. Identifying bed X with
    ``wrist_extension`` rejects any pull that crosses X=0 (this trial:
    ``-0.001 m``). Check bed crop and arm *stroke* ``|Δx|`` instead.
    The executor uses live ``base_link``: bed X → arm (−Y_base), bed Y → drive.
    """

    grasp = np.asarray(grasp_xy, dtype=np.float64).reshape(2)
    release = np.asarray(release_xy, dtype=np.float64).reshape(2)
    waypoints = interpolate_xy_line(grasp, release, speed_m_s=0.05, dt_s=0.05)
    x_lo, x_hi = x_limits
    y_lo, y_hi = y_limits
    for index, xy in enumerate(waypoints):
        if not (x_lo <= xy[0] <= x_hi and y_lo <= xy[1] <= y_hi):
            return ReachabilityResult(
                False,
                f"waypoint {index} bed XY {xy.round(3).tolist()} outside "
                f"x{list(x_limits)} y{list(y_limits)}",
                len(waypoints),
                index,
            )
    stroke = abs(float(release[0] - grasp[0]))
    if stroke > workspace.arm_max_m + 1e-3:
        return ReachabilityResult(
            False,
            f"arm stroke |Δx|={stroke:.3f} m exceeds "
            f"{workspace.arm_max_m:.2f} m",
            len(waypoints),
            0,
        )
    return ReachabilityResult(True, "ok_bed_frame", len(waypoints), None)
