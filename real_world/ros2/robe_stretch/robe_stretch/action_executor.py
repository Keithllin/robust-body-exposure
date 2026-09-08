#!/usr/bin/env python3
"""BedPull action server: pregrasp, guarded descend, grasp, lift, linear pull.

Does not import the GNN. Policy metres come from the goal 4-vector in frame
``bed`` (canonical). ``T_odom_layout`` comes from Stretch-local PnP
(session ``stretch_origin_corrected.json`` if present); this node never
subscribes to the D435i.
Park parallel to the bed long side. Drive is ``+X_base``
(``translate_mobile_base``); telescoping arm is ``-Y_base``
(``joint_arm``). No auto ``rotate_mobile_base``. ``T_odom_layout`` is
the raw Stretch PnP (same Z-down convention as the ceiling overlay).
Do not ``ensure_layout_z_up`` — that Rx(π) flips bed +Y and drives
away from the grasp. The pickle is never remirrored and never scaled
to sim bed size.

``pull.clearance_above_bed_m`` is EE grasp-center height above the cloth
plane, not an absolute ``joint_lift`` command.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import TransformStamped
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as RclDuration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint, MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist
from tf2_ros import Buffer, TransformListener

from .path_setup import add_workstation_code

add_workstation_code()

from .hello_motion import (  # noqa: E402
    fill_streaming_qpos,
    follow_joint_trajectory_goal,
)

from canonical_bed import (  # noqa: E402
    CanonicalBedFrame,
    canonical_xy_to_layout,
    layout_points_to_canonical,
    load_canonical_frame,
)
from execution_manifest import write_execution_manifest  # noqa: E402
from marker_utils import invert_transform  # noqa: E402
from stretch_cartesian import (  # noqa: E402
    StretchWorkspace,
    apply_wrist_down_grasp_offset,
    command_toward_target,
    ee_base_from_tf,
    execution_ee_from_snapshot,
    interpolate_xy_line,
    plan_wrist_down_steps,
    rewind_ee_to_arm0,
    trajectory_metrics,
    wrist_from_params,
)
from stretch_contact import (  # noqa: E402
    FUNMAP_MOVE_INCREMENT_M,
    FUNMAP_PERIOD_S,
    LiftContactDetector,
    contact_trace_row,
    descent_floor_q,
    format_effort_pct,
    is_force_contact,
    passed_stopping_position,
    traj_error_code,
    write_contact_trace_csv,
)
from stretch_grasp_stages import (  # noqa: E402
    FAULT_CANCELED,
    FAULT_DRIVER,
    FAULT_DRY_RUN,
    FAULT_GOAL_REJECT,
    FAULT_INTENTIONAL_STOP,
    FAULT_OK,
    FAULT_UNREACHABLE,
    TUNE_PARAM_NAMES,
    IntentionalStop,
    descend_budget_ok,
    intentional_stop_message,
    normalize_recipe,
    normalize_stop_after,
    resolve_stop_after,
    should_stop_after,
    verify_allows_descend,
)
from stretch_safety import (  # noqa: E402
    DEFAULT_LIVE_GEOMETRY_POLICY,
    SafetyReject,
    apply_live_geometry_gate,
    assert_execution_wrist_joints,
    assert_goal_params,
    normalize_controller,
    normalize_live_geometry_policy,
)
from stretch_hello_pose import HelloPoseError  # noqa: E402
from session_paths import (  # noqa: E402
    SessionPaths,
    SessionRegistrationError,
    assert_registration_frozen,
    load_active_trial,
    print_session_resolved,
    resolve_session_dir_once,
    session_paths,
)
from stretch_localize import (  # noqa: E402
    bedside_arm_into_bed_error_rad,
    bedside_parallel_error_rad,
    bedside_parking_error_rad,
    layout_canonical_disagreement_m,
    mat_to_quat_xyzw,
    resolve_stretch_origin_path,
    t_odom_layout_for_motion,
    layout_xy_at_height,
    transform_to_mat,
)
from stretch_limits import (  # noqa: E402
    EXEC_ARM_SLACK_M,
    HARDWARE_ARM_MAX_M,
    LIVE_GEOMETRY_MATCH_M,
    load_reach_snapshot,
)
from stretch_reachability import check_bed_pull_reachability  # noqa: E402

try:
    from robe_stretch_interfaces.action import BedPull
except ImportError:  # pragma: no cover - until the interface package is built
    BedPull = None


def _cfg_workspace(node: Node) -> StretchWorkspace:
    return StretchWorkspace(
        arm_min_m=float(node.declare_parameter("workspace.arm_min_m", 0.0).value),
        arm_max_m=float(node.declare_parameter("workspace.arm_max_m", 0.52).value),
        lift_min_m=float(node.declare_parameter("workspace.lift_min_m", 0.0).value),
        lift_max_m=float(node.declare_parameter("workspace.lift_max_m", 1.10).value),
        base_forward_in_base=tuple(
            node.declare_parameter("workspace.base_forward_in_base", [1.0, 0.0]).value
        ),
        arm_extends_in_base=tuple(
            node.declare_parameter("workspace.arm_extends_in_base", [0.0, -1.0]).value
        ),
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


class BedPullExecutor(Node):
    def __init__(self) -> None:
        super().__init__("bed_pull_executor")
        self.callback_group = ReentrantCallbackGroup()
        self.declare_parameter("pull.clearance_above_bed_m", 0.40)
        self.declare_parameter("pull.stop_after", "")
        self.declare_parameter("pull.stop_after_pregrasp", False)
        self.declare_parameter("pull.round_trip", False)
        self.declare_parameter("pull.xy_speed_m_s", 0.05)
        self.declare_parameter("pull.dt_s", 0.08)
        self.declare_parameter("pull.controller", "streaming")
        self.declare_parameter("pull.recipe", "full-pull")
        self.declare_parameter("pull.dry_run", False)
        self.declare_parameter("live_geometry.policy", DEFAULT_LIVE_GEOMETRY_POLICY)
        self.declare_parameter("pull.contact_effort", 40.0)
        self.declare_parameter("pull.contact_abort_m", 0.02)
        self.declare_parameter("contact.threshold_mode", "robot_default")
        # Debug only when threshold_mode=custom_symmetric (clean_surface ±e).
        self.declare_parameter("contact.lift_effort", 33.7)
        self.declare_parameter("contact.approach_effort", 55.0)
        self.declare_parameter("contact.raise_effort", 60.0)
        # High enough that a raise against gravity never looks like contact.
        self.declare_parameter("contact.disable_effort", 100.0)
        self.declare_parameter("contact.breakaway_m", 0.03)
        # Deprecated. Contact uses driver error 100, not a Python spike.
        self.declare_parameter("contact.effort_spike", 12.0)
        self.declare_parameter("contact.max_descent_m", 0.50)
        self.declare_parameter("contact.min_descent_m", 0.05)
        self.declare_parameter("contact.cloth_margin_m", 0.02)
        self.declare_parameter("contact.guard_band_m", 0.06)
        self.declare_parameter("contact.lower_speed_m_s", 0.05)
        self.declare_parameter("contact.lower_accel_m_s2", 0.12)
        self.declare_parameter("contact.hover_positive_warn_pct", 55.0)
        self.declare_parameter("contact.lift_effort_threshold", 20.0)
        self.declare_parameter("contact.move_increment_m", 0.008)
        self.declare_parameter("contact.lowest_allowed_m", 0.0)
        self.declare_parameter("contact.cloth_below_estimate_m", 0.10)
        self.declare_parameter("contact.log_trace", False)
        self.declare_parameter("contact.trace_path", "")
        self.declare_parameter("contact.coarse_standoff_m", 0.03)
        self.declare_parameter("contact.coarse_speed_m_s", 0.07)
        self.declare_parameter("contact.coarse_accel_m_s2", 0.15)
        self.declare_parameter("contact.probe_step_m", 0.02)
        self.declare_parameter("contact.probe_max_m", 0.06)
        self.declare_parameter("contact.probe_speed_m_s", 0.04)
        self.declare_parameter("contact.probe_accel_m_s2", 0.10)
        self.declare_parameter("contact.baseline_settle_s", 0.3)
        self.declare_parameter("contact.baseline_samples", 15)
        self.declare_parameter("contact.window_samples", 8)
        self.declare_parameter("contact.unload_drop_pct", 15.0)
        self.declare_parameter("contact.unload_consecutive", 2)
        self.declare_parameter("contact.compress_max_m", 0.03)
        self.declare_parameter("contact.detect_above_cloth_m", 0.008)
        self.declare_parameter("approach.retract_arm_m", 0.05)
        self.declare_parameter("approach.clear_above_cloth_m", 0.12)
        self.declare_parameter("approach.pregrasp_pass_m", 0.015)
        self.declare_parameter("approach.pregrasp_reject_m", 0.025)
        self.declare_parameter("approach.refine_passes", 1)
        self.declare_parameter("grasp.lift_after_contact_m", 0.07)
        self.declare_parameter("grasp.lower_after_open_m", 0.06)
        self.declare_parameter("grasp.probe_extra_m", 0.02)
        self.declare_parameter("grasp.pause_after_bump_s", 1.0)
        self.declare_parameter("grasp.pause_after_probe_s", 2.0)
        self.declare_parameter("grasp.pause_after_close_s", 2.0)
        self.declare_parameter("grasp.gripper_open", 0.35)
        self.declare_parameter("grasp.gripper_closed", -0.20)
        self.declare_parameter("gripper_down.yaw", 1.5708)
        self.declare_parameter("gripper_down.pitch", -1.57)
        self.declare_parameter("gripper_down.roll", 0.0)
        self.declare_parameter("gripper_down.settle_s", 0.5)
        self.declare_parameter("canonical_frame_path", "")
        self.declare_parameter("layout_snapshot_path", "")
        self.declare_parameter("layout_path", "")
        self.declare_parameter("manifest_dir", "")
        self.declare_parameter("frames.odom", "odom")
        self.declare_parameter("frames.base", "base_link")
        self.declare_parameter("frames.layout", "layout")
        self.declare_parameter("frames.camera_optical", "camera_color_optical_frame")
        self.declare_parameter("frames.grasp_center", "link_grasp_center")
        self.declare_parameter("frames.wrist_tag", "link_aruco_inner_wrist")
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
        self.declare_parameter("localization.origin_disagree_max_m", 0.05)
        self.declare_parameter("localization.head_tilt", -0.15)
        self.declare_parameter("localization.pan_start", -1.05)
        self.declare_parameter("localization.pan_end", -2.59)
        self.declare_parameter("localization.n_stops", 2)
        self.declare_parameter("localization.yaw_tolerance_rad", 0.15)
        self.workspace = _cfg_workspace(self)
        self._session_dir: Path | None = None
        self._session_paths: SessionPaths | None = None
        try:
            self._session_dir = resolve_session_dir_once()
            self._session_paths = session_paths(self._session_dir)
            origin_hint = (
                self._session_paths.origin_corrected
                if self._session_paths.origin_corrected.is_file()
                else self._session_paths.origin_raw
            )
            print_session_resolved(self._session_dir, origin_hint)
            self.get_logger().info(
                f"SESSION RESOLVED:\n  {self._session_dir}\n"
                f"STRETCH ORIGIN:\n  {origin_hint}"
            )
        except FileNotFoundError as exc:
            self.get_logger().warn(f"No session directory: {exc}")
        self.frozen_t_odom_layout = None
        self._cancel_event = threading.Event()
        self._active_traj_handle = None
        self._joints: dict[str, float] = {}
        self._jsp_joints: dict[str, float] = {}
        self._effort: dict[str, float] = {}
        self._tune: dict[str, object] | None = None
        self._contact_ok = False
        self._q_at_verify: float | None = None
        self._lift_detector: LiftContactDetector | None = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
            callback_group=self.callback_group,
        )
        self.runstop = self.create_client(SetBool, "/runstop")
        self.activate_streaming = self.create_client(
            Trigger, "/activate_streaming_position"
        )
        self.deactivate_streaming = self.create_client(
            Trigger, "/deactivate_streaming_position"
        )
        self.switch_position = self.create_client(
            Trigger, "/stretch/switch_to_position_mode"
        )
        self.switch_trajectory = self.create_client(
            Trigger, "/stretch/switch_to_trajectory_mode"
        )
        self.stop_the_robot = self.create_client(Trigger, "/stop_the_robot")
        self.joint_pose_pub = self.create_publisher(
            Float64MultiArray, "/joint_pose_cmd", 1
        )
        self.frozen_tf_pub = self.create_publisher(
            TransformStamped, "/robe_stretch/frozen_layout_tf", 1
        )
        self.create_subscription(
            JointState, "/stretch/joint_states", self._on_joints, 10
        )
        self.create_subscription(
            JointState, "/joint_states", self._on_jsp_joints, 10
        )
        if BedPull is None:
            self.get_logger().error(
                "robe_stretch_interfaces is not built; BedPull server disabled"
            )
            self._server = None
            return
        self._server = ActionServer(
            self,
            BedPull,
            "bed_pull",
            execute_callback=self.execute_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
            callback_group=self.callback_group,
        )
        self.get_logger().info(
            "BedPull action server ready on /bed_pull "
            "(EE = URDF-zero TF + lift/arm JSP-lag + wrist-down 0.23 m; "
            "no live TF broadcast)"
        )

    def _on_joints(self, msg: JointState) -> None:
        self._joints = dict(zip(msg.name, msg.position))
        if msg.effort and len(msg.effort) == len(msg.name):
            self._effort = dict(zip(msg.name, msg.effort))
        det = self._lift_detector
        if det is not None and det.active:
            effort = self._lift_effort_pct()
            if effort is not None:
                det.update(self._lift_q(), effort)

    def _on_jsp_joints(self, msg: JointState) -> None:
        self._jsp_joints = dict(zip(msg.name, msg.position))

    def goal_cb(self, goal_request):
        if goal_request.frame_id and goal_request.frame_id != "bed":
            self.get_logger().error(f"Rejected frame_id={goal_request.frame_id}")
            return GoalResponse.REJECT
        if self._session_dir is not None:
            try:
                assert_registration_frozen(self._session_dir)
            except SessionRegistrationError as exc:
                self.get_logger().error(f"Rejected /bed_pull: {exc}")
                return GoalResponse.REJECT
        try:
            self._tune = self._snapshot_tune_params()
        except (ValueError, SafetyReject) as exc:
            self.get_logger().error(f"Rejected /bed_pull: {exc}")
            return GoalResponse.REJECT
        self._contact_ok = False
        self._q_at_verify = None
        return GoalResponse.ACCEPT

    def cancel_cb(self, goal_handle):
        self._cancel_event.set()
        self._cancel_driver_goal()
        return CancelResponse.ACCEPT

    def _cancel_driver_goal(self) -> None:
        handle = self._active_traj_handle
        if handle is not None:
            try:
                handle.cancel_goal_async()
                return
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"cancel_goal_async failed: {exc}")
        self._trigger_runstop()

    def _trigger_runstop(self) -> None:
        if self.runstop.service_is_ready():
            req = SetBool.Request()
            req.data = True
            self.runstop.call_async(req)

    def _feedback(self, goal_handle, stage: str):
        if BedPull is None:
            return
        msg = BedPull.Feedback()
        msg.stage = stage
        goal_handle.publish_feedback(msg)
        self.get_logger().info(stage)

    def _result(
        self,
        success: bool,
        stage: str,
        message: str,
        *,
        fault_code: str = "",
    ):
        result = BedPull.Result()
        result.success = success
        if success and fault_code in ("", FAULT_OK):
            result.failure_stage = ""
        elif success:
            result.failure_stage = stage or fault_code
        else:
            result.failure_stage = fault_code or stage
        prefix = f"[{fault_code}] " if fault_code else ""
        result.message = f"{prefix}{message}" if not message.startswith("[") else message
        return result

    def _snapshot_tune_params(self) -> dict[str, object]:
        snap: dict[str, object] = {}
        for name in TUNE_PARAM_NAMES:
            snap[name] = self.get_parameter(name).value
        recipe = normalize_recipe(snap.get("pull.recipe", "full-pull"))
        stop = resolve_stop_after(
            recipe=recipe, stop_after=snap.get("pull.stop_after", "")
        )
        if not stop and bool(self.get_parameter("pull.stop_after_pregrasp").value):
            stop = "VERIFY_PREGRASP"
        snap["pull.recipe"] = recipe
        snap["pull.stop_after"] = stop
        snap["pull.controller"] = normalize_controller(snap.get("pull.controller"))
        snap["live_geometry.policy"] = normalize_live_geometry_policy(
            snap.get("live_geometry.policy")
        )
        production = recipe == "full-pull" and not bool(snap.get("pull.dry_run"))
        assert_goal_params(snap, production=production)
        self.get_logger().info(
            f"tune snapshot recipe={recipe} "
            f"stop_after={stop or 'DONE(full pull)'} "
            f"controller={snap['pull.controller']} "
            f"live_geometry={snap['live_geometry.policy']} "
            f"threshold_mode={snap.get('contact.threshold_mode')} "
            f"breakaway={snap.get('contact.breakaway_m')}"
        )
        return snap

    def _execution_wrist(self):
        return wrist_from_params(
            yaw_rad=float(self.get_parameter("gripper_down.yaw").value),
            pitch_rad=float(self.get_parameter("gripper_down.pitch").value),
            roll_rad=float(self.get_parameter("gripper_down.roll").value),
            planning_safe_extension_m=float(
                self.get_parameter("approach.retract_arm_m").value
            ),
        )

    def _build_probe_report(self, *, action=None, dry_run: bool = False) -> dict:
        pose = self._execution_wrist()
        joints = dict(self._joints)
        arm = self._wrist_extension() if joints else None
        report = {
            "dry_run": bool(dry_run),
            "controller": self._tune_val("pull.controller") if self._tune else None,
            "recipe": self._tune_val("pull.recipe") if self._tune else None,
            "stop_after": (
                self._tune_val("pull.stop_after") if self._tune else ""
            ),
            "live_geometry_policy": (
                self._tune_val("live_geometry.policy") if self._tune else None
            ),
            "wrist_target": {
                "yaw_rad": pose.yaw_rad,
                "pitch_rad": pose.pitch_rad,
                "roll_rad": pose.roll_rad,
            },
            "joints": {
                "joint_wrist_yaw": joints.get("joint_wrist_yaw"),
                "joint_wrist_pitch": joints.get("joint_wrist_pitch"),
                "joint_wrist_roll": joints.get("joint_wrist_roll"),
                "wrist_extension": arm,
                "joint_lift": joints.get("joint_lift"),
            },
            "jsp_lag": None,
            "snapshot_age_s": None,
            "expected_wrist_steps": [
                {
                    "name": step.name,
                    "joints": step.joints,
                    "duration_s": step.duration_s,
                    "contact_off": step.contact_off,
                    "reason": step.reason,
                }
                for step in plan_wrist_down_steps(
                    pitch_rad=float(joints.get("joint_wrist_pitch", 0.0)),
                    yaw_rad=float(joints.get("joint_wrist_yaw", 0.0)),
                    roll_rad=float(joints.get("joint_wrist_roll", 0.0)),
                    wrist_extension_m=float(arm or 0.0),
                    pose=pose,
                )
            ],
            "motion_sent": False,
        }
        if action is not None:
            report["action_bed"] = [float(v) for v in action]
        return report

    def _tune_val(self, name: str):
        if self._tune is None:
            raise RuntimeError(f"tune snapshot missing; cannot read {name}")
        if name not in self._tune:
            raise RuntimeError(f"tune snapshot has no {name}")
        return self._tune[name]

    def _stop_after_if(self, stage: str, extra: dict | None = None) -> None:
        if should_stop_after(stage, str(self._tune_val("pull.stop_after") or "")):
            raise IntentionalStop(stage, extra=extra or {})

    def _wait_future(self, future, timeout_sec: float):
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            if self._cancel_event.is_set():
                raise RuntimeError("canceled")
            if time.monotonic() > deadline:
                raise TimeoutError("timed out waiting for driver")
            time.sleep(0.02)
        return future.result()

    def _call_trigger(self, client, timeout_sec: float = 5.0) -> None:
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(f"{client.srv_name} not available")
        future = client.call_async(Trigger.Request())
        self._wait_future(future, timeout_sec)

    def _lookup(self, parent: str, child: str) -> np.ndarray:
        stamped = self.tf_buffer.lookup_transform(
            parent, child, rclpy.time.Time(), timeout=RclDuration(seconds=2.0)
        )
        return _stamp_to_mat(stamped)

    def _try_lookup(self, parent: str, child: str) -> np.ndarray | None:
        if not self.tf_buffer.can_transform(
            parent, child, rclpy.time.Time(), timeout=RclDuration(seconds=0.0)
        ):
            return None
        try:
            return _stamp_to_mat(
                self.tf_buffer.lookup_transform(
                    parent, child, rclpy.time.Time()
                )
            )
        except Exception:  # noqa: BLE001
            return None

    def _wrist_extension(self) -> float:
        if "wrist_extension" in self._joints:
            return float(self._joints["wrist_extension"])
        if "joint_arm" in self._joints:
            return float(self._joints["joint_arm"])
        return float(sum(self._joints.get(f"joint_arm_l{i}", 0.0) for i in range(4)))

    def _joint(self, name: str, default: float = 0.0) -> float:
        return float(self._joints.get(name, default))

    def _lift_q(self) -> float:
        for name in ("joint_lift", "lift"):
            if name in self._joints:
                return float(self._joints[name])
        return float(self._joints.get("joint_lift", 0.0))

    def _jsp_arm(self) -> float:
        if "wrist_extension" in self._jsp_joints:
            return float(self._jsp_joints["wrist_extension"])
        if "joint_arm" in self._jsp_joints:
            return float(self._jsp_joints["joint_arm"])
        return float(
            sum(self._jsp_joints.get(f"joint_arm_l{i}", 0.0) for i in range(4))
        )

    def _tf_joint_delta(self) -> tuple[float, float]:
        """stretch_driver minus JSP. RSP reads JSP, which is often stuck at 0."""

        d_lift = self._lift_q() - float(self._jsp_joints.get("joint_lift", 0.0))
        d_arm = self._wrist_extension() - self._jsp_arm()
        return float(d_lift), float(d_arm)

    def _correct_ee_base(self, p_base: np.ndarray) -> np.ndarray:
        corrected, _, _ = ee_base_from_tf(
            p_base,
            lift_real=self._lift_q(),
            arm_real=self._wrist_extension(),
            lift_jsp=float(self._jsp_joints.get("joint_lift", 0.0)),
            arm_jsp=self._jsp_arm(),
            workspace=self.workspace,
        )
        return apply_wrist_down_grasp_offset(
            corrected,
            pitch_driver=self._joint("joint_wrist_pitch"),
            pitch_jsp=float(self._jsp_joints.get("joint_wrist_pitch", 0.0)),
            workspace=self.workspace,
        )

    def _move_to_pose(
        self,
        pose: dict,
        *,
        custom_contact_thresholds: bool = False,
        duration: float = 2.0,
        expect_contact: bool = False,
        blocking: bool = True,
    ):
        if self._cancel_event.is_set():
            raise RuntimeError("canceled")
        # Do not call wait_for_server() from BedPull execute_cb (nested wait).
        deadline = time.monotonic() + 8.0
        while not self.traj_client.server_is_ready():
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "FollowJointTrajectory has no server — stretch_driver "
                    "is not running. On Stretch: ros2 launch stretch_core "
                    "stretch_driver.launch.py broadcast_odom_tf:=True "
                    "(keep this action_executor; do not relaunch "
                    "execute_pull.launch.py)"
                )
            time.sleep(0.05)
        goal = follow_joint_trajectory_goal(
            pose,
            custom_contact_thresholds=custom_contact_thresholds,
            duration_s=duration,
        )
        send = self.traj_client.send_goal_async(goal)
        handle = self._wait_future(send, duration + 5.0)
        if handle is None or not handle.accepted:
            raise RuntimeError("FollowJointTrajectory rejected")
        self._active_traj_handle = handle
        if not blocking:
            return handle
        result_future = handle.get_result_async()
        wrapped = self._wait_future(result_future, max(float(duration), 8.0) + 8.0)
        self._active_traj_handle = None
        result = wrapped.result if wrapped is not None else None
        if expect_contact:
            return result
        if result is not None and getattr(result, "error_code", 0) not in (0, None):
            raise RuntimeError(
                f"trajectory error_code={result.error_code} {getattr(result, 'error_string', '')}"
            )
        return result

    def _layout_snapshot_file(self) -> Path:
        explicit = str(self.get_parameter("layout_snapshot_path").value).strip()
        if explicit:
            return Path(explicit)
        if self._session_paths is not None:
            if self._session_paths.origin_corrected.is_file():
                return self._session_paths.origin_corrected
            if self._session_paths.origin_raw.is_file():
                return self._session_paths.origin_raw
            return self._session_paths.origin_json
        manifest = str(self.get_parameter("manifest_dir").value).strip()
        if manifest:
            return resolve_stretch_origin_path(Path(manifest))
        return Path("/tmp/robe/stretch_origin.json")

    def _load_layout_snapshot(self) -> np.ndarray:
        if self._session_dir is not None:
            try:
                assert_registration_frozen(self._session_dir)
            except SessionRegistrationError as exc:
                raise RuntimeError(str(exc)) from exc
        path = self._layout_snapshot_file()
        if not path.is_file():
            raise RuntimeError(
                f"No layout snapshot at {path}. Run sample_origin on Stretch "
                "and ceiling 136 XY registration; freeze session.json."
            )
        data = json.loads(path.read_text())
        if not data.get("ok") or "T_odom_layout" not in data:
            raise RuntimeError(f"{path} has no T_odom_layout (ok={data.get('ok')})")
        t_ol = t_odom_layout_for_motion(
            np.asarray(data["T_odom_layout"], dtype=np.float64)
        )
        if t_ol.shape != (4, 4):
            raise RuntimeError(f"{path} T_odom_layout shape {t_ol.shape}")
        self.get_logger().info(
            f"Loaded T_odom_layout from {path} R[2,2]={t_ol[2, 2]:.2f} "
            "(raw / overlay convention; not ensure_layout_z_up)"
        )
        return t_ol

    def _freeze_snapshot(self, t_odom_layout: np.ndarray) -> None:
        self.frozen_t_odom_layout = np.asarray(t_odom_layout, dtype=np.float64)
        stamp = self.get_clock().now().to_msg()
        msg = _mat_to_stamp(
            self.frozen_t_odom_layout,
            str(self.get_parameter("frames.odom").value),
            str(self.get_parameter("frames.layout").value),
            stamp,
        )
        self.frozen_tf_pub.publish(msg)

    def _trial_paths(self) -> tuple[Path, Path]:
        """canonical_bed_frame.json and trial dir. Read at /bed_pull time.

        Launch args still override. Otherwise use session/active_trial.json
        written by run_trial / send_bed_pull — no trial paths at launch.
        """

        explicit = str(self.get_parameter("canonical_frame_path").value).strip()
        manifest = str(self.get_parameter("manifest_dir").value).strip()
        if explicit:
            canon = Path(explicit)
            dest = Path(manifest) if manifest else canon.parent
            return canon, dest
        if self._session_dir is not None:
            active = load_active_trial(self._session_dir)
            if active is not None:
                pose = Path(active["pose_dir"])
                return Path(active["canonical_frame_path"]), pose
        raise RuntimeError(
            "No trial canonical_bed_frame. Start the executor with no "
            "trial args; run_trial writes session/active_trial.json "
            "before /bed_pull. Or pass canonical_frame_path:=..."
        )

    def _canonical_frame(self):
        canon, _dest = self._trial_paths()
        if not canon.is_file():
            raise RuntimeError(f"canonical_frame_path missing: {canon}")
        return load_canonical_frame(canon)

    def _cloth_z_layout(self) -> float:
        """Median blanket Z in the layout/bed frame. Fallback 0.08 m."""

        try:
            _canon, pose = self._trial_paths()
        except RuntimeError:
            pose = None
        candidates = []
        if pose is not None:
            candidates.extend(
                [
                    pose / "initial" / "blanket_pcd.pcd",
                    pose / "blanket_pcd.pcd",
                ]
            )
        for path in candidates:
            if not path.is_file():
                continue
            try:
                from overlay_action_on_ceiling import _load_blanket_points

                pts = _load_blanket_points(path)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"cloth Z read {path} failed: {exc}")
                continue
            if pts is None or not len(pts):
                continue
            z = float(np.median(pts[:, 2]))
            self.get_logger().info(f"cloth Z layout median={z:.4f} m from {path}")
            return z
        if pose is not None:
            snap = pose / "stretch_reach_snapshot.json"
            if snap.is_file():
                try:
                    data = json.loads(snap.read_text())
                    z_snap = data.get("cloth_z_layout")
                    if z_snap is not None:
                        z = float(z_snap)
                        self.get_logger().info(
                            f"cloth Z layout={z:.4f} m from {snap}"
                        )
                        return z
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"cloth Z snapshot {snap} failed: {exc}")
        z = 0.08
        self.get_logger().warn(f"cloth Z fallback {z:.3f} m (no blanket PCD)")
        return z

    def _map_canonical_to_layout(
        self, xy: np.ndarray, frame: CanonicalBedFrame, z_layout: float
    ) -> np.ndarray:
        """Canonical XY → layout XY; force layout Z to the cloth plane.

        ``z_layout`` is blanket height in the layout/bed frame (PCD median).
        Do not stuff that number into canonical Z — tilt then inflates
        layout Z and makes odom "above cloth" come out inverted.
        """

        xy_l = canonical_xy_to_layout(
            np.asarray(xy, dtype=np.float64).reshape(2),
            frame,
            z=float(z_layout),
        )
        return np.array(
            [float(xy_l[0]), float(xy_l[1]), float(z_layout)], dtype=np.float64
        )

    def _layout_to_base(
        self, p_layout: np.ndarray, *, ee_z_base: float | None = None
    ) -> np.ndarray:
        if self.frozen_t_odom_layout is None:
            raise RuntimeError("T_odom_layout snapshot missing")
        p_h = np.append(np.asarray(p_layout, dtype=np.float64).reshape(3), 1.0)
        p_odom = self.frozen_t_odom_layout @ p_h
        t_odom_base = self._lookup(
            str(self.get_parameter("frames.odom").value),
            str(self.get_parameter("frames.base").value),
        )
        p_base_h = invert_transform(t_odom_base) @ p_odom
        p_base = p_base_h[:3].copy()
        if ee_z_base is not None:
            p_base[2] = float(ee_z_base)
        return p_base

    def _hover_to_base(self, p_layout: np.ndarray) -> np.ndarray:
        """Target in base_link for hovering over a layout XY at current EE height."""

        return self._layout_to_base(
            layout_xy_at_height(p_layout, float(self._ee_layout()[2])),
            ee_z_base=float(self._ee_base()[2]),
        )

    def _ee_odom(self) -> np.ndarray:
        t_ob = self._lookup(
            str(self.get_parameter("frames.odom").value),
            str(self.get_parameter("frames.base").value),
        )
        p = t_ob @ np.append(self._ee_base(), 1.0)
        return p[:3].copy()

    def _lift_for_ee_z_odom(self, z_ee_target_odom: float) -> float:
        z_now = float(self._ee_odom()[2])
        q_now = self._lift_q()
        q = q_now + (float(z_ee_target_odom) - z_now)
        lo = float(self.workspace.lift_min_m)
        hi = float(self.workspace.lift_max_m)
        return float(np.clip(q, lo, hi))

    def _point_odom(self, p_layout: np.ndarray) -> np.ndarray:
        p_h = np.append(np.asarray(p_layout, dtype=np.float64).reshape(3), 1.0)
        return (self.frozen_t_odom_layout @ p_h)[:3]

    def _ee_layout(self) -> np.ndarray:
        t_odom_base = self._lookup(
            str(self.get_parameter("frames.odom").value),
            str(self.get_parameter("frames.base").value),
        )
        p_odom = t_odom_base @ np.append(self._ee_base(), 1.0)
        p_layout = invert_transform(self.frozen_t_odom_layout) @ p_odom
        return p_layout[:3]

    def _retract_arm(self, *, label: str = "RETRACT") -> None:
        """Pull the arm in. Contact off: leftover cloth/firmware 100 is not a fail."""

        target = float(self._tune_val("approach.retract_arm_m"))
        current = self._wrist_extension()
        if current <= target + 0.01:
            self.get_logger().info(
                f"{label} keep arm={current:.3f} m (already <= {target:.3f})"
            )
            return
        effort = self._disable_effort()
        self.get_logger().info(
            f"{label} joint_arm {current:.3f} -> {target:.3f} m (contact off)"
        )
        result = self._move_to_pose(
            {"joint_arm": (float(target), effort)},
            custom_contact_thresholds=True,
            duration=3.0,
            expect_contact=True,
        )
        after = self._wrist_extension()
        if after <= target + 0.03:
            return
        if is_force_contact(result):
            self.get_logger().warn(
                f"{label} error 100 retracting arm {current:.3f}->{after:.3f}; "
                "clear latch and retry"
            )
            self._clear_lift_guarded_event()
            self._move_to_pose(
                {"joint_arm": (float(target), effort)},
                custom_contact_thresholds=True,
                duration=3.0,
                expect_contact=True,
            )
            after = self._wrist_extension()
        if after > target + 0.05:
            raise RuntimeError(
                f"{label} arm still {after:.3f} m (want {target:.3f})"
            )

    def _safe_lift_up_only(
        self,
        z_ee_target_odom: float,
        *,
        label: str,
        z_min_ok: float | None = None,
    ) -> None:
        """Raise-only. Never command a Z the lift cannot reach.

        ``pull.clearance_above_bed_m`` (40 cm) is the post-grasp pull
        height, not a pregrasp requirement. Wrist-down EE Z sits below
        ``joint_lift``; cloth+40 cm is often above lift_max.
        """

        z_now = float(self._ee_odom()[2])
        q_now = self._lift_q()
        q_target = self._lift_for_ee_z_odom(z_ee_target_odom)
        hi = float(self.workspace.lift_max_m)
        z_reachable = z_now + (q_target - q_now)
        if z_now + 0.005 >= min(float(z_ee_target_odom), z_reachable + 0.01):
            self.get_logger().info(
                f"{label} keep lift q={q_now:.3f} EE odom Z={z_now:.3f} "
                f"(want {z_ee_target_odom:.3f}, reachable {z_reachable:.3f})"
            )
            return
        if q_target <= q_now + 0.005:
            self.get_logger().info(
                f"{label} lift already at q={q_now:.3f} (max {hi:.3f}), "
                f"EE odom Z={z_now:.3f}"
            )
            return
        self.get_logger().info(
            f"{label} lift up q {q_now:.3f} -> {q_target:.3f} "
            f"EE odom Z {z_now:.3f} -> reachable {z_reachable:.3f} "
            f"(want {z_ee_target_odom:.3f}; contact disabled)"
        )
        result = self._lift_up(q_target, label=label, duration=5.0)
        z_after = float(self._ee_odom()[2])
        q_after = self._lift_q()
        err = traj_error_code(result)
        err_s = getattr(result, "error_string", "") if result is not None else ""
        at_max = q_after >= hi - 0.03
        reached = z_after + 0.02 >= min(float(z_ee_target_odom), z_reachable)
        above_min = z_min_ok is None or z_after + 0.01 >= float(z_min_ok)
        if reached or (at_max and above_min and z_after > z_now + 0.01):
            if not reached:
                self.get_logger().warn(
                    f"{label} lift at max q={q_after:.3f}; "
                    f"EE odom Z {z_after:.3f} < want {z_ee_target_odom:.3f}, "
                    "continuing (wrist-down cannot reach cloth+40 cm)"
                )
            return
        raise RuntimeError(
            f"{label} raise stalled: EE odom Z {z_now:.3f} -> {z_after:.3f} "
            f"(want {z_ee_target_odom:.3f}, reachable {z_reachable:.3f}); "
            f"joint_lift {q_now:.3f} -> {q_after:.3f}. "
            f"trajectory error_code={err} {err_s}."
        )

    def _arm_unreachable_msg(self, wrist_target: float) -> str:
        hi = float(self.workspace.arm_max_m)
        lo = float(self.workspace.arm_min_m)
        w = float(wrist_target)
        if w < lo:
            return (
                f"REJECT retract past zero: joint_arm={w:.3f} m. "
                "Reported EE is already past the grasp into the bed; "
                "arm cannot go negative. Not a 'mast closer' problem "
                "(that is only when joint_arm > 0.52)."
            )
        excess = w - hi
        return (
            f"REJECT unreachable (do not clip): joint_arm={w:.3f} m "
            f"outside [{lo}, {hi}]. "
            f"Need the mast ~{max(excess, 0.0)*100:.0f} cm closer to the bed. "
            "Into-bed is arm −Y_base; translate_mobile_base cannot eat this. "
            "Not a home/lift problem; SAFE_LIFT already finished."
        )

    def _base_translate_only(self, target_ee_base: np.ndarray) -> float:
        cmd = command_toward_target(
            current_ee_base=self._ee_base(),
            target_ee_base=target_ee_base,
            current_wrist_extension=self._wrist_extension(),
            current_lift=self._joint("joint_lift"),
            workspace=self.workspace,
        )
        d_base = float(cmd.translate_mobile_base)
        if abs(d_base) <= 0.005:
            self.get_logger().info("BASE_APPROACH skip translate_mobile_base (<5 mm)")
            return 0.0
        duration = float(max(3.0, min(16.0, abs(d_base) / 0.08)))
        self.get_logger().info(f"BASE_APPROACH translate_mobile_base={d_base:.3f} m")
        self._move_to_pose(
            {"translate_mobile_base": d_base},
            duration=duration,
        )
        time.sleep(0.3)
        return d_base

    def _clamp_arm_command(self, wrist: float) -> float:
        """Restrict the commanded arm to hardware [0, 0.52].

        Negative means TF already past the grasp — retract to 0, do not
        abort as 'mast closer'. Over-max still rejects (cannot invent stroke).
        """

        lo = float(self.workspace.arm_min_m)
        hi = float(self.workspace.arm_max_m)
        w = float(wrist)
        if w > hi + EXEC_ARM_SLACK_M:
            raise RuntimeError(self._arm_unreachable_msg(w))
        if w < lo:
            self.get_logger().warn(
                f"joint_arm plan {w:.3f} m < 0; restricting to {lo:.3f} "
                "(EE already past grasp in TF, not a mast-closer reject)"
            )
        return float(min(max(w, lo), hi))

    def _arm_extend_only(self, target_ee_base: np.ndarray) -> None:
        cmd = command_toward_target(
            current_ee_base=self._ee_base(),
            target_ee_base=target_ee_base,
            current_wrist_extension=self._wrist_extension(),
            current_lift=self._joint("joint_lift"),
            workspace=self.workspace,
        )
        wrist = self._clamp_arm_command(cmd.wrist_extension)
        effort = self._disable_effort()
        self.get_logger().info(
            f"ARM_APPROACH joint_arm={wrist:.3f} m "
            f"(plan {cmd.wrist_extension:.3f}, contact off)"
        )
        result = self._move_to_pose(
            {"joint_arm": (wrist, effort)},
            custom_contact_thresholds=True,
            duration=4.0,
            expect_contact=True,
        )
        after = self._wrist_extension()
        if after < wrist - 0.03 and is_force_contact(result):
            self.get_logger().warn(
                f"ARM_APPROACH error 100 arm {after:.3f} want {wrist:.3f}; "
                "clear latch and retry"
            )
            self._clear_lift_guarded_event()
            self._move_to_pose(
                {"joint_arm": (wrist, effort)},
                custom_contact_thresholds=True,
                duration=4.0,
                expect_contact=True,
            )

    def _refine_pregrasp_xy(self, grasp_layout: np.ndarray, *, passes: int) -> float:
        """Re-query TF and eat leftover base/arm after the first approach."""

        extra = 0.0
        for i in range(max(0, int(passes))):
            target = self._hover_to_base(grasp_layout)
            cmd = command_toward_target(
                current_ee_base=self._ee_base(),
                target_ee_base=target,
                current_wrist_extension=self._wrist_extension(),
                current_lift=self._joint("joint_lift"),
                workspace=self.workspace,
            )
            d_base = float(cmd.translate_mobile_base)
            d_arm = float(cmd.wrist_extension - self._wrist_extension())
            self.get_logger().info(
                f"PREGRASP_REFINE pass {i + 1}/{passes} "
                f"d_base={d_base:+.3f} d_arm={d_arm:+.3f}"
            )
            moved = False
            if abs(d_base) > 0.005:
                extra += self._base_translate_only(target)
                moved = True
            target = self._hover_to_base(grasp_layout)
            cmd = command_toward_target(
                current_ee_base=self._ee_base(),
                target_ee_base=target,
                current_wrist_extension=self._wrist_extension(),
                current_lift=self._joint("joint_lift"),
                workspace=self.workspace,
            )
            if abs(cmd.wrist_extension - self._wrist_extension()) > 0.005:
                self._arm_extend_only(target)
                moved = True
            if not moved:
                break
        return extra

    def _verify_pregrasp(
        self,
        grasp_canonical_xy: np.ndarray,
        frame: CanonicalBedFrame,
        *,
        z_cloth_odom: float,
        z_cloth_layout: float,
        d_base_cmd: float,
    ) -> dict:
        target = np.asarray(grasp_canonical_xy, float).reshape(2)
        ee = self._ee_layout()
        # Same layout Z as the mapped grasp (cloth). High-Z EE otherwise
        # leaks T's Z column into canonical XY (~3 cm at 40 cm standoff).
        ee_same_z = layout_xy_at_height(ee, float(z_cloth_layout))
        p_can = layout_points_to_canonical(ee_same_z.reshape(1, 3), frame)[0]
        delta = p_can[:2] - target
        err = float(np.linalg.norm(delta))
        z_ee_odom = float(self._ee_odom()[2])
        z_above_cloth = z_ee_odom - float(z_cloth_odom)
        arm = float(self._wrist_extension())
        ee_base = self._ee_base()
        payload = {
            "target_grasp_canonical_xy": target.round(6).tolist(),
            "actual_grasp_center_canonical_xy": p_can[:2].round(6).tolist(),
            "dx_m": float(delta[0]),
            "dy_m": float(delta[1]),
            "e_g_m": err,
            "grasp_center_z_above_cloth_m": z_above_cloth,
            "base_translate_commanded_m": float(d_base_cmd),
            "final_arm_extension_m": arm,
            "ee_base": ee_base.round(6).tolist(),
            "ee_source": "jsp_lag_wrist_down",
            "live_tf_prismatic": False,
        }
        self.get_logger().info(
            "VERIFY_PREGRASP "
            f"target grasp canonical XY={payload['target_grasp_canonical_xy']} "
            f"actual grasp_center canonical XY="
            f"{payload['actual_grasp_center_canonical_xy']} "
            f"dx={delta[0]:+.4f} dy={delta[1]:+.4f} e_g={err:.4f} m "
            f"Z_above_cloth={z_above_cloth:.3f} m "
            f"base_cmd={d_base_cmd:+.3f} m arm={arm:.3f} m"
        )
        return payload

    def _ee_base(self) -> np.ndarray:
        t = self._lookup(
            str(self.get_parameter("frames.base").value),
            str(self.get_parameter("frames.grasp_center").value),
        )
        raw = t[:3, 3].copy()
        corrected = self._correct_ee_base(raw)
        d_lift, d_arm = self._tf_joint_delta()
        if abs(d_lift) > 0.05 or abs(d_arm) > 0.05:
            self.get_logger().warn(
                f"TF EE stale vs /stretch/joint_states: "
                f"Δlift={d_lift:+.3f} Δarm={d_arm:+.3f} "
                f"tf={np.round(raw, 3).tolist()} "
                f"corr={np.round(corrected, 3).tolist()}",
                throttle_duration_sec=2.0,
            )
        return corrected

    def _ee_layout_xy(self) -> np.ndarray:
        t_odom_base = self._lookup(
            str(self.get_parameter("frames.odom").value),
            str(self.get_parameter("frames.base").value),
        )
        p_odom = t_odom_base @ np.append(self._ee_base(), 1.0)
        p_layout = invert_transform(self.frozen_t_odom_layout) @ p_odom
        return p_layout[:2]

    def _command_ee_base(self, target_ee_base: np.ndarray, *, duration: float = 2.0):
        """One Hello-style combined pose (used on the pull line)."""

        cmd = command_toward_target(
            current_ee_base=self._ee_base(),
            target_ee_base=target_ee_base,
            current_wrist_extension=self._wrist_extension(),
            current_lift=self._joint("joint_lift"),
            workspace=self.workspace,
        )
        self._move_to_pose(
            {
                "translate_mobile_base": cmd.translate_mobile_base,
                "joint_arm": cmd.wrist_extension,
                "joint_lift": cmd.joint_lift,
            },
            duration=duration,
        )
        return cmd

    def _approach_grasp_hello(self, target_ee_base: np.ndarray) -> None:
        """Official split: translate_mobile_base, re-query TF, then joint_arm."""

        def _plan():
            return command_toward_target(
                current_ee_base=self._ee_base(),
                target_ee_base=target_ee_base,
                current_wrist_extension=self._wrist_extension(),
                current_lift=self._joint("joint_lift"),
                workspace=self.workspace,
            )

        cmd = _plan()
        self.get_logger().info(
            f"pregrasp translate_mobile_base={cmd.translate_mobile_base:.3f} m "
            f"wrist_extension={cmd.wrist_extension:.3f} m "
            f"(+X_base drive, -Y_base arm)"
        )
        if abs(cmd.translate_mobile_base) > 0.005:
            duration = float(
                max(3.0, min(16.0, abs(cmd.translate_mobile_base) / 0.08))
            )
            self._move_to_pose(
                {"translate_mobile_base": float(cmd.translate_mobile_base)},
                duration=duration,
            )
            time.sleep(0.3)
        cmd = _plan()
        ee = self._ee_base()
        self.get_logger().info(
            f"after base ee={ee.round(3).tolist()} "
            f"arm_target={cmd.wrist_extension:.3f} m"
        )
        wrist = self._clamp_arm_command(cmd.wrist_extension)
        self._move_to_pose(
            {"joint_arm": (wrist, self._disable_effort())},
            custom_contact_thresholds=True,
            duration=4.0,
            expect_contact=True,
        )

    def _wrist_down(self, *, stage: str = "WRIST_DOWN") -> None:
        pose = self._execution_wrist()
        steps = plan_wrist_down_steps(
            pitch_rad=self._joint("joint_wrist_pitch"),
            yaw_rad=self._joint("joint_wrist_yaw"),
            roll_rad=self._joint("joint_wrist_roll"),
            wrist_extension_m=self._wrist_extension(),
            pose=pose,
        )
        disable = self._disable_effort()
        for step in steps:
            joints = dict(step.joints)
            if step.contact_off and "joint_arm" in joints:
                joints["joint_arm"] = (float(joints["joint_arm"]), disable)
            self.get_logger().info(
                f"{stage} {step.name}: {step.reason or step.joints}"
            )
            self._move_to_pose(
                joints,
                custom_contact_thresholds=step.contact_off,
                duration=step.duration_s,
                expect_contact=step.contact_off,
            )
        time.sleep(float(self.get_parameter("gripper_down.settle_s").value))
        assert_execution_wrist_joints(
            pitch_rad=self._joint("joint_wrist_pitch"),
            yaw_rad=self._joint("joint_wrist_yaw"),
            roll_rad=self._joint("joint_wrist_roll"),
            wrist_extension_m=self._wrist_extension(),
            pose=pose,
            stage=stage,
        )

    def _wrist_tag_transform(self) -> list | None:
        try:
            t_parent_tag = self._lookup(
                str(self.get_parameter("frames.odom").value),
                str(self.get_parameter("frames.wrist_tag").value),
            )
            t_parent_grasp = self._lookup(
                str(self.get_parameter("frames.odom").value),
                str(self.get_parameter("frames.grasp_center").value),
            )
            t = invert_transform(t_parent_tag) @ t_parent_grasp
            return t.tolist()
        except Exception:  # noqa: BLE001
            return None

    def _lower_duration(self, travel_m: float) -> float:
        speed = max(float(self._tune_val("contact.lower_speed_m_s")), 0.02)
        return float(max(3.0, min(16.0, abs(travel_m) / speed)))

    def _disable_effort(self) -> float:
        return float(self._tune_val("contact.disable_effort"))

    def _lift_effort_pct(self) -> float | None:
        for name in ("joint_lift", "lift"):
            if name in self._effort:
                return float(self._effort[name])
        return None

    def _lower_until_contact(
        self, *, z_cloth_odom: float, z_before: float, q_before: float
    ) -> dict:
        """Relative lift-effort unload vs in-motion hover. Floor is panic only."""

        drop_pct = float(self._tune_val("contact.unload_drop_pct") or 15.0)
        consecutive = int(self._tune_val("contact.unload_consecutive") or 2)
        n_base = int(self._tune_val("contact.baseline_samples") or 15)
        settle = float(self._tune_val("contact.baseline_settle_s") or 0.5)
        increment = float(
            self._tune_val("contact.move_increment_m") or FUNMAP_MOVE_INCREMENT_M
        )
        period = FUNMAP_PERIOD_S
        log_trace = bool(self._tune_val("contact.log_trace"))
        max_d = float(self._tune_val("contact.max_descent_m"))
        q0 = self._lift_q()
        z0 = float(self._ee_odom()[2])
        z_cloth = float(z_cloth_odom)
        lowest = descent_floor_q(
            q_start=q0,
            max_descent_m=max_d,
            lift_min_m=float(self.workspace.lift_min_m),
            lowest_allowed_m=float(self._tune_val("contact.lowest_allowed_m")),
        )
        detector = LiftContactDetector(
            unload_drop_pct=drop_pct,
            unload_consecutive=consecutive,
            baseline_samples=n_base,
        )
        self._lift_detector = detector
        detector.reset()
        detector.active = True
        t0 = time.monotonic()
        rows: list[dict] = []
        csv_path = None
        settle_end = time.monotonic() + max(0.0, settle)
        while time.monotonic() < settle_end:
            if self._cancel_event.is_set():
                raise RuntimeError("canceled")
            time.sleep(0.05)
        hold_effort = detector.sample_median()
        self.get_logger().info(
            f"LOWER_UNTIL_CONTACT q={q0:.3f} EE={z0:.3f} "
            f"hold_effort={format_effort_pct(hold_effort)} (diagnostic) "
            f"unload>={drop_pct:g}% after in-motion baseline "
            f"panic_floor_q={lowest:.3f} (mechanical start-{max_d:.3f}m) "
            f"increment={increment:.3f}"
        )
        motion_started = False
        motion_base_end = None
        # One Funmap step, then collect descending-hover samples.
        arm_after_drop = max(float(increment), 0.008)
        try:
            deadline = time.monotonic() + 25.0
            while True:
                if self._cancel_event.is_set():
                    raise RuntimeError("canceled")
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "DESCEND_CONTACT CONTACT_NOT_FOUND: timed out "
                        f"(q={self._lift_q():.3f} lowest={lowest:.3f})"
                    )
                q = self._lift_q()
                effort = self._lift_effort_pct()
                if not motion_started and q0 - q >= arm_after_drop:
                    motion_started = True
                    detector.begin_motion_baseline()
                    if effort is not None:
                        detector.update(q, effort)
                    motion_base_end = time.monotonic() + max(0.3, settle)
                    self.get_logger().info(
                        f"MOTION_BASELINE collecting after downward travel "
                        f"{q0 - q:.4f} m for {max(0.3, settle):.2f}s"
                    )
                if (
                    motion_started
                    and not detector.armed
                    and motion_base_end is not None
                    and time.monotonic() >= motion_base_end
                    and detector.has_motion_samples()
                ):
                    detector.lock_motion_baseline()
                    need = abs(float(detector.baseline)) * drop_pct / 100.0
                    self.get_logger().info(
                        f"RELATIVE_EFFORT_ARMED "
                        f"baseline={format_effort_pct(detector.baseline)} "
                        f"unload>={drop_pct:g}% (need {need:.1f})"
                    )
                avg = detector.av_effort
                if effort is not None and avg is None:
                    avg = effort
                z_now = float(self._ee_odom()[2])
                if not motion_started:
                    phase = "hover"
                elif not detector.armed:
                    phase = "motion_baseline"
                else:
                    phase = "descend"
                rows.append(
                    contact_trace_row(
                        t0=t0,
                        q_lift=q,
                        z_ee=z_now,
                        effort=effort,
                        effort_filtered=avg,
                        baseline=detector.baseline,
                        phase=phase,
                    )
                )
                self.get_logger().info(
                    f"q={q:.3f} EE={z_now:.3f} "
                    f"effort={format_effort_pct(effort)} "
                    f"filt={format_effort_pct(avg)} "
                    f"phase={phase}"
                )
                if detector.armed and detector.in_contact:
                    self._try_trigger(self.stop_the_robot)
                    time.sleep(0.15)
                    q_c = (
                        detector.contact_q
                        if detector.contact_q is not None
                        else self._lift_q()
                    )
                    e_c = (
                        detector.contact_effort
                        if detector.contact_effort is not None
                        else effort
                    )
                    z_c = float(self._ee_odom()[2])
                    self.get_logger().info(
                        f"CONTACT reason=relative_effort "
                        f"baseline={format_effort_pct(detector.baseline)} "
                        f"effort={format_effort_pct(e_c)} "
                        f"filt={format_effort_pct(avg)} "
                        f"q={q_c:.3f} EE={z_c:.3f}"
                    )
                    return {
                        "contact_z": z_c,
                        "contact_effort": e_c,
                        "avg_effort": avg,
                        "contact_reason": "relative_effort",
                        "hover_baseline": detector.baseline,
                        "q_start": q0,
                        "q_contact": q_c,
                        "z_start": z0,
                        "ee_odom_z_before_m": z_before,
                        "ee_odom_z_after_m": z_c,
                        "joint_lift_before_m": q_before,
                        "joint_lift_after_m": q_c,
                        "dropped_m": z_before - z_c,
                        "dropped_q_m": q0 - float(q_c),
                        "z_cloth_estimate_m": z_cloth,
                        "lowest_q": lowest,
                        "max_descent_m": max_d,
                        "log_trace": log_trace,
                        "steps": rows,
                        "force_contact": False,
                    }
                if passed_stopping_position(q, lowest, -1):
                    self._try_trigger(self.stop_the_robot)
                    z_ee = float(self._ee_odom()[2])
                    raise RuntimeError(
                        "DESCEND_CONTACT CONTACT_NOT_FOUND: hit panic floor "
                        f"q {q0:.3f}->{q:.3f} (lowest {lowest:.3f}) "
                        f"EE {z0:.3f}->{z_ee:.3f} "
                        f"effort={format_effort_pct(effort)} "
                        f"filt={format_effort_pct(avg)} "
                        f"(relative unload {drop_pct:g}% never crossed, "
                        f"baseline={format_effort_pct(detector.baseline)})"
                    )
                target = max(float(self.workspace.lift_min_m), q - increment)
                self.get_logger().info(f"target={target:.3f}")
                self._move_to_pose(
                    {"joint_lift": float(target)},
                    custom_contact_thresholds=False,
                    duration=0.0,
                    expect_contact=True,
                    blocking=False,
                )
                time.sleep(period)
        finally:
            detector.active = False
            self._lift_detector = None
            if log_trace and rows:
                try:
                    csv_path = write_contact_trace_csv(
                        self._contact_trace_path(), rows
                    )
                    self.get_logger().info(f"contact trace csv={csv_path}")
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().error(f"contact trace write failed: {exc}")
        raise RuntimeError("DESCEND_CONTACT: lower loop exited")

    def _contact_trace_path(self) -> Path:
        raw = str(self._tune_val("contact.trace_path") or "").strip()
        if raw:
            return Path(raw).expanduser()
        try:
            _canon, pose = self._trial_paths()
            dest = pose
        except RuntimeError:
            dest = Path.cwd()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return Path(dest) / f"contact_trace_{stamp}.csv"

    def _lift_up(self, q_target: float, *, label: str, duration: float = 2.0):
        """Official raise: ``{'joint_lift': q}``, no contact thresholds.

        ``duration`` is ignored; HelloNode position-mode uses time_from_start=0.
        """

        q_now = self._lift_q()
        hi = float(self.workspace.lift_max_m)
        target = min(hi, float(q_target))
        if target <= q_now + 0.003:
            return None
        self.get_logger().info(
            f"{label} joint_lift {q_now:.3f} -> {target:.3f} "
            f"(raise, no custom contact)"
        )
        result = self._move_to_pose(
            {"joint_lift": float(target)},
            duration=0.0,
            expect_contact=True,
        )
        if is_force_contact(result):
            self.get_logger().warn(
                f"{label} error 100 on raise; /stop_the_robot then retry"
            )
            self._clear_lift_guarded_event()
            result = self._move_to_pose(
                {"joint_lift": float(target)},
                duration=0.0,
                expect_contact=True,
            )
        return result

    def _raise_unlatch(self, bump_m: float = 0.015) -> None:
        """Nudge lift up so a prior contact latch does not stall the next drop."""

        q_now = self._lift_q()
        target = min(float(self.workspace.lift_max_m), q_now + float(bump_m))
        self._lift_up(target, label="DESCEND_UNLATCH", duration=1.2)

    def _breakaway_down(self, travel_m: float) -> None:
        """Force a short drop with contact off. Needed when lift is already at max."""

        q_now = self._lift_q()
        target = max(self.workspace.lift_min_m, q_now - float(travel_m))
        if target >= q_now - 0.005:
            return
        effort = self._disable_effort()
        duration = max(1.5, self._lower_duration(q_now - target))
        self.get_logger().info(
            f"DESCEND_BREAKAWAY joint_lift {q_now:.3f} -> {target:.3f} "
            f"(contact off, {float(travel_m):.3f} m)"
        )
        self._move_to_pose(
            {"joint_lift": (target, effort)},
            custom_contact_thresholds=True,
            duration=duration,
            expect_contact=True,
        )
        q_after = self._lift_q()
        if q_now - q_after < 0.008:
            raise RuntimeError(
                f"DESCEND_BREAKAWAY did not move q {q_now:.3f}->{q_after:.3f}"
            )

    def _wait_lift_still(self, timeout_s: float = 2.0) -> None:
        """hello_world: wait until the previous lift command has actually stopped."""

        deadline = time.monotonic() + float(timeout_s)
        last = self._lift_q()
        still = 0
        while time.monotonic() < deadline:
            time.sleep(0.1)
            q = self._lift_q()
            if abs(q - last) < 0.002:
                still += 1
                if still >= 3:
                    return
            else:
                still = 0
            last = q

    def _try_trigger(self, client, timeout_sec: float = 3.0) -> bool:
        if not client.service_is_ready():
            return False
        try:
            future = client.call_async(Trigger.Request())
            self._wait_future(future, timeout_sec)
            return True
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"{client.srv_name} failed: {exc}")
            return False

    def _clear_lift_guarded_event(self) -> None:
        """Clear firmware ``in_guarded_event`` without arming contact.

        ``/stop_the_robot`` then a completed **raise** to a new q
        (plain ``{'joint_lift': q+bump}``, no effort). A hold at the
        current q is zero travel and re-trips error 100.
        Do not call ``_lift_up`` here (it would recurse on 100).
        """

        stopped = self._try_trigger(self.stop_the_robot)
        switched = self._try_trigger(self.switch_position)
        self.get_logger().info(
            f"CLEAR_GUARDED /stop_the_robot={'ok' if stopped else 'skip'} "
            f"position={'ok' if switched else 'skip'}"
        )
        time.sleep(0.3)
        q = self._lift_q()
        hi = float(self.workspace.lift_max_m)
        target = min(hi, q + 0.012)
        if target <= q + 0.003:
            self.get_logger().info(
                f"CLEAR_GUARDED skip raise: q={q:.3f} at lift_max {hi:.3f}"
            )
            time.sleep(0.5)
            self._wait_lift_still()
            return
        self.get_logger().info(
            f"CLEAR_GUARDED plain raise q {q:.3f} -> {target:.3f} "
            "(no custom contact)"
        )
        result = self._move_to_pose(
            {"joint_lift": float(target)},
            duration=0.0,
            expect_contact=True,
        )
        q1 = self._lift_q()
        if is_force_contact(result) and q1 - q < 0.004:
            self.get_logger().warn(
                f"CLEAR_GUARDED raise also error 100 with no travel "
                f"(q {q:.3f}->{q1:.3f}); firmware latch may remain"
            )
        time.sleep(0.5)
        self._wait_lift_still()

    def _gripper_joint_names(self) -> list[str]:
        if "stretch_gripper" in self._joints:
            return ["stretch_gripper"]
        if "joint_gripper_finger_left" in self._joints:
            return ["joint_gripper_finger_left"]
        return ["joint_gripper_finger_left"]

    def _gripper(self, opening: float, duration: float = 1.5) -> None:
        pose = {name: float(opening) for name in self._gripper_joint_names()}
        self.get_logger().info(f"gripper -> {opening:.3f} ({', '.join(pose)})")
        self._move_to_pose(pose, duration=duration)

    def _assert_not_runstopped(self) -> None:
        if self._cancel_event.is_set():
            raise RuntimeError("runstop/cancel is set; will not descend")

    def _assert_position_mode(self) -> None:
        if not self.traj_client.server_is_ready():
            raise RuntimeError(
                "FollowJointTrajectory is not ready (need position-mode stretch_driver)"
            )

    def _assert_descend_entry(
        self,
        verify: dict,
        *,
        z_cloth_odom: float,
        z_cloth_layout: float,
        action_xy,
        frame,
    ) -> None:
        self._assert_not_runstopped()
        self._assert_position_mode()
        reject_m = float(self._tune_val("approach.pregrasp_reject_m"))
        e_g = float(verify.get("e_g_m", 1e9))
        if not verify_allows_descend(e_g_m=e_g, reject_m=reject_m):
            raise RuntimeError(
                f"DESCEND invariant: VERIFY_PREGRASP e_g={e_g:.4f} "
                f"> reject {reject_m:.3f} m"
            )
        live = self._verify_pregrasp(
            action_xy,
            frame,
            z_cloth_odom=z_cloth_odom,
            z_cloth_layout=float(z_cloth_layout),
            d_base_cmd=float(verify.get("base_translate_commanded_m", 0.0)),
        )
        if not verify_allows_descend(
            e_g_m=float(live["e_g_m"]), reject_m=reject_m
        ):
            raise RuntimeError(
                f"DESCEND invariant: live grasp residual e_g="
                f"{live['e_g_m']:.4f} > {reject_m:.3f} m"
            )
        q_now = self._lift_q()
        dropped = 0.0 if self._q_at_verify is None else max(
            0.0, float(self._q_at_verify) - q_now
        )
        max_d = float(self._tune_val("contact.max_descent_m"))
        if not descend_budget_ok(dropped_m=dropped, max_descent_m=max_d):
            raise RuntimeError(
                f"DESCEND invariant: no descent budget "
                f"(dropped {dropped:.3f} vs max {max_d:.3f})"
            )

    def _assert_tap2_entry(self) -> None:
        if not self._contact_ok:
            raise RuntimeError(
                "GRASP_TAP2 invariant: DESCEND_CONTACT has not succeeded this goal"
            )

    def _stage_descend_unlatch(self) -> None:
        self.get_logger().info(
            "DESCEND_UNLATCH skip: Funmap lower-until-contact does not unlatch"
        )

    def _stage_descend_breakaway(self) -> None:
        self.get_logger().info(
            "DESCEND_BREAKAWAY skip: Funmap lower-until-contact does not breakaway"
        )

    def _stage_descend_coarse(self, *, z_cloth_odom: float) -> None:
        self.get_logger().info(
            "DESCEND_COARSE skip: Funmap lower-until-contact from pregrasp"
        )

    def _stage_descend_contact(
        self,
        *,
        z_cloth_odom: float,
        z_before: float,
        q_before: float,
    ) -> dict:
        payload = self._lower_until_contact(
            z_cloth_odom=float(z_cloth_odom),
            z_before=z_before,
            q_before=q_before,
        )
        self._contact_ok = True
        return payload

    def _second_contact_tap(self, drop_m: float, *, z_cloth_odom: float) -> None:
        """After open, lower onto the cloth. Driver error 100 is the tap."""

        q0 = self._lift_q()
        target = max(float(self.workspace.lift_min_m), q0 - float(drop_m))
        self.get_logger().info(
            f"GRASP_TAP2 joint_lift {q0:.3f} -> {target:.3f} "
            f"(plain -{float(drop_m):.3f} m; error 100 = cloth hit, success)"
        )
        if target >= q0 - 0.003:
            return
        result = self._move_to_pose(
            {"joint_lift": float(target)},
            duration=0.0,
            expect_contact=True,
        )
        if is_force_contact(result):
            self.get_logger().info(
                f"GRASP_TAP2 error 100 at q={self._lift_q():.3f} (cloth tap ok)"
            )

    def _linear_pull_position(
        self,
        waypoints_xy_layout,
        pull_z: float,
        dt: float,
        contact_effort: float,
        *,
        layout_z: float,
    ) -> list:
        actual = []
        for xy in waypoints_xy_layout:
            if self._cancel_event.is_set():
                raise RuntimeError("canceled")
            target_base = self._hover_to_base(
                np.array([xy[0], xy[1], layout_z], dtype=np.float64)
            )
            cmd = command_toward_target(
                current_ee_base=self._ee_base(),
                target_ee_base=target_base,
                current_wrist_extension=self._wrist_extension(),
                current_lift=self._joint("joint_lift"),
                workspace=self.workspace,
            )
            pose = {
                "translate_mobile_base": (cmd.translate_mobile_base, contact_effort),
                "joint_arm": (cmd.wrist_extension, contact_effort),
                "joint_lift": (cmd.joint_lift, contact_effort),
            }
            self._move_to_pose(
                pose,
                custom_contact_thresholds=True,
                duration=max(dt, 0.05),
                expect_contact=True,
            )
            ee_layout = np.append(self._ee_layout_xy(), self._ee_base()[2])
            actual.append(ee_layout)
            remaining = np.linalg.norm(self._ee_layout_xy() - xy)
            abort_m = float(self.get_parameter("pull.contact_abort_m").value)
            if remaining > abort_m:
                raise RuntimeError("execution_failure: contact trip during pull")
        return actual

    def _linear_pull_streaming(
        self, waypoints_xy_layout, pull_z: float, dt: float, *, layout_z: float
    ) -> list:
        self._call_trigger(self.activate_streaming)
        actual = []
        try:
            for xy in waypoints_xy_layout:
                if self._cancel_event.is_set():
                    raise RuntimeError("canceled")
                target_base = self._hover_to_base(
                    np.array([xy[0], xy[1], layout_z], dtype=np.float64)
                )
                cmd = command_toward_target(
                    current_ee_base=self._ee_base(),
                    target_ee_base=target_base,
                    current_wrist_extension=self._wrist_extension(),
                    current_lift=self._joint("joint_lift"),
                    workspace=self.workspace,
                )
                grip_name = self._gripper_joint_names()[0]
                qpos = fill_streaming_qpos(
                    arm=cmd.wrist_extension,
                    lift=cmd.joint_lift,
                    gripper=self._joint(grip_name),
                    wrist_yaw=self._joint("joint_wrist_yaw"),
                    wrist_pitch=self._joint("joint_wrist_pitch"),
                    wrist_roll=self._joint("joint_wrist_roll"),
                    head_pan=self._joint("joint_head_pan"),
                    head_tilt=self._joint("joint_head_tilt"),
                    base_translate=cmd.translate_mobile_base,
                    base_rotate=0.0,
                )
                msg = Float64MultiArray()
                msg.data = qpos
                self.joint_pose_pub.publish(msg)
                time.sleep(dt)
                actual.append(np.append(self._ee_layout_xy(), self._ee_base()[2]))
        finally:
            try:
                self._call_trigger(self.deactivate_streaming)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"deactivate streaming failed: {exc}")
        abort_m = float(self.get_parameter("pull.contact_abort_m").value)
        if waypoints_xy_layout is not None and len(waypoints_xy_layout) and len(actual):
            remaining = float(
                np.linalg.norm(np.asarray(actual[-1][:2]) - waypoints_xy_layout[-1])
            )
            if remaining > abort_m:
                raise RuntimeError("execution_failure: contact trip or miss during pull")
        return actual

    def _linear_pull_trajectory(
        self,
        waypoints_xy_layout,
        pull_z: float,
        speed: float,
        dt: float,
        *,
        layout_z: float,
    ) -> list:
        """Trajectory-mode base uses MultiDOF joint ``position``, not translate_mobile_base."""

        self._call_trigger(self.switch_trajectory)
        arm_pts = []
        mdof_pts = []
        t_from_start = 0.0
        start_base = self._ee_base().copy()
        wrist = self._wrist_extension()
        for xy in waypoints_xy_layout:
            target_base = self._hover_to_base(
                np.array([xy[0], xy[1], layout_z], dtype=np.float64)
            )
            cmd = command_toward_target(
                current_ee_base=start_base,
                target_ee_base=target_base,
                current_wrist_extension=wrist,
                current_lift=self._joint("joint_lift"),
                workspace=self.workspace,
            )
            t_from_start += dt
            point = JointTrajectoryPoint()
            point.positions = [
                self._clamp_arm_command(cmd.wrist_extension),
                float(cmd.joint_lift),
            ]
            sec = int(t_from_start)
            point.time_from_start = Duration(
                sec=sec, nanosec=int(round((t_from_start - sec) * 1e9))
            )
            arm_pts.append(point)
            mdof = MultiDOFJointTrajectoryPoint()
            tf = Transform()
            tf.translation.x = float(target_base[0])
            tf.translation.y = float(target_base[1])
            tf.rotation.w = 1.0
            mdof.transforms = [tf]
            mdof.velocities = [Twist()]
            mdof.time_from_start = point.time_from_start
            mdof_pts.append(mdof)
            start_base = target_base
            wrist = cmd.wrist_extension
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["joint_arm", "joint_lift"]
        goal.trajectory.points = arm_pts
        # Humble stretch_driver trajectory mode reads base from MultiDOF joint "position".
        if hasattr(goal, "multi_dof_trajectory"):
            goal.multi_dof_trajectory.joint_names = ["position"]
            goal.multi_dof_trajectory.points = mdof_pts
        else:
            self.get_logger().warn(
                "FollowJointTrajectory has no multi_dof_trajectory; "
                "base will not move in trajectory mode — use streaming"
            )
        send = self.traj_client.send_goal_async(goal)
        handle = self._wait_future(send, 5.0)
        if handle is None or not handle.accepted:
            raise RuntimeError("trajectory pull rejected")
        self._active_traj_handle = handle
        self._wait_future(handle.get_result_async(), t_from_start + 10.0)
        self._active_traj_handle = None
        try:
            self._call_trigger(self.switch_position)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"switch_to_position_mode failed: {exc}")
        return [np.append(self._ee_layout_xy(), self._ee_base()[2])]

    def _round_trip(self) -> bool:
        if bool(self._tune_val("pull.round_trip")):
            return True
        stop = str(self._tune_val("pull.stop_after") or "")
        return stop.startswith("REVERSE_")

    def _pull_canonical(
        self,
        start_xy,
        end_xy,
        *,
        frame,
        z_cloth: float,
        speed: float,
        dt: float,
    ):
        waypoints_can = interpolate_xy_line(
            start_xy, end_xy, speed_m_s=speed, dt_s=dt
        )
        waypoints_layout = np.array(
            [
                self._map_canonical_to_layout(xy, frame, z_cloth)[:2]
                for xy in waypoints_can
            ]
        )
        return self._linear_pull(
            waypoints_layout,
            float(self._ee_base()[2]),
            speed,
            dt,
            layout_z=z_cloth,
        )

    def _linear_pull(
        self,
        waypoints_xy_layout,
        pull_z: float,
        speed: float,
        dt: float,
        *,
        layout_z: float,
    ):
        controller = str(self.get_parameter("pull.controller").value)
        contact_effort = float(self.get_parameter("pull.contact_effort").value)
        self.get_logger().info(
            f"PULL controller={controller} n={len(waypoints_xy_layout)} "
            f"speed={speed:.3f} m/s ee_z_base={pull_z:.3f} layout_z={layout_z:.3f}"
        )
        if controller == "position":
            return self._linear_pull_position(
                waypoints_xy_layout, pull_z, dt, contact_effort, layout_z=layout_z
            )
        if controller == "trajectory":
            raise SafetyReject(
                FAULT_GOAL_REJECT,
                "pull.controller=trajectory is not validated; refusing motion",
            )
        return self._linear_pull_streaming(
            waypoints_xy_layout, pull_z, dt, layout_z=layout_z
        )

    def execute_cb(self, goal_handle):
        self._cancel_event.clear()
        goal = goal_handle.request
        stage = "LOCALIZING"
        t0 = time.monotonic()
        actual_xy = None
        frame = None
        reach = None
        z_cloth = 0.0
        descend = None
        action = np.array(
            [goal.grasp_x, goal.grasp_y, goal.release_x, goal.release_y],
            dtype=np.float64,
        )
        if self._tune is None:
            self._tune = self._snapshot_tune_params()
        self._contact_ok = False
        self._q_at_verify = None
        try:
            self._clear_lift_guarded_event()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"pre-goal CLEAR_GUARDED: {exc}")
        try:
            if goal.frame_id and goal.frame_id != "bed":
                goal_handle.abort()
                return self._result(
                    False, stage, "frame_id must be bed", fault_code=FAULT_GOAL_REJECT
                )
            if bool(self._tune_val("pull.dry_run")):
                probe = self._build_probe_report(action=action, dry_run=True)
                self.get_logger().info(
                    f"DRY_RUN probe {json.dumps(probe, default=str)}"
                )
                goal_handle.succeed()
                return self._result(
                    True,
                    "DRY_RUN",
                    "dry-run: no FollowJointTrajectory sent",
                    fault_code=FAULT_DRY_RUN,
                )
            frame = self._canonical_frame()
            clearance = float(
                self.get_parameter("pull.clearance_above_bed_m").value
            )
            speed = float(self.get_parameter("pull.xy_speed_m_s").value)
            dt = float(self.get_parameter("pull.dt_s").value)
            disagree_max = float(
                self.get_parameter("localization.origin_disagree_max_m").value
            )
            origin_err = layout_canonical_disagreement_m(frame.origin)
            # ||O|| is the 4-tag centroid vs layout-board origin (this bed
            # ~6 cm toward the head). That offset is applied by the trial
            # (O,R) map. It is not Stretch-vs-ceiling disagreement and must
            # not abort. T_odom_layout is Stretch-local PnP in stretch_origin.json;
            # RCHI never subscribes to the D435i.
            self.get_logger().info(
                f"canonical centroid O is {origin_err:.3f} m from layout "
                f"origin; pickle maps through trial (O,R)"
            )
            if origin_err > max(disagree_max, 0.25):
                raise RuntimeError(
                    f"canonical origin_layout XY {origin_err:.3f} m is past "
                    f"the sanity cap; wrong canonical_bed_frame.json?"
                )

            # Bed pose is frozen in odom. Live T_odom_base tracks the
            # robot. Do not rotate_mobile_base. Park parallel to the
            # long side; raw Z-down T may make +X_base vs +Y look 180°.
            # That is a frame sign, not a request to spin the base.
            stage = "LOCALIZING"
            self._feedback(goal_handle, "LOCALIZING")
            t_odom_layout = self._load_layout_snapshot()
            if float(t_odom_layout[2, 2]) < 0.0:
                self.get_logger().info(
                    "T_odom_layout is Z-down "
                    f"(R[2,2]={t_odom_layout[2, 2]:.2f}), same as overlay. "
                    "Not applying ensure_layout_z_up (that flips bed +Y and "
                    "drives translate_mobile_base the wrong way)."
                )
            self._freeze_snapshot(t_odom_layout)
            t_odom_base = self._lookup(
                str(self.get_parameter("frames.odom").value),
                str(self.get_parameter("frames.base").value),
            )
            yaw_signed = bedside_parking_error_rad(
                t_odom_layout, t_odom_base, layout_y_axis=frame.y_axis
            )
            yaw_along = bedside_parallel_error_rad(
                t_odom_layout, t_odom_base, layout_y_axis=frame.y_axis
            )
            yaw_into = bedside_arm_into_bed_error_rad(
                t_odom_layout, t_odom_base, layout_x_axis=frame.x_axis
            )
            yaw_tol = float(
                self.get_parameter("localization.yaw_tolerance_rad").value
            )
            self.get_logger().info(
                f"parking: +X_base vs bed +Y = {float(np.degrees(yaw_signed)):.1f} deg "
                f"(parallel {float(np.degrees(yaw_along)):.1f}, "
                f"arm-into-bed {float(np.degrees(yaw_into)):.1f}, "
                f"limit {float(np.degrees(yaw_tol)):.1f}); no rotate_mobile_base"
            )
            if yaw_along > yaw_tol or abs(yaw_into) > yaw_tol:
                raise RuntimeError(
                    f"REJECT: base not parallel to the long side "
                    f"(along {float(np.degrees(yaw_along)):.1f} deg, "
                    f"arm-into-bed {float(np.degrees(yaw_into)):.1f} deg). "
                    "Park with the arm toward the bed. "
                    "Executor will not rotate_mobile_base."
                )

            stage = "EXEC_PREFLIGHT"
            self._feedback(goal_handle, "EXEC_PREFLIGHT")
            _canon, pose_dir = self._trial_paths()
            snap_path = pose_dir / "stretch_reach_snapshot.json"
            if not snap_path.is_file():
                raise RuntimeError(
                    f"REJECT: missing {snap_path}. "
                    "CMA and executor share the same snapshot."
                )
            reach_snap = load_reach_snapshot(snap_path)
            t_ob_snap = np.asarray(reach_snap["T_odom_base"], dtype=np.float64)
            base_shift = float(
                np.linalg.norm(t_odom_base[:2, 3] - t_ob_snap[:2, 3])
            )
            warn = apply_live_geometry_gate(
                error_m=base_shift,
                policy=self._tune_val("live_geometry.policy"),
                label="T_odom_base XY",
            )
            if warn:
                self.get_logger().warn(f"{warn}. Executing anyway (policy=warn).")
            reach = check_bed_pull_reachability(
                action,
                frame=frame,
                snapshot=reach_snap,
                workspace=StretchWorkspace(arm_max_m=HARDWARE_ARM_MAX_M),
                slack_m=EXEC_ARM_SLACK_M,
                speed_m_s=speed,
                dt_s=dt,
            )
            self.get_logger().info(
                "EXEC_PREFLIGHT execution_wrist_down "
                f"reachable={reach.reachable} "
                f"max_wrist={reach.max_wrist_extension} "
                f"n={reach.n_waypoints} {reach.reason}"
            )
            if not reach.reachable:
                raise SafetyReject(
                    FAULT_UNREACHABLE,
                    "REJECT unreachable (do not clip): "
                    f"{reach.reason}. Drive is +X_base; arm is -Y_base.",
                )

            z_cloth = self._cloth_z_layout()
            grasp_layout = self._map_canonical_to_layout(action[:2], frame, z_cloth)
            release_layout = self._map_canonical_to_layout(action[2:], frame, z_cloth)
            grasp_roundtrip = layout_points_to_canonical(
                grasp_layout.reshape(1, 3), frame
            )[0]
            release_roundtrip = layout_points_to_canonical(
                release_layout.reshape(1, 3), frame
            )[0]
            d_g = grasp_roundtrip[:2] - action[:2]
            d_r = release_roundtrip[:2] - action[2:]
            self.get_logger().info(
                "CMA grasp canonical XY="
                f"{np.asarray(action[:2]).round(4).tolist()} "
                "executor grasp canonical XY="
                f"{grasp_roundtrip[:2].round(4).tolist()} "
                f"difference XY={d_g.round(6).tolist()} "
                f"|d|={float(np.linalg.norm(d_g)):.6f} m"
            )
            self.get_logger().info(
                "CMA release canonical XY="
                f"{np.asarray(action[2:]).round(4).tolist()} "
                "executor release canonical XY="
                f"{release_roundtrip[:2].round(4).tolist()} "
                f"difference XY={d_r.round(6).tolist()} "
                f"|d|={float(np.linalg.norm(d_r)):.6f} m"
            )

            z_cloth_odom = float(self._point_odom(grasp_layout)[2])
            clear_m = float(self._tune_val("approach.clear_above_cloth_m"))
            z_approach = z_cloth_odom + clear_m
            z_safe_ee_odom = z_cloth_odom + clearance
            z_ee_now = float(self._ee_odom()[2])
            self.get_logger().info(
                f"approach EE odom Z={z_approach:.3f} "
                f"(cloth {z_cloth_odom:.3f}+{clear_m:.2f}); "
                f"pull Z={z_safe_ee_odom:.3f} "
                f"(cloth+{clearance:.2f}, after grasp only); "
                f"current EE odom Z={z_ee_now:.3f} "
                f"joint_lift={self._joint('joint_lift'):.3f}. "
                "Lateral arm/base only after EE is at or above this plane. "
                "Never command joint_lift=clearance."
            )

            self.get_logger().info("start: open then close gripper")
            self._gripper(float(self._tune_val("grasp.gripper_open")))
            self._gripper(float(self._tune_val("grasp.gripper_closed")))

            stage = "SAFE_LIFT"
            self._feedback(goal_handle, "SAFE_LIFT")
            self._safe_lift_up_only(
                z_approach,
                label="SAFE_LIFT",
                z_min_ok=z_cloth_odom + 0.05,
            )
            self._stop_after_if("SAFE_LIFT")

            stage = "RETRACT"
            self._feedback(goal_handle, "RETRACT")
            self._retract_arm()
            self._stop_after_if("RETRACT")

            stage = "WRIST_DOWN"
            self._feedback(goal_handle, "WRIST_DOWN")
            self._wrist_down(stage="WRIST_DOWN")
            self._stop_after_if("WRIST_DOWN")

            stage = "SAFE_LIFT_RECHECK"
            self._feedback(goal_handle, "SAFE_LIFT_RECHECK")
            self._safe_lift_up_only(
                z_approach,
                label="SAFE_LIFT_RECHECK",
                z_min_ok=z_cloth_odom + 0.05,
            )
            self._stop_after_if("SAFE_LIFT_RECHECK")

            ee = self._ee_base()
            grasp_base = self._hover_to_base(grasp_layout)
            release_base = self._hover_to_base(release_layout)
            plan0 = command_toward_target(
                current_ee_base=ee,
                target_ee_base=grasp_base,
                current_wrist_extension=self._wrist_extension(),
                current_lift=self._joint("joint_lift"),
                workspace=self.workspace,
            )
            self.get_logger().info(
                f"grasp_base_xy={grasp_base[:2].round(3).tolist()} "
                f"release_base_xy={release_base[:2].round(3).tolist()} "
                f"ee_base={ee.round(3).tolist()} "
                f"d_base={plan0.translate_mobile_base:.3f} "
                f"arm_target={plan0.wrist_extension:.3f} "
                f"lift_cmd={plan0.joint_lift:.3f} (unused until LIFT)"
            )
            model_ee0 = execution_ee_from_snapshot(reach_snap, self.workspace)
            live_ee0 = rewind_ee_to_arm0(
                ee, self._wrist_extension(), self.workspace
            )
            match_xy = float(np.linalg.norm(live_ee0[:2] - model_ee0[:2]))
            self.get_logger().info(
                f"WRIST_DOWN geometry match XY={match_xy:.3f} m "
                f"(model arm0={model_ee0[:2].round(3).tolist()} "
                f"live arm0={live_ee0[:2].round(3).tolist()})"
            )
            warn = apply_live_geometry_gate(
                error_m=match_xy,
                policy=self._tune_val("live_geometry.policy"),
                label="wrist-down grasp_center XY",
            )
            if warn:
                self.get_logger().warn(f"{warn}. Executing anyway (policy=warn).")
            if plan0.wrist_extension < self.workspace.arm_min_m:
                self.get_logger().warn(
                    f"joint_arm={plan0.wrist_extension:.3f} m < 0; "
                    "restricting to 0 (EE past grasp in TF). Not mast-closer."
                )
            stop_requested = str(self._tune_val("pull.stop_after") or "")

            stage = "BASE_APPROACH"
            self._feedback(goal_handle, "BASE_APPROACH")
            d_base_cmd = self._base_translate_only(self._hover_to_base(grasp_layout))
            self._stop_after_if("BASE_APPROACH")

            stage = "ARM_APPROACH"
            self._feedback(goal_handle, "ARM_APPROACH")
            self._arm_extend_only(self._hover_to_base(grasp_layout))
            refine_n = int(self._tune_val("approach.refine_passes"))
            if refine_n > 0:
                d_base_cmd += self._refine_pregrasp_xy(grasp_layout, passes=refine_n)
            self._stop_after_if("ARM_APPROACH")

            stage = "VERIFY_PREGRASP"
            self._feedback(goal_handle, "VERIFY_PREGRASP")
            verify = self._verify_pregrasp(
                action[:2],
                frame,
                z_cloth_odom=z_cloth_odom,
                z_cloth_layout=z_cloth,
                d_base_cmd=d_base_cmd,
            )
            err = float(verify["e_g_m"])
            pass_m = float(self._tune_val("approach.pregrasp_pass_m"))
            reject_m = float(self._tune_val("approach.pregrasp_reject_m"))
            if err > reject_m:
                msg = (
                    f"REJECT pregrasp e_g={err:.4f} m "
                    f"dx={verify['dx_m']:+.4f} dy={verify['dy_m']:+.4f} "
                    f"> {reject_m:.3f} m; will not descend "
                    f"(arm={verify['final_arm_extension_m']:.3f} "
                    f"ee_base={verify['ee_base']})"
                )
                duration_s = time.monotonic() - t0
                self._write_manifest(
                    goal,
                    action,
                    frame,
                    reach,
                    np.asarray([self._ee_layout()], dtype=np.float64),
                    z_cloth,
                    duration_s,
                    success=False,
                    failure_stage="VERIFY_PREGRASP",
                    extra_update={"verify_pregrasp": verify, "grade": "REJECT"},
                )
                goal_handle.abort()
                return self._result(False, "VERIFY_PREGRASP", msg)
            grade = "PASS" if err < pass_m else "WARN"
            self.get_logger().info(
                f"VERIFY_PREGRASP {grade}: e_g={err:.4f} m "
                f"dx={verify['dx_m']:+.4f} dy={verify['dy_m']:+.4f} "
                f"(pass<{pass_m:.3f}, reject>{reject_m:.3f})"
            )
            self._q_at_verify = self._lift_q()
            stop_extra = {
                "verify_pregrasp": verify,
                "grade": grade,
                "tune_snapshot": dict(self._tune or {}),
            }
            self._stop_after_if("VERIFY_PREGRASP", extra=stop_extra)

            descend_kw = dict(
                verify=verify,
                z_cloth_odom=z_cloth_odom,
                z_cloth_layout=z_cloth,
                action_xy=action[:2],
                frame=frame,
            )
            z_before = float(self._ee_odom()[2])
            q_before = self._lift_q()

            stage = "DESCEND_UNLATCH"
            self._feedback(goal_handle, "DESCEND_UNLATCH")
            self._assert_descend_entry(**descend_kw)
            self._stage_descend_unlatch()
            self._stop_after_if("DESCEND_UNLATCH", extra=stop_extra)

            stage = "DESCEND_BREAKAWAY"
            self._feedback(goal_handle, "DESCEND_BREAKAWAY")
            self._assert_descend_entry(**descend_kw)
            self._stage_descend_breakaway()
            self._stop_after_if("DESCEND_BREAKAWAY", extra=stop_extra)

            stage = "DESCEND_COARSE"
            self._feedback(goal_handle, "DESCEND_COARSE")
            self._assert_descend_entry(**descend_kw)
            self._stage_descend_coarse(z_cloth_odom=z_cloth_odom)
            self._stop_after_if("DESCEND_COARSE", extra=stop_extra)

            stage = "DESCEND_CONTACT"
            self._feedback(goal_handle, "DESCEND_CONTACT")
            self._assert_descend_entry(**descend_kw)
            descend = self._stage_descend_contact(
                z_cloth_odom=z_cloth_odom,
                z_before=z_before,
                q_before=q_before,
            )
            stop_extra["descend"] = descend
            self._stop_after_if("DESCEND_CONTACT", extra=stop_extra)

            bump = float(self._tune_val("grasp.lift_after_contact_m"))
            stage = "GRASP_BUMP"
            self._feedback(goal_handle, "GRASP_BUMP")
            q_now = self._lift_q()
            if bump > 0.003:
                z_b0 = float(self._ee_odom()[2])
                self._lift_up(q_now + bump, label="GRASP_BUMP", duration=1.2)
                z_b1 = float(self._ee_odom()[2])
                self.get_logger().info(
                    f"GRASP_BUMP EE odom Z {z_b0:.3f} -> {z_b1:.3f} "
                    f"(+{z_b1 - z_b0:.3f} m, want {bump:.3f})"
                )
            time.sleep(float(self._tune_val("grasp.pause_after_bump_s")))
            self._stop_after_if("GRASP_BUMP", extra=stop_extra)

            stage = "GRASP_OPEN"
            self._feedback(goal_handle, "GRASP_OPEN")
            self._gripper(float(self._tune_val("grasp.gripper_open")))
            self._stop_after_if("GRASP_OPEN", extra=stop_extra)

            stage = "GRASP_TAP2"
            self._feedback(goal_handle, "GRASP_TAP2")
            self._assert_tap2_entry()
            self._second_contact_tap(
                float(self._tune_val("grasp.lower_after_open_m")),
                z_cloth_odom=z_cloth_odom,
            )
            time.sleep(float(self._tune_val("grasp.pause_after_probe_s")))
            descend["second_tap_ee_odom_z"] = float(self._ee_odom()[2])
            self._stop_after_if("GRASP_TAP2", extra=stop_extra)

            stage = "GRASP_CLOSE"
            self._feedback(goal_handle, "GRASP_CLOSE")
            self._gripper(float(self._tune_val("grasp.gripper_closed")))
            time.sleep(float(self._tune_val("grasp.pause_after_close_s")))
            self._stop_after_if("GRASP_CLOSE", extra=stop_extra)

            stage = "LIFTING"
            self._feedback(goal_handle, "LIFTING")
            z_contact = float(self._ee_odom()[2])
            z_lift = z_contact + clearance
            self.get_logger().info(
                f"LIFT raise-only toward EE odom Z {z_lift:.3f} "
                f"(contact {z_contact:.3f}+{clearance:.2f}); "
                "clips to lift_max if wrist-down cannot reach it"
            )
            self._safe_lift_up_only(
                z_lift, label="LIFT", z_min_ok=z_contact + 0.05
            )
            self._stop_after_if("LIFTING", extra=stop_extra)

            stage = "PULLING"
            self._feedback(goal_handle, "PULLING")
            actual = self._pull_canonical(
                action[:2],
                action[2:],
                frame=frame,
                z_cloth=z_cloth,
                speed=speed,
                dt=dt,
            )
            actual_xy = np.asarray(actual, dtype=np.float64)
            self._stop_after_if("PULLING", extra=stop_extra)

            stage = "RELEASING"
            self._feedback(goal_handle, "RELEASING")
            self._gripper(float(self._tune_val("grasp.gripper_open")))
            self._stop_after_if("RELEASING", extra=stop_extra)

            if self._round_trip():
                self.get_logger().info(
                    "round-trip: pick at uncover place, "
                    "place back at uncover pick"
                )
                self._gripper(float(self._tune_val("grasp.gripper_closed")))
                self._q_at_verify = self._lift_q()
                self._contact_ok = False
                z_before = float(self._ee_odom()[2])
                q_before = self._lift_q()

                stage = "REVERSE_CONTACT"
                self._feedback(goal_handle, "REVERSE_CONTACT")
                self._assert_not_runstopped()
                self._assert_position_mode()
                descend = self._stage_descend_contact(
                    z_cloth_odom=z_cloth_odom,
                    z_before=z_before,
                    q_before=q_before,
                )
                stop_extra["reverse_descend"] = descend
                self._stop_after_if("REVERSE_CONTACT", extra=stop_extra)

                bump = float(self._tune_val("grasp.lift_after_contact_m"))
                stage = "REVERSE_BUMP"
                self._feedback(goal_handle, "REVERSE_BUMP")
                q_now = self._lift_q()
                if bump > 0.003:
                    self._lift_up(q_now + bump, label="REVERSE_BUMP", duration=1.2)
                time.sleep(float(self._tune_val("grasp.pause_after_bump_s")))
                self._stop_after_if("REVERSE_BUMP", extra=stop_extra)

                stage = "REVERSE_OPEN"
                self._feedback(goal_handle, "REVERSE_OPEN")
                self._gripper(float(self._tune_val("grasp.gripper_open")))
                self._stop_after_if("REVERSE_OPEN", extra=stop_extra)

                stage = "REVERSE_TAP2"
                self._feedback(goal_handle, "REVERSE_TAP2")
                self._assert_tap2_entry()
                self._second_contact_tap(
                    float(self._tune_val("grasp.lower_after_open_m")),
                    z_cloth_odom=z_cloth_odom,
                )
                time.sleep(float(self._tune_val("grasp.pause_after_probe_s")))
                self._stop_after_if("REVERSE_TAP2", extra=stop_extra)

                stage = "REVERSE_CLOSE"
                self._feedback(goal_handle, "REVERSE_CLOSE")
                self._gripper(float(self._tune_val("grasp.gripper_closed")))
                time.sleep(float(self._tune_val("grasp.pause_after_close_s")))
                self._stop_after_if("REVERSE_CLOSE", extra=stop_extra)

                stage = "REVERSE_LIFT"
                self._feedback(goal_handle, "REVERSE_LIFT")
                z_contact = float(self._ee_odom()[2])
                z_lift = z_contact + clearance
                self._safe_lift_up_only(
                    z_lift, label="REVERSE_LIFT", z_min_ok=z_contact + 0.05
                )
                self._stop_after_if("REVERSE_LIFT", extra=stop_extra)

                stage = "REVERSE_PULL"
                self._feedback(goal_handle, "REVERSE_PULL")
                actual = self._pull_canonical(
                    action[2:],
                    action[:2],
                    frame=frame,
                    z_cloth=z_cloth,
                    speed=speed,
                    dt=dt,
                )
                actual_xy = np.asarray(actual, dtype=np.float64)
                self._stop_after_if("REVERSE_PULL", extra=stop_extra)

                stage = "REVERSE_PLACE"
                self._feedback(goal_handle, "REVERSE_PLACE")
                self._gripper(float(self._tune_val("grasp.gripper_open")))
                self._stop_after_if("REVERSE_PLACE", extra=stop_extra)

            stage = "RETRACTING"
            self._feedback(goal_handle, "RETRACTING")
            self._retract_arm(label="RETRACTING")
            self._stop_after_if("RETRACTING")

            # Known pose before the next snapshot / Recover CMA. Leftover
            # Uncover pitch (~−70°) made the EE model ~8 cm wrong.
            stage = "RESET_EE"
            self._feedback(goal_handle, "RESET_EE")
            self._wrist_down(stage="RESET_EE")
            self._gripper(float(self._tune_val("grasp.gripper_open")))
            self._stop_after_if("RESET_EE")

            self._feedback(goal_handle, "DONE")
            duration_s = time.monotonic() - t0
            self._write_manifest(
                goal,
                action,
                frame,
                reach,
                actual_xy,
                z_cloth,
                duration_s,
                success=True,
                extra_update={
                    "verify_pregrasp": verify,
                    "grade": grade,
                    "descend": descend,
                    "pull_ee_z_base_m": float(self._ee_base()[2]),
                    "stopped_after": "",
                    "full_pull_completed": True,
                    "tune_snapshot": dict(self._tune or {}),
                },
            )
            goal_handle.succeed()
            return self._result(
                True,
                "DONE",
                f"ok e_g={err:.4f} m {grade} "
                f"descend={(descend or {}).get('dropped_m', 0):.3f} m "
                f"base={d_base_cmd:+.3f} arm={verify['final_arm_extension_m']:.3f}",
                fault_code=FAULT_OK,
            )
        except IntentionalStop as stop:
            duration_s = time.monotonic() - t0
            extra = {
                "stopped_after": stop.stage,
                "full_pull_completed": False,
                "failure_stage": f"INTENTIONAL_STOP_AFTER:{stop.stage}",
                "tune_snapshot": dict(self._tune or {}),
            }
            extra.update(stop.extra)
            try:
                xy = actual_xy
                if xy is None:
                    xy = np.asarray([self._ee_layout()], dtype=np.float64)
                self._write_manifest(
                    goal,
                    action,
                    frame,
                    reach
                    if reach is not None
                    else {"reachable": True, "reason": "intentional_stop"},
                    xy,
                    z_cloth,
                    duration_s,
                    success=True,
                    extra_update=extra,
                )
            except Exception:  # noqa: BLE001
                pass
            detail = stop.detail or f"hold after {stop.stage}"
            msg = intentional_stop_message(stop.stage, detail)
            self.get_logger().info(msg)
            goal_handle.succeed()
            return self._result(
                True,
                f"INTENTIONAL_STOP_AFTER:{stop.stage}",
                msg,
                fault_code=FAULT_INTENTIONAL_STOP,
            )
        except SafetyReject as exc:
            self.get_logger().error(f"{stage}: {exc}")
            goal_handle.abort()
            return self._result(
                False, stage, exc.detail, fault_code=exc.fault_code
            )
        except HelloPoseError as exc:
            self.get_logger().error(f"{stage}: {exc}")
            goal_handle.abort()
            return self._result(False, stage, str(exc), fault_code=FAULT_DRIVER)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"{stage}: {exc}")
            try:
                self._write_manifest(
                    goal,
                    action,
                    frame,
                    {"reachable": False, "reason": str(exc)},
                    actual_xy,
                    float(self.get_parameter("pull.clearance_above_bed_m").value),
                    time.monotonic() - t0,
                    success=False,
                    failure_stage=stage,
                )
            except Exception:  # noqa: BLE001
                pass
            if self._cancel_event.is_set():
                goal_handle.canceled()
                return self._result(
                    False, stage, "canceled", fault_code=FAULT_CANCELED
                )
            goal_handle.abort()
            return self._result(False, stage, str(exc), fault_code=FAULT_DRIVER)

    def _write_manifest(
        self,
        goal,
        action,
        frame,
        reach,
        actual_xy,
        pull_z,
        duration_s,
        *,
        success: bool,
        failure_stage: str = "",
        extra_update: dict | None = None,
    ) -> None:
        try:
            canon_path, dest_dir = self._trial_paths()
        except RuntimeError:
            canon_path = Path()
            dest_dir = Path(str(self.get_parameter("manifest_dir").value) or "/tmp")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{goal.trial_id or 'bed_pull'}_execution_manifest.json"
        reach_dict = reach.to_dict() if hasattr(reach, "to_dict") else dict(reach)
        extra = {
            "success": success,
            "failure_stage": failure_stage,
            "T_odom_layout": None
            if self.frozen_t_odom_layout is None
            else self.frozen_t_odom_layout.tolist(),
            "T_wristTag_linkGraspCenter": self._wrist_tag_transform(),
            "canonical_origin_layout": None if frame is None else frame.origin.tolist(),
            "mirror_x_note": "executor never remirrors; pickle is canonical metres",
            "metres": "identity",
        }
        session_json = None
        corrected = None
        session_id = None
        pcd_config = None
        registration = None
        if self._session_paths is not None:
            session_id = self._session_paths.session_id
            session_json = self._session_paths.session_json
            corrected = self._session_paths.origin_corrected
            pcd_config = self._session_paths.pcd_config
            registration = self._session_paths.registration_json
            extra["session_dir"] = str(self._session_dir)
        if extra_update:
            extra.update(extra_update)
        if actual_xy is not None and len(actual_xy) > 0:
            logged_xy = np.asarray(actual_xy, dtype=np.float64)
            if frame is not None and logged_xy.shape[1] >= 2:
                pts = np.column_stack(
                    [
                        logged_xy[:, 0],
                        logged_xy[:, 1],
                        np.full(len(logged_xy), pull_z),
                    ]
                )
                logged_xy = layout_points_to_canonical(pts, frame)
            extra["trajectory_metrics"] = trajectory_metrics(
                logged_xy[:, :2],
                action[:2],
                action[2:],
                actual_z=np.asarray(actual_xy)[:, 2]
                if np.asarray(actual_xy).shape[1] > 2
                else None,
                pull_z=pull_z,
                duration_s=duration_s,
            )
        write_execution_manifest(
            dest,
            trial_id=str(goal.trial_id or ""),
            planned_action_bed=action.tolist(),
            mirror_x_applied=False,
            reachability=reach_dict,
            canonical_frame_path=canon_path,
            session_id=session_id,
            session_json_path=session_json,
            stretch_origin_corrected_path=corrected,
            pcd_config_path=pcd_config,
            registration_path=registration,
            extra=extra,
        )
        self.get_logger().info(f"Wrote {dest}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BedPullExecutor()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
