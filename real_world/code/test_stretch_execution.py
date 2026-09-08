#!/usr/bin/env python3
"""Tests for Stretch Cartesian decomposition, reachability, and localization fuse."""

from __future__ import annotations

import sys
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from blanket_grasp import (  # noqa: E402
    DEFAULT_MAX_GRASP_DISTANCE_M,
    DEFAULT_SNAP_INWARD_M,
    nearest_grasp_xy_distance,
    snap_grasp_inward,
)
from stretch_cartesian import (  # noqa: E402
    StretchWorkspace,
    apply_wrist_down_grasp_offset,
    command_toward_target,
    correct_ee_base_for_jsp_lag,
    ee_base_from_tf,
    execution_ee_from_snapshot,
    grasp_xy_correction,
    interpolate_xy_line,
    plan_wrist_down_steps,
    trajectory_metrics,
)
from stretch_live_tf import (  # noqa: E402
    MOVING_JOINTS,
    PRISMATIC_JOINTS,
    apply_joint_to_zero_tf,
    tf_looks_zero_pose,
)
from stretch_limits import (  # noqa: E402
    BED_Y_LIMITS,
    CmaReachContext,
    DEFAULT_ACTION_SCALE,
    PLANNER_ARM_MAX_M,
    assert_selected_action_reachable,
    clip_action_to_bounds,
    cma_policy_bounds,
    into_bed_canonical_x_limits,
    load_cma_reach,
    planner_workspace,
    restrict_cma_bounds_to_reach,
    snapshot_path,
    snapshot_uses_live_tf,
)
from stretch_contact import (  # noqa: E402
    CONTACT_ERROR_CODE,
    CLEAN_SURFACE_LIFT_EFFORT,
    CONTACT_TRACE_COLUMNS,
    FUNMAP_LIFT_EFFORT_THRESHOLD,
    FUNMAP_MOVE_INCREMENT_M,
    HOVER_BASELINE_S,
    PATH_TOLERANCE_VIOLATED,
    THRESHOLD_MODE_CUSTOM,
    THRESHOLD_MODE_DEFAULT,
    LiftContactDetector,
    contact_trace_row,
    descent_floor_q,
    effort_spike,
    format_effort_pct,
    guarded_lower_ok,
    guarded_too_sensitive,
    is_force_contact,
    lift_contact_func,
    median_effort,
    normalize_threshold_mode,
    passed_stopping_position,
    relative_drop,
    relative_unload,
    update_average_effort,
    write_contact_trace_csv,
)
from stretch_grasp_stages import (  # noqa: E402
    STAGES,
    intentional_stop_message,
    normalize_stop_after,
    parse_intentional_stop,
    parse_set_override,
    production_stop_after_ok,
    relevant_params,
    motion_lines,
    should_stop_after,
    verify_allows_descend,
    descend_budget_ok,
)
from stretch_hello_pose import (  # noqa: E402
    HelloPoseError,
    encode_move_to_pose,
)
from stretch_localize import (  # noqa: E402
    LocalizationSample,
    bedside_arm_into_bed_error_rad,
    bedside_parallel_error_rad,
    bedside_parking_error_rad,
    apply_base_yaw,
    compose_t_odom_layout,
    ensure_layout_z_up,
    fuse_localization_samples,
    fuse_odom_layout,
    fusion_acceptable,
    layout_canonical_disagreement_m,
    sample_acceptable,
    t_odom_layout_for_motion,
    layout_xy_at_height,
    union_posts,
    yaw_align_error_rad,
)
from marker_utils import invert_transform  # noqa: E402
from stretch_reachability import check_bed_pull_reachability, check_pull_path  # noqa: E402
from canonical_bed import (  # noqa: E402
    canonical_bed_frame,
    transform_action_canonical_to_layout,
)


class CartesianTests(unittest.TestCase):
    def test_constant_speed_waypoints_include_endpoints(self):
        pts = interpolate_xy_line([0.0, 0.0], [0.15, 0.20], speed_m_s=0.05, dt_s=0.1)
        np.testing.assert_allclose(pts[0], [0.0, 0.0])
        np.testing.assert_allclose(pts[-1], [0.15, 0.20])
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.assertTrue(np.all(step <= 0.05 + 1e-9))

    def test_metrics_straight_path(self):
        grasp = np.array([0.0, 0.0])
        release = np.array([0.15, 0.20])
        pts = interpolate_xy_line(grasp, release, speed_m_s=0.05, dt_s=0.05)
        metrics = trajectory_metrics(pts, grasp, release, duration_s=5.0)
        self.assertLess(metrics["cross_track_max_m"], 1e-9)
        self.assertLess(metrics["grasp_error_m"], 1e-9)
        self.assertLess(metrics["release_error_m"], 1e-9)

    def test_command_splits_base_x_and_arm_neg_y(self):
        # Official: ΔX_base → translate_mobile_base; ΔY_base → −arm.
        cmd = command_toward_target(
            current_ee_base=np.array([0.3, 0.0, 0.4]),
            target_ee_base=np.array([0.4, 0.1, 0.4]),
            current_wrist_extension=0.30,
            current_lift=0.40,
            workspace=StretchWorkspace(),
        )
        self.assertAlmostEqual(cmd.translate_mobile_base, 0.1, places=6)
        self.assertAlmostEqual(cmd.wrist_extension, 0.20, places=6)

    def test_base_x_standoff_is_not_wrist_extension(self):
        # Regression: grasp at base X≈0.80 used to become wrist 0.83 m.
        cmd = command_toward_target(
            current_ee_base=np.array([-0.02, -0.16, 0.40]),
            target_ee_base=np.array([0.80, -0.15, 0.40]),
            current_wrist_extension=0.01,
            current_lift=0.40,
            workspace=StretchWorkspace(),
        )
        self.assertAlmostEqual(cmd.translate_mobile_base, 0.82, places=2)
        self.assertLess(abs(cmd.wrist_extension - 0.00), 0.05)
        self.assertAlmostEqual(cmd.joint_lift, 0.40, places=6)

    def test_live_tf_prismatic_is_t0_times_joint(self):
        t0 = np.eye(4)
        t0[:3, 3] = [0.1, 0.2, 0.3]
        lift = next(j for j in MOVING_JOINTS if j.name == "joint_lift")
        out = apply_joint_to_zero_tf(t0, lift, 1.099)
        np.testing.assert_allclose(out[:3, 3], [0.1, 0.2, 1.399], atol=1e-9)
        np.testing.assert_allclose(apply_joint_to_zero_tf(t0, lift, 0.0), t0)

    def test_live_tf_detects_urdf_zero_ee(self):
        self.assertTrue(tf_looks_zero_pose(ee_z_base=0.096, joint_lift=1.10))
        self.assertFalse(tf_looks_zero_pose(ee_z_base=1.19, joint_lift=1.10))
        self.assertFalse(tf_looks_zero_pose(ee_z_base=0.096, joint_lift=0.0))

    def test_live_tf_broadcasts_prismatic_only(self):
        names = [j.name for j in PRISMATIC_JOINTS]
        self.assertEqual(
            names,
            [
                "joint_lift",
                "joint_arm_l3",
                "joint_arm_l2",
                "joint_arm_l1",
                "joint_arm_l0",
            ],
        )
        self.assertTrue(all(j.kind == "prismatic" for j in PRISMATIC_JOINTS))
        self.assertGreater(len(MOVING_JOINTS), len(PRISMATIC_JOINTS))

    def test_jsp_lag_adds_lift_and_neg_y_arm(self):
        ee, d_lift, d_arm = correct_ee_base_for_jsp_lag(
            [0.0, 0.0, 0.096],
            lift_real=0.916,
            arm_real=0.100,
            lift_jsp=0.0,
            arm_jsp=0.0,
        )
        self.assertAlmostEqual(d_lift, 0.916, places=6)
        self.assertAlmostEqual(d_arm, 0.100, places=6)
        np.testing.assert_allclose(ee, [0.0, -0.100, 1.012], atol=1e-6)

    def test_live_tf_does_not_double_count_arm(self):
        ee, d_lift, d_arm = ee_base_from_tf(
            [-0.005, -0.354, 1.011],
            lift_real=1.099,
            arm_real=0.029,
            lift_jsp=0.0,
            arm_jsp=0.0,
        )
        self.assertEqual(d_lift, 0.0)
        self.assertEqual(d_arm, 0.0)
        np.testing.assert_allclose(ee, [-0.005, -0.354, 1.011], atol=1e-9)

    def test_zero_pose_tf_still_gets_jsp_lag(self):
        ee, d_lift, d_arm = ee_base_from_tf(
            [0.0, -0.16, 0.096],
            lift_real=1.10,
            arm_real=0.10,
            lift_jsp=0.0,
            arm_jsp=0.0,
        )
        self.assertAlmostEqual(d_lift, 1.10, places=6)
        np.testing.assert_allclose(ee, [0.0, -0.26, 1.196], atol=1e-6)

    def test_wrist_down_undoes_along_arm_offset(self):
        ee = apply_wrist_down_grasp_offset(
            [-0.010431, -0.421946, 1.195192],
            pitch_driver=-1.57,
            pitch_jsp=0.0,
        )
        np.testing.assert_allclose(ee, [-0.010431, -0.191946, 0.965192], atol=1e-6)

    def test_wrist_offset_noop_when_pitch_already_in_tf(self):
        ee = apply_wrist_down_grasp_offset(
            [0.0, -0.16, 0.95],
            pitch_driver=-1.57,
            pitch_jsp=-1.57,
        )
        np.testing.assert_allclose(ee, [0.0, -0.16, 0.95], atol=1e-9)

    def test_wrist_offset_noop_before_wrist_down(self):
        ee = apply_wrist_down_grasp_offset(
            [0.0, -0.16, 0.95],
            pitch_driver=0.0,
            pitch_jsp=0.0,
        )
        np.testing.assert_allclose(ee, [0.0, -0.16, 0.95], atol=1e-9)

    def test_force_contact_is_error_100_only(self):
        class _R:
            def __init__(self, code):
                self.error_code = code

        self.assertEqual(CONTACT_ERROR_CODE, 100)
        self.assertTrue(is_force_contact(_R(100)))
        self.assertFalse(is_force_contact(_R(0)))
        self.assertFalse(is_force_contact(None))
        self.assertTrue(guarded_lower_ok(_R(100), 0.02))
        self.assertFalse(guarded_lower_ok(_R(100), 0.0))
        self.assertFalse(guarded_lower_ok(_R(100), 0.002))
        self.assertFalse(guarded_lower_ok(_R(100), -0.003))
        self.assertFalse(guarded_lower_ok(_R(0), 0.05))
        self.assertEqual(PATH_TOLERANCE_VIOLATED, -4)
        self.assertFalse(guarded_lower_ok(_R(-4), 0.243))
        self.assertFalse(guarded_lower_ok(_R(-4), 0.0))
        self.assertTrue(lift_contact_func(20.0, 21.0))
        self.assertTrue(lift_contact_func(21.0, 19.9))
        self.assertFalse(lift_contact_func(20.1, 20.0))
        self.assertAlmostEqual(FUNMAP_LIFT_EFFORT_THRESHOLD, 20.0)
        self.assertAlmostEqual(FUNMAP_MOVE_INCREMENT_M, 0.008)
        self.assertAlmostEqual(update_average_effort(None, 24.0), 24.0)
        self.assertAlmostEqual(update_average_effort(24.0, 18.0), 22.0)
        det = LiftContactDetector(
            baseline_samples=3,
            unload_drop_pct=15.0,
            unload_consecutive=1,
            window=1,
        )
        # Trace 20260902_113202: hold +36, first 8 mm still +27.
        self.assertFalse(det.update(1.094, 36.0))
        self.assertFalse(det.update(1.094, 38.7))
        self.assertFalse(det.in_contact)
        det.begin_motion_baseline()
        self.assertFalse(det.update(1.086, 26.8))
        self.assertFalse(det.update(1.082, 24.5))
        self.assertFalse(det.update(1.074, 24.4))
        self.assertFalse(det.in_contact)
        det.lock_motion_baseline()
        self.assertAlmostEqual(det.baseline, 24.5)
        self.assertFalse(det.update(1.066, 24.4))
        self.assertFalse(det.update(1.058, 24.0))
        self.assertTrue(det.update(1.020, 15.6))
        self.assertTrue(det.in_contact)
        self.assertFalse(passed_stopping_position(0.90, 0.80, -1))
        self.assertTrue(passed_stopping_position(0.79, 0.80, -1))
        self.assertAlmostEqual(HOVER_BASELINE_S, 0.5)
        self.assertAlmostEqual(relative_drop(25.0, 23.0), 2.0)
        self.assertTrue(relative_unload(25.0, 23.4, drop_pct=6.0))
        self.assertFalse(relative_unload(25.0, 24.5, drop_pct=6.0))
        self.assertAlmostEqual(median_effort([25.1, 24.9, 25.0]), 25.0)
        self.assertAlmostEqual(
            descent_floor_q(
                q_start=1.03,
                max_descent_m=0.50,
                lift_min_m=0.0,
            ),
            0.53,
        )
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "contact_trace.csv"
            row = contact_trace_row(
                t0=0.0,
                q_lift=1.03,
                z_ee=0.88,
                effort=25.1,
                effort_filtered=25.0,
                baseline=25.0,
                phase="hover",
            )
            row["t_s"] = 0.0
            written = write_contact_trace_csv(dest, [row])
            text = written.read_text()
            self.assertIn(",".join(CONTACT_TRACE_COLUMNS), text)
            self.assertIn("hover", text)
            self.assertIn("25.100", text)
        self.assertTrue(guarded_too_sensitive(_R(100), 0.002))
        self.assertTrue(guarded_too_sensitive(_R(100), -0.003))
        self.assertFalse(guarded_too_sensitive(_R(100), 0.02))
        self.assertAlmostEqual(CLEAN_SURFACE_LIFT_EFFORT, 33.7)
        self.assertEqual(normalize_threshold_mode(""), THRESHOLD_MODE_DEFAULT)
        self.assertEqual(normalize_threshold_mode("robot_default"), THRESHOLD_MODE_DEFAULT)
        self.assertEqual(normalize_threshold_mode("custom_symmetric"), THRESHOLD_MODE_CUSTOM)
        with self.assertRaises(ValueError):
            normalize_threshold_mode("hover_plus_12")
        self.assertEqual(format_effort_pct(34.0), "+34.0")
        self.assertEqual(format_effort_pct(None), "none")
        self.assertFalse(effort_spike(20.0, 20.0, 12.0))
        self.assertTrue(effort_spike(33.0, 20.0, 12.0))
        self.assertTrue(effort_spike(-33.0, 20.0, 12.0))
        cloth = np.asarray([[0.0, 0.0, 0.1], [0.10, 0.0, 0.1]])
        self.assertAlmostEqual(
            nearest_grasp_xy_distance([0.019, 0.0, 0.3, 0.3], cloth),
            0.019,
        )
        self.assertLess(0.019, DEFAULT_MAX_GRASP_DISTANCE_M)
        self.assertGreater(
            nearest_grasp_xy_distance([0.043, 0.0, 0.3, 0.3], cloth),
            DEFAULT_MAX_GRASP_DISTANCE_M,
        )
        on_cloth = snap_grasp_inward([0.019, 0.0, 0.3, 0.3], cloth)
        self.assertFalse(on_cloth.snapped)
        np.testing.assert_allclose(on_cloth.action[:2], [0.019, 0.0])
        snapped = snap_grasp_inward(
            [-0.043, 0.0, 0.3, 0.3],
            cloth,
            inward_m=DEFAULT_SNAP_INWARD_M,
        )
        self.assertTrue(snapped.snapped)
        self.assertAlmostEqual(snapped.distance_before_m, 0.043)
        # Planned (-0.043, 0) → nearest (0, 0) → inland +1.5 cm toward +X.
        np.testing.assert_allclose(snapped.action[:2], [0.015, 0.0], atol=1e-9)
        np.testing.assert_allclose(snapped.action[2:], [0.3, 0.3])
        thin = np.asarray([[0.0, 0.0, 0.1]])
        fallback = snap_grasp_inward(
            [-0.05, 0.0, 0.2, 0.2],
            thin,
            inward_m=0.03,
        )
        self.assertTrue(fallback.used_nearest_only)
        np.testing.assert_allclose(fallback.action[:2], [0.0, 0.0])

    def test_stop_after_is_complete_then_stop(self):
        self.assertEqual(normalize_stop_after(""), "")
        self.assertEqual(normalize_stop_after("DONE"), "")
        self.assertEqual(normalize_stop_after("DESCEND_CONTACT"), "DESCEND_CONTACT")
        with self.assertRaises(ValueError):
            normalize_stop_after("GRASPING")
        self.assertTrue(should_stop_after("DESCEND_CONTACT", "DESCEND_CONTACT"))
        self.assertFalse(should_stop_after("DESCEND_BREAKAWAY", "DESCEND_CONTACT"))
        self.assertFalse(should_stop_after("GRASP_CLOSE", ""))
        self.assertTrue("DESCEND_CONTACT" in STAGES)
        self.assertTrue("REVERSE_PLACE" in STAGES)
        self.assertTrue("RESET_EE" in STAGES)
        self.assertTrue("DESCEND_COARSE" in STAGES)
        self.assertFalse("PREGRASP_PLAN" in STAGES)
        self.assertFalse("GRASPING" in STAGES)

    def test_intentional_stop_token_roundtrip(self):
        msg = intentional_stop_message("GRASP_CLOSE", "hold after GRASP_CLOSE")
        self.assertIn("INTENTIONAL_STOP_AFTER:GRASP_CLOSE", msg)
        self.assertEqual(parse_intentional_stop(msg), "GRASP_CLOSE")
        self.assertIsNone(parse_intentional_stop("ok e_g=0.01"))
        self.assertTrue(production_stop_after_ok(""))
        self.assertTrue(production_stop_after_ok("DONE"))
        self.assertFalse(production_stop_after_ok("LIFTING"))

    def test_parse_set_override_and_verify_gate(self):
        self.assertEqual(
            parse_set_override("contact.lift_effort:=20"),
            ("contact.lift_effort", "20"),
        )
        self.assertTrue(verify_allows_descend(e_g_m=0.02, reject_m=0.025))
        self.assertFalse(verify_allows_descend(e_g_m=0.03, reject_m=0.025))
        self.assertTrue(descend_budget_ok(dropped_m=0.0, max_descent_m=0.50))
        self.assertFalse(descend_budget_ok(dropped_m=0.50, max_descent_m=0.50))
        self.assertIn("contact.unload_drop_pct", relevant_params("DESCEND_CONTACT"))
        self.assertIn("contact.move_increment_m", relevant_params("DESCEND_CONTACT"))
        self.assertIn("contact.max_descent_m", relevant_params("DESCEND_CONTACT"))
        self.assertNotIn("contact.effort_spike", relevant_params("DESCEND_CONTACT"))
        self.assertNotIn("contact.cloth_below_estimate_m", relevant_params("DESCEND_CONTACT"))
        self.assertIn("grasp.lift_after_contact_m", relevant_params("GRASP_BUMP"))
        self.assertIn("grasp.lower_after_open_m", relevant_params("GRASP_TAP2"))
        self.assertIn("NO GRIPPER CLOSE", motion_lines("DESCEND_CONTACT"))
        self.assertIn("relative lift-effort unload", motion_lines("DESCEND_CONTACT")[0])
        self.assertIn("NO PULL", motion_lines("LIFTING"))
        self.assertIn("place at uncover pick", motion_lines("REVERSE_PLACE")[0])

    def test_jsp_lag_noop_when_tf_matches_driver(self):
        ee, d_lift, d_arm = correct_ee_base_for_jsp_lag(
            [0.01, -0.15, 0.95],
            lift_real=0.92,
            arm_real=0.05,
            lift_jsp=0.92,
            arm_jsp=0.05,
        )
        self.assertAlmostEqual(d_lift, 0.0, places=6)
        self.assertAlmostEqual(d_arm, 0.0, places=6)
        np.testing.assert_allclose(ee, [0.01, -0.15, 0.95], atol=1e-9)

    def test_lift_follows_ee_z_delta_not_absolute(self):
        cmd = command_toward_target(
            current_ee_base=np.array([0.3, 0.0, 0.80]),
            target_ee_base=np.array([0.3, 0.0, 0.90]),
            current_wrist_extension=0.20,
            current_lift=0.55,
            workspace=StretchWorkspace(),
        )
        self.assertAlmostEqual(cmd.joint_lift, 0.65, places=6)
        self.assertAlmostEqual(cmd.translate_mobile_base, 0.0, places=6)


class ReachabilityTests(unittest.TestCase):
    def test_reachable_short_pull(self):
        result = check_pull_path(
            [0.20, 0.0],
            [0.30, 0.10],
            pull_z=0.40,
            current_ee_base=np.array([0.20, 0.0, 0.40]),
            current_wrist_extension=0.20,
            current_lift=0.40,
            workspace=StretchWorkspace(),
        )
        self.assertTrue(result.reachable, result.reason)

    def test_exec_slack_rejects_16mm_past_hardware(self):
        from stretch_limits import EXEC_ARM_SLACK_M, HARDWARE_ARM_MAX_M

        result = check_pull_path(
            [0.20, 0.0],
            [0.20, -0.536],
            pull_z=0.40,
            current_ee_base=np.array([0.20, 0.0, 0.40]),
            current_wrist_extension=0.0,
            current_lift=0.40,
            workspace=StretchWorkspace(arm_max_m=HARDWARE_ARM_MAX_M),
            slack_m=EXEC_ARM_SLACK_M,
        )
        self.assertFalse(result.reachable)
        self.assertIn("wrist_extension", result.reason)

    def test_mid_uncover_pitch_is_not_full_wrist_down(self):
        """Leftover Uncover pitch −1.22 still has ~8 cm tool XY."""

        from stretch_cartesian import apply_execution_wrist_down

        ee_obs = np.array([-0.008, -0.223, 0.948], dtype=np.float64)
        mapped = apply_execution_wrist_down(ee_obs, pitch_rad=-1.215)
        already = apply_execution_wrist_down(ee_obs, pitch_rad=-0.5 * np.pi)
        parking = apply_execution_wrist_down(ee_obs, pitch_rad=0.0)
        self.assertGreater(float(mapped[1] - ee_obs[1]), 0.06)
        self.assertLess(float(mapped[1] - ee_obs[1]), 0.12)
        np.testing.assert_allclose(already, ee_obs, atol=1e-9)
        self.assertAlmostEqual(float(parking[1] - ee_obs[1]), 0.23, places=2)

        from stretch_cartesian import parking_is_execution_wrist_down

        self.assertFalse(
            parking_is_execution_wrist_down(pitch_rad=-1.215, wrist_extension_m=0.05)
        )
        self.assertTrue(
            parking_is_execution_wrist_down(
                pitch_rad=-0.5 * np.pi,
                yaw_rad=0.5 * np.pi,
                wrist_extension_m=0.05,
            )
        )
        self.assertFalse(
            parking_is_execution_wrist_down(
                pitch_rad=-0.5 * np.pi, wrist_extension_m=0.40
            )
        )
        # Tucked + full pitch-down is still not parking if yaw is ~145°.
        self.assertFalse(
            parking_is_execution_wrist_down(
                pitch_rad=-1.563,
                yaw_rad=2.534,
                roll_rad=-0.002,
                wrist_extension_m=0.018,
            )
        )
        from stretch_cartesian import parking_wrist_down_failures

        reasons = parking_wrist_down_failures(
            pitch_rad=-1.563,
            yaw_rad=2.534,
            roll_rad=-0.002,
            wrist_extension_m=0.018,
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("yaw 2.534", reasons[0])
        steps = plan_wrist_down_steps(
            pitch_rad=-1.563,
            yaw_rad=2.534,
            roll_rad=-0.002,
            wrist_extension_m=0.018,
        )
        self.assertEqual(steps[0].name, "yaw_sweep_clearance")

        snap = {
            "ee_base": [-0.008, -0.273, 0.948],
            "wrist_extension": 0.05,
            "joint_wrist_pitch": -1.215,
            "tf_live": True,
            "ee_base_arm0_wrist_down": [-0.008, -0.223, 0.948],
        }
        model = execution_ee_from_snapshot(snap)
        self.assertGreater(float(model[1] - (-0.223)), 0.06)
        self.assertLess(float(np.linalg.norm(model[:2] - mapped[:2])), 0.01)

    def test_execution_reachability_ignores_parking_arm(self):
        from canonical_bed import canonical_bed_frame

        frame = canonical_bed_frame(
            {
                0: [-0.42, -0.92, 0.0],
                1: [0.42, -0.92, 0.0],
                2: [0.42, 0.92, 0.0],
                3: [-0.42, 0.92, 0.0],
            }
        )
        eye = np.eye(4).tolist()
        snap_a = {
            "T_odom_layout": eye,
            "T_odom_base": eye,
            "ee_base": [0.0, -0.20, 0.40],
            "wrist_extension": 0.04,
            "joint_lift": 0.40,
            "joint_wrist_pitch": 0.0,
            "cloth_z_layout": 0.08,
        }
        snap_b = {
            "T_odom_layout": eye,
            "T_odom_base": eye,
            "ee_base": [0.0, -0.40, 0.40],
            "wrist_extension": 0.24,
            "joint_lift": 0.90,
            "joint_wrist_pitch": 0.0,
            "cloth_z_layout": 0.08,
        }
        np.testing.assert_allclose(
            execution_ee_from_snapshot(snap_a),
            execution_ee_from_snapshot(snap_b),
            atol=1e-9,
        )
        action = np.array([0.10, 0.0, 0.20, 0.15], dtype=np.float64)
        a = check_bed_pull_reachability(action, frame=frame, snapshot=snap_a)
        b = check_bed_pull_reachability(action, frame=frame, snapshot=snap_b)
        self.assertEqual(a.geometry, "execution_wrist_down")
        self.assertEqual(a.reachable, b.reachable)
        self.assertAlmostEqual(
            a.max_wrist_extension, b.max_wrist_extension, places=6
        )

    def test_negative_arm_is_restricted_to_zero(self):
        # TF EE already past the grasp (−Y). Restrict to 0; do not reject.
        result = check_pull_path(
            [0.20, -0.20],
            [0.40, -0.20],
            pull_z=1.195,
            current_ee_base=np.array([0.20, -0.35, 1.195]),
            current_wrist_extension=0.03,
            current_lift=1.10,
            workspace=StretchWorkspace(),
        )
        self.assertTrue(result.reachable, result.reason)
        self.assertGreaterEqual(result.max_wrist_extension, 0.0)

    def test_rejects_overextended_arm(self):
        # Arm is −Y_base, not +X_base. 0.70 m of −Y overextends.
        result = check_pull_path(
            [0.20, 0.0],
            [0.20, -0.70],
            pull_z=0.40,
            current_ee_base=np.array([0.20, 0.0, 0.40]),
            current_wrist_extension=0.20,
            current_lift=0.40,
            workspace=StretchWorkspace(arm_max_m=0.52),
        )
        self.assertFalse(result.reachable)
        self.assertIn("wrist_extension", result.reason)

    def test_canonical_snapshot_rejects_overextended_arm(self):
        from canonical_bed import canonical_bed_frame
        from stretch_reachability import check_canonical_action_from_snapshot

        frame = canonical_bed_frame(
            {
                0: [-0.42, -0.92, 0.0],
                1: [0.42, -0.92, 0.0],
                2: [0.42, 0.92, 0.0],
                3: [-0.42, 0.92, 0.0],
            }
        )
        snapshot = {
            "T_odom_layout": np.eye(4).tolist(),
            "T_odom_base": np.eye(4).tolist(),
            "ee_base": [0.0, 0.0, 0.40],
            "wrist_extension": 0.05,
            "joint_lift": 0.40,
            "cloth_z_layout": 0.08,
        }
        # Into-bed is −Y_base. 0.70 m from a retracted arm is past 0.52.
        result = check_canonical_action_from_snapshot(
            [0.0, 0.0, 0.0, -0.70],
            frame=frame,
            snapshot=snapshot,
            workspace=StretchWorkspace(arm_max_m=0.52),
        )
        self.assertFalse(result.reachable)
        self.assertIn("wrist_extension", result.reason)

    def test_ee_z_above_lift_max_is_still_reachable(self):
        # Wrist-zero TF: grasp_center Z ≈ 0.096 + joint_lift. Not joint_lift.
        result = check_pull_path(
            [0.20, -0.30],
            [0.40, -0.30],
            pull_z=1.195,
            current_ee_base=np.array([0.20, -0.30, 1.195]),
            current_wrist_extension=0.05,
            current_lift=1.10,
            workspace=StretchWorkspace(),
        )
        self.assertTrue(result.reachable, result.reason)

    def test_joint_lift_over_max_still_rejects(self):
        result = check_pull_path(
            [0.20, -0.30],
            [0.40, -0.30],
            pull_z=1.30,
            current_ee_base=np.array([0.20, -0.30, 1.10]),
            current_wrist_extension=0.05,
            current_lift=1.10,
            workspace=StretchWorkspace(),
        )
        self.assertFalse(result.reachable)
        self.assertIn("lift", result.reason)

    def test_long_base_x_move_is_reachable(self):
        result = check_pull_path(
            [0.20, 0.0],
            [0.80, 0.0],
            pull_z=0.40,
            current_ee_base=np.array([0.20, 0.0, 0.40]),
            current_wrist_extension=0.20,
            current_lift=0.40,
            workspace=StretchWorkspace(arm_max_m=0.52),
        )
        self.assertTrue(result.reachable, result.reason)

    def test_shared_cma_bounds_match_yesterday(self):
        lo, hi = cma_policy_bounds("symmetric")
        np.testing.assert_array_equal(lo, [-1, -1, -1, -1])
        np.testing.assert_array_equal(hi, [1, 1, 1, 1])
        lo_a, hi_a = cma_policy_bounds("asymmetric")
        np.testing.assert_array_equal(lo_a, [0.0, -0.5, 0.0, -1.0])
        np.testing.assert_array_equal(hi_a, [1.0, 1.0, 1.0, 1.0])
        x0 = clip_action_to_bounds(np.array([-0.2, -0.8, 0.3, 0.2]), lo_a, hi_a)
        np.testing.assert_allclose(x0, [0.0, -0.5, 0.3, 0.2])
        self.assertEqual(planner_workspace().arm_max_m, PLANNER_ARM_MAX_M)

    def test_restrict_cma_bounds_tightens_into_bed_x(self):
        from canonical_bed import load_canonical_frame

        pose = Path(os.environ.get("ROBE_TEST_POSE_DIR", ""))
        snap_p = pose / "stretch_reach_snapshot.json"
        frame_p = pose / "canonical_bed_frame.json"
        if not snap_p.is_file() or not frame_p.is_file():
            self.skipTest("current TL2 study data not present")
        import json

        snapshot = json.loads(snap_p.read_text())
        frame = load_canonical_frame(frame_p)
        lo, hi = restrict_cma_bounds_to_reach(
            np.array([-1.0, -1.0, -1.0, -1.0]),
            np.array([1.0, 1.0, 1.0, 1.0]),
            snapshot=snapshot,
            frame=frame,
            arm_max_m=0.51,
            action_scale=DEFAULT_ACTION_SCALE,
        )
        y_n = float(BED_Y_LIMITS[1]) / float(DEFAULT_ACTION_SCALE[1])
        self.assertAlmostEqual(lo[1], -y_n)
        self.assertAlmostEqual(hi[1], y_n)
        self.assertLess(float(BED_Y_LIMITS[1]), 1.038)
        self.assertLess(float(BED_Y_LIMITS[1]), 0.925)
        self.assertLess(hi[0] - lo[0], 2.0)
        grasp_n = -0.2948910267114222 / float(DEFAULT_ACTION_SCALE[0])
        self.assertGreaterEqual(grasp_n, lo[0] - 1e-6)
        self.assertLessEqual(grasp_n, hi[0] + 1e-6)

    def test_snapshot_uses_live_tf_flag(self):
        self.assertTrue(snapshot_uses_live_tf({"tf_live": True}))
        self.assertFalse(snapshot_uses_live_tf({"tf_live": False}))
        self.assertFalse(snapshot_uses_live_tf({"ee_source": "jsp_lag"}))

    def test_cma_x_strip_endpoints_pass_checker(self):
        """Analytical X-strip corners stay under the planning arm limit."""

        from canonical_bed import canonical_bed_frame

        frame = canonical_bed_frame(
            {
                0: [-0.42, -0.92, 0.0],
                1: [0.42, -0.92, 0.0],
                2: [0.42, 0.92, 0.0],
                3: [-0.42, 0.92, 0.0],
            }
        )
        # +X_base = +Y_layout (along bed); −Y_base = +X_layout (into bed).
        t_ob = np.eye(4)
        t_ob[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        t_ob[:3, 3] = np.array([-0.70, 0.0, 0.0])
        snap = {
            "T_odom_layout": np.eye(4).tolist(),
            "T_odom_base": t_ob.tolist(),
            "ee_base": [0.05, -0.12, 0.40],
            "wrist_extension": 0.04,
            "joint_lift": 0.40,
            "joint_wrist_pitch": 0.0,
            "cloth_z_layout": 0.08,
        }
        x_lo, x_hi = into_bed_canonical_x_limits(snap, frame, PLANNER_ARM_MAX_M)
        self.assertGreater(x_hi - x_lo, 0.20)
        ctx = CmaReachContext(
            snap, None, planner_workspace(), frame, PLANNER_ARM_MAX_M
        )
        y_g, y_r = BED_Y_LIMITS
        for gx, rx in ((x_lo, x_lo), (x_hi, x_hi), (x_lo, x_hi), (x_hi, x_lo)):
            action = np.array([gx, y_g, rx, y_r], dtype=np.float64)
            result = ctx.check(action)
            self.assertTrue(result.reachable, f"{action.tolist()} {result.reason}")
            assert_selected_action_reachable(action, ctx)
            self.assertLessEqual(
                float(result.max_wrist_extension), PLANNER_ARM_MAX_M + 1e-3
            )

    def test_x_strip_covers_full_cma_y_on_current_parking(self):
        """EE-Y-only invert leaked at y=+1.05; worst-Y invert must not."""

        from canonical_bed import load_canonical_frame

        pose = Path(os.environ.get("ROBE_TEST_POSE_DIR", ""))
        snap_p = pose / "stretch_reach_snapshot.json"
        frame_p = pose / "canonical_bed_frame.json"
        if not snap_p.is_file() or not frame_p.is_file():
            self.skipTest("current TL2 study data not present")
        import json

        snapshot = json.loads(snap_p.read_text())
        frame = load_canonical_frame(frame_p)
        y_limits = (
            -float(DEFAULT_ACTION_SCALE[1]),
            float(DEFAULT_ACTION_SCALE[1]),
        )
        leaked = check_bed_pull_reachability(
            [-0.0615, 1.05, -0.0615, -1.05],
            frame=frame,
            snapshot=snapshot,
            workspace=planner_workspace(),
        )
        self.assertFalse(leaked.reachable, leaked.reason)
        x_lo, x_hi = into_bed_canonical_x_limits(
            snapshot, frame, PLANNER_ARM_MAX_M, y_limits=y_limits
        )
        ctx = CmaReachContext(
            snapshot, None, planner_workspace(), frame, PLANNER_ARM_MAX_M
        )
        for gx, gy, rx, ry in (
            (x_lo, y_limits[0], x_hi, y_limits[1]),
            (x_hi, y_limits[1], x_lo, y_limits[0]),
            (x_hi, y_limits[1], x_hi, y_limits[0]),
        ):
            action = np.array([gx, gy, rx, ry], dtype=np.float64)
            result = ctx.check(action)
            self.assertTrue(result.reachable, f"{action.tolist()} {result.reason}")
            self.assertLessEqual(
                float(result.max_wrist_extension), PLANNER_ARM_MAX_M + 1e-3
            )

    def test_load_cma_reach_reads_pose_snapshot(self):
        pose = Path(os.environ.get("ROBE_TEST_POSE_DIR", ""))
        snap = snapshot_path(pose)
        if not snap.is_file() or not (pose / "canonical_bed_frame.json").is_file():
            self.skipTest("yesterday TL2 study data not present")
        ctx = load_cma_reach(
            pose_dir=pose,
            snapshot_path_or_none=snap,
            arm_max_m=PLANNER_ARM_MAX_M,
        )
        self.assertTrue(ctx.enabled)
        # Yesterday's executed Uncover was reachable from that parking pose.
        result = ctx.check([-0.379, -0.505, -0.258, 0.635])
        self.assertEqual(result.geometry, "execution_wrist_down")

    def test_bed_frame_allows_negative_x(self):
        from stretch_reachability import check_bed_frame_pull

        result = check_bed_frame_pull(
            [0.026, -0.638],
            [-0.195, 0.463],
            workspace=StretchWorkspace(),
        )
        self.assertTrue(result.reachable, result.reason)


class LocalizeTests(unittest.TestCase):
    def test_three_ids_on_two_posts_fail(self):
        self.assertFalse(sample_acceptable((0, 10, 1), min_posts=3))
        self.assertTrue(sample_acceptable((0, 1, 2), min_posts=3))

    def test_multi_view_union_and_fuse(self):
        a = np.eye(4)
        b = np.eye(4)
        b[:3, 3] = [0.02, 0.0, 0.0]
        samples = [
            LocalizationSample(a, (0, 3), 1.0, frozenset({0, 3})),
            LocalizationSample(b, (0, 1), 1.2, frozenset({0, 1})),
        ]
        self.assertEqual(union_posts(samples), frozenset({0, 1, 3}))
        fused = fuse_odom_layout(samples)
        np.testing.assert_allclose(fused[:3, 3], [0.01, 0.0, 0.0], atol=1e-12)
        ok, reason = fusion_acceptable(samples)
        self.assertTrue(ok, reason)
        merged = fuse_localization_samples(samples)
        self.assertEqual(merged.posts, frozenset({0, 1, 3}))

    def test_two_posts_per_view_need_union(self):
        a = np.eye(4)
        b = np.eye(4)
        samples = [
            LocalizationSample(a, (0, 1), 0.8, frozenset({0, 1})),
            LocalizationSample(b, (1, 2), 0.9, frozenset({1, 2})),
        ]
        self.assertTrue(sample_acceptable((0, 1), min_posts=2))
        ok, reason = fusion_acceptable(samples, min_posts=3)
        self.assertTrue(ok, reason)

    def test_fusion_rejects_inconsistent_views(self):
        a = np.eye(4)
        b = np.eye(4)
        b[:3, 3] = [0.20, 0.0, 0.0]
        samples = [
            LocalizationSample(a, (0, 1), 1.0, frozenset({0, 1})),
            LocalizationSample(b, (2, 3), 1.0, frozenset({2, 3})),
        ]
        ok, reason = fusion_acceptable(samples)
        self.assertFalse(ok)
        self.assertIn("spread", reason)


class HelloPoseTests(unittest.TestCase):
    def test_two_tuple_is_velocity_unless_custom_contact(self):
        encoded = encode_move_to_pose({"joint_lift": (0.2, 0.05)})
        self.assertEqual(encoded["velocities"], [0.05])
        self.assertIsNone(encoded["effort"])
        contact = encode_move_to_pose(
            {"joint_lift": (0.2, 25.0)}, custom_contact_thresholds=True
        )
        self.assertEqual(contact["effort"], [25.0])
        self.assertIsNone(contact["velocities"])

    def test_three_tuple_is_vel_acc_without_effort(self):
        encoded = encode_move_to_pose({"joint_lift": (0.8, 0.025, 0.05)})
        self.assertEqual(encoded["positions"], [0.8])
        self.assertEqual(encoded["velocities"], [0.025])
        self.assertEqual(encoded["accelerations"], [0.05])
        self.assertIsNone(encoded["effort"])
        self.assertFalse(encoded["custom_contact_thresholds"])

    def test_mixed_effort_is_rejected(self):
        with self.assertRaises(HelloPoseError):
            encode_move_to_pose(
                {"joint_lift": (0.2, 25.0), "joint_arm": 0.3},
                custom_contact_thresholds=True,
            )


class FrameContractTests(unittest.TestCase):
    def test_yaw_then_relocalize_zeros_error(self):
        t_odom_layout = np.eye(4)
        yaw = np.deg2rad(25.0)
        t_odom_layout[:3, :3] = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        t_odom_base = np.eye(4)
        err = yaw_align_error_rad(t_odom_layout, t_odom_base)
        self.assertAlmostEqual(err, yaw, places=6)
        aligned = apply_base_yaw(t_odom_base, err)
        self.assertAlmostEqual(
            yaw_align_error_rad(t_odom_layout, aligned), 0.0, places=6
        )

    def test_bedside_parking_matches_bed_y_not_bed_x(self):
        t_odom_layout = np.eye(4)
        t_odom_base = np.eye(4)
        # Identity: +X_base is bed +X, not bed +Y → ~90°.
        err_wrong = bedside_parking_error_rad(t_odom_layout, t_odom_base)
        self.assertGreater(abs(err_wrong), 1.0)
        yaw = np.deg2rad(90.0)
        parked = apply_base_yaw(t_odom_base, yaw)
        self.assertAlmostEqual(
            bedside_parking_error_rad(t_odom_layout, parked), 0.0, places=5
        )
        self.assertAlmostEqual(
            bedside_parallel_error_rad(t_odom_layout, parked), 0.0, places=5
        )
        self.assertAlmostEqual(
            bedside_arm_into_bed_error_rad(t_odom_layout, parked), 0.0, places=5
        )

    def test_raw_z_down_t_is_parallel_and_into_bed(self):
        # Session-like Stretch PnP: Z-down, +Y_layout ≈ −X_odom.
        t_raw = np.array(
            [
                [-0.041, -0.997, -0.066, -0.144],
                [-0.996, 0.046, -0.080, -0.690],
                [0.082, 0.062, -0.995, 0.877],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        t_ob = np.eye(4)
        # +X_base vs +Y_raw is ~180°, but the base is still parallel and
        # the arm (−Y_base) still points along +X_bed.
        signed = bedside_parking_error_rad(t_raw, t_ob)
        self.assertGreater(abs(signed), np.deg2rad(150.0))
        self.assertLess(bedside_parallel_error_rad(t_raw, t_ob), np.deg2rad(15.0))
        self.assertLess(
            abs(bedside_arm_into_bed_error_rad(t_raw, t_ob)), np.deg2rad(15.0)
        )
        t_zup = ensure_layout_z_up(t_raw)
        self.assertLess(
            abs(bedside_parking_error_rad(t_zup, t_ob)), np.deg2rad(15.0)
        )

    def test_ensure_layout_z_up_reverses_along_bed_drive(self):
        t_raw = t_odom_layout_for_motion(
            np.array(
                [
                    [-0.041, -0.997, -0.066, -0.144],
                    [-0.996, 0.046, -0.080, -0.690],
                    [0.082, 0.062, -0.995, 0.877],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        )
        t_ob = np.eye(4)
        ee = np.array([-0.011, -0.155, 0.95])
        grasp_layout = np.array([-0.31, -0.42, 0.07])

        def d_base(t_ol: np.ndarray) -> float:
            p = invert_transform(t_ob) @ (t_ol @ np.append(grasp_layout, 1.0))
            target = p[:3].copy()
            target[2] = ee[2]
            return command_toward_target(
                current_ee_base=ee,
                target_ee_base=target,
                current_wrist_extension=0.05,
                current_lift=1.0,
                workspace=StretchWorkspace(),
            ).translate_mobile_base

        raw_cmd = d_base(t_raw)
        zup_cmd = d_base(ensure_layout_z_up(t_raw))
        self.assertLess(raw_cmd * zup_cmd, 0.0)
        self.assertGreater(abs(zup_cmd), 0.2)

    def test_hover_uses_ee_height_not_cloth_z(self):
        t_raw = np.array(
            [
                [-0.041, -0.997, -0.066, -0.144],
                [-0.996, 0.046, -0.080, -0.690],
                [0.082, 0.062, -0.995, 0.877],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        cloth = np.array([-0.31, -0.42, 0.07])
        hover = layout_xy_at_height(cloth, 0.47)
        p_cloth = t_raw @ np.append(cloth, 1.0)
        p_hover = t_raw @ np.append(hover, 1.0)
        leak = float(np.linalg.norm(p_hover[:2] - p_cloth[:2]))
        self.assertGreater(leak, 0.025)
        self.assertAlmostEqual(hover[0], cloth[0])
        self.assertAlmostEqual(hover[1], cloth[1])

    def test_compose_odom_layout(self):
        t_odom_cam = np.eye(4)
        t_odom_cam[:3, 3] = [1.0, 2.0, 0.5]
        t_cam_layout = np.eye(4)
        t_cam_layout[:3, 3] = [0.1, 0.0, 0.0]
        fused = compose_t_odom_layout(t_odom_cam, t_cam_layout)
        np.testing.assert_allclose(fused[:3, 3], [1.1, 2.0, 0.5])

    def test_canonical_pickle_maps_without_scale_or_mirror(self):
        centers = {
            0: [-0.4225, -0.925, 0.0],
            1: [0.4225, -0.925, 0.0],
            2: [0.4225, 0.925, 0.0],
            3: [-0.4225, 0.925, 0.0],
        }
        frame = canonical_bed_frame(centers)
        action = np.array([0.10, -0.20, 0.15, 0.05], dtype=np.float64)
        mapped = transform_action_canonical_to_layout(action, frame)
        np.testing.assert_allclose(mapped, action, atol=1e-9)
        self.assertLess(np.max(np.abs(mapped[:2])), 0.42)
        self.assertNotAlmostEqual(0.10 * (0.88 / 0.845), mapped[0])

    def test_layout_canonical_disagreement(self):
        self.assertAlmostEqual(
            layout_canonical_disagreement_m([0.03, 0.04, 0.0]), 0.05, places=6
        )

    def test_grasp_xy_correction_after_wrist_down(self):
        delta = grasp_xy_correction([0.10, 0.20], [0.12, 0.19])
        np.testing.assert_allclose(delta, [0.02, -0.01])


if __name__ == "__main__":
    unittest.main()
