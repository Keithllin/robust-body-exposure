#!/usr/bin/env python3
"""Unit tests for safety gates, trial state/resume, artifacts, and recipes."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from artifact_contract import (  # noqa: E402
    ArtifactError,
    assert_bed_frame_transform,
    assert_session_filters,
    build_capture_contract,
    validate_capture_contract,
    write_capture_contract,
)
from session_paths import sha256_file  # noqa: E402
from stretch_cartesian import (  # noqa: E402
    EXECUTION_WRIST_YAW_RAD,
    YAW_SWEEP_CLEARANCE_M,
    default_execution_wrist,
    parking_is_execution_wrist_down,
    plan_wrist_down_steps,
    validate_execution_wrist,
    wrist_from_params,
)
from stretch_grasp_stages import (  # noqa: E402
    FAULT_INTENTIONAL_STOP,
    FAULT_OK,
    IntentionalStop,
    normalize_recipe,
    recipe_stop_after,
    resolve_stop_after,
)
from stretch_safety import (  # noqa: E402
    SafetyReject,
    apply_live_geometry_gate,
    assert_goal_params,
    normalize_controller,
    validate_goal_params,
)
from trial_config import (  # noqa: E402
    add_trial_arguments,
    identity_fields,
    named_stage_from_start_here,
    production_profile,
    resolve_profile,
)
from trial_state import (  # noqa: E402
    ResumeError,
    assert_safe_start,
    empty_progress,
    latest_trial_dir,
    next_pending_stage,
    progress_path,
    resolve_start_stage,
    should_run_stage,
    update_stage,
    write_identity,
    write_progress,
)


class WristSafetyTests(unittest.TestCase):
    def test_clearance_before_plus_90_yaw(self) -> None:
        steps = plan_wrist_down_steps(
            pitch_rad=0.0,
            yaw_rad=0.0,
            roll_rad=0.0,
            wrist_extension_m=0.02,
        )
        names = [step.name for step in steps]
        self.assertEqual(names[0], "yaw_sweep_clearance")
        self.assertAlmostEqual(
            float(steps[0].joints["joint_arm"]), YAW_SWEEP_CLEARANCE_M
        )
        self.assertIn("wrist_yaw_roll", names)
        self.assertIn("wrist_down", names)

    def test_already_down_is_noop(self) -> None:
        pose = default_execution_wrist()
        steps = plan_wrist_down_steps(
            pitch_rad=pose.pitch_rad,
            yaw_rad=pose.yaw_rad,
            roll_rad=pose.roll_rad,
            wrist_extension_m=pose.planning_safe_extension_m,
            skip_if_already_down=True,
        )
        self.assertEqual(steps, [])
        self.assertTrue(
            parking_is_execution_wrist_down(
                pitch_rad=pose.pitch_rad,
                yaw_rad=pose.yaw_rad,
                wrist_extension_m=0.05,
            )
        )

    def test_validate_execution_wrist_rejects_tiny_clearance(self) -> None:
        pose = wrist_from_params(yaw_sweep_clearance_m=0.01)
        self.assertTrue(validate_execution_wrist(pose))


class GoalParamTests(unittest.TestCase):
    def _ok(self) -> dict:
        return {
            "pull.controller": "streaming",
            "pull.recipe": "full-pull",
            "pull.stop_after": "",
            "live_geometry.policy": "reject",
            "pull.clearance_above_bed_m": 0.40,
            "workspace.arm_min_m": 0.0,
            "workspace.arm_max_m": 0.52,
            "workspace.lift_min_m": 0.0,
            "workspace.lift_max_m": 1.10,
            "approach.retract_arm_m": 0.05,
            "approach.clear_above_cloth_m": 0.12,
            "approach.pregrasp_pass_m": 0.015,
            "approach.pregrasp_reject_m": 0.025,
            "contact.max_descent_m": 0.50,
            "contact.min_descent_m": 0.05,
            "contact.breakaway_m": 0.03,
            "contact.unload_drop_pct": 15.0,
            "contact.disable_effort": 100.0,
        }

    def test_streaming_ok_trajectory_forbidden(self) -> None:
        self.assertEqual(normalize_controller("streaming"), "streaming")
        with self.assertRaises(SafetyReject):
            normalize_controller("trajectory")
        errors = validate_goal_params(self._ok())
        self.assertEqual(errors, [])

    def test_leftover_stop_after_rejected_in_production(self) -> None:
        params = self._ok()
        params["pull.stop_after"] = "LIFTING"
        errors = validate_goal_params(params, production=True)
        self.assertTrue(any("stop_after" in item for item in errors))

    def test_recipe_maps_to_stop_after(self) -> None:
        self.assertEqual(normalize_recipe("verify-only"), "verify-only")
        self.assertEqual(recipe_stop_after("contact-only"), "DESCEND_CONTACT")
        self.assertEqual(
            resolve_stop_after(recipe="grasp-only", stop_after="DONE"),
            "GRASP_CLOSE",
        )
        self.assertEqual(resolve_stop_after(recipe="full-pull", stop_after=""), "")

    def test_live_geometry_reject_and_warn(self) -> None:
        self.assertIsNone(
            apply_live_geometry_gate(error_m=0.01, label="xy", policy="reject")
        )
        with self.assertRaises(SafetyReject) as ctx:
            apply_live_geometry_gate(error_m=0.05, label="xy", policy="reject")
        self.assertEqual(ctx.exception.fault_code, "LIVE_GEOMETRY")
        warn = apply_live_geometry_gate(error_m=0.05, label="xy", policy="warn")
        self.assertIn("0.050", warn)

    def test_intentional_stop_is_not_ok(self) -> None:
        stop = IntentionalStop("GRASP_CLOSE")
        self.assertEqual(stop.fault_code, FAULT_INTENTIONAL_STOP)
        self.assertNotEqual(FAULT_INTENTIONAL_STOP, FAULT_OK)


class TrialStateTests(unittest.TestCase):
    def test_named_stage_compat_and_safe_start(self) -> None:
        self.assertEqual(named_stage_from_start_here("closed", 5), "recover-plan")
        self.assertEqual(named_stage_from_start_here("closed", 3), "uncover-exec")
        with self.assertRaises(ResumeError):
            assert_safe_start(
                start_here=4,
                start_stage="",
                tl_code="random",
                sub_id=None,
                resume="",
            )
        assert_safe_start(
            start_here=4,
            start_stage="",
            tl_code="12",
            sub_id=123,
            resume="",
        )

    def test_progress_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pose = Path(tmp) / "pose"
            pose.mkdir()
            write_progress(pose, empty_progress("closed", start_stage="pose"))
            update_stage(pose, "pose", status="succeeded")
            update_stage(pose, "initial-capture", status="succeeded")
            progress = json.loads((pose / "trial_progress.json").read_text())
            self.assertEqual(next_pending_stage(progress), "uncover-plan")
            self.assertFalse(should_run_stage(progress, "pose"))
            self.assertTrue(should_run_stage(progress, "uncover-plan"))
            self.assertFalse(should_run_stage(progress, "score"))

    def test_production_profile_and_cli_overrides(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        add_trial_arguments(parser)
        args = parser.parse_args(
            ["--subject-id", "S", "--pose-num", "1", "--profile", "production"]
        )
        profile = resolve_profile(args, ["--profile", "production"])
        self.assertEqual(profile.loop, "closed")
        self.assertTrue(profile.send_bed_pull)
        self.assertEqual(profile.live_geometry_policy, "reject")
        self.assertEqual(profile.mask_backend, "sam2")
        args2 = parser.parse_args(
            [
                "--subject-id",
                "S",
                "--pose-num",
                "1",
                "--profile",
                "production",
                "--loop",
                "open",
                "--no-ros2-execute",
            ]
        )
        profile2 = resolve_profile(
            args2, ["--profile", "production", "--loop", "open", "--no-ros2-execute"]
        )
        self.assertEqual(profile2.loop, "open")
        self.assertFalse(profile2.send_bed_pull)
        self.assertIn("loop", profile2.debug_overrides)

    def test_identity_records_random_tl(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        add_trial_arguments(parser)
        args = parser.parse_args(["--subject-id", "S", "--pose-num", "2"])
        profile = production_profile()
        fields = identity_fields(
            args=args,
            profile=profile,
            target_limb_code="12",
            sub_id=99,
            pose_dir=Path("/tmp/pose"),
        )
        self.assertEqual(fields["tl_code"], 12)
        self.assertEqual(fields["wrist"]["yaw_rad"], EXECUTION_WRIST_YAW_RAD)


class ArtifactContractTests(unittest.TestCase):
    def test_missing_t_bed_camera_rejected(self) -> None:
        with self.assertRaises(ArtifactError):
            assert_bed_frame_transform(None, role="ceiling")
        with self.assertRaises(ArtifactError):
            assert_session_filters({}, roles=("ceiling",))

    def test_capture_contract_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            rgb = dest / "covered_rgb_ceiling.png"
            mask = dest / "blanket_mask_ceiling.png"
            hsv = dest / "blanket_mask_ceiling_hsv.png"
            pcd = dest / "pcd_filtered_ceiling.pcd"
            merged = dest / "blanket_pcd.pcd"
            filt = dest / "filter.json"
            for path in (rgb, mask, hsv, pcd, merged, filt):
                path.write_bytes(b"x")
            metadata = {
                "frame": "canonical_bed",
                "mask_backend": "sam2",
                "extrinsics": str(filt),
                "roles": {
                    "ceiling": {
                        "T_bed_camera": np.eye(4).tolist(),
                        "filter_json": str(filt),
                        "mask_backend": "sam2",
                        "hsv_px": 10,
                        "mask_px": 12,
                        "num_points_bed": 100,
                    }
                },
                "canonical": {"source": "pack", "num_points_after_crop": 80},
            }
            contract = build_capture_contract(
                dest,
                metadata=metadata,
                session_hashes={"zed_extrinsics_sha256": sha256_file(filt)},
            )
            write_capture_contract(dest, contract)
            self.assertEqual(validate_capture_contract(contract), [])
            self.assertEqual(
                contract["session_hashes"]["zed_extrinsics_sha256"],
                sha256_file(filt),
            )
            missing = dict(contract)
            missing["roles"] = {
                "ceiling": {**contract["roles"]["ceiling"], "T_bed_camera": None}
            }
            self.assertTrue(validate_capture_contract(missing))


class MockExecutorTests(unittest.TestCase):
    def test_dry_run_recipe_has_no_motion_flag(self) -> None:
        from robe_ops import cmd_dry_run, cmd_replay
        import argparse

        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp)
            write_identity(trial, {"trial_id": "t", "tl_code": 2, "sub_id": 1})
            write_progress(trial, empty_progress("closed"))
            args = argparse.Namespace(
                trial=trial, recipe="verify-only", action=None
            )
            rc = cmd_dry_run(args)
            self.assertEqual(rc, 0)

    def test_golden_replay_capture_contract(self) -> None:
        from robe_ops import cmd_replay
        import argparse

        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp)
            write_identity(
                trial,
                {"trial_id": "gold", "tl_code": 12, "sub_id": 1, "profile": {"name": "production"}},
            )
            write_progress(trial, empty_progress("closed", start_stage="score"))
            capture = trial / "initial"
            capture.mkdir()
            for name in (
                "covered_rgb_ceiling.png",
                "blanket_mask_ceiling.png",
                "blanket_mask_ceiling_hsv.png",
                "pcd_filtered_ceiling.pcd",
                "blanket_pcd.pcd",
                "filter.json",
            ):
                (capture / name).write_bytes(b"g")
            metadata = {
                "frame": "canonical_bed",
                "mask_backend": "sam2",
                "extrinsics": str(capture / "filter.json"),
                "roles": {
                    "ceiling": {
                        "T_bed_camera": np.eye(4).tolist(),
                        "filter_json": str(capture / "filter.json"),
                        "mask_backend": "sam2",
                    }
                },
                "canonical": {"source": "pack", "num_points_after_crop": 10},
            }
            write_capture_contract(capture, build_capture_contract(capture, metadata=metadata))
            rc = cmd_replay(argparse.Namespace(trial=trial))
            self.assertEqual(rc, 0)

    def test_latest_trial_picker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = Path(tmp)
            a = subject / "pose_a"
            b = subject / "pose_b"
            a.mkdir()
            b.mkdir()
            write_progress(a, empty_progress("closed"))
            write_progress(b, empty_progress("closed"))
            import os
            os.utime(progress_path(b), (1_700_000_100, 1_700_000_100))
            os.utime(progress_path(a), (1_700_000_000, 1_700_000_000))
            self.assertEqual(latest_trial_dir(subject), b)


if __name__ == "__main__":
    unittest.main()
