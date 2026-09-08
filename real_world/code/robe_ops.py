#!/usr/bin/env python3
"""Operator toolkit: doctor | validate | dry-run | replay | bundle.

Read-only except ``bundle`` (writes an archive). Formal motion gates stay in
``preflight_bed_pull.py``. Do not send /bed_pull from this tool.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
REAL_WORLD_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from artifact_contract import (  # noqa: E402
    ArtifactError,
    assert_capture_contract,
    assert_pose_pack,
    load_capture_contract,
    validate_capture_contract,
)
from session_paths import evaluate_session, resolve_session_dir_once  # noqa: E402
from stretch_cartesian import (  # noqa: E402
    default_execution_wrist,
    execution_wrist_as_dict,
    plan_wrist_down_steps,
)
from stretch_grasp_stages import (  # noqa: E402
    FAULT_OK,
    FAULT_PREFLIGHT,
    RECIPES,
    recipe_stop_after,
)
from stretch_safety import (  # noqa: E402
    DEFAULT_LIVE_GEOMETRY_POLICY,
    validate_goal_params,
)
from trial_layout import (  # noqa: E402
    final_dir,
    initial_dir,
    intermediate_dir,
)
from trial_state import (  # noqa: E402
    IDENTITY_NAME,
    PROGRESS_NAME,
    load_identity,
    load_progress,
    next_pending_stage,
)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _print_lines(lines: list[str]) -> int:
    for line in lines:
        print(line)
    return 0 if all(not line.startswith("FAIL") for line in lines) else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    lines = ["ROBE OPS doctor"]
    try:
        session = resolve_session_dir_once()
        report = evaluate_session(session)
        lines.append(report.text())
        lines.append("READY FOR TRIAL: YES" if report.ready else "READY FOR TRIAL: NO")
    except FileNotFoundError as exc:
        lines.append(f"FAIL session: {exc}")
    wrist = default_execution_wrist()
    lines.append(f"execution wrist: {execution_wrist_as_dict(wrist)}")
    lines.append(f"recipes: {', '.join(RECIPES)}")
    lines.append(f"live_geometry.policy default={DEFAULT_LIVE_GEOMETRY_POLICY}")
    domain = os.environ.get("ROS_DOMAIN_ID", "12")
    setup = REAL_WORLD_DIR / "ros2" / "source_humble.sh"
    if args.probe_ros and setup.is_file():
        probe = (
            "set +u; "
            f"source {setup}; "
            f"export ROS_DOMAIN_ID={domain}; "
            "ros2 action info /bed_pull >/dev/null 2>&1; echo bed_pull=$?; "
            "ros2 topic info /stretch/joint_states >/dev/null 2>&1; echo jsp=$?"
        )
        completed = subprocess.run(["bash", "-lc", probe], capture_output=True, text=True)
        lines.append(completed.stdout.strip() or "ROS probe produced no output")
        if completed.returncode != 0:
            lines.append("FAIL ROS probe (source Humble / DOMAIN 12)")
    else:
        lines.append("ROS probe skipped (pass --probe-ros after sourcing Humble)")
    return _print_lines(lines)


def _trial_dir(args: argparse.Namespace) -> Path:
    if args.trial:
        return Path(args.trial).expanduser().resolve()
    raise SystemExit("--trial is required")


def cmd_validate(args: argparse.Namespace) -> int:
    trial = _trial_dir(args)
    errors: list[str] = []
    try:
        identity = load_identity(trial)
        print(f"identity {identity.get('trial_id')} TL{identity.get('tl_code')}")
    except FileNotFoundError as exc:
        errors.append(str(exc))
    try:
        progress = load_progress(trial)
        print(f"progress current={progress.get('current')} next={next_pending_stage(progress)}")
    except FileNotFoundError as exc:
        errors.append(str(exc))
    try:
        assert_pose_pack(trial)
        print("pose pack OK")
    except ArtifactError as exc:
        errors.append(str(exc))
    for label, directory in (
        ("initial", initial_dir(trial)),
        ("intermediate", intermediate_dir(trial)),
        ("final", final_dir(trial)),
    ):
        contract_path = directory / "capture_contract.json"
        if not contract_path.is_file():
            if (directory / "blanket_pcd.pcd").is_file():
                errors.append(f"{label}: PCD exists but capture_contract.json is missing")
            continue
        try:
            assert_capture_contract(directory)
            print(f"{label} capture contract OK")
        except ArtifactError as exc:
            errors.append(str(exc))
    action = trial / "scaled_action.pkl"
    uncover = trial / "uncover" / "uncover_scaled_action.pkl"
    if not uncover.is_file():
        uncover = trial / "uncover_scaled_action.pkl"
    if args.preflight and (action.is_file() or uncover.is_file()):
        target = action if action.is_file() else uncover
        cmd = [
            sys.executable,
            str(CODE_DIR / "preflight_bed_pull.py"),
            "--action",
            str(target),
            "--pose-dir",
            str(trial),
        ]
        pcd = intermediate_dir(trial) / "blanket_pcd.pcd"
        if not pcd.is_file():
            pcd = initial_dir(trial) / "blanket_pcd.pcd"
        if pcd.is_file():
            cmd.extend(["--blanket-pcd", str(pcd)])
        print("canonical preflight:", " ".join(cmd))
        completed = subprocess.run(cmd)
        if completed.returncode != 0:
            errors.append(f"{FAULT_PREFLIGHT}: preflight_bed_pull exit {completed.returncode}")
    if errors:
        for item in errors:
            print(f"FAIL {item}")
        return 1
    print("validate OK")
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    trial = _trial_dir(args)
    recipe = args.recipe
    stop = recipe_stop_after(recipe)
    params = {
        "pull.recipe": recipe,
        "pull.stop_after": stop,
        "pull.controller": "streaming",
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
        "pull.dry_run": True,
    }
    errors = validate_goal_params(params, production=recipe == "full-pull")
    wrist = default_execution_wrist()
    steps = plan_wrist_down_steps(
        pitch_rad=0.0,
        yaw_rad=0.0,
        roll_rad=0.0,
        wrist_extension_m=0.02,
        pose=wrist,
    )
    report = {
        "trial": str(trial),
        "recipe": recipe,
        "stop_after": stop or "DONE",
        "controller": "streaming",
        "live_geometry_policy": "reject",
        "goal_param_errors": errors,
        "expected_wrist_steps": [
            {
                "name": step.name,
                "joints": step.joints,
                "duration_s": step.duration_s,
                "reason": step.reason,
            }
            for step in steps
        ],
        "motion_sent": False,
        "fault_code": FAULT_OK if not errors else FAULT_PREFLIGHT,
    }
    if args.action and Path(args.action).is_file():
        cmd = [
            sys.executable,
            str(CODE_DIR / "preflight_bed_pull.py"),
            "--action",
            str(args.action),
            "--pose-dir",
            str(trial),
        ]
        print("preflight (no /bed_pull):", " ".join(cmd))
        completed = subprocess.run(cmd)
        report["preflight_exit"] = completed.returncode
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


def cmd_replay(args: argparse.Namespace) -> int:
    trial = _trial_dir(args)
    identity = load_identity(trial) if (trial / IDENTITY_NAME).is_file() else {}
    progress = load_progress(trial) if (trial / PROGRESS_NAME).is_file() else {}
    checks: dict[str, Any] = {
        "trial": str(trial),
        "identity_tl": identity.get("tl_code"),
        "profile": (identity.get("profile") or {}).get("name"),
        "progress_current": progress.get("current"),
        "captures": {},
    }
    ok = True
    for label, directory in (
        ("initial", initial_dir(trial)),
        ("intermediate", intermediate_dir(trial)),
        ("final", final_dir(trial)),
    ):
        path = directory / "capture_contract.json"
        if not path.is_file():
            checks["captures"][label] = "missing"
            continue
        contract = load_capture_contract(directory)
        errors = validate_capture_contract(contract)
        checks["captures"][label] = "ok" if not errors else errors
        ok = ok and not errors
    print(json.dumps(checks, indent=2))
    return 0 if ok else 1


def cmd_bundle(args: argparse.Namespace) -> int:
    trial = _trial_dir(args)
    dest = Path(args.output) if args.output else trial / f"ops_bundle_{_now_stamp()}.tgz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    names = [
        IDENTITY_NAME,
        PROGRESS_NAME,
        "trial_manifest.json",
        "stretch_reach_snapshot.json",
        "uncover_action_overlay.png",
        "recover_action_overlay.png",
        "canonical_bed_frame.json",
    ]
    dirs = [
        initial_dir(trial),
        intermediate_dir(trial),
        final_dir(trial),
    ]
    with tarfile.open(dest, "w:gz") as archive:
        for name in names:
            path = trial / name
            if path.is_file():
                archive.add(path, arcname=path.name)
        for directory in dirs:
            contract = directory / "capture_contract.json"
            if contract.is_file():
                archive.add(contract, arcname=f"{directory.name}/{contract.name}")
            meta = directory / "pcd_capture_metadata.json"
            if meta.is_file():
                archive.add(meta, arcname=f"{directory.name}/{meta.name}")
        try:
            session = resolve_session_dir_once()
            session_json = session / "session.json"
            if session_json.is_file():
                archive.add(session_json, arcname="session.json")
        except FileNotFoundError:
            pass
        rev = subprocess.run(
            ["git", "-C", str(REAL_WORLD_DIR.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if rev.returncode == 0:
            info = dest.with_suffix(".revision.txt")
            info.write_text(rev.stdout)
            archive.add(info, arcname="git_revision.txt")
            info.unlink(missing_ok=True)
    print(f"bundle → {dest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="session READY + wrist + optional ROS probe")
    doctor.add_argument("--probe-ros", action="store_true")
    validate = sub.add_parser("validate", help="pose pack + capture contracts + preflight")
    validate.add_argument("--trial", type=Path, required=True)
    validate.add_argument("--preflight", action="store_true")
    dry = sub.add_parser("dry-run", help="recipe/probe report; never sends motion")
    dry.add_argument("--trial", type=Path, required=True)
    dry.add_argument("--recipe", default="full-pull", choices=list(RECIPES))
    dry.add_argument("--action", type=Path, default=None)
    replay = sub.add_parser("replay", help="offline artifact replay of an archived trial")
    replay.add_argument("--trial", type=Path, required=True)
    bundle = sub.add_parser("bundle", help="collect identity/progress/contracts/overlays")
    bundle.add_argument("--trial", type=Path, required=True)
    bundle.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    dispatch = {
        "doctor": cmd_doctor,
        "validate": cmd_validate,
        "dry-run": cmd_dry_run,
        "replay": cmd_replay,
        "bundle": cmd_bundle,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
