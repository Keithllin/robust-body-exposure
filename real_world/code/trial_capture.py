"""Capture-stage command builders. Algorithms stay in capture_blanket / pose."""

from __future__ import annotations

from pathlib import Path

from artifact_contract import ArtifactError, assert_session_filters
from stretch_limits import snapshot_path, snapshot_uses_live_tf
from trial_context import TrialContext
from trial_layout import pose_pack_complete


def body_reuse_flags(ctx: TrialContext) -> list[str]:
    if ctx.args.reuse_subject_body:
        return ["--reuse-subject-body"]
    return []


def pose_capture_cmd(
    ctx: TrialContext, *, recapture: bool = False, remeasure: bool = False
) -> list[str]:
    cmd = [
        ctx.robe_py,
        str(ctx.code_dir / "capture_human_pose.py"),
        "--subject-dir",
        str(ctx.subject_dir),
        "--pose-dir",
        str(ctx.pose_dir),
        "--manikin",
        ctx.args.manikin,
    ]
    if ctx.args.ceiling_serial:
        cmd.extend(["--serial", str(ctx.args.ceiling_serial)])
    if ctx.args.require_aruco:
        cmd.append("--require-aruco")
    if recapture:
        cmd.append("--recapture-pose")
    if remeasure:
        cmd.append("--remeasure-body")
    if ctx.args.reuse_subject_body:
        cmd.append("--reuse-subject-body")
    return cmd


def blanket_capture_cmd(
    ctx: TrialContext, target_dir: Path, *, require_calibration: bool = True
) -> list[str]:
    if ctx.session_dir is not None and ctx.session_filters:
        roles = tuple(
            role.strip()
            for role in str(ctx.args.camera_roles).split(",")
            if role.strip()
        )
        assert_session_filters(
            ctx.session_filters,
            roles=roles,
            allow_calibration_fallback=bool(ctx.args.allow_unfrozen_session),
        )
    command = [
        ctx.robe_py,
        str(ctx.code_dir / "capture_blanket.py"),
        "--subject-dir",
        str(ctx.subject_dir),
        "--pose-dir",
        str(target_dir),
        "--manikin",
        ctx.args.manikin,
        "--roles",
        ctx.args.camera_roles,
        "--merge-mode",
        ctx.args.merge_mode,
        "--support-radius-3d",
        str(ctx.args.support_radius),
        "--mask-backend",
        ctx.args.mask_backend,
    ]
    if ctx.profile.marker_policy == "force_redetect" or ctx.args.detect_top_markers:
        command.append("--detect-top-markers")
    if ctx.args.camera_extrinsics:
        command.extend(["--extrinsics", ctx.args.camera_extrinsics])
    if require_calibration or ctx.args.require_camera_calibration or ctx.is_recover:
        command.append("--require-calibration")
    if ctx.session_dir is not None and not ctx.args.allow_unfrozen_session:
        command.append("--forbid-calibration-fallback")
    if ctx.args.pcd_exposure is not None:
        command.extend(["--exposure", str(int(ctx.args.pcd_exposure))])
    if ctx.args.pcd_gain is not None:
        command.extend(["--gain", str(int(ctx.args.pcd_gain))])
    filters = ctx.session_filters
    if filters.get("ceiling") and Path(filters["ceiling"]).is_file():
        command.extend(["--filter-json", str(filters["ceiling"])])
    if filters.get("side_left") and Path(filters["side_left"]).is_file():
        command.extend(["--filter-json-side-left", str(filters["side_left"])])
    if filters.get("side_right") and Path(filters["side_right"]).is_file():
        command.extend(["--filter-json-side-right", str(filters["side_right"])])
    return command


def prepare_wrist_cmd(ctx: TrialContext) -> list[str]:
    return ["bash", "-lc", _ros2_python(ctx, ctx.code_dir / "prepare_wrist_down.py")]


def snapshot_cmd(
    ctx: TrialContext,
    dest: Path,
    *,
    prepared: bool,
    debug_override: bool,
    save_uncover_home: bool,
) -> list[str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    extra = (
        f" --planning-wrist-down-prepared {1 if prepared else 0}"
        f" --planning-debug-override {1 if debug_override else 0}"
        + (" --save-uncover-home" if save_uncover_home else "")
    )
    inner = (
        _ros2_python(
            ctx,
            ctx.code_dir / "snapshot_stretch_reach.py",
            f"--pose-dir { _q(ctx.pose_dir) } --output { _q(dest) }{extra}",
        )
    )
    return ["bash", "-lc", inner]


def wrist_match_cmd(ctx: TrialContext, image: Path, dest: Path) -> list[str]:
    if ctx.session_dir is None:
        raise ArtifactError("136 rematch requires a frozen session")
    if ctx.profile.registration_policy == "refreeze_session":
        raise ArtifactError(
            "registration_policy=refreeze_session needs an explicit backup "
            "and operator confirm; refusing from capture."
        )
    register_py = ctx.real_world_dir / "sessions" / "register_robot_to_planning.py"
    inner = _ros2_python(
        ctx,
        register_py,
        f"--session-dir { _q(ctx.session_dir) } "
        f"--image { _q(image) } --output-json { _q(dest) } "
        "--on-missing-136 keep",
    )
    return ["bash", "-lc", inner]


def cma_snapshot_dest(ctx: TrialContext) -> Path:
    return snapshot_path(ctx.pose_dir, ctx.args.stretch_reach_snapshot)


def should_reuse_snapshot(ctx: TrialContext, dest: Path, *, force: bool) -> bool:
    reuse = dest.is_file() and not ctx.args.resnapshot_stretch and not force
    if reuse and snapshot_uses_live_tf(dest):
        return False
    return reuse


def pose_stage_needed(ctx: TrialContext) -> str:
    """Return recapture | remeasure | reuse | capture."""

    pack_ok = pose_pack_complete(ctx.pose_dir, require_body=True)
    if ctx.args.recapture_pose:
        return "recapture"
    if pack_ok and ctx.args.remeasure_body:
        return "remeasure"
    if pack_ok:
        return "reuse"
    return "capture"


def _q(path: Path | str) -> str:
    import shlex

    return shlex.quote(str(path))


def _ros2_python(ctx: TrialContext, script: Path, extra: str = "") -> str:
    import os
    import shlex

    domain = os.environ.get("ROS_DOMAIN_ID", "12")
    setup = ctx.real_world_dir / "ros2" / "source_humble.sh"
    tail = f"python3 {shlex.quote(str(script))}"
    if extra:
        tail = f"{tail} {extra}"
    return (
        "set -eo pipefail; set +u; "
        f"source {shlex.quote(str(setup))}; "
        f"export ROS_DOMAIN_ID={shlex.quote(domain)}; "
        + tail
    )
