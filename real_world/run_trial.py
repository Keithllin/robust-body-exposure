#!/usr/bin/env python3
"""Interactive real-world trial orchestrator (named stages + resume).

Capture uses conda env ``robe-zed``; planning uses ``robe``. Study data lands
under ``real_world/STUDY_DATA/``. This file only parses CLI, loads trial
state, and dispatches stages. Command builders live in ``trial_*.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

REAL_WORLD_DIR = Path(__file__).resolve().parent
ROBE_ROOT = REAL_WORLD_DIR.parent
CODE_DIR = REAL_WORLD_DIR / "code"
sys.path.insert(0, str(CODE_DIR))

from conda_python import (  # noqa: E402
    drop_foreign_site_packages,
    env_for_python,
    robe_python,
    zed_python,
)

drop_foreign_site_packages()

import numpy as np  # noqa: E402
from compare_recover_sources import (  # noqa: E402
    write_comparison_montage,
    write_pose_comparison,
)
from session_paths import (  # noqa: E402
    evaluate_session,
    load_pcd_config,
    resolve_pcd_filters,
    resolve_session_dir_once,
    session_manifest_fields,
    session_paths,
    write_active_trial,
)
from trial_config import (  # noqa: E402
    STUDY_TLS,
    TL_LABELS,
    add_trial_arguments,
    argv_has,
    identity_fields,
    resolve_profile,
)
from trial_context import TrialContext  # noqa: E402
from trial_capture import (  # noqa: E402
    blanket_capture_cmd,
    cma_snapshot_dest,
    pose_capture_cmd,
    pose_stage_needed,
    prepare_wrist_cmd,
    should_reuse_snapshot,
    snapshot_cmd,
    wrist_match_cmd,
)
from trial_execute import (  # noqa: E402
    preflight_cmd,
    preflight_must_stop,
    recover_grasp_was_snapped,
    return_home_cmd,
    send_bed_pull_cmd,
)
from trial_layout import (  # noqa: E402
    apply_pose_pack,
    export_pose_pack_to_session,
    recover_pred_dir as layout_recover_pred_dir,
    recover_sensor_dir as layout_recover_sensor_dir,
    recover_snap_dir as layout_recover_snap_dir,
    resolve_ceiling_rgb,
    resolve_uncover_file,
    session_pose_dir,
)
from trial_operator import (  # noqa: E402
    OperatorQuit,
    mark_running,
    mark_succeeded,
    prompt_execution_quality,
    prompt_failure_type,
    record_failure,
    run_command,
    wait_yes as operator_wait_yes,
)
from trial_plan import (  # noqa: E402
    overlay_cmd,
    publish_cma_box,
    recover_plan_cmd,
    uncover_plan_cmd,
)
from trial_score import (  # noqa: E402
    build_trial_manifest,
    recover_f1_cmd,
    uncover_f1_cmd,
)
from trial_state import (  # noqa: E402
    ResumeError,
    assert_safe_start,
    empty_progress,
    load_identity,
    load_progress,
    next_pending_stage,
    resolve_resume_dir,
    resolve_start_stage,
    should_run_stage,
    write_identity,
    write_progress,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="")
    return add_trial_arguments(parser)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_bed_action(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        action = np.asarray(pickle.load(handle), dtype=np.float64)
    return action.reshape(4)


def _action_disagreement(pred_action: np.ndarray, sensor_action: np.ndarray) -> dict:
    grasp = float(np.linalg.norm(pred_action[:2] - sensor_action[:2]))
    release = float(np.linalg.norm(pred_action[2:] - sensor_action[2:]))
    return {
        "grasp_l2_m": grasp,
        "release_l2_m": release,
        "mean_endpoint_l2_m": float(0.5 * (grasp + release)),
        "full_action_l2_m": float(np.linalg.norm(pred_action - sensor_action)),
    }


def _open_overlay(path: Path) -> None:
    if not path.is_file():
        return
    print(f"  overlay: {path}")
    if not os.environ.get("DISPLAY"):
        return
    opener = shutil.which("xdg-open") or shutil.which("eog")
    if opener is None:
        return
    try:
        subprocess.Popen(
            [opener, str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        print(f"WARN: could not open {path.name}: {exc}")


def _render_pcd_overlap(ctx: TrialContext, pcd_dir: Path, body_data: Path | None = None) -> None:
    vis_py = ctx.code_dir / "visualize_pcd_overlap.py"
    if not vis_py.is_file():
        return
    cmd = [ctx.robe_py, str(vis_py), "--pose-dir", str(pcd_dir)]
    if body_data is not None and body_data.is_file():
        cmd.extend(["--body-data", str(body_data)])
    result = subprocess.run(cmd, check=False, env=env_for_python(ctx.robe_py))
    if result.returncode:
        print(f"WARN: PCD overlap failed for {pcd_dir} (exit {result.returncode})")


def _run_ctx(ctx: TrialContext, cmd, *, context: str | None = None, stage: str | None = None) -> None:
    run_command(
        cmd,
        pose_dir=ctx.pose_dir,
        stage=stage,
        context=context,
        env_for=lambda exe: env_for_python(exe) if "python" in Path(exe).name else None,
    )


def _overlay_action(
    ctx: TrialContext,
    image: Path,
    output: Path,
    actions: list[tuple[str, Path]],
    title: str,
    *,
    stage: str | None = None,
) -> None:
    sim_origin = ctx.pose_dir / "sim_origin_data.pkl"
    if not image.is_file():
        raise SystemExit(f"Overlay RGB missing: {image}.")
    if not sim_origin.is_file():
        raise SystemExit(f"Overlay sim_origin missing: {sim_origin}.")
    _run_ctx(
        ctx,
        overlay_cmd(ctx, image, output, actions, title),
        context=f"action overlay → {output.name}",
        stage=stage,
    )
    if not output.is_file():
        raise SystemExit(f"Action overlay did not write {output}")
    pcd_output = output.with_name(f"{output.stem}_pcd_on_rgb{output.suffix}")
    _run_ctx(
        ctx,
        overlay_cmd(ctx, image, pcd_output, actions, title, draw_pcd=True),
        context=f"action overlay PCD-on-RGB → {pcd_output.name}",
        stage=stage,
    )
    print("Overlays (review before YES):")
    _open_overlay(output)
    _open_overlay(pcd_output)


def _capture_blanket(ctx: TrialContext, target_dir: Path, *, stage: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    _run_ctx(
        ctx,
        blanket_capture_cmd(ctx, target_dir),
        context=f"blanket capture → {target_dir.name}",
        stage=stage,
    )
    _render_pcd_overlap(ctx, target_dir)


def _run_prepare_wrist_down(ctx: TrialContext, *, stage: str) -> None:
    try:
        _run_ctx(ctx, prepare_wrist_cmd(ctx), context="prepare wrist-down", stage=stage)
    except (subprocess.CalledProcessError, SystemExit):
        raise SystemExit(
            "Wrist-down failed. Keep stretch_driver up, clear runstop, "
            "or pass --skip-prepare-wrist-down (debug only)."
        )


def _ensure_snapshot(
    ctx: TrialContext,
    *,
    required: bool,
    force: bool = False,
    save_uncover_home: bool = False,
    already_prepared: bool = False,
    stage: str | None = None,
) -> Path | None:
    if ctx.args.no_stretch_reach:
        print("CMA stretch reach / EE limits disabled (--no-stretch-reach).")
        return None
    dest = cma_snapshot_dest(ctx)
    if should_reuse_snapshot(ctx, dest, force=force):
        print(f"Reusing Stretch parking snapshot: {dest}")
        publish_cma_box(ctx, dest)
        return dest
    prepared = not ctx.args.skip_prepare_wrist_down
    debug_override = bool(ctx.args.skip_prepare_wrist_down)
    if already_prepared:
        print("Snapshot Stretch parking after wrist-down + first capture.")
    elif force:
        wait_yes(
            ctx,
            "Robot is still after Uncover and ready for wrist-down + Recover snapshot?",
        )
    else:
        wait_yes(ctx, "Stretch is parked and ready for wrist-down + this trial's snapshot?")
    if debug_override and not already_prepared:
        print("DEBUG OVERRIDE: --skip-prepare-wrist-down.")
        if ctx.closed_loop:
            wait_yes(
                ctx,
                "Closed-loop production should wrist-down first. "
                "Type YES to continue as debug?",
            )
    if prepared and not already_prepared:
        _run_prepare_wrist_down(ctx, stage=stage or "initial-capture")
    try:
        _run_ctx(
            ctx,
            snapshot_cmd(
                ctx,
                dest,
                prepared=prepared,
                debug_override=debug_override,
                save_uncover_home=save_uncover_home,
            ),
            context="stretch reach snapshot",
            stage=stage,
        )
    except (subprocess.CalledProcessError, SystemExit):
        if required:
            raise SystemExit("Stretch parking snapshot failed.")
        return None
    if dest.is_file():
        publish_cma_box(ctx, dest)
    return dest


def _capture_with_wrist(
    ctx: TrialContext,
    target_dir: Path,
    *,
    stage: str,
    save_uncover_home: bool,
    match_stem: str,
    prompt: str,
) -> None:
    print("Wrist-down first so ceiling 136 is in the same frame as the PCD.")
    wait_yes(ctx, prompt)
    if not ctx.args.no_stretch_reach and not ctx.args.skip_prepare_wrist_down:
        _run_prepare_wrist_down(ctx, stage=stage)
    _capture_blanket(ctx, target_dir, stage=stage)
    if ctx.profile.registration_policy == "trial_check" and not ctx.args.no_stretch_reach:
        image = target_dir / "covered_rgb_ceiling.png"
        dest = ctx.pose_dir / f"{match_stem}.json"
        _run_ctx(
            ctx,
            wrist_match_cmd(ctx, image, dest),
            context=f"{match_stem} rematch",
            stage=stage,
        )
    if not ctx.args.no_stretch_reach:
        _ensure_snapshot(
            ctx,
            required=True,
            force=True,
            save_uncover_home=save_uncover_home,
            already_prepared=True,
            stage=stage,
        )


def wait_yes(ctx: TrialContext, message: str) -> None:
    if ctx.wait_yes is not None:
        ctx.wait_yes(message)
        return
    operator_wait_yes(message)


def _preflight(ctx: TrialContext, action_path: Path, label: str) -> bool:
    if not action_path.is_file():
        print(f"WARN: missing action file {action_path}")
        return False
    print("Reachability preflight (reject, do not clip):")
    completed = subprocess.run(
        preflight_cmd(ctx, action_path, label),
        env=env_for_python(ctx.robe_py),
    )
    return completed.returncode == 0


def _confirm_and_send(ctx: TrialContext, action_path: Path, label: str, *, stage: str) -> None:
    if not ctx.profile.send_bed_pull:
        return
    wait_yes(
        ctx,
        f"Review {label} overlays, then send {label} /bed_pull now?",
    )
    trial_id = str(ctx.args.trial_id or ctx.pose_dir.name)
    if ctx.session_dir is not None:
        dest = write_active_trial(ctx.session_dir, ctx.pose_dir)
        print(f"active trial → {dest}")
    _run_ctx(
        ctx,
        send_bed_pull_cmd(ctx, action_path, trial_id),
        context="send /bed_pull",
        stage=stage,
    )


def _return_home(ctx: TrialContext, *, stage: str) -> None:
    if ctx.args.skip_return_home:
        print("Skipping return to Uncover home (--skip-return-home).")
        return
    wait_yes(ctx, "Bedside is clear; drive back to Uncover home now?")
    _run_ctx(ctx, return_home_cmd(ctx), context="return to Uncover home", stage=stage)


def _score_json(ctx: TrialContext, builder, dest: Path, *, stage: str, force: bool = False) -> dict | None:
    if dest.is_file() and not force:
        return _read_json(dest)
    cmd = builder(ctx)
    if cmd is None:
        return None
    try:
        _run_ctx(ctx, cmd, context=dest.name, stage=stage)
        return _read_json(dest) if dest.is_file() else None
    except subprocess.CalledProcessError as exc:
        print(f"WARN: {dest.name} failed: {exc}")
        return None


def run_pose(ctx: TrialContext) -> dict:
    mark_running(ctx.pose_dir, "pose")
    mode = pose_stage_needed(ctx)
    if mode == "recapture":
        wait_yes(ctx, "Ready to recapture the subject's pose?")
        _run_ctx(
            ctx,
            pose_capture_cmd(ctx, recapture=True, remeasure=ctx.args.remeasure_body),
            context="human pose / ArUco capture",
            stage="pose",
        )
    elif mode == "remeasure":
        wait_yes(ctx, "Ready to remeasure the 9 body diameters?")
        _run_ctx(
            ctx,
            pose_capture_cmd(ctx, remeasure=True),
            context="remeasure body diameters",
            stage="pose",
        )
    elif mode == "reuse":
        print(f"Reusing pose pack on this trial ({ctx.pose_dir.name}).")
    else:
        wait_yes(ctx, "Ready to capture the subject's pose?")
        _run_ctx(
            ctx,
            pose_capture_cmd(ctx, remeasure=ctx.args.remeasure_body),
            context="human pose / ArUco capture",
            stage="pose",
        )
    if ctx.session_dir is not None:
        from trial_layout import pose_pack_complete

        if pose_pack_complete(ctx.pose_dir, require_body=True):
            exported = export_pose_pack_to_session(
                ctx.pose_dir, ctx.session_dir, ctx.args.pose_num, overwrite=True
            )
            if exported:
                print(
                    "Session pose pack "
                    f"{session_pose_dir(ctx.session_dir, ctx.args.pose_num)}: "
                    f"{', '.join(exported)}"
                )
    mark_succeeded(ctx.pose_dir, "pose", outputs={"mode": mode})
    return {"mode": mode}


def run_initial_capture(ctx: TrialContext) -> dict:
    mark_running(ctx.pose_dir, "initial-capture")
    print("Cover the subject with the blanket")
    _capture_with_wrist(
        ctx,
        ctx.initial_dir,
        stage="initial-capture",
        save_uncover_home=True,
        match_stem="wrist_xy_match",
        prompt="Stretch is parked and ready for wrist-down + initial capture?",
    )
    mark_succeeded(
        ctx.pose_dir,
        "initial-capture",
        outputs={"pcd": str(ctx.initial_dir / "blanket_pcd.pcd")},
    )
    return {}


def run_uncover_plan(ctx: TrialContext) -> dict:
    if not (ctx.closed_loop and ctx.is_recover):
        mark_succeeded(ctx.pose_dir, "uncover-plan", outputs={"skipped": True})
        return {"skipped": True}
    mark_running(ctx.pose_dir, "uncover-plan")
    wait_yes(ctx, "Ready to compute the Uncover action?")
    _ensure_snapshot(ctx, required=True, save_uncover_home=True, stage="uncover-plan")
    _run_ctx(ctx, uncover_plan_cmd(ctx), context="uncover CMA", stage="uncover-plan")
    _overlay_action(
        ctx,
        resolve_ceiling_rgb(ctx.pose_dir, "initial"),
        ctx.pose_dir / "uncover_action_overlay.png",
        [("uncover", resolve_uncover_file(ctx.pose_dir, "uncover_scaled_action.pkl"))],
        title=f"TL{ctx.target_limb_code} Uncover — pull grasp→release",
        stage="uncover-plan",
    )
    mark_succeeded(ctx.pose_dir, "uncover-plan")
    return {}


def run_uncover_exec(ctx: TrialContext) -> dict:
    if not (ctx.closed_loop and ctx.is_recover):
        mark_succeeded(ctx.pose_dir, "uncover-exec", outputs={"skipped": True})
        return {"skipped": True}
    mark_running(ctx.pose_dir, "uncover-exec")
    uncover_action = resolve_uncover_file(ctx.pose_dir, "uncover_scaled_action.pkl")
    if ctx.args.execute_uncover == "skip":
        print("Skipping Uncover execution.")
        mark_succeeded(ctx.pose_dir, "uncover-exec", outputs={"skipped": True})
        return {"skipped": True}
    if not _preflight(ctx, uncover_action, "uncover"):
        record_failure(
            ctx.pose_dir,
            "uncover-exec",
            fault_code="PREFLIGHT",
            error="Uncover preflight rejected",
        )
        preflight_must_stop(False, label="Uncover")
    if ctx.profile.send_bed_pull:
        _confirm_and_send(ctx, uncover_action, "uncover", stage="uncover-exec")
    else:
        print("Planning only: /bed_pull was NOT sent.")
        wait_yes(
            ctx,
            "Uncover already finished on the robot (this YES does not send /bed_pull)?",
        )
    mark_succeeded(ctx.pose_dir, "uncover-exec")
    return {}


def run_intermediate_capture(ctx: TrialContext) -> dict:
    mark_running(ctx.pose_dir, "intermediate-capture")
    if ctx.closed_loop and ctx.is_recover:
        prompt = "Robot is still after Uncover and ready for wrist-down + intermediate capture?"
        print("Capture the actual intermediate blanket after Uncover.")
    else:
        prompt = "Target limb is exposed; ready for wrist-down + intermediate capture?"
        print(
            f"Manually expose TL{ctx.target_limb_code} "
            f"({TL_LABELS.get(int(ctx.target_limb_code), 'custom target')})."
        )
    _capture_with_wrist(
        ctx,
        ctx.intermediate_dir,
        stage="intermediate-capture",
        save_uncover_home=False,
        match_stem="wrist_xy_match_recover",
        prompt=prompt,
    )
    ctx.recover_aligned_at_capture = True
    if ctx.closed_loop:
        _score_json(
            ctx,
            uncover_f1_cmd,
            ctx.pose_dir / "real_uncover_f1.json",
            stage="intermediate-capture",
            force=True,
        )
    mark_succeeded(ctx.pose_dir, "intermediate-capture")
    return {}


def run_recover_plan(ctx: TrialContext) -> dict:
    mark_running(ctx.pose_dir, "recover-plan")
    wait_yes(ctx, "Ready to compute the Recover action?")
    _ensure_snapshot(
        ctx,
        required=True,
        force=ctx.is_recover and not ctx.recover_aligned_at_capture,
        stage="recover-plan",
    )
    if ctx.compare_loop:
        pred_dir = layout_recover_pred_dir(ctx.pose_dir)
        sensor_dir = layout_recover_sensor_dir(ctx.pose_dir)
        snap_dir = layout_recover_snap_dir(ctx.pose_dir)
        for directory in (pred_dir, sensor_dir, snap_dir):
            directory.mkdir(parents=True, exist_ok=True)
        uncover_npz = resolve_uncover_file(ctx.pose_dir, "uncover_prediction.npz")
        intermediate_pcd = ctx.intermediate_dir / "blanket_pcd.pcd"
        _run_ctx(
            ctx,
            recover_plan_cmd(ctx, output_dir=pred_dir, prediction=uncover_npz, pcd=None),
            context="recover pred CMA",
            stage="recover-plan",
        )
        _run_ctx(
            ctx,
            recover_plan_cmd(ctx, output_dir=sensor_dir, prediction=None, pcd=intermediate_pcd),
            context="recover sensor CMA",
            stage="recover-plan",
        )
        _run_ctx(
            ctx,
            recover_plan_cmd(
                ctx, output_dir=snap_dir, prediction=uncover_npz, pcd=intermediate_pcd
            ),
            context="recover snap CMA",
            stage="recover-plan",
        )
        pred_action = _load_bed_action(pred_dir / "scaled_action.pkl")
        sensor_action = _load_bed_action(sensor_dir / "scaled_action.pkl")
        disagreement = _action_disagreement(pred_action, sensor_action)
        (ctx.pose_dir / "action_disagreement.json").write_text(
            json.dumps(disagreement, indent=2) + "\n"
        )
        write_pose_comparison(ctx.pose_dir)
        write_comparison_montage(ctx.pose_dir)
        ceiling_image = ctx.intermediate_dir / "covered_rgb_ceiling.png"
        if not ceiling_image.is_file():
            ceiling_image = ctx.pose_dir / "covered_rgb_ceiling.png"
        label_for = {
            ctx.args.a_is: "A",
            ("sensor" if ctx.args.a_is == "pred" else "pred"): "B",
        }
        overlay_actions = [
            (label_for["pred"], pred_dir / "scaled_action.pkl"),
            (label_for["sensor"], sensor_dir / "scaled_action.pkl"),
        ]
        overlay_actions.sort(key=lambda item: item[0])
        _overlay_action(
            ctx,
            ceiling_image,
            ctx.pose_dir / "recover_actions_overlay.png",
            overlay_actions,
            title=f"TL{ctx.target_limb_code} Recover — execute the assigned Action",
            stage="recover-plan",
        )
    else:
        _run_ctx(ctx, recover_plan_cmd(ctx), context="recover CMA", stage="recover-plan")
        recover_pkl = ctx.pose_dir / "scaled_action.pkl"
        if recover_pkl.is_file():
            try:
                recover_rgb = resolve_ceiling_rgb(ctx.pose_dir, "intermediate")
            except FileNotFoundError:
                recover_rgb = resolve_ceiling_rgb(ctx.pose_dir, "initial")
            _overlay_action(
                ctx,
                recover_rgb,
                ctx.pose_dir / "recover_action_overlay.png",
                [("recover", recover_pkl)],
                title=f"TL{ctx.target_limb_code} Recover — pull grasp→release",
                stage="recover-plan",
            )
    mark_succeeded(ctx.pose_dir, "recover-plan")
    return {}


def run_recover_exec(ctx: TrialContext) -> dict:
    mark_running(ctx.pose_dir, "recover-exec")
    if ctx.compare_loop:
        execute_dir = (
            layout_recover_pred_dir(ctx.pose_dir)
            if ctx.args.execute_source == "pred"
            else layout_recover_sensor_dir(ctx.pose_dir)
        )
        shutil.copy2(execute_dir / "scaled_action.pkl", ctx.pose_dir / "scaled_action.pkl")
    recover_pkl = ctx.pose_dir / "scaled_action.pkl"
    if not _preflight(ctx, recover_pkl, "recover"):
        record_failure(
            ctx.pose_dir,
            "recover-exec",
            fault_code="PREFLIGHT",
            error="Recover preflight rejected",
        )
        preflight_must_stop(False, label="Recover")
    if recover_grasp_was_snapped(ctx, "recover"):
        print("Recover grasp snapped onto cloth.")
    _confirm_and_send(ctx, recover_pkl, "recover", stage="recover-exec")
    _return_home(ctx, stage="recover-exec")
    mark_succeeded(ctx.pose_dir, "recover-exec")
    return {}


def run_final_capture(ctx: TrialContext) -> dict:
    if not (ctx.closed_loop and ctx.is_recover):
        mark_succeeded(ctx.pose_dir, "final-capture", outputs={"skipped": True})
        return {"skipped": True}
    mark_running(ctx.pose_dir, "final-capture")
    wait_yes(ctx, "Ready to capture the final blanket point cloud?")
    _capture_blanket(ctx, ctx.final_dir, stage="final-capture")
    mark_succeeded(ctx.pose_dir, "final-capture")
    return {}


def run_score(ctx: TrialContext) -> dict:
    mark_running(ctx.pose_dir, "score")
    execution_quality = prompt_execution_quality()
    failure_type = prompt_failure_type()
    uncover_f1 = _score_json(
        ctx, uncover_f1_cmd, ctx.pose_dir / "real_uncover_f1.json", stage="score"
    )
    recover_f1 = _score_json(
        ctx, recover_f1_cmd, ctx.pose_dir / "real_recover_f1.json", stage="score"
    )
    session_fields = (
        session_manifest_fields(ctx.session_dir) if ctx.session_dir is not None else None
    )
    manifest = build_trial_manifest(
        ctx,
        execution_quality=execution_quality,
        failure_type=failure_type,
        uncover_f1=uncover_f1,
        recover_f1=recover_f1,
        session_fields=session_fields,
    )
    dest = ctx.pose_dir / "trial_manifest.json"
    dest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {dest}")
    mark_succeeded(ctx.pose_dir, "score", outputs={"manifest": str(dest)})
    return manifest


STAGE_RUNNERS = {
    "pose": run_pose,
    "initial-capture": run_initial_capture,
    "uncover-plan": run_uncover_plan,
    "uncover-exec": run_uncover_exec,
    "intermediate-capture": run_intermediate_capture,
    "recover-plan": run_recover_plan,
    "recover-exec": run_recover_exec,
    "final-capture": run_final_capture,
    "score": run_score,
}


def _bind_session(args):
    session_dir = None
    session_filters: dict = {}
    paths = None
    try:
        session_dir = resolve_session_dir_once()
    except FileNotFoundError as exc:
        if not args.allow_unfrozen_session:
            raise SystemExit(f"READY FOR TRIAL: NO\n{exc}") from exc
        print(f"WARN no session: {exc}")
    if session_dir is not None:
        report = evaluate_session(session_dir)
        print(report.text())
        if not report.ready and not args.allow_unfrozen_session:
            raise SystemExit("READY FOR TRIAL: NO")
        cfg = load_pcd_config(session_dir)
        paths = session_paths(session_dir)
        if not argv_has(sys.argv[1:], "--camera-extrinsics") and paths.zed_extrinsics.is_file():
            args.camera_extrinsics = str(paths.zed_extrinsics)
        if not argv_has(sys.argv[1:], "--camera-roles") and paths.zed_extrinsics.is_file():
            try:
                cams = json.loads(paths.zed_extrinsics.read_text()).get("cameras") or {}
                order = ("ceiling", "side_left", "side_right")
                present = [role for role in order if cams.get(role)]
                if present:
                    args.camera_roles = ",".join(present)
            except (OSError, json.JSONDecodeError):
                pass
        if not argv_has(sys.argv[1:], "--pcd-exposure"):
            args.pcd_exposure = int(cfg["pcd_exposure"])
        if not argv_has(sys.argv[1:], "--pcd-gain"):
            args.pcd_gain = int(cfg["pcd_gain"])
        if not argv_has(sys.argv[1:], "--merge-mode"):
            args.merge_mode = str(cfg["merge_mode"])
        if not argv_has(sys.argv[1:], "--support-radius-3d", "--support-radius-xy"):
            args.support_radius = float(cfg["support_radius_3d"])
        try:
            session_filters = resolve_pcd_filters(session_dir)
        except ValueError as exc:
            if not args.allow_unfrozen_session:
                raise SystemExit(f"READY FOR TRIAL: NO\n{exc}") from exc
            print(f"WARN {exc}")
    return session_dir, session_filters, paths


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    profile = resolve_profile(args, sys.argv[1:])
    args.loop = profile.loop
    args.ros2_execute = profile.send_bed_pull
    args.live_geometry_policy = profile.live_geometry_policy
    if args.loop == "closed-compare":
        if args.execute_source is None or args.a_is is None:
            parser.error("--loop closed-compare requires --execute-source and --a-is")
    if profile.loop in ("closed", "closed-compare") and args.approach not in (
        "recover",
        "dyn",
    ):
        parser.error(f"--loop {profile.loop} currently requires --approach recover")

    try:
        assert_safe_start(
            start_here=args.start_here,
            start_stage=args.start_stage,
            tl_code=args.tl_code,
            sub_id=args.sub_id,
            resume=args.resume,
        )
    except ResumeError as exc:
        raise SystemExit(str(exc)) from exc

    session_dir, session_filters, sess_paths = _bind_session(args)
    t0 = time.time()
    study_root = (
        Path(os.environ["ROBE_STUDY_DATA"])
        if "ROBE_STUDY_DATA" in os.environ
        else (REAL_WORLD_DIR / "STUDY_DATA")
    )
    subject_dir = study_root / f"subject_{args.subject_id}"

    if args.resume:
        pose_dir = resolve_resume_dir(
            resume=args.resume, subject_dir=subject_dir
        )
        identity = load_identity(pose_dir)
        progress = load_progress(pose_dir)
        target_limb_code = str(identity["tl_code"])
        sub_id = int(identity["sub_id"])
        start_stage = next_pending_stage(progress) or "score"
        print(f"RESUME {pose_dir.name} from {start_stage}")
    else:
        target_limb_code = (
            args.tl_code
            if args.tl_code != "random"
            else str(random.choice(list(STUDY_TLS)))
        )
        sub_id = int(t0) if args.sub_id is None else args.sub_id
        pose_dir = subject_dir / f"pose_{args.pose_num}_TL{target_limb_code}_{sub_id}"
        pose_dir.mkdir(parents=True, exist_ok=True)
        start_stage = resolve_start_stage(
            loop=profile.loop,
            start_here=args.start_here,
            start_stage=args.start_stage,
        )
        hashes = {}
        if session_dir is not None:
            hashes = session_manifest_fields(session_dir)
        write_identity(
            pose_dir,
            identity_fields(
                args=args,
                profile=profile,
                target_limb_code=target_limb_code,
                sub_id=sub_id,
                pose_dir=pose_dir,
                session_hashes=hashes,
            ),
        )
        write_progress(pose_dir, empty_progress(profile.loop, start_stage=start_stage))
        progress = load_progress(pose_dir)

    pack_source, pack_copied = apply_pose_pack(
        pose_dir,
        subject_dir,
        args.pose_num,
        prefer=args.reuse_human_pose,
        session_dir=session_dir,
        copy_sibling=not (args.no_reuse_human_pose or args.recapture_pose),
    )

    print("===============================================================================")
    print(
        f"          SUBJECT: {args.subject_id},  POSE: {args.pose_num}, "
        f"TL CODE: {target_limb_code},  SUB_ID: {sub_id}"
    )
    print("===============================================================================")
    print(f"profile={profile.name} loop={profile.loop} send_bed_pull={profile.send_bed_pull}")
    print(f"Blanket mask backend: {args.mask_backend}")
    print(
        f"Target TL{target_limb_code}: "
        f"{TL_LABELS.get(int(target_limb_code), 'custom target')}"
        + (" (random)" if str(args.tl_code) == "random" and not args.resume else "")
    )
    if profile.debug_overrides:
        print(f"debug overrides: {profile.debug_overrides}")
    if pack_source is not None:
        print(f"Reusing pose files from: {pack_source}")

    ctx = TrialContext(
        args=args,
        profile=profile,
        pose_dir=pose_dir,
        subject_dir=subject_dir,
        study_root=study_root,
        target_limb_code=str(target_limb_code),
        sub_id=int(sub_id),
        robe_py=robe_python(),
        zed_py=zed_python(),
        code_dir=CODE_DIR,
        real_world_dir=REAL_WORLD_DIR,
        session_dir=session_dir,
        session_filters=session_filters,
        session_paths=sess_paths,
        pack_source=pack_source,
        pack_copied=list(pack_copied or []),
    )

    for name in progress["stages"]:
        stage = name["name"]
        if not should_run_stage(progress, stage):
            continue
        runner = STAGE_RUNNERS[stage]
        try:
            runner(ctx)
        except OperatorQuit:
            raise
        except SystemExit:
            raise
        progress = load_progress(pose_dir)

    print((time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
