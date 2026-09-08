"""Scoring and trial-manifest builders. F1 scripts stay unchanged."""

from __future__ import annotations

import json
from pathlib import Path

from trial_capture import body_reuse_flags, cma_snapshot_dest
from trial_context import TrialContext
from trial_layout import (
    recover_pred_dir,
    recover_sensor_dir,
    recover_snap_dir,
    resolve_initial_pcd,
)


def recover_model_dir(ctx: TrialContext) -> Path | None:
    if ctx.args.model_dir is not None:
        return Path(ctx.args.model_dir)
    meta = ctx.pose_dir / "recover_runtime_metadata.json"
    if meta.is_file():
        raw = json.loads(meta.read_text()).get("model_dir")
        if raw:
            return Path(raw)
    return None


def uncover_f1_cmd(ctx: TrialContext) -> list[str] | None:
    if ctx.args.skip_real_f1:
        return None
    initial_pcd = resolve_initial_pcd(ctx.pose_dir)
    intermediate_pcd = ctx.intermediate_dir / "blanket_pcd.pcd"
    if not initial_pcd.is_file() or not intermediate_pcd.is_file():
        return None
    score_json = ctx.pose_dir / "real_uncover_f1.json"
    cmd = [
        ctx.robe_py,
        str(ctx.code_dir / "score_real_uncover_f1.py"),
        "--subject-dir",
        str(ctx.subject_dir),
        "--pose-dir",
        str(ctx.pose_dir),
        "--tl-code",
        ctx.target_limb_code,
        "--initial-pcd",
        str(initial_pcd),
        "--intermediate-pcd",
        str(intermediate_pcd),
        "--output-json",
        str(score_json),
        "--max-points",
        str(ctx.args.max_points),
        "--manikin",
        ctx.args.manikin,
    ]
    model_dir = recover_model_dir(ctx)
    if model_dir is not None:
        cmd.extend(["--model-dir", str(model_dir)])
    cmd.extend(body_reuse_flags(ctx))
    return cmd


def recover_f1_cmd(ctx: TrialContext) -> list[str] | None:
    if ctx.args.skip_real_f1:
        return None
    model_dir = recover_model_dir(ctx)
    if model_dir is None:
        return None
    score_json = ctx.pose_dir / "real_recover_f1.json"
    cmd = [
        ctx.robe_py,
        str(ctx.code_dir / "score_real_recover_f1.py"),
        "--subject-dir",
        str(ctx.subject_dir),
        "--pose-dir",
        str(ctx.pose_dir),
        "--tl-code",
        ctx.target_limb_code,
        "--model-dir",
        str(model_dir),
        "--initial-pcd",
        str(resolve_initial_pcd(ctx.pose_dir)),
        "--intermediate-pcd",
        str(ctx.intermediate_dir / "blanket_pcd.pcd"),
        "--final-pcd",
        str(ctx.final_dir / "blanket_pcd.pcd"),
        "--output-json",
        str(score_json),
        "--max-points",
        str(ctx.args.max_points),
        "--manikin",
        ctx.args.manikin,
    ]
    cmd.extend(body_reuse_flags(ctx))
    return cmd


def planning_manifest_fields(ctx: TrialContext) -> dict:
    dest = cma_snapshot_dest(ctx)
    snap = json.loads(dest.read_text()) if dest.is_file() else {}
    prepared = snap.get(
        "planning_wrist_down_prepared",
        None if ctx.args.no_stretch_reach else (not ctx.args.skip_prepare_wrist_down),
    )
    debug_override = bool(
        snap.get("planning_debug_override", ctx.args.skip_prepare_wrist_down)
    )
    standard = snap.get(
        "standard_experiment_config",
        bool(prepared) and not debug_override,
    )
    return {
        "planning_wrist_down_prepared": prepared,
        "debug_override": debug_override,
        "planning_debug_override": debug_override,
        "planning_safe_extension_m": snap.get("planning_safe_extension_m"),
        "standard_experiment_config": standard,
        "reuse_subject_body": bool(ctx.args.reuse_subject_body),
    }


def build_trial_manifest(
    ctx: TrialContext,
    *,
    execution_quality: str,
    failure_type: str | None,
    uncover_f1: dict | None,
    recover_f1: dict | None,
    session_fields: dict | None = None,
) -> dict:
    exclude = execution_quality == "bad"
    recover_model = recover_model_dir(ctx)
    manifest = {
        "trial_id": ctx.args.trial_id or ctx.pose_dir.name,
        "subject_id": ctx.args.subject_id,
        "pose_num": ctx.args.pose_num,
        "sub_id": ctx.sub_id,
        "tl_code": int(ctx.target_limb_code),
        "tl_code_mode": ctx.profile.tl_code_mode,
        "trial_wrist_match": bool(ctx.args.trial_wrist_match),
        "detect_top_markers": bool(ctx.args.detect_top_markers),
        "marker_policy": ctx.profile.marker_policy,
        "registration_policy": ctx.profile.registration_policy,
        "remeasure_body": bool(ctx.args.remeasure_body),
        "loop": ctx.profile.loop,
        "profile": ctx.profile.name,
        "debug_overrides": dict(ctx.profile.debug_overrides),
        "uncover_model_dir": (
            None if ctx.args.uncover_model_dir is None else str(ctx.args.uncover_model_dir)
        ),
        "recover_model_dir": None if recover_model is None else str(recover_model),
        "camera_roles": ctx.args.camera_roles,
        "mask_backend": ctx.args.mask_backend,
        "max_points": ctx.args.max_points,
        "execution_quality": execution_quality,
        "failure_type": failure_type,
        "exclude_from_source_comparison": exclude,
        "real_uncover_f1": None if uncover_f1 is None else uncover_f1.get("real_uncover_f1"),
        "real_uncover_f1_detail": uncover_f1,
        "real_recover_f1": None if recover_f1 is None else recover_f1.get("real_recover_f1"),
        "real_recover_f1_detail": recover_f1,
        **planning_manifest_fields(ctx),
        "artifacts": {
            "pose_dir": str(ctx.pose_dir),
            "intermediate_dir": str(ctx.intermediate_dir),
            "final_dir": str(ctx.final_dir),
            "identity": str(ctx.pose_dir / "trial_identity.json"),
            "progress": str(ctx.pose_dir / "trial_progress.json"),
            "uncover_overlay": str(ctx.pose_dir / "uncover_action_overlay.png"),
            "recover_overlay": str(
                ctx.pose_dir / "recover_actions_overlay.png"
                if ctx.compare_loop
                else ctx.pose_dir / "recover_action_overlay.png"
            ),
        },
    }
    if ctx.compare_loop:
        disagreement_path = ctx.pose_dir / "action_disagreement.json"
        disagreement = (
            json.loads(disagreement_path.read_text())
            if disagreement_path.is_file()
            else {}
        )

        def _meta(directory: Path) -> dict:
            path = directory / "recover_runtime_metadata.json"
            return json.loads(path.read_text()) if path.is_file() else {}

        pred_dir = recover_pred_dir(ctx.pose_dir)
        sensor_dir = recover_sensor_dir(ctx.pose_dir)
        snap_dir = recover_snap_dir(ctx.pose_dir)
        pred_meta = _meta(pred_dir)
        sensor_meta = _meta(sensor_dir)
        snap_meta = _meta(snap_dir)
        execute_label = "A" if ctx.args.execute_source == ctx.args.a_is else "B"
        manifest.update(
            {
                "a_is": ctx.args.a_is,
                "b_is": "sensor" if ctx.args.a_is == "pred" else "pred",
                "execute_source": ctx.args.execute_source,
                "execute_label": execute_label,
                "action_disagreement": disagreement,
                "recover_pred": {
                    "output_dir": str(pred_dir),
                    "reward": pred_meta.get("reward"),
                    "graph_stats": pred_meta.get("graph_stats"),
                    "scaled_action": pred_meta.get("scaled_action"),
                },
                "recover_sensor": {
                    "output_dir": str(sensor_dir),
                    "reward": sensor_meta.get("reward"),
                    "graph_stats": sensor_meta.get("graph_stats"),
                    "scaled_action": sensor_meta.get("scaled_action"),
                },
                "recover_snap": {
                    "output_dir": str(snap_dir),
                    "reward": snap_meta.get("reward"),
                    "graph_stats": snap_meta.get("graph_stats"),
                    "scaled_action": snap_meta.get("scaled_action"),
                    "sensor_snap": snap_meta.get("sensor_snap"),
                },
            }
        )
    if session_fields:
        manifest.update(session_fields)
    return manifest
