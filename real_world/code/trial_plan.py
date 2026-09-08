"""Planning-stage command builders. CMA / overlay internals stay unchanged."""

from __future__ import annotations

from pathlib import Path

from stretch_limits import CMA_BOUNDS_NAME, load_reach_snapshot, write_restricted_cma_bounds
from trial_capture import body_reuse_flags, cma_snapshot_dest
from trial_context import TrialContext
from trial_layout import resolve_uncover_file


def cma_reach_flags(ctx: TrialContext) -> list[str]:
    if ctx.args.no_stretch_reach:
        return []
    dest = cma_snapshot_dest(ctx)
    if not dest.is_file():
        return []
    return [
        "--stretch-reach-snapshot",
        str(dest),
        "--arm-max-m",
        str(ctx.args.arm_max_m),
    ]


def uncover_plan_cmd(ctx: TrialContext) -> list[str]:
    cmd = [
        ctx.robe_py,
        str(ctx.code_dir / "get_action.py"),
        "--subject-dir",
        str(ctx.subject_dir),
        "--pose-dir",
        str(ctx.pose_dir),
        "--tl-code",
        ctx.target_limb_code,
        "--approach",
        "uncover",
        "--max-fevals",
        str(ctx.uncover_fevals),
        "--device",
        ctx.args.device,
        "--max-points",
        str(ctx.args.max_points),
        "--entropy-weight",
        str(ctx.args.entropy_weight),
        "--entropy-grid-size",
        str(ctx.args.entropy_grid_size),
        "--bounds",
        ctx.args.cma_bounds,
        "--manikin",
        ctx.args.manikin,
        "--output-dir",
        str(ctx.uncover_dir),
    ]
    cmd.extend(cma_reach_flags(ctx))
    cmd.extend(body_reuse_flags(ctx))
    if ctx.args.uncover_model_dir is not None:
        cmd.extend(["--model-dir", str(ctx.args.uncover_model_dir)])
    if ctx.args.uncover_checkpoint_number is not None:
        cmd.extend(["--checkpoint-number", str(ctx.args.uncover_checkpoint_number)])
    return cmd


def recover_plan_cmd(
    ctx: TrialContext,
    *,
    output_dir: Path | None = None,
    prediction: Path | None = None,
    pcd: Path | None = None,
) -> list[str]:
    dest = output_dir or ctx.pose_dir
    cmd = [
        ctx.robe_py,
        str(ctx.code_dir / "get_action.py"),
        "--subject-dir",
        str(ctx.subject_dir),
        "--pose-dir",
        str(ctx.pose_dir),
        "--tl-code",
        ctx.target_limb_code,
        "--approach",
        ctx.args.approach,
        "--max-fevals",
        str(ctx.args.max_fevals),
        "--device",
        ctx.args.device,
        "--max-points",
        str(ctx.args.max_points),
        "--manikin",
        ctx.args.manikin,
    ]
    if dest != ctx.pose_dir:
        cmd.extend(["--output-dir", str(dest)])
    cmd.extend(cma_reach_flags(ctx))
    cmd.extend(body_reuse_flags(ctx))
    if ctx.is_recover:
        intermediate = pcd or (ctx.intermediate_dir / "blanket_pcd.pcd")
        if prediction is None and not intermediate.is_file():
            raise FileNotFoundError(
                f"Recover needs the recaptured intermediate PCD ({intermediate}). "
                "Resume at intermediate-capture."
            )
        if prediction is not None:
            cmd.extend(["--intermediate-prediction", str(prediction)])
            cmd.append("--skip-intermediate-validation")
        if pcd is not None or prediction is None:
            cmd.extend(["--intermediate-pcd", str(intermediate)])
        cmd.extend(["--warm-start", "line"])
        uncover_meta = resolve_uncover_file(
            ctx.pose_dir, "uncover_runtime_metadata.json", required=False
        )
        if ctx.closed_loop and uncover_meta.is_file():
            cmd.extend(["--uncover-policy-action", str(uncover_meta)])
    if ctx.args.model_dir:
        cmd.extend(["--model-dir", str(ctx.args.model_dir)])
    if ctx.args.checkpoint_number is not None:
        cmd.extend(["--checkpoint-number", str(ctx.args.checkpoint_number)])
    if ctx.args.residual_model_dir is not None:
        cmd.extend(["--residual-model-dir", str(ctx.args.residual_model_dir)])
    if ctx.args.residual_checkpoint_number is not None:
        cmd.extend(
            ["--residual-checkpoint-number", str(ctx.args.residual_checkpoint_number)]
        )
    return cmd


def overlay_cmd(
    ctx: TrialContext,
    image: Path,
    output: Path,
    actions: list[tuple[str, Path]],
    title: str,
    *,
    draw_pcd: bool = False,
) -> list[str]:
    command = [
        ctx.robe_py,
        str(ctx.code_dir / "overlay_action_on_ceiling.py"),
        "--image",
        str(image),
        "--sim-origin",
        str(ctx.pose_dir / "sim_origin_data.pkl"),
        "--output",
        str(output),
        "--title",
        title,
    ]
    for candidate in (
        image.parent / "blanket_pcd.pcd",
        ctx.initial_dir / "blanket_pcd.pcd",
        ctx.intermediate_dir / "blanket_pcd.pcd",
    ):
        if candidate.is_file():
            command.extend(["--blanket-pcd", str(candidate)])
            break
    for label, path in actions:
        command.extend(["--action", label, str(path)])
    if draw_pcd:
        command.append("--draw-pcd")
    return command


def publish_cma_box(ctx: TrialContext, dest: Path) -> dict | None:
    from canonical_bed import maybe_load_canonical_frame

    snap = load_reach_snapshot(dest)
    frame = maybe_load_canonical_frame(ctx.pose_dir)
    if frame is None:
        print("WARN: no canonical_bed_frame.json; cannot write stretch_cma_bounds.json")
        return None
    box_path = dest.parent / CMA_BOUNDS_NAME
    return write_restricted_cma_bounds(
        box_path,
        snapshot=snap,
        frame=frame,
        arm_max_m=float(ctx.args.arm_max_m),
        bounds_mode=str(ctx.args.cma_bounds),
        mirror_x=True,
    )
