#!/usr/bin/env python3
"""Plan a real-world Uncover action with CMA over the Uncover GNN.

Cost matches remote W200 Uncover evals (``-reward + W * (1 - XY entropy)``)
plus ``run_robe_sim`` CMA options.  Search box defaults to sim
``[-1,1]^4``.  Pass ``--bounds asymmetric`` for the legacy real-world
box.  ``run_trial`` restricts that box to the execution wrist-down
X-strip from ``stretch_limits.py`` when a parking snapshot is present.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import cma
import numpy as np

REAL_WORLD_CODE = Path(__file__).resolve().parents[1]
if str(REAL_WORLD_CODE) not in sys.path:
    sys.path.insert(0, str(REAL_WORLD_CODE))

from trial_layout import (  # noqa: E402
    ensure_trial_body_info,
    resolve_body_info,
    resolve_initial_pcd,
    uncover_write_dir,
)
from canonical_bed import canonicalize_body_xy, maybe_load_canonical_frame  # noqa: E402
from stretch_limits import (  # noqa: E402
    PLANNER_ARM_MAX_M,
    assert_selected_action_reachable,
    clip_action_to_bounds,
    cma_policy_bounds,
    load_cma_reach,
    restrict_cma_bounds_to_reach,
)
from recover_runtime import (  # noqa: E402
    ACTION_SCALE,
    DEFAULT_UNCOVER_MODEL_DIR,
    DEFAULT_RECOVER_POINTS,
    DRAPING_EDGE_X,
    ModelFrameAdapter,
    RecoverRuntimeGraph,
    load_uncover_model,
    read_pcd_points,
    scale_action,
)


def _parse_limits(value: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError("limits must be MIN,MAX with MIN < MAX")
    return parts[0], parts[1]


def uncover_x0(target_limb_code: int) -> np.ndarray:
    from cma_gnn_util import set_x0_for_cmaes

    return np.asarray(set_x0_for_cmaes(int(target_limb_code)), dtype=np.float64)


def asymmetric_cma_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Original real-world Uncover CMA box (legacy ``run_cma_over_dyn_model``)."""

    return cma_policy_bounds("asymmetric")


def grasp_threshold_m(voxel_size: float) -> float:
    """Match bu_gnn_util: 0.028 dense mesh, 0.05 when voxel-subsampled."""

    if np.isfinite(voxel_size) and float(voxel_size) > 0:
        return max(0.05, float(voxel_size))
    return 0.028


def normalized_xy_occupancy_entropy(
    cloth_final_2d: np.ndarray,
    cloth_initial_2d: np.ndarray,
    grid_size: float = 0.05,
) -> float:
    """Remote Uncover CMA occupancy-entropy term."""

    cloth_final_2d = np.asarray(cloth_final_2d, dtype=np.float32)
    cloth_initial_2d = np.asarray(cloth_initial_2d, dtype=np.float32)
    xy_stack = np.vstack([cloth_initial_2d[:, :2], cloth_final_2d[:, :2]])
    xy_min = np.min(xy_stack, axis=0) - float(grid_size)
    xy_max = np.max(xy_stack, axis=0) + float(grid_size)
    shape = np.maximum(
        np.ceil((xy_max - xy_min) / float(grid_size)).astype(np.int64) + 1,
        1,
    )
    cell_idx = np.floor(
        (cloth_final_2d[:, :2] - xy_min) / float(grid_size)
    ).astype(np.int64)
    valid = np.all((cell_idx >= 0) & (cell_idx < shape), axis=1)
    flat_idx = cell_idx[valid, 0] * shape[1] + cell_idx[valid, 1]
    counts = np.bincount(flat_idx, minlength=int(shape[0] * shape[1]))
    counts = counts[counts > 0].astype(np.float64)
    total_cells = int(shape[0] * shape[1])
    if counts.size == 0 or total_cells <= 1:
        return 0.0
    probs = counts / np.sum(counts)
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / math.log(float(total_cells))


def _load_body_points(
    pose_dir: Path,
    subject_dir: Path,
    manikin: bool,
    target_limb_code: int,
    allow_subject: bool = False,
) -> np.ndarray:
    from pickle_compat import load_pickle

    pose_path = (subject_dir if manikin else pose_dir) / "human_pose.pkl"
    body_path = (
        subject_dir / "body_info.pkl"
        if manikin
        else ensure_trial_body_info(
            pose_dir, subject_dir, allow_subject=allow_subject
        )
        or resolve_body_info(pose_dir, subject_dir, allow_subject=allow_subject)
    )
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing pose file: {pose_path}")
    if not body_path.is_file():
        raise FileNotFoundError(
            f"Missing {body_path}; run get_body_info_from_img.py before planning."
        )
    human_pose = np.asarray(load_pickle(pose_path), dtype=np.float64)
    body_info = load_pickle(body_path)
    required_body_parts = (
        "head",
        "upperchest",
        "waist",
        "upperarm",
        "forearm",
        "hand",
        "thigh",
        "shin",
        "foot",
    )
    invalid_parts = [
        name
        for name in required_body_parts
        if name not in body_info
        or len(body_info[name]) < 2
        or float(body_info[name][1]) <= 0
    ]
    if invalid_parts:
        raise ValueError(
            f"Invalid body_info radii for {invalid_parts}; "
            "measure body_info before planning."
        )
    from assistive_gym.envs.bu_gnn_util import get_body_points_from_obs

    body_points = get_body_points_from_obs(
        human_pose=human_pose,
        target_limb_code=target_limb_code,
        body_info=body_info,
    )
    return np.asarray(body_points, dtype=np.float64)


def _evaluate(
    normalized_action: np.ndarray,
    graph: RecoverRuntimeGraph,
    model,
    device: str,
    cloth_initial_xy: np.ndarray,
    body_points: np.ndarray,
    use_displacement: bool,
    entropy_weight: float,
    entropy_grid_size: float,
    grasp_thres: float,
) -> dict[str, object]:
    """Mirror ``run_robe_sim.cost_function``, then add optional entropy."""

    from assistive_gym.envs.bu_gnn_util import (
        check_grasp_on_cloth,
        get_uncovering_reward,
    )

    action_world = scale_action(normalized_action)
    _, is_on_cloth = check_grasp_on_cloth(
        action_world,
        cloth_initial_xy,
        clipping_thres=grasp_thres,
    )
    is_on_cloth = bool(is_on_cloth)

    if is_on_cloth:
        prediction = graph.predict(
            model,
            normalized_action,
            device=device,
            use_displacement=use_displacement,
        )
    else:
        # Same as sim: off-cloth keeps the initial state; cost is still -reward.
        prediction = cloth_initial_xy.copy()

    reward, covered_status = get_uncovering_reward(
        action_world,
        body_points,
        cloth_initial_xy,
        prediction,
    )
    xy_entropy = normalized_xy_occupancy_entropy(
        prediction,
        cloth_initial_xy,
        entropy_grid_size,
    )
    overlap_penalty = 1.0 - float(xy_entropy)
    if is_on_cloth:
        # run_robe_sim.get_cost: cost = -reward, plus optional W200 entropy.
        cost = -float(reward) + float(entropy_weight) * overlap_penalty
    else:
        # Keep search pressure toward on-cloth grasps; selection still uses
        # feasible-only best (joint_opt style).
        cost = 500.0 + float(entropy_weight) * overlap_penalty

    return {
        "cost": float(cost),
        "reward": float(reward),
        "xy_entropy": float(xy_entropy),
        "overlap_penalty": float(overlap_penalty),
        "prediction": np.asarray(prediction, dtype=np.float64),
        "covered_status": np.asarray(covered_status),
        "is_on_cloth": is_on_cloth,
        "action_world": np.asarray(action_world, dtype=np.float64),
    }


def _draw_uncover_prediction(
    output_path: Path,
    cloth_initial_xy: np.ndarray,
    predicted_xy: np.ndarray,
    body_points: np.ndarray,
    action_model: np.ndarray,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, points, name in (
        (axes[0], cloth_initial_xy, "Initial"),
        (axes[1], predicted_xy, "Uncover GNN prediction"),
    ):
        ax.scatter(points[:, 0], points[:, 1], s=6, c="#63BEF2", label="blanket")
        ax.scatter(
            body_points[:, 0],
            body_points[:, 1],
            s=18,
            c="#FFBA47",
            label="body",
        )
        ax.scatter(action_model[0], action_model[1], c="k", s=40)
        ax.annotate(
            "",
            xy=(action_model[2], action_model[3]),
            xytext=(action_model[0], action_model[1]),
            arrowprops={"arrowstyle": "->", "color": "k", "lw": 2},
        )
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for Uncover artifacts. Defaults to <pose-dir>/uncover.",
    )
    parser.add_argument("--tl-code", type=int, required=True)
    parser.add_argument("--manikin", type=int, default=0)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_UNCOVER_MODEL_DIR)
    parser.add_argument("--checkpoint-number", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-fevals", type=int, default=150)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--sigma0", type=float, default=0.2)
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=95.0,
        help="Early-stop when best uncover reward reaches this (run_robe_sim)",
    )
    parser.add_argument(
        "--feasible-only-best",
        action="store_true",
        default=True,
        help="Only keep the best on-cloth candidate (joint_opt feasible tracker)",
    )
    parser.add_argument(
        "--allow-off-cloth-best",
        action="store_true",
        help="Disable feasible-only selection (legacy; may return no-op off-cloth)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=DEFAULT_RECOVER_POINTS)
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=200.0,
        help="Overlap penalty weight. 200 matches remote W200 Uncover CMA; "
        "0 reproduces plain run_robe_sim cost=-reward.",
    )
    parser.add_argument("--entropy-grid-size", type=float, default=0.05)
    parser.add_argument(
        "--bounds",
        choices=("symmetric", "asymmetric"),
        default="symmetric",
        help="symmetric: sim [-1,1]^4 (default). asymmetric: legacy real-world "
        "box [0,-0.5,0,-1]..[1,1,1,1], which clips half the bed in x.",
    )
    parser.add_argument(
        "--action-feature-mode",
        choices=("normalized", "scaled"),
        default="normalized",
        help="Node-feature action convention. 'normalized' matches the sim CMA "
        "runtime code/build_runtime_graph.py used by run_robe_sim; 'scaled' "
        "matches code/bm_dataset.py.",
    )
    parser.add_argument("--x-limits", type=_parse_limits, default=(-0.55, 0.55))
    parser.add_argument("--y-limits", type=_parse_limits, default=(-1.10, 1.10))
    parser.add_argument("--no-draping-rotation", action="store_true")
    parser.add_argument(
        "--draping-edge-x",
        type=float,
        default=DRAPING_EDGE_X,
        help="Only drape points beyond this |x| (bed edge). Pass 0.0 for the "
        "literal bm_dataset rule, which folds mid-bed points into a strip.",
    )
    parser.add_argument(
        "--mirror-x",
        dest="mirror_x",
        action="store_true",
        default=True,
        help="Mirror bed x so the real scene matches the simulation handedness "
        "(sim TL2 target sits at x>0). Enabled by default.",
    )
    parser.add_argument(
        "--no-mirror-x",
        dest="mirror_x",
        action="store_false",
        help="Keep the raw bed x orientation.",
    )
    parser.add_argument(
        "--stretch-reach-snapshot",
        type=Path,
        default=None,
        help="Parking snapshot JSON from snapshot_stretch_reach.py. "
        "On-cloth candidates that need joint_arm outside the workspace "
        "are treated as infeasible.",
    )
    parser.add_argument(
        "--arm-max-m",
        type=float,
        default=PLANNER_ARM_MAX_M,
        help="CMA arm-extension cap (m). Default matches stretch_limits "
        "(2 cm inside hardware 0.52) so pull waypoints pass preflight.",
    )
    parser.add_argument(
        "--reuse-subject-body",
        action="store_true",
        help="Copy subject_dir/body_info.pkl into this trial. Default is "
        "trial then this exp's poses/pose_n only.",
    )
    args = parser.parse_args()
    if args.max_points < 0:
        parser.error("--max-points must be >= 0 (0 disables the warn threshold)")
    if args.entropy_grid_size <= 0:
        parser.error("--entropy-grid-size must be positive")

    args.subject_dir = args.subject_dir.resolve()
    args.pose_dir = args.pose_dir.resolve()
    output_dir = (
        uncover_write_dir(args.pose_dir)
        if args.output_dir is None
        else args.output_dir.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_pcd = resolve_initial_pcd(args.pose_dir)

    model, config = load_uncover_model(
        model_dir=args.model_dir,
        checkpoint_number=args.checkpoint_number,
        device=args.device,
    )
    adapter = ModelFrameAdapter(
        x_limits=args.x_limits,
        y_limits=args.y_limits,
        mirror_x=bool(args.mirror_x),
    )
    apply_draping = config.rot_draping and not args.no_draping_rotation
    _, points_model = read_pcd_points(
        initial_pcd,
        adapter,
        max_points=args.max_points,
        voxel_size=config.voxel_size,
        apply_draping=apply_draping,
        draping_edge_x=args.draping_edge_x,
    )
    cloth_initial_xy = points_model[:, :2].copy()
    graph = RecoverRuntimeGraph(
        points_model,
        config,
        apply_draping=False,
        action_feature_mode=args.action_feature_mode,
    )
    body_points = adapter.mirror_body_points(
        canonicalize_body_xy(
            _load_body_points(
                args.pose_dir,
                args.subject_dir,
                bool(args.manikin),
                args.tl_code,
                allow_subject=bool(args.reuse_subject_body),
            ),
            maybe_load_canonical_frame(args.pose_dir),
        )
    )
    grasp_thres = grasp_threshold_m(config.voxel_size)
    feasible_only = bool(args.feasible_only_best) and not bool(
        args.allow_off_cloth_best
    )
    use_asymmetric = args.bounds == "asymmetric"
    bounds_min, bounds_max = cma_policy_bounds(args.bounds)
    reach_ctx = load_cma_reach(
        pose_dir=args.pose_dir,
        snapshot_path_or_none=args.stretch_reach_snapshot,
        arm_max_m=float(args.arm_max_m),
    )
    if reach_ctx.enabled:
        bounds_min, bounds_max = restrict_cma_bounds_to_reach(
            bounds_min,
            bounds_max,
            snapshot=reach_ctx.snapshot,
            frame=reach_ctx.frame,
            arm_max_m=reach_ctx.arm_max_m,
            action_scale=ACTION_SCALE,
            mirror_x=bool(args.mirror_x),
        )
        print(
            f"CMA stretch reach: snapshot={reach_ctx.snapshot_path} "
            f"arm_max={reach_ctx.arm_max_m:.3f} m "
            f"(hardware 0.52; planning {reach_ctx.arm_max_m:.3f}); "
            f"box = execution wrist-down X strip "
            f"[{bounds_min.round(3).tolist()}].."
            f"[{bounds_max.round(3).tolist()}] "
            "(CMA does not filter waypoints)"
        )
    print(
        f"Uncover CMA: "
        f"popsize={args.popsize} sigma0={args.sigma0} "
        f"maxfevals={args.max_fevals} CMA_stds=1 "
        f"bounds={'asymmetric_rw' if use_asymmetric else 'symmetric_sim'} "
        f"[{bounds_min.tolist()}]..[{bounds_max.tolist()}] "
        f"action_features={args.action_feature_mode} "
        f"mirror_x={bool(args.mirror_x)} "
        f"reward_threshold={args.reward_threshold:g} "
        f"grasp_thres={grasp_thres:.3f}m "
        f"entropy_weight={args.entropy_weight:g} "
        f"feasible_only_best={feasible_only}"
    )

    rng = np.random.default_rng(args.seed)
    x0 = clip_action_to_bounds(uncover_x0(args.tl_code), bounds_min, bounds_max)
    # CMA options match run_robe_sim; bounds default to legacy real-world box.
    opts = cma.CMAOptions(
        {
            "verb_disp": 1,
            "popsize": args.popsize,
            "maxfevals": args.max_fevals,
            "tolfun": 1e-11,
            "tolflatfitness": 20,
            "tolfunhist": 1e-20,
            "seed": int(rng.integers(0, 2**31 - 1)),
        }
    )
    opts.set("bounds", [bounds_min.tolist(), bounds_max.tolist()])
    opts.set("CMA_stds", bounds_max)

    es = cma.CMAEvolutionStrategy(x0.tolist(), args.sigma0, opts)
    best = None
    best_cost = None
    feasible_best = None
    feasible_best_cost = None
    evaluations = 0
    iterations = 0
    num_on_cloth = 0
    while not es.stop():
        iterations += 1
        actions = es.ask()
        costs = []
        results = []
        for action in actions:
            action_n = np.asarray(action, dtype=np.float64)
            result = _evaluate(
                action_n,
                graph,
                model,
                args.device,
                cloth_initial_xy,
                body_points,
                config.use_displacement,
                args.entropy_weight,
                args.entropy_grid_size,
                grasp_thres,
            )
            result["is_reachable"] = True
            result["reach_reason"] = "cma_x_strip"
            costs.append(result["cost"])
            results.append(result)
            evaluations += 1
            if result["is_on_cloth"]:
                num_on_cloth += 1
                candidate = dict(result)
                candidate["normalized_action"] = action_n
                candidate["evaluations"] = evaluations
                candidate["iterations"] = iterations
                if (
                    feasible_best_cost is None
                    or float(result["cost"]) < feasible_best_cost
                ):
                    feasible_best_cost = float(result["cost"])
                    feasible_best = candidate
        es.tell(actions, costs)

        gen_best_idx = int(np.argmin(costs))
        gen_best_cost = float(costs[gen_best_idx])
        if best_cost is None or gen_best_cost < best_cost:
            best_cost = gen_best_cost
            candidate = dict(results[gen_best_idx])
            candidate["normalized_action"] = np.asarray(
                actions[gen_best_idx], dtype=np.float64
            )
            candidate["evaluations"] = evaluations
            candidate["iterations"] = iterations
            best = candidate

        selected = feasible_best if feasible_only else best
        if selected is not None and selected["reward"] >= args.reward_threshold:
            print(
                f"Early stop: uncover reward {selected['reward']:.2f} "
                f">= {args.reward_threshold:g} after {evaluations} fevals"
            )
            break

    if feasible_only:
        if feasible_best is None:
            raise RuntimeError(
                "Uncover CMA found no on-cloth (feasible) grasp. "
                f"Tried {evaluations} fevals with grasp_thres={grasp_thres:.3f}m; "
                "check blanket PCD coverage, human pose, and parking."
            )
        best = feasible_best
        print(
            f"Selected feasible-only best: reward={best['reward']:.3f} "
            f"cost={best['cost']:.3f} "
            f"on_cloth_evals={num_on_cloth}/{evaluations}"
        )
    elif best is None:
        raise RuntimeError("Uncover CMA-ES produced no candidate action")
    elif not best["is_on_cloth"]:
        print(
            "WARN: selected best action is off-cloth "
            f"(on_cloth_evals={num_on_cloth}/{evaluations})"
        )

    action_model = np.asarray(best["action_world"], dtype=np.float64)
    action_bed = adapter.model_action_to_bed(action_model)
    reach = assert_selected_action_reachable(action_bed, reach_ctx)
    if reach_ctx.enabled:
        print(
            "Selected-action workspace check: "
            f"max_wrist={reach.max_wrist_extension} "
            f"planning_max={reach_ctx.arm_max_m:.3f} {reach.reason}"
        )
    prediction = np.asarray(best["prediction"], dtype=np.float64)
    pred_delta = np.linalg.norm(prediction - cloth_initial_xy, axis=1)
    prediction_path = output_dir / "uncover_prediction.npz"
    predicted_xyz = np.column_stack(
        [
            prediction.astype(np.float32),
            points_model[:, 2].astype(np.float32),
        ]
    )
    np.savez_compressed(
        prediction_path,
        predicted_model_xy=prediction.astype(np.float32),
        predicted_model_xyz=predicted_xyz.astype(np.float32),
        cloth_initial_xy=cloth_initial_xy.astype(np.float32),
        cloth_initial_xyz=points_model.astype(np.float32),
        action_model=action_model.astype(np.float32),
        action_bed=action_bed.astype(np.float32),
        normalized_action=np.asarray(
            best["normalized_action"], dtype=np.float32
        ),
    )
    action_path = output_dir / "uncover_scaled_action.pkl"
    with action_path.open("wb") as handle:
        pickle.dump(action_bed, handle, protocol=2)

    eval_data = {
        "stage": "uncover",
        "model_dir": str(config.model_dir),
        "checkpoint": str(config.checkpoint_path),
        "entropy_weight": float(args.entropy_weight),
        "entropy_grid_size": float(args.entropy_grid_size),
        "grasp_threshold_m": float(grasp_thres),
        "reward_threshold": float(args.reward_threshold),
        "coordinate_frame": "uncover_model_xy",
        "action_frame": "bed",
        "initial_pcd": str(initial_pcd),
        "normalized_action": best["normalized_action"],
        "scaled_action": action_bed,
        "action_model": action_model,
        "cloth_initial": cloth_initial_xy,
        "cloth_predicted_intermediate": prediction,
        "body_points": body_points,
        "covered_status": best["covered_status"],
        "cost": best["cost"],
        "reward": best["reward"],
        "xy_entropy": best["xy_entropy"],
        "overlap_penalty": best["overlap_penalty"],
        "is_on_cloth": best["is_on_cloth"],
        "evaluations": best["evaluations"],
        "iterations": best.get("iterations"),
        "pred_delta_median_m": float(np.median(pred_delta)),
        "pred_delta_p95_m": float(np.percentile(pred_delta, 95)),
        "prediction_path": str(prediction_path),
    }
    with (output_dir / "uncover_cma_eval_data.pkl").open("wb") as handle:
        pickle.dump(eval_data, handle, protocol=2)
    with (output_dir / "uncover_runtime_metadata.json").open("w") as handle:
        json.dump(
            {
                "stage": "uncover",
                "aligned_with": "code/run_robe_sim.py::gnn_cma (build_runtime_graph)",
                "bounds_mode": (
                    "asymmetric_rw" if use_asymmetric else "symmetric_sim"
                ),
                "action_feature_mode": args.action_feature_mode,
                "mirror_x": bool(args.mirror_x),
                "bounds_min": bounds_min.tolist(),
                "bounds_max": bounds_max.tolist(),
                "model_dir": str(config.model_dir),
                "checkpoint": str(config.checkpoint_path),
                "entropy_weight": float(args.entropy_weight),
                "entropy_grid_size": float(args.entropy_grid_size),
                "grasp_threshold_m": float(grasp_thres),
                "reward_threshold": float(args.reward_threshold),
                "feasible_only_best": bool(feasible_only),
                "on_cloth_evaluations": int(num_on_cloth),
                "max_wrist_extension": reach.max_wrist_extension,
                "stretch_reach_snapshot": (
                    None
                    if reach_ctx.snapshot_path is None
                    else str(reach_ctx.snapshot_path)
                ),
                "arm_max_m": (
                    None if not reach_ctx.enabled else float(reach_ctx.arm_max_m)
                ),
                "is_reachable": bool(best.get("is_reachable", True)),
                "scaled_action": action_bed.tolist(),
                "action_model": action_model.tolist(),
                "normalized_action": np.asarray(
                    best["normalized_action"], dtype=float
                ).tolist(),
                "reward": float(best["reward"]),
                "xy_entropy": float(best["xy_entropy"]),
                "overlap_penalty": float(best["overlap_penalty"]),
                "cost": float(best["cost"]),
                "is_on_cloth": bool(best["is_on_cloth"]),
                "evaluations": int(best["evaluations"]),
                "iterations": int(best.get("iterations") or 0),
                "pred_delta_median_m": float(np.median(pred_delta)),
                "pred_delta_p95_m": float(np.percentile(pred_delta, 95)),
                "prediction_path": str(prediction_path),
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    _draw_uncover_prediction(
        output_dir / "uncover_prediction.png",
        cloth_initial_xy,
        prediction,
        body_points,
        action_model,
        title=(
            f"TL{args.tl_code} Uncover CMA · "
            f"W={args.entropy_weight:g} g={args.entropy_grid_size:g} · "
            f"R={best['reward']:.1f}"
        ),
    )
    print(f"Uncover checkpoint: {config.checkpoint_path}")
    print(
        f"Entropy: weight={args.entropy_weight:g} "
        f"grid={args.entropy_grid_size:g} "
        f"xy_entropy={best['xy_entropy']:.3f} "
        f"overlap_penalty={best['overlap_penalty']:.3f}"
    )
    print(
        f"Uncover reward: {best['reward']:.3f}  cost: {best['cost']:.3f}  "
        f"on_cloth={best['is_on_cloth']}  "
        f"pred_delta med/p95="
        f"{float(np.median(pred_delta)):.3f}/"
        f"{float(np.percentile(pred_delta, 95)):.3f}m"
    )
    print(f"Action in bed frame: {action_bed}")
    print(f"Saved uncover action: {action_path}")
    print(f"Saved predicted intermediate: {prediction_path}")


if __name__ == "__main__":
    main()
