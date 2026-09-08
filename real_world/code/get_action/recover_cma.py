#!/usr/bin/env python3
"""Plan a Recover action from original and intermediate blanket states."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cma
import numpy as np

REAL_WORLD_CODE = Path(__file__).resolve().parents[1]
if str(REAL_WORLD_CODE) not in sys.path:
    sys.path.insert(0, str(REAL_WORLD_CODE))

from blanket_grasp import (  # noqa: E402
    DEFAULT_MAX_GRASP_DISTANCE_M,
    DEFAULT_SNAP_INWARD_M,
    nearest_grasp_xy_distance,
    read_pcd_xyz,
    snap_grasp_inward,
)
from pickle_compat import load_pickle  # noqa: E402
from trial_layout import (  # noqa: E402
    ensure_trial_body_info,
    resolve_body_info,
    resolve_initial_pcd,
    resolve_uncover_file,
)
from canonical_bed import canonicalize_body_xy, maybe_load_canonical_frame  # noqa: E402
from stretch_limits import (  # noqa: E402
    PLANNER_ARM_MAX_M,
    assert_selected_action_reachable,
    load_cma_reach,
    restrict_cma_bounds_to_reach,
)
from recover_runtime import (
    ACTION_SCALE,
    DEFAULT_MODEL_DIR,
    ModelFrameAdapter,
    RecoverRuntimeGraph,
    DEFAULT_RECOVER_POINTS,
    DRAPING_EDGE_X,
    compute_graph_stats,
    draw_recover_prediction,
    load_recover_model,
    read_pcd_points,
    scale_action,
)
from density_complete import complete_sensor_density  # noqa: E402
from sensor_snap import (  # noqa: E402
    DEFAULT_COLUMN_VOXEL_M,
    DEFAULT_XY_RADIUS_M,
    DEFAULT_Z_RADIUS_M,
    load_predicted_state_3d,
    snap_visible_layer,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _parse_limits(value: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError("limits must be MIN,MAX with MIN < MAX")
    return parts[0], parts[1]


def reverse_uncover_policy_action(uncover_policy: np.ndarray) -> np.ndarray:
    """Legacy Recover warm-start: swap uncover grasp/release in policy space."""

    action = np.asarray(uncover_policy, dtype=np.float64).reshape(-1)
    if action.size != 4:
        raise ValueError(
            f"Uncover policy action must have 4 values, got shape {action.shape}"
        )
    return np.asarray(
        [action[2], action[3], action[0], action[1]], dtype=np.float64
    )


def line_field_recover_policy_action(
    cloth_intermediate_3d: np.ndarray,
    body_points: np.ndarray,
    *,
    step_size: float = 0.16,
) -> np.ndarray | None:
    """LineInvField Recover warm-start (``gnn_dc_recover`` field-guided line).

    Builds the recover reward field on the intermediate cloth, then takes the
    field-guided pick→place segment from ``compute_field_guided_action``
    (``task_type='cover'``) and maps it into normalized CMA policy space.
    """

    from assistive_gym.envs.field_guided_policy import compute_field_guided_action
    from assistive_gym.gnn_dc_recover import build_recover_reward_field

    cloth = np.asarray(cloth_intermediate_3d, dtype=np.float32)
    if cloth.ndim != 2 or cloth.shape[1] < 2:
        raise ValueError(f"Unexpected intermediate cloth shape: {cloth.shape}")
    if cloth.shape[1] == 2:
        cloth = np.concatenate(
            [cloth, np.zeros((len(cloth), 1), dtype=np.float32)], axis=1
        )
    elif cloth.shape[1] > 3:
        cloth = cloth[:, :3]

    body = np.asarray(body_points, dtype=np.float32)
    reward_field = build_recover_reward_field(cloth, body)
    target_pts = body[body[:, 2] == 1] if body.shape[1] >= 3 else body
    if len(target_pts) > 0:
        target_center = np.mean(target_pts[:, :3], axis=0)
    else:
        target_center = np.mean(body[:, :3], axis=0)

    field_result = compute_field_guided_action(
        cloth_positions=cloth,
        reward_field=reward_field,
        target_center=target_center,
        task_type="cover",
        threshold=0.0,
        step_size=float(step_size),
        debug_mode=False,
        p_id=None,
    )
    if field_result is None:
        return None

    pick_pos, place_pos, _ = field_result
    action_world = np.asarray(
        [pick_pos[0], pick_pos[1], place_pos[0], place_pos[1]],
        dtype=np.float64,
    )
    # Match gnn_dc_recover.policy_action_from_world / scale_action inverse.
    from recover_runtime import ACTION_SCALE

    return np.clip(action_world / ACTION_SCALE, -1.0, 1.0)


def _load_uncover_policy_action(path: Path) -> np.ndarray:
    """Load normalized uncover action from metadata / eval / npz / raw pickle."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Uncover policy action file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r") as handle:
            payload = json.load(handle)
        if "normalized_action" in payload:
            return np.asarray(payload["normalized_action"], dtype=np.float64)
        raise KeyError(f"{path} missing normalized_action")
    if suffix == ".npz":
        data = np.load(path)
        if "normalized_action" not in data:
            raise KeyError(f"{path} missing normalized_action")
        return np.asarray(data["normalized_action"], dtype=np.float64)
    if suffix in (".pkl", ".pickle"):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict) and "normalized_action" in payload:
            return np.asarray(payload["normalized_action"], dtype=np.float64)
        return np.asarray(payload, dtype=np.float64).reshape(-1)
    raise ValueError(f"Unsupported uncover-action file type: {path}")


def _try_reverse_x0(
    pose_dir: Path, uncover_policy_action: Path | None
) -> tuple[np.ndarray, str] | None:
    candidates: list[Path] = []
    if uncover_policy_action is not None:
        candidates.append(uncover_policy_action)
    else:
        candidates.extend(
            [
                resolve_uncover_file(
                    pose_dir, "uncover_runtime_metadata.json", required=False
                ),
                resolve_uncover_file(
                    pose_dir, "uncover_cma_eval_data.pkl", required=False
                ),
                resolve_uncover_file(
                    pose_dir, "uncover_prediction.npz", required=False
                ),
            ]
        )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            uncover = _load_uncover_policy_action(candidate)
            x0 = np.clip(reverse_uncover_policy_action(uncover), -1.0, 1.0)
            return x0, f"reverse({candidate.name})"
        except Exception:
            continue
    return None


def _resolve_recover_x0(
    pose_dir: Path,
    uncover_policy_action: Path | None,
    warm_start: str,
    cloth_intermediate_3d: np.ndarray | None = None,
    body_points: np.ndarray | None = None,
    line_step_size: float = 0.16,
) -> tuple[np.ndarray, str]:
    """Recover CMA x0. Default ``line`` = LineInvField field-guided segment."""

    if warm_start in ("none", "zero"):
        return np.zeros(4, dtype=np.float64), warm_start

    if warm_start == "line":
        if cloth_intermediate_3d is None or body_points is None:
            raise ValueError("line warm-start requires intermediate cloth and body points")
        try:
            x0 = line_field_recover_policy_action(
                cloth_intermediate_3d,
                body_points,
                step_size=line_step_size,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Recover line warm-start failed ({exc}); trying reverse fallback")
            x0 = None
        if x0 is not None:
            return np.asarray(x0, dtype=np.float64), "line_field"
        reverse = _try_reverse_x0(pose_dir, uncover_policy_action)
        if reverse is not None:
            print(
                "Recover line warm-start returned None; "
                f"falling back to {reverse[1]}"
            )
            return reverse
        print("Recover line warm-start failed; falling back to zeros")
        return np.zeros(4, dtype=np.float64), "zero_fallback"

    if warm_start == "reverse":
        reverse = _try_reverse_x0(pose_dir, uncover_policy_action)
        if reverse is not None:
            return reverse
        if uncover_policy_action is not None:
            raise RuntimeError(
                f"Failed to load uncover policy action from {uncover_policy_action}"
            )
        print(
            "Recover reverse warm-start: no uncover normalized_action found; "
            "falling back to zeros"
        )
        return np.zeros(4, dtype=np.float64), "zero_fallback"

    raise ValueError(f"Unknown warm-start strategy: {warm_start}")


def _resolve_pose_file(pose_dir: Path, name: str, manikin: bool, subject_dir: Path) -> Path:
    path = (subject_dir if manikin else pose_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"Required runtime artifact does not exist: {path}")
    return path


def _prepare_state(
    path: str | Path,
    adapter: ModelFrameAdapter,
    apply_draping: bool,
    max_points: int,
    voxel_size: float,
    draping_edge_x: float = DRAPING_EDGE_X,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_bed, points_model = read_pcd_points(
        path,
        adapter,
        max_points=max_points,
        voxel_size=voxel_size,
        apply_draping=apply_draping,
        draping_edge_x=draping_edge_x,
    )
    return points_bed, points_model, points_model[:, :2].copy()


def _load_body_points(
    pose_dir: Path,
    subject_dir: Path,
    manikin: bool,
    target_limb_code: int,
    allow_subject: bool = False,
) -> np.ndarray:
    pose_path = _resolve_pose_file(pose_dir, "human_pose.pkl", manikin, subject_dir)
    body_path = (
        subject_dir / "body_info.pkl"
        if manikin
        else ensure_trial_body_info(
            pose_dir, subject_dir, allow_subject=allow_subject
        )
        or resolve_body_info(pose_dir, subject_dir, allow_subject=allow_subject)
    )
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
            "measure body_info before running Recover."
        )

    from assistive_gym.envs.bu_gnn_util import get_body_points_from_obs

    body_points = get_body_points_from_obs(
        human_pose=human_pose,
        target_limb_code=target_limb_code,
        body_info=body_info,
    )
    if body_points.ndim != 2 or body_points.shape[1] < 2:
        raise ValueError(f"Unexpected body point shape: {body_points.shape}")
    return np.asarray(body_points, dtype=np.float64)


def _covered_status(
    body_points: np.ndarray,
    cloth_xy: np.ndarray,
) -> np.ndarray:
    from assistive_gym.envs.bu_gnn_util import get_covered_status

    return np.asarray(get_covered_status(body_points, cloth_xy))


def _validate_intermediate_state(
    original_xy: np.ndarray,
    intermediate_xy: np.ndarray,
    body_points: np.ndarray,
) -> None:
    """Reject a Recover run when manual uncovering did not occur."""

    initial_status = _covered_status(body_points, original_xy)
    intermediate_status = _covered_status(body_points, intermediate_xy)
    target = body_points[:, 2] == 1
    initially_covered = target & initial_status[:, 1].astype(bool)
    newly_uncovered = (
        initially_covered & ~intermediate_status[:, 1].astype(bool)
    )
    initial_target_count = int(np.count_nonzero(initially_covered))
    exposed_target_count = int(np.count_nonzero(newly_uncovered))
    if exposed_target_count == 0:
        raise ValueError(
            "Recover intermediate state is invalid: 0 target body points "
            "became uncovered. Keep the initial blanket state covered, "
            "manually expose the selected TL, then press ENTER to capture "
            "intermediate/blanket_pcd.pcd. "
            f"Initially covered target points: {initial_target_count}."
        )
    print(
        "Recover intermediate validation: "
        f"exposed_target={exposed_target_count}/{initial_target_count}"
    )


def _evaluate(
    normalized_action: np.ndarray,
    graph: RecoverRuntimeGraph,
    model,
    device: str,
    original_xy: np.ndarray,
    intermediate_xy: np.ndarray,
    execution_cloth_xy: np.ndarray | None,
    adapter: ModelFrameAdapter,
    body_points: np.ndarray,
    use_displacement: bool,
    grasp_thres: float = 0.028,
) -> tuple[float, np.ndarray, np.ndarray, bool, np.ndarray]:
    action_world = scale_action(normalized_action)
    distances = np.linalg.norm(
        intermediate_xy - action_world[:2][None, :],
        axis=1,
    )
    model_on_cloth = bool(np.any(distances < grasp_thres))
    execution_on_cloth = True
    if execution_cloth_xy is not None:
        action_bed = adapter.model_action_to_bed(action_world)
        execution_distances = np.linalg.norm(
            execution_cloth_xy - action_bed[:2][None, :],
            axis=1,
        )
        execution_on_cloth = bool(
            np.any(execution_distances <= DEFAULT_MAX_GRASP_DISTANCE_M)
        )
    is_on_cloth = model_on_cloth and execution_on_cloth
    if not is_on_cloth:
        prediction = intermediate_xy.copy()
        return 500.0, prediction, np.empty((0, 3)), False, action_world

    detail = graph.predict_detail(
        model,
        normalized_action,
        device=device,
        use_displacement=use_displacement,
    )
    prediction = detail["prediction"]
    from assistive_gym.envs.bu_gnn_util import get_recovering_reward

    reward, covered_status = get_recovering_reward(
        action_world,
        body_points,
        original_xy,
        intermediate_xy,
        prediction,
    )
    # Optional planner-score corrector: CMA optimizes -J_corrected.
    if hasattr(model, "correct_single_score") and detail.get("latent") is not None:
        corrected = model.correct_single_score(
            latent=detail["latent"],
            j_base=float(reward),
            action_policy=normalized_action,
            points_xy=graph.initial_state,
            base_delta=detail.get("base_delta"),
        )
        reward = float(corrected["j_corrected"])
    return -float(reward), prediction, covered_status, True, action_world


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=Path("TEST"))
    parser.add_argument("--pose-dir", type=Path, default=Path("TEST"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for Recover artifacts. Defaults to --pose-dir. "
        "Use a source-specific subdirectory when planning both pred and sensor.",
    )
    parser.add_argument("--intermediate-pcd", type=Path, default=None)
    parser.add_argument(
        "--intermediate-prediction",
        type=Path,
        default=None,
        help="NPZ from uncover_cma.py (predicted_model_xy). Closed-loop Recover input. "
        "With --intermediate-pcd, builds G_snap = visible-layer sensor update of G_pred.",
    )
    parser.add_argument(
        "--snap-xy-radius",
        type=float,
        default=DEFAULT_XY_RADIUS_M,
        help="Sensor-to-prediction XY gate for visible-layer snap (m).",
    )
    parser.add_argument(
        "--snap-z-radius",
        type=float,
        default=DEFAULT_Z_RADIUS_M,
        help="Sensor-to-prediction Z gate for visible-layer snap (m).",
    )
    parser.add_argument(
        "--snap-column-voxel",
        type=float,
        default=DEFAULT_COLUMN_VOXEL_M,
        help="XY column size used to pick the single top node that may snap. "
        "0 disables the column gate.",
    )
    parser.add_argument(
        "--graph-correction",
        choices=("snap", "density-1", "density-50", "density-full"),
        default="snap",
        help="When both --intermediate-prediction and --intermediate-pcd are "
        "set: snap = pred-anchored visible snap; density-* = sensor-anchored "
        "local density completion.",
    )
    parser.add_argument("--tl-code", type=int, required=True)
    parser.add_argument("--manikin", type=int, default=0)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint-number", type=int, default=None)
    parser.add_argument(
        "--residual-model-dir",
        type=Path,
        default=None,
        help=(
            "Optional planner residual bundle. Prefer score-corrector bundles "
            "([ScoreCorrector] config.ini); legacy displacement residual still loads."
        ),
    )
    parser.add_argument(
        "--residual-checkpoint-number",
        type=int,
        default=None,
        help="Optional residual epoch; defaults to residual_model_best.pth.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-fevals", type=int, default=150)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--sigma0", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_RECOVER_POINTS,
        help="Sensor-to-model warn threshold. For voxel models, 0 disables "
        "the warning and the full XYZ voxel set is always kept. Legacy "
        "voxel_size=nan still uses this as a hard XY-centroid cap.",
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
        "--skip-intermediate-validation",
        action="store_true",
        help="Do not require the intermediate to expose the target limb. "
        "Used when Recover consumes an Uncover GNN prediction.",
    )
    parser.add_argument(
        "--uncover-policy-action",
        type=Path,
        default=None,
        help="Normalized uncover action (json/pkl/npz with normalized_action). "
        "Default: auto-load from pose_dir uncover_* artifacts.",
    )
    parser.add_argument(
        "--warm-start",
        choices=("line", "reverse", "zero", "none"),
        default="line",
        help="Recover x0. Default 'line' = LineInvField field-guided pick→place "
        "(gnn_dc_recover). 'reverse' swaps uncover grasp/release.",
    )
    parser.add_argument(
        "--line-step-size",
        type=float,
        default=0.16,
        help="Field-guided place offset (m) for --warm-start line.",
    )
    parser.add_argument(
        "--mirror-x",
        dest="mirror_x",
        action="store_true",
        default=True,
        help="Mirror bed x to match the simulation handedness. Must match the "
        "Uncover stage when consuming its predicted intermediate.",
    )
    parser.add_argument(
        "--no-mirror-x",
        dest="mirror_x",
        action="store_false",
        help="Keep the raw bed x orientation.",
    )
    parser.add_argument(
        "--action-feature-mode",
        choices=("normalized", "scaled"),
        default="normalized",
        help="Node-feature action convention. Default 'normalized' matches "
        "Uncover / code/build_runtime_graph.py (run_robe_sim).",
    )
    parser.add_argument(
        "--stretch-reach-snapshot",
        type=Path,
        default=None,
        help="Parking snapshot JSON. Unreachable EE paths get a high CMA cost.",
    )
    parser.add_argument(
        "--arm-max-m",
        type=float,
        default=PLANNER_ARM_MAX_M,
        help="Planner EE/arm cap (m). Default 0.50 leaves 2 cm vs hardware.",
    )
    parser.add_argument(
        "--reuse-subject-body",
        action="store_true",
        help="Copy subject_dir/body_info.pkl into this trial. Default is "
        "trial then this exp's poses/pose_n only.",
    )
    parser.add_argument(
        "--grasp-snap-inward-m",
        type=float,
        default=DEFAULT_SNAP_INWARD_M,
        help="When the selected grasp is off the measured cloth, snap to "
        "the nearest point plus this inward margin (m). Default 0.015.",
    )
    args = parser.parse_args()
    if args.max_points < 0:
        parser.error("--max-points must be >= 0 (0 disables the warn threshold)")
    if args.intermediate_pcd is None and args.intermediate_prediction is None:
        parser.error("Provide --intermediate-pcd and/or --intermediate-prediction")
    if args.snap_xy_radius <= 0 or args.snap_z_radius <= 0:
        parser.error("--snap-xy-radius and --snap-z-radius must be positive")

    args.subject_dir = args.subject_dir.resolve()
    args.pose_dir = args.pose_dir.resolve()
    output_dir = (
        args.pose_dir if args.output_dir is None else args.output_dir.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_pcd = resolve_initial_pcd(args.pose_dir)
    if args.intermediate_pcd is not None:
        args.intermediate_pcd = args.intermediate_pcd.resolve()
    if args.intermediate_prediction is not None:
        args.intermediate_prediction = args.intermediate_prediction.resolve()

    model, config = load_recover_model(
        model_dir=args.model_dir,
        checkpoint_number=args.checkpoint_number,
        device=args.device,
        residual_model_dir=args.residual_model_dir,
        residual_checkpoint_number=args.residual_checkpoint_number,
    )
    adapter = ModelFrameAdapter(
        x_limits=args.x_limits,
        y_limits=args.y_limits,
        mirror_x=bool(args.mirror_x),
    )
    apply_draping = config.rot_draping and not args.no_draping_rotation

    _, original_model, original_xy = _prepare_state(
        initial_pcd,
        adapter,
        apply_draping,
        args.max_points,
        config.voxel_size,
        draping_edge_x=args.draping_edge_x,
    )
    snap_metadata: dict | None = None
    density_metadata: dict | None = None
    if (
        args.intermediate_prediction is not None
        and args.intermediate_pcd is not None
    ):
        if not args.intermediate_prediction.is_file():
            raise FileNotFoundError(
                f"Missing Uncover prediction: {args.intermediate_prediction}"
            )
        if not args.intermediate_pcd.is_file():
            raise FileNotFoundError(
                f"Missing intermediate PCD: {args.intermediate_pcd}"
            )
        predicted_model = load_predicted_state_3d(
            args.intermediate_prediction,
            g0_model=original_model,
        )
        if args.graph_correction.startswith("density"):
            _, sensor_model, _ = _prepare_state(
                args.intermediate_pcd,
                adapter,
                apply_draping,
                args.max_points,
                config.voxel_size,
                draping_edge_x=args.draping_edge_x,
            )
            intermediate_model, density_metadata = complete_sensor_density(
                sensor_model,
                predicted_model,
                mode=args.graph_correction,
                voxel_size=(
                    float(config.voxel_size)
                    if np.isfinite(config.voxel_size)
                    else 0.05
                ),
            )
            intermediate_model = np.asarray(intermediate_model, dtype=np.float64)
            intermediate_xy = intermediate_model[:, :2].copy()
            intermediate_source = f"sensor_{args.graph_correction.replace('-', '_')}"
            args.skip_intermediate_validation = True
            print(
                "Recover density completion: "
                f"mode={args.graph_correction} "
                f"N(G_sensor)={density_metadata['num_sensor']} "
                f"N(G_pred)={density_metadata['num_pred']} "
                f"N(G_corr)={density_metadata['num_completed']} "
                f"added={density_metadata['num_added']} "
                f"cells={density_metadata['num_cells_completed']}/"
                f"{density_metadata['num_density_gap_cells']}"
            )
        else:
            # Observation cloud: same frame as G_pred, but not re-voxelized.
            _, sensor_model = read_pcd_points(
                args.intermediate_pcd,
                adapter,
                max_points=0,
                voxel_size=float("nan"),
                apply_draping=apply_draping,
                draping_edge_x=args.draping_edge_x,
            )
            intermediate_model, snap_metadata = snap_visible_layer(
                predicted_model,
                sensor_model,
                xy_radius=float(args.snap_xy_radius),
                z_radius=float(args.snap_z_radius),
                column_voxel_size=float(args.snap_column_voxel),
            )
            intermediate_model = np.asarray(intermediate_model, dtype=np.float64)
            intermediate_xy = intermediate_model[:, :2].copy()
            intermediate_source = "prediction_snapped_by_sensor"
            args.skip_intermediate_validation = True
            print(
                "Recover sensor snap: "
                f"N(G_pred)={len(predicted_model)}, "
                f"N(P_sensor)={len(sensor_model)}, "
                f"N(G_snap)={len(intermediate_model)}, "
                f"visible={snap_metadata['num_visible']}/"
                f"{snap_metadata['num_eligible_top']} top nodes, "
                f"hidden_unchanged={snap_metadata['num_hidden_unchanged']}, "
                f"mean_visible_snap={snap_metadata['mean_visible_snap_m']:.4f}m"
            )
    elif args.intermediate_prediction is not None:
        if not args.intermediate_prediction.is_file():
            raise FileNotFoundError(
                f"Missing Uncover prediction: {args.intermediate_prediction}"
            )
        intermediate_model = np.asarray(
            load_predicted_state_3d(
                args.intermediate_prediction,
                g0_model=original_model,
            ),
            dtype=np.float64,
        )
        intermediate_xy = intermediate_model[:, :2].copy()
        intermediate_source = "uncover_prediction"
        args.skip_intermediate_validation = True
    else:
        _, intermediate_model, intermediate_xy = _prepare_state(
            args.intermediate_pcd,
            adapter,
            apply_draping,
            args.max_points,
            config.voxel_size,
            draping_edge_x=args.draping_edge_x,
        )
        intermediate_source = "sensor_pcd"
    execution_cloth_xy = None
    if args.intermediate_pcd is not None:
        measured_bed = adapter.filter_bed_points(
            read_pcd_xyz(args.intermediate_pcd)
        )
        execution_cloth_xy = measured_bed[:, :2].copy()
    print(
        "Recover point sampling: "
        f"initial={len(original_model)}, "
        f"intermediate={len(intermediate_model)}, "
        f"source={intermediate_source}, "
        f"max={args.max_points}"
    )
    graph = RecoverRuntimeGraph(
        intermediate_model,
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
    initial_covered_status = _covered_status(body_points, original_xy)
    intermediate_covered_status = _covered_status(body_points, intermediate_xy)
    if args.skip_intermediate_validation:
        print(
            "Recover intermediate validation skipped "
            f"(source={intermediate_source})"
        )
    else:
        _validate_intermediate_state(original_xy, intermediate_xy, body_points)

    reach_ctx = load_cma_reach(
        pose_dir=args.pose_dir,
        snapshot_path_or_none=args.stretch_reach_snapshot,
        arm_max_m=float(args.arm_max_m),
    )
    if reach_ctx.enabled:
        print(
            f"Recover stretch reach: snapshot={reach_ctx.snapshot_path} "
            f"arm_max={reach_ctx.arm_max_m:.3f} m"
        )

    rng = np.random.default_rng(args.seed)
    x0, warm_start_source = _resolve_recover_x0(
        args.pose_dir,
        args.uncover_policy_action,
        args.warm_start,
        cloth_intermediate_3d=intermediate_model,
        body_points=body_points,
        line_step_size=float(args.line_step_size),
    )
    bounds_min = -np.ones(4, dtype=np.float64)
    bounds_max = np.ones(4, dtype=np.float64)
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
            f"Recover CMA box = execution wrist-down X strip "
            f"[{bounds_min.round(3).tolist()}].."
            f"[{bounds_max.round(3).tolist()}] "
            "(CMA does not filter waypoints)"
        )
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
    x0 = np.minimum(np.maximum(np.asarray(x0, dtype=np.float64), bounds_min), bounds_max)
    print(
        f"Recover CMA warm-start={warm_start_source} "
        f"x0={np.asarray(x0).round(3).tolist()} "
        f"action_features={args.action_feature_mode} "
        f"mirror_x={bool(args.mirror_x)} "
        f"stretch_reach={'on' if reach_ctx.enabled else 'off'} "
        f"(LineInvField line / gnn_dc_recover field-guided)"
    )
    es = cma.CMAEvolutionStrategy(x0.tolist(), args.sigma0, opts)
    best = None
    evaluations = 0
    while not es.stop():
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
                original_xy,
                intermediate_xy,
                execution_cloth_xy,
                adapter,
                body_points,
                config.use_displacement,
                grasp_thres=(
                    max(0.05, float(config.voxel_size))
                    if np.isfinite(config.voxel_size)
                    and float(config.voxel_size) > 0
                    else 0.028
                ),
            )
            costs.append(result[0])
            results.append(result)
            evaluations += 1
        es.tell(actions, costs)
        best_index = int(np.argmin(costs))
        candidate = {
            "cost": float(costs[best_index]),
            "normalized_action": np.asarray(
                actions[best_index], dtype=np.float64
            ),
            "prediction": results[best_index][1],
            "covered_status": results[best_index][2],
            "is_on_cloth": results[best_index][3],
            "action_world": results[best_index][4],
            "is_reachable": True,
            "reach_reason": "cma_x_strip",
            "evaluations": evaluations,
        }
        if best is None or candidate["cost"] < best["cost"]:
            best = candidate

    if best is None:
        raise RuntimeError("CMA-ES produced no candidate action")

    action_model = np.asarray(best["action_world"], dtype=np.float64)
    action_bed = adapter.model_action_to_bed(action_model)
    planned_action_bed = action_bed.copy()
    execution_grasp_distance = None
    grasp_snap = None
    if execution_cloth_xy is not None:
        cloth_xyz = np.column_stack(
            [execution_cloth_xy, np.zeros(len(execution_cloth_xy))]
        )
        execution_grasp_distance = nearest_grasp_xy_distance(action_bed, cloth_xyz)
        grasp_snap = snap_grasp_inward(
            action_bed,
            cloth_xyz,
            on_cloth_m=DEFAULT_MAX_GRASP_DISTANCE_M,
            inward_m=float(args.grasp_snap_inward_m),
        )
        if grasp_snap.snapped:
            action_bed = np.asarray(grasp_snap.action, dtype=np.float64)
            print(
                "SNAP recover grasp onto cloth: "
                f"planned {grasp_snap.distance_before_m * 100.0:.1f} cm away → "
                f"nearest + {grasp_snap.inward_m * 100.0:.1f} cm inward "
                f"(after {grasp_snap.distance_after_m * 100.0:.1f} cm)"
            )
        else:
            print(
                "Selected-action execution cloth check: "
                f"{execution_grasp_distance * 100.0:.1f} cm "
                f"<= {DEFAULT_MAX_GRASP_DISTANCE_M * 100.0:.1f} cm"
            )
    reach = assert_selected_action_reachable(action_bed, reach_ctx)
    if reach_ctx.enabled:
        print(
            "Selected-action workspace check: "
            f"max_wrist={reach.max_wrist_extension} "
            f"planning_max={reach_ctx.arm_max_m:.3f} {reach.reason}"
        )
    prediction = np.asarray(best["prediction"], dtype=np.float64)
    final_covered_status = np.asarray(best["covered_status"])
    if (
        final_covered_status.ndim != 2
        or final_covered_status.shape[0] != len(body_points)
        or final_covered_status.shape[1] < 2
    ):
        final_covered_status = _covered_status(body_points, prediction)
    graph_stats = compute_graph_stats(
        intermediate_xy,
        graph.edge_index,
        voxel_size=config.voxel_size,
        max_points=args.max_points,
    )
    output_action = output_dir / "scaled_action.pkl"
    with output_action.open("wb") as handle:
        pickle.dump(action_bed, handle, protocol=2)

    eval_data = {
        "model_dir": str(config.model_dir),
        "checkpoint": str(config.checkpoint_path),
        "residual_model_dir": (
            None
            if args.residual_model_dir is None
            else str(args.residual_model_dir)
        ),
        "residual_checkpoint_number": args.residual_checkpoint_number,
        "coordinate_frame": "recover_model_xy",
        "action_frame": "bed",
        "initial_pcd": str(initial_pcd),
        "intermediate_pcd": (
            None if args.intermediate_pcd is None else str(args.intermediate_pcd)
        ),
        "intermediate_prediction": (
            None
            if args.intermediate_prediction is None
            else str(args.intermediate_prediction)
        ),
        "intermediate_source": intermediate_source,
        "sensor_snap": _json_safe(snap_metadata),
        "density_completion": _json_safe(density_metadata),
        "output_dir": str(output_dir),
        "warm_start": args.warm_start,
        "warm_start_source": warm_start_source,
        "warm_start_x0": np.asarray(x0, dtype=np.float64),
        "mirror_x": bool(args.mirror_x),
        "action_feature_mode": args.action_feature_mode,
        "aligned_with": "assistive_gym/gnn_dc_recover.py::field_guided_line",
        "normalized_action": best["normalized_action"],
        "scaled_action": action_bed,
        "planned_action_bed": planned_action_bed,
        "grasp_snap": None if grasp_snap is None else grasp_snap.to_dict(),
        "action_model": action_model,
        "cloth_initial": original_xy,
        "cloth_intermediate": intermediate_xy,
        "cloth_final": prediction,
        "body_points": body_points,
        "initial_covered_status": initial_covered_status,
        "intermediate_covered_status": intermediate_covered_status,
        "covered_status": best["covered_status"],
        "final_covered_status": final_covered_status,
        "cost": best["cost"],
        "reward": -best["cost"],
        "is_on_cloth": bool(best["is_on_cloth"])
        or (grasp_snap is not None and grasp_snap.snapped),
        "execution_grasp_distance_m": execution_grasp_distance,
        "execution_grasp_distance_limit_m": DEFAULT_MAX_GRASP_DISTANCE_M,
        "is_reachable": True,
        "max_wrist_extension": reach.max_wrist_extension,
        "stretch_reach_snapshot": (
            None
            if reach_ctx.snapshot_path is None
            else str(reach_ctx.snapshot_path)
        ),
        "arm_max_m": (
            None if not reach_ctx.enabled else float(reach_ctx.arm_max_m)
        ),
        "evaluations": best["evaluations"],
        "graph": {
            "edge_threshold": config.edge_threshold,
            "num_points": len(intermediate_xy),
            "num_edges": int(graph.edge_index.shape[1]),
            "sensor_max_points": args.max_points,
            "voxel_size": (
                None if np.isnan(config.voxel_size) else config.voxel_size
            ),
            "rot_draping": apply_draping,
        },
        "graph_stats": graph_stats,
        "coordinate_adapter": {
            "xy": "bed_identity",
            "model_surface_z": adapter.model_surface_z,
            "x_limits": list(adapter.x_limits),
            "y_limits": list(adapter.y_limits),
        },
    }
    with (output_dir / "cma_eval_data.pkl").open("wb") as handle:
        pickle.dump(eval_data, handle, protocol=2)
    with (output_dir / "recover_runtime_metadata.json").open("w") as handle:
        json.dump(
            {
                "model_dir": str(config.model_dir),
                "checkpoint": str(config.checkpoint_path),
                "residual_model_dir": (
                    None
                    if args.residual_model_dir is None
                    else str(args.residual_model_dir)
                ),
                "residual_checkpoint_number": args.residual_checkpoint_number,
                "coordinate_frame": "recover_model_xy",
                "action_frame": "bed",
                "initial_pcd": str(initial_pcd),
                "intermediate_pcd": (
                    None
                    if args.intermediate_pcd is None
                    else str(args.intermediate_pcd)
                ),
                "intermediate_prediction": (
                    None
                    if args.intermediate_prediction is None
                    else str(args.intermediate_prediction)
                ),
                "intermediate_source": intermediate_source,
                "sensor_snap": _json_safe(snap_metadata),
        "density_completion": _json_safe(density_metadata),
                "output_dir": str(output_dir),
                "aligned_with": "assistive_gym/gnn_dc_recover.py::field_guided_line",
                "warm_start": args.warm_start,
                "warm_start_source": warm_start_source,
                "warm_start_x0": np.asarray(x0, dtype=float).tolist(),
                "mirror_x": bool(args.mirror_x),
                "action_feature_mode": args.action_feature_mode,
                "scaled_action": action_bed.tolist(),
                "planned_action_bed": planned_action_bed.tolist(),
                "grasp_snap": None if grasp_snap is None else grasp_snap.to_dict(),
                "action_model": action_model.tolist(),
                "reward": -best["cost"],
                "is_on_cloth": bool(best["is_on_cloth"])
                or (grasp_snap is not None and grasp_snap.snapped),
                "is_reachable": True,
                "max_wrist_extension": reach.max_wrist_extension,
                "stretch_reach_snapshot": (
                    None
                    if reach_ctx.snapshot_path is None
                    else str(reach_ctx.snapshot_path)
                ),
                "arm_max_m": (
                    None if not reach_ctx.enabled else float(reach_ctx.arm_max_m)
                ),
                "evaluations": best["evaluations"],
                "graph": eval_data["graph"],
                "graph_stats": graph_stats,
                "coordinate_adapter": eval_data["coordinate_adapter"],
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    if snap_metadata is not None:
        np.savez_compressed(
            output_dir / "g_snap.npz",
            g_snap_xyz=np.asarray(intermediate_model, dtype=np.float32),
            cloth_final_xy=prediction.astype(np.float32),
        )
    if density_metadata is not None:
        np.savez_compressed(
            output_dir / "g_density.npz",
            g_density_xyz=np.asarray(intermediate_model, dtype=np.float32),
            cloth_final_xy=prediction.astype(np.float32),
        )
    source_title = {
        "uncover_prediction": "recover_pred",
        "sensor_pcd": "recover_sensor",
        "prediction_snapped_by_sensor": "recover_snap",
        "sensor_density_1": "recover_density1",
        "sensor_density_50": "recover_density50",
        "sensor_density_full": "recover_density_full",
    }.get(intermediate_source, intermediate_source)
    draw_recover_prediction(
        output_dir / "recover_prediction.png",
        original_xy,
        intermediate_xy,
        prediction,
        body_points,
        action_model,
        initial_covered_status,
        intermediate_covered_status,
        final_covered_status,
        title=f"Target: TL{args.tl_code} | {source_title}",
    )
    print(f"Recover checkpoint: {config.checkpoint_path}")
    print(f"Initial PCD: {initial_pcd}")
    print(
        "Intermediate: "
        f"{args.intermediate_prediction or args.intermediate_pcd} "
        f"({intermediate_source})"
    )
    print(f"Best reward: {-best['cost']:.3f}")
    print(f"Action in model frame: {action_model}")
    print(f"Action in bed frame: {action_bed}")
    print(f"graph_stats: {graph_stats}")
    print(f"Saved action: {output_action}")
    print(f"Saved visualization: {output_dir / 'recover_prediction.png'}")


if __name__ == "__main__":
    main()
