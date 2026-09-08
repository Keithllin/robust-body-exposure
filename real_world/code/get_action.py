import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a real-world blanket action")
    parser.add_argument("--subject-dir", type=Path, default=Path("TEST"))
    parser.add_argument("--pose-dir", type=Path, default=Path("TEST"))
    parser.add_argument("--tl-code", type=int, required=True)
    parser.add_argument("--manikin", type=int, default=0)
    parser.add_argument(
        "--approach",
        choices=("recover", "dyn", "uncover", "naive"),
        default="recover",
        help="recover/dyn: Recover CMA; uncover: Uncover CMA with entropy cost",
    )
    parser.add_argument("--intermediate-pcd", type=Path)
    parser.add_argument(
        "--intermediate-prediction",
        type=Path,
        help="NPZ written by uncover_cma.py for closed-loop Recover",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint-number", type=int)
    parser.add_argument("--residual-model-dir", type=Path)
    parser.add_argument("--residual-checkpoint-number", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-fevals", type=int, default=150)
    parser.add_argument("--max-points", type=int, default=1061)
    parser.add_argument("--entropy-weight", type=float, default=200.0)
    parser.add_argument("--entropy-grid-size", type=float, default=0.05)
    parser.add_argument("--skip-intermediate-validation", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Recover-only: write artifacts here instead of --pose-dir",
    )
    parser.add_argument(
        "--uncover-policy-action",
        type=Path,
        default=None,
        help="Recover warm-start: path to uncover normalized_action "
        "(json/pkl/npz). Default: auto from pose_dir.",
    )
    parser.add_argument(
        "--warm-start",
        choices=("line", "reverse", "zero", "none"),
        default="line",
        help="Recover x0: default 'line' = LineInvField field-guided segment.",
    )
    parser.add_argument(
        "--bounds",
        choices=("symmetric", "asymmetric"),
        default="symmetric",
        help="Uncover-only: CMA search box. symmetric matches run_robe_sim.",
    )
    parser.add_argument(
        "--action-feature-mode",
        choices=("normalized", "scaled"),
        default="normalized",
        help="Uncover/Recover: node-feature action convention "
        "(normalized matches run_robe_sim).",
    )
    parser.add_argument(
        "--graph-correction",
        choices=("snap", "density-1", "density-50", "density-full"),
        default="snap",
        help="Recover: when both prediction and PCD are given, choose "
        "pred-anchored snap or sensor-anchored density completion.",
    )
    parser.add_argument(
        "--no-mirror-x",
        dest="mirror_x",
        action="store_false",
        default=True,
        help="Keep raw bed x instead of mirroring to sim handedness.",
    )
    parser.add_argument(
        "--stretch-reach-snapshot",
        type=Path,
        default=None,
        help="Constrain Uncover/Recover CMA to Stretch reach from this parking snapshot.",
    )
    parser.add_argument(
        "--arm-max-m",
        type=float,
        default=None,
        help="Planner EE/arm cap (m). Default 0.50 when a snapshot is set.",
    )
    parser.add_argument(
        "--reuse-subject-body",
        action="store_true",
        help="Copy subject_dir/body_info.pkl into this trial. Default is "
        "trial then this exp's poses/pose_n only.",
    )
    args = parser.parse_args()

    if args.approach == "uncover":
        script = CODE_DIR / "get_action" / "uncover_cma.py"
        command = [
            str(Path(sys.executable)),
            str(script),
            "--subject-dir",
            str(args.subject_dir),
            "--pose-dir",
            str(args.pose_dir),
            "--tl-code",
            str(args.tl_code),
            "--device",
            args.device,
            "--max-fevals",
            str(args.max_fevals),
            "--max-points",
            str(args.max_points),
            "--entropy-weight",
            str(args.entropy_weight),
            "--entropy-grid-size",
            str(args.entropy_grid_size),
            "--bounds",
            args.bounds,
            "--action-feature-mode",
            args.action_feature_mode,
        ]
        if not args.mirror_x:
            command.append("--no-mirror-x")
        if args.manikin:
            command.extend(["--manikin", str(args.manikin)])
        if args.model_dir is not None:
            command.extend(["--model-dir", str(args.model_dir)])
        if args.checkpoint_number is not None:
            command.extend(["--checkpoint-number", str(args.checkpoint_number)])
        if args.output_dir is not None:
            command.extend(["--output-dir", str(args.output_dir)])
        if args.stretch_reach_snapshot is not None:
            command.extend(
                ["--stretch-reach-snapshot", str(args.stretch_reach_snapshot)]
            )
        if args.arm_max_m is not None:
            command.extend(["--arm-max-m", str(args.arm_max_m)])
        if args.reuse_subject_body:
            command.append("--reuse-subject-body")
    elif args.approach in ("recover", "dyn"):
        script = CODE_DIR / "get_action" / "recover_cma.py"
        if args.intermediate_pcd is None and args.intermediate_prediction is None:
            parser.error(
                "Recover requires --intermediate-pcd and/or --intermediate-prediction"
            )
        command = [
            str(Path(sys.executable)),
            str(script),
            "--subject-dir",
            str(args.subject_dir),
            "--pose-dir",
            str(args.pose_dir),
            "--tl-code",
            str(args.tl_code),
            "--device",
            args.device,
            "--max-fevals",
            str(args.max_fevals),
            "--max-points",
            str(args.max_points),
            "--warm-start",
            args.warm_start,
            "--action-feature-mode",
            args.action_feature_mode,
        ]
        if not args.mirror_x:
            command.append("--no-mirror-x")
        if args.output_dir is not None:
            command.extend(["--output-dir", str(args.output_dir)])
        if args.intermediate_pcd is not None:
            command.extend(["--intermediate-pcd", str(args.intermediate_pcd)])
        if args.intermediate_prediction is not None:
            command.extend(
                ["--intermediate-prediction", str(args.intermediate_prediction)]
            )
        if args.graph_correction != "snap":
            command.extend(["--graph-correction", args.graph_correction])
        if args.uncover_policy_action is not None:
            command.extend(
                ["--uncover-policy-action", str(args.uncover_policy_action)]
            )
        if args.skip_intermediate_validation:
            command.append("--skip-intermediate-validation")
        if args.manikin:
            command.extend(["--manikin", str(args.manikin)])
        if args.model_dir is not None:
            command.extend(["--model-dir", str(args.model_dir)])
        if args.checkpoint_number is not None:
            command.extend(["--checkpoint-number", str(args.checkpoint_number)])
        if args.residual_model_dir is not None:
            command.extend(["--residual-model-dir", str(args.residual_model_dir)])
        if args.residual_checkpoint_number is not None:
            command.extend(
                [
                    "--residual-checkpoint-number",
                    str(args.residual_checkpoint_number),
                ]
            )
        if args.stretch_reach_snapshot is not None:
            command.extend(
                ["--stretch-reach-snapshot", str(args.stretch_reach_snapshot)]
            )
        if args.arm_max_m is not None:
            command.extend(["--arm-max-m", str(args.arm_max_m)])
        if args.reuse_subject_body:
            command.append("--reuse-subject-body")
    else:
        command = [
            str(Path(sys.executable)),
            str(CODE_DIR / "get_action" / "get_naive_action.py"),
            "--subject-dir",
            str(args.subject_dir),
            "--pose-dir",
            str(args.pose_dir),
            "--tl-code",
            str(args.tl_code),
        ]
        command.extend(["--manikin", str(args.manikin)])

    print(">>", " ".join(command))
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
