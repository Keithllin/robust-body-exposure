import argparse
import json
import subprocess
from pathlib import Path


def resolve_output_dirs(run, outputs_root):
    run_id = run.get("id", "unknown")

    output_dir = run.get("output_dir", None)
    raw_dir = run.get("raw_dir", None)

    if output_dir:
        out_dir = Path(output_dir)
    elif raw_dir and "ABS/PATH" not in str(raw_dir):
        out_dir = Path(raw_dir).parent
    else:
        out_dir = outputs_root / run_id

    out_dir = out_dir.resolve()
    resolved_raw = (out_dir / "raw").resolve()
    return out_dir, resolved_raw


def build_cmd(python_exec, run_robe_script, run, output_dir):
    script_name = Path(run_robe_script).name
    is_joint = "joint_opt" in script_name

    required = ["graph_config", "env_var", "max_fevals", "num_rollouts"]
    if is_joint:
        required += ["uncover_model_path", "recover_model_path"]
    else:
        required += ["model_path"]
    missing = [k for k in required if k not in run]
    if missing:
        raise ValueError(f"Run {run.get('id','unknown')} missing required fields: {missing}")

    cmd = [python_exec, str(run_robe_script)]
    if is_joint:
        cmd += [
            "--uncover-model-path", str(run["uncover_model_path"]),
            "--recover-model-path", str(run["recover_model_path"]),
            "--graph-config", str(run["graph_config"]),
            "--env-var", str(run["env_var"]),
            "--max-fevals", str(run["max_fevals"]),
            "--num-rollouts", str(run["num_rollouts"]),
            "--output-dir", str(output_dir),
        ]
    else:
        search_method = str(run.get("search_method", "")).strip().lower()
        if not search_method:
            search_label = str(run.get("search", "RandomSearch")).strip().lower()
            if "cma" in search_label:
                search_method = "cma"
            else:
                search_method = "random"

        if search_method not in {"random", "cma"}:
            raise ValueError(f"Run {run.get('id','unknown')} has invalid search_method={search_method}")

        cmd += [
            "--model-path", str(run["model_path"]),
            "--graph-config", str(run["graph_config"]),
            "--env-var", str(run["env_var"]),
            "--max-fevals", str(run["max_fevals"]),
            "--num-rollouts", str(run["num_rollouts"]),
            "--search-method", search_method,
            "--output-dir", str(output_dir),
        ]

    value_flags = {
        "optimization_mode": "--optimization-mode",
        "uncover_max_fevals": "--uncover-max-fevals",
        "recover_max_fevals": "--recover-max-fevals",
        "uncover_f1_threshold": "--uncover-f1-threshold",
        "screen_uncover_f1_threshold": "--screen-uncover-f1-threshold",
        "uncover_weight": "--uncover-weight",
        "recover_weight": "--recover-weight",
        "popsize": "--popsize",
        "sigma": "--sigma",
        "recover_search_method": "--recover-search-method",
        "recover_warm_start_strategy": "--recover-warm-start-strategy",
        "outer_init_source": "--outer-init-source",
        "baseline_raw_dir": "--baseline-raw-dir",
        "outer_baseline_seeds": "--outer-baseline-seeds",
        "outer_random_seeds": "--outer-random-seeds",
        "baseline_seed_selection": "--baseline-seed-selection",
        "inner_baseline_recover_topk": "--inner-baseline-recover-topk",
        "arg_seed": "--arg-seed",
        "target_limb_code": "--target-limb-code",
    }
    for key, flag in value_flags.items():
        if key in run and run[key] is not None:
            cmd += [flag, str(run[key])]

    bool_flags = {
        "feasible_only_best": "--feasible-only-best",
        "recover_feasible_only_best": "--recover-feasible-only-best",
        "inner_include_baseline_recover": "--inner-include-baseline-recover",
    }
    for key, flag in bool_flags.items():
        if bool(run.get(key, False)):
            cmd.append(flag)

    if run.get("inner_include_baseline_recover") is False and "inner_include_baseline_recover" in run:
        cmd.append("--no-inner-include-baseline-recover")

    if not is_joint and "warm_start_strategy" in run:
        cmd += ["--warm-start-strategy", str(run["warm_start_strategy"])]

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run experiment matrix for run_robe_sim_new_opt.py or run_robe_sim_joint_opt.py")
    parser.add_argument("--matrix", required=True, help="Path to matrix json")
    parser.add_argument("--python", default="python", help="Python executable")
    parser.add_argument("--script", default="run_robe_sim_new_opt.py", help="Path to run_robe script")
    parser.add_argument("--outputs-root", default="matrix_outputs", help="Directory for per-run outputs when raw_dir/output_dir is not provided")
    parser.add_argument("--resolved-matrix", default="", help="Write resolved matrix json (with concrete raw_dir/output_dir)")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands")
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    run_robe_script = Path(args.script)
    outputs_root = Path(args.outputs_root)

    cfg = json.loads(matrix_path.read_text())
    runs = cfg.get("runs", [])
    if not runs:
        raise RuntimeError("No runs found in matrix file")

    resolved_runs = []

    for i, run in enumerate(runs, start=1):
        out_dir, resolved_raw = resolve_output_dirs(run, outputs_root)
        run["output_dir"] = str(out_dir)
        run["raw_dir"] = str(resolved_raw)
        resolved_runs.append(run)

        if run.get("enabled", True) is False:
            print(f"[{i}/{len(runs)}] skip {run.get('id','unknown')} (enabled=false)")
            continue

        run_id = run.get("id", f"run_{i}")
        cmd = build_cmd(args.python, run_robe_script, run, out_dir)

        print("=" * 80)
        print(f"[{i}/{len(runs)}] RUN {run_id}")
        print("CMD:", " ".join(cmd))

        if args.dry_run:
            continue

        completed = subprocess.run(cmd)
        if completed.returncode != 0:
            raise RuntimeError(f"Run failed: {run_id}, returncode={completed.returncode}")

    resolved_cfg = {"runs": resolved_runs}
    resolved_path = Path(args.resolved_matrix) if args.resolved_matrix else matrix_path.with_name(matrix_path.stem + ".resolved.json")
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(resolved_cfg, indent=2))
    print(f"Wrote resolved matrix: {resolved_path}")

    print("All enabled runs completed.")


if __name__ == "__main__":
    main()
