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
    required = ["model_path", "graph_config", "env_var", "max_fevals", "num_rollouts"]
    missing = [k for k in required if k not in run]
    if missing:
        raise ValueError(f"Run {run.get('id','unknown')} missing required fields: {missing}")

    search_method = str(run.get("search_method", "")).strip().lower()
    if not search_method:
        search_label = str(run.get("search", "RandomSearch")).strip().lower()
        if "cma" in search_label:
            search_method = "cma"
        else:
            search_method = "random"

    if search_method not in {"random", "cma"}:
        raise ValueError(f"Run {run.get('id','unknown')} has invalid search_method={search_method}")

    cmd = [
        python_exec,
        str(run_robe_script),
        "--model-path", str(run["model_path"]),
        "--graph-config", str(run["graph_config"]),
        "--env-var", str(run["env_var"]),
        "--max-fevals", str(run["max_fevals"]),
        "--num-rollouts", str(run["num_rollouts"]),
        "--search-method", search_method,
        "--output-dir", str(output_dir),
    ]

    # Optional fields supported by current run_robe_sim_new_opt.py
    if "warm_start_strategy" in run:
        cmd += ["--warm-start-strategy", str(run["warm_start_strategy"])]

    if bool(run.get("feasible_only_best", False)):
        cmd += ["--feasible-only-best"]

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run experiment matrix for run_robe_sim_new_opt.py")
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
