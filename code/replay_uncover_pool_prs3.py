#!/usr/bin/env python3
"""Replay uncover-action pool under fixed PRS and filter by sim uncover F1."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "assistive-gym-fem"))


def compute_fscore_uncover(initial_covered_status, final_covered_status):
    targ_uncov = 0
    nontarg_uncov = 0
    targ_cov = 0
    total_nontarg = 0
    for i in range(len(final_covered_status)):
        bod_point_type = final_covered_status[i][0]
        is_covered = final_covered_status[i][1]
        is_initially_covered = initial_covered_status[i][1]
        if bod_point_type == 1:
            if is_covered:
                targ_cov += 1
            else:
                targ_uncov += 1
        elif bod_point_type == 0 and is_initially_covered:
            total_nontarg += 1
            if not is_covered:
                nontarg_uncov += 1
    total_targ = targ_cov + targ_uncov
    if (total_targ + total_nontarg) == 0:
        return 0.0
    weight = total_targ / (total_targ + total_nontarg)
    penalties = []
    for i in range(1, nontarg_uncov + 1):
        penalty = i * weight
        penalties.append(penalty if penalty <= 1 else 1)
    tp = targ_uncov
    fp = float(np.sum(penalties))
    fn = targ_cov
    denom = tp + 0.5 * (fp + fn)
    if denom == 0:
        return 0.0
    return float(tp / denom)


def load_exclude_keys(exclude_json):
    if not exclude_json:
        return set()
    payload = json.load(open(Path(exclude_json).expanduser().resolve()))
    records = payload.get("records", payload if isinstance(payload, list) else [])
    return {(int(r["target_limb_code"]), str(r["seed"])) for r in records}


def list_pool_jobs(pool_raw_dir, exclude_keys):
    jobs = []
    for path in sorted(Path(pool_raw_dir).expanduser().resolve().glob("*.pkl")):
        parts = path.stem.split("_")
        tl = int(parts[0][2:])
        seed = parts[2]
        if (tl, seed) in exclude_keys:
            continue
        jobs.append({"path": str(path), "target_limb_code": tl, "seed": seed})
    return jobs


def load_completed_keys(out_raw_dir):
    """Resume helper: keys already written as tl{TL}_c0_{seed}_pid*.pkl."""
    done = {}
    out_raw = Path(out_raw_dir)
    if not out_raw.exists():
        return done
    for path in out_raw.glob("*.pkl"):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        try:
            tl = int(parts[0][2:])
            seed = parts[2]
        except Exception:
            continue
        done[(tl, seed)] = str(path)
    return done


def result_from_existing(path, threshold):
    path = Path(path)
    with open(path, "rb") as handle:
        raw = pickle.load(handle)
    info = raw.get("data_collection_info", {})
    f1 = float(info.get("sim_uncover_f1", float("nan")))
    executed = bool(info.get("execute_uncover_action", True))
    return {
        "status": "ok",
        "target_limb_code": int(raw.get("target_limb_code", path.name.split("_")[0][2:])),
        "seed": str(path.stem.split("_")[2]),
        "source_uncover_pkl": str(info.get("source_uncover_pkl", "")),
        "output_pkl": str(path),
        "sim_uncover_f1": f1,
        "execute_uncover_action": executed,
        "grasp_on_cloth_uncover": bool(info.get("grasp_on_cloth_uncover", True)),
        "pass_threshold": bool(f1 >= float(threshold) and executed),
        "resumed": True,
    }


def replay_one(job):
    from assistive_gym.envs.bu_gnn_util import get_body_points_from_obs, get_covered_status
    from assistive_gym.learn import make_env

    path = Path(job["path"])
    with open(path, "rb") as handle:
        raw = pickle.load(handle)

    seed = int(job["seed"])
    target = int(job["target_limb_code"])
    uncover_action = np.asarray(raw["uncover_action"], dtype=np.float32)

    env = make_env(job["env_name"], coop=False, seed=seed)
    try:
        env.set_env_variations(
            collect_data=False,
            blanket_pose_var=False,
            high_pose_var=False,
            body_shape_var=False,
        )
        env.set_singulate(True)
        env.set_target_limb_code(target)
        # Match April PRS3 recover-eval replay path: recover=True, then only uncover_step.
        env.set_recover(True)
        env.set_seed_val(seed)
        observation = env.reset()
        if isinstance(observation, (list, tuple)):
            human_pose = np.reshape(observation[0], (-1, 2))
        else:
            human_pose = np.reshape(observation, (-1, 2))
        if hasattr(env, "set_release_sim_steps"):
            env.set_release_sim_steps(post_release_steps=int(job["prs"]))
        cloth_initial, cloth_final, executed = env.uncover_step(uncover_action)
        body_info = env.get_human_body_info()
        all_body_points = get_body_points_from_obs(
            human_pose, target_limb_code=target, body_info=body_info
        )
        initial_status = get_covered_status(
            all_body_points, np.delete(np.asarray(cloth_initial[1]), 2, axis=1)
        )
        final_status = get_covered_status(
            all_body_points, np.delete(np.asarray(cloth_final[1]), 2, axis=1)
        )
        f1 = compute_fscore_uncover(initial_status, final_status)
        info = {}
        if hasattr(env, "get_info"):
            try:
                _obs, _ur, _rr, _done, info = env.get_info()
            except Exception:
                info = {}
        if not isinstance(info, dict):
            info = {}
        grasp_ok = bool(info.get("grasp_on_cloth_uncover", executed))

        # Keep uncover-eval layout expected by gnn_dc_recover F1 filter:
        # cloth_final == uncovered/intermediate state.
        info_payload = {
            "cloth_initial": cloth_initial,
            "cloth_final": cloth_final,
            "cloth_intermediate": cloth_final,
            "target_limb_code": int(target),
            "human_body_info": body_info,
            "grasp_on_cloth_uncover": grasp_ok,
        }
        out = {
            "recovering": False,
            "uncover_action": uncover_action,
            "recover_action": [],
            "human_pose": human_pose,
            "target_limb_code": int(target),
            "observation": observation,
            "info": info_payload,
            "sim_info": {"info": info_payload},
            "data_collection_info": {
                "source_uncover_pkl": str(path),
                "uncover_post_release_steps": int(job["prs"]),
                "sim_uncover_f1": float(f1),
                "execute_uncover_action": bool(executed),
                "grasp_on_cloth_uncover": grasp_ok,
            },
        }

        stem = f"tl{target}_c0_{seed}_pid{os.getpid()}"
        out_path = Path(job["out_raw_dir"]) / f"{stem}.pkl"
        with open(out_path, "wb") as handle:
            pickle.dump(out, handle)

        return {
            "status": "ok",
            "target_limb_code": int(target),
            "seed": str(seed),
            "source_uncover_pkl": str(path),
            "output_pkl": str(out_path),
            "sim_uncover_f1": float(f1),
            "execute_uncover_action": bool(executed),
            "grasp_on_cloth_uncover": grasp_ok,
            "pass_threshold": bool(f1 >= float(job["threshold"]) and executed),
        }
    except Exception as exc:
        import traceback

        return {
            "status": "error",
            "target_limb_code": int(target),
            "seed": str(seed),
            "source_uncover_pkl": str(path),
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=8),
            "pass_threshold": False,
        }
    finally:
        try:
            env.disconnect()
        except Exception:
            pass


def summarize(results, threshold):
    ok = [r for r in results if r.get("status") == "ok"]
    passed = [r for r in ok if r.get("pass_threshold")]
    by_tl = defaultdict(list)
    pass_by_tl = Counter()
    for r in ok:
        by_tl[int(r["target_limb_code"])].append(float(r["sim_uncover_f1"]))
    for r in passed:
        pass_by_tl[int(r["target_limb_code"])] += 1
    per_tl = {}
    for tl, vals in sorted(by_tl.items()):
        arr = np.asarray(vals, dtype=np.float64)
        per_tl[str(tl)] = {
            "n": int(len(arr)),
            "pass": int(pass_by_tl[tl]),
            "mean": float(arr.mean()) if len(arr) else float("nan"),
            "min": float(arr.min()) if len(arr) else float("nan"),
            "max": float(arr.max()) if len(arr) else float("nan"),
        }
    return {
        "threshold": float(threshold),
        "attempted": int(len(results)),
        "ok": int(len(ok)),
        "errors": int(sum(1 for r in results if r.get("status") != "ok")),
        "pass": int(len(passed)),
        "per_target_limb": per_tl,
        "records": results,
        "passed_records": passed,
    }


def write_markdown(path, summary):
    lines = [
        "# PRS3 Uncover Pool Replay Filter",
        "",
        f"- threshold: `{summary['threshold']}`",
        f"- attempted: {summary['attempted']}",
        f"- ok: {summary['ok']}",
        f"- errors: {summary['errors']}",
        f"- pass: {summary['pass']}",
        "",
        "| TL | n | pass | mean | min | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tl, row in summary["per_target_limb"].items():
        lines.append(
            f"| {tl} | {row['n']} | {row['pass']} | {row['mean']:.3f} | {row['min']:.3f} | {row['max']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-raw-dir", required=True)
    parser.add_argument("--exclude-eval-set", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prs", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.745)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--env-name", default="RobeReversible-v1")
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining")
    parser.add_argument("--resume", action="store_true",
                        help="Skip seeds that already have output PKLs under output-dir/raw.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_raw = out_dir / "raw"
    out_pass = out_dir / "raw_pass_0p745"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_pass.mkdir(parents=True, exist_ok=True)

    exclude = load_exclude_keys(args.exclude_eval_set)
    jobs = list_pool_jobs(args.pool_raw_dir, exclude)
    if args.limit > 0:
        jobs = jobs[: args.limit]

    results = []
    if args.resume:
        completed = load_completed_keys(out_raw)
        kept = []
        for job in jobs:
            key = (int(job["target_limb_code"]), str(job["seed"]))
            if key in completed:
                results.append(result_from_existing(completed[key], args.threshold))
            else:
                kept.append(job)
        print(f"[replay] resume: loaded {len(results)} existing, pending {len(kept)}")
        jobs = kept

    for job in jobs:
        job.update(
            {
                "prs": int(args.prs),
                "threshold": float(args.threshold),
                "env_name": args.env_name,
                "out_raw_dir": str(out_raw),
            }
        )

    print(
        f"[replay] pending={len(jobs)} exclude={len(exclude)} "
        f"prs={args.prs} threshold={args.threshold} workers={args.num_processes}"
    )
    t0 = time.time()
    workers = max(1, min(int(args.num_processes), max(1, len(jobs)))) if jobs else 1
    if not jobs:
        print("[replay] nothing pending")
    elif workers == 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs, 1):
            row = replay_one(job)
            results.append(row)
            print(
                f"[{i}/{len(jobs)}] tl={row.get('target_limb_code')} "
                f"f1={row.get('sim_uncover_f1', float('nan')):.3f} "
                f"pass={row.get('pass_threshold')} status={row.get('status')}"
            )
    else:
        with mp.Pool(processes=workers) as pool:
            for i, row in enumerate(pool.imap_unordered(replay_one, jobs), 1):
                results.append(row)
                print(
                    f"[{i}/{len(jobs)}] tl={row.get('target_limb_code')} "
                    f"f1={row.get('sim_uncover_f1', float('nan')):.3f} "
                    f"pass={row.get('pass_threshold')} status={row.get('status')}"
                )

    # Copy/link passers into raw_pass dir by rewriting path references only (copy files).
    for row in results:
        if not row.get("pass_threshold"):
            continue
        src = Path(row["output_pkl"])
        dst = out_pass / src.name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())

    summary = summarize(results, args.threshold)
    summary["elapsed_sec"] = float(time.time() - t0)
    summary["exclude_count"] = int(len(exclude))
    summary["pool_raw_dir"] = str(Path(args.pool_raw_dir).expanduser().resolve())
    summary["exclude_eval_set"] = str(args.exclude_eval_set)
    summary["prs"] = int(args.prs)

    with open(out_dir / "replay_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(out_dir / "replay_summary.md", summary)
    print(f"[replay] done pass={summary['pass']}/{summary['ok']} elapsed={summary['elapsed_sec']:.1f}s")
    print(f"[replay] summary={out_dir / 'replay_summary.md'}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
