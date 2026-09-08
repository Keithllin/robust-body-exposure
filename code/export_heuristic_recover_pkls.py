#!/usr/bin/env python3
"""Export recover training PKLs by replaying line-heuristic actions from results.json."""

from __future__ import print_function

import argparse
import json
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np

from eval_recover_heuristics import apply_release_settings, worker_dict_to_args
from assistive_gym.learn import make_env


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-json",
        default=(
            "evaluation_sets/decoupled_eval_40x10_nonzero_uncover_line_heuristic_eval/results.json"
        ),
    )
    parser.add_argument(
        "--output-raw-dir",
        default="DATASETS/Recover_Data/TL_All_Recover_heuristic_prs3_field_400/raw",
    )
    parser.add_argument("--num-processes", type=int, default=16)
    parser.add_argument("--post-release-steps", type=int, default=3)
    parser.add_argument(
        "--patch-results",
        action="store_true",
        help="Normalize result paths after moving a results file between machines.",
    )
    return parser.parse_args()


def patch_results_paths(results_path):
    rows = json.load(open(results_path))
    for row in rows:
        row["pkl"] = str(Path(row["pkl"]).expanduser())
    json.dump(rows, open(results_path, "w"), indent=2)
    return rows


def build_worker_args(post_release_steps):
    return worker_dict_to_args(
        {
            "mode": "line",
            "env_name": "RobeReversible-v1",
            "blanket_pose_var": False,
            "high_pose_var": False,
            "body_shape_var": False,
            "singulate_layers": True,
            "save_images": False,
            "output_format": "png",
            "image_dir": "",
            "post_release_steps": int(post_release_steps),
            "quiet_settle": False,
            "quiet_max_steps": 20,
            "quiet_speed_threshold": 0.15,
            "uncover_release_gravity_boost": False,
            "uncover_release_gravity_z": -39.24,
        }
    )


def export_one(job):
    idx, total, row, args_dict, out_raw_str = job
    args = worker_dict_to_args(args_dict)
    out_raw = Path(out_raw_str)
    pkl = Path(row["pkl"]).expanduser().resolve()
    if not pkl.exists():
        return idx, row.get("eval_id"), "missing_pkl", str(pkl)

    seed = int(row["seed"])
    tl = int(row["target_limb_code"])
    uncover = np.asarray(row["uncover_action_policy"], dtype=np.float32)
    recover = np.asarray(row["recover_action_policy"], dtype=np.float32)

    env = make_env(args.env_name, coop=False, seed=seed)
    env.set_env_variations(False, False, False, False)
    env.set_singulate(True)
    env.set_target_limb_code(tl)
    env.set_recover(True)
    env.set_seed_val(seed)
    try:
        env.reset()
        apply_release_settings(env, args)
        env.uncover_step(uncover)
        _, execute_recover_action = env.recover_step(recover)
        observation, _, _, _, info = env.get_info()
        try:
            anchor_idx = [int(v) for v in list(getattr(env, "anchor_idx", []) or [])]
        except Exception:
            anchor_idx = []
        if "anchor_idx" not in info:
            info["anchor_idx"] = anchor_idx
            info["anchor_count"] = int(len(anchor_idx))
        payload = {
            "recovering": True,
            "observation": observation,
            "info": info,
            "uncover_action": uncover,
            "recover_action": recover,
            "data_collection_info": {
                "post_release_steps": int(args.post_release_steps),
                "action_mode": "line_heuristic",
                "heuristic_mode": "line",
                "eval_id": row.get("eval_id"),
                "target_limb_code": tl,
                "seed": seed,
                "source_uncover_pkl": str(pkl),
                "source_results_json": str(row.get("source_results_json", "")),
                "execute_recover_action": bool(execute_recover_action),
                "anchor_idx": anchor_idx,
                "anchor_count": int(len(anchor_idx)),
            },
        }
        out_path = out_raw / ("heur_prs3_%s.pkl" % row["eval_id"])
        with open(out_path, "wb") as handle:
            pickle.dump(payload, handle)
        return idx, row.get("eval_id"), "ok", str(out_path)
    except Exception as exc:
        return idx, row.get("eval_id"), "error", "%s: %s" % (type(exc).__name__, exc)
    finally:
        try:
            env.disconnect()
        except Exception:
            try:
                env.close()
            except Exception:
                pass


def main():
    args = parse_args()
    results_path = Path(args.results_json).expanduser().resolve()
    out_raw = Path(args.output_raw_dir).expanduser().resolve()
    out_raw.mkdir(parents=True, exist_ok=True)

    if args.patch_results:
        rows = patch_results_paths(results_path)
    else:
        rows = json.load(open(results_path))

    missing = [row for row in rows if not Path(row["pkl"]).expanduser().resolve().exists()]
    if missing:
        raise FileNotFoundError(
            "missing %d source pkls; first=%s" % (len(missing), missing[0]["pkl"])
        )

    args_dict = {
        "mode": "line",
        "env_name": "RobeReversible-v1",
        "blanket_pose_var": False,
        "high_pose_var": False,
        "body_shape_var": False,
        "singulate_layers": True,
        "save_images": False,
        "output_format": "png",
        "image_dir": "",
        "post_release_steps": int(args.post_release_steps),
        "quiet_settle": False,
        "quiet_max_steps": 20,
        "quiet_speed_threshold": 0.15,
        "uncover_release_gravity_boost": False,
        "uncover_release_gravity_z": -39.24,
    }
    for row in rows:
        row["source_results_json"] = str(results_path)

    jobs = [
        (idx + 1, len(rows), row, args_dict, str(out_raw))
        for idx, row in enumerate(rows)
    ]
    workers = min(int(args.num_processes), len(jobs))

    ctx = mp.get_context("fork")
    ok = err = 0
    with ctx.Pool(processes=workers) as pool:
        for idx, eval_id, status, msg in pool.imap_unordered(export_one, jobs):
            if status == "ok":
                ok += 1
            else:
                err += 1
                print("[%d/%d] %s %s: %s" % (idx, len(rows), eval_id, status, msg))
            if idx % 25 == 0 or idx == len(rows):
                print("progress %d/%d ok=%d err=%d" % (idx, len(rows), ok, err))

    print(
        "DONE ok=%d err=%d raw_count=%d output=%s"
        % (ok, err, len(list(out_raw.glob("*.pkl"))), out_raw)
    )


if __name__ == "__main__":
    main()
