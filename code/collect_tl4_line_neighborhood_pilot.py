#!/usr/bin/env python3
"""Line-neighborhood recover DC collector (varaware uncover seeds x 20 families).

Aligned with ``assistive_gym/gnn_dc_recover.py`` rollout protocol:

* jobs are ``(state, family_index)`` — ``--num-processes`` parallelizes **families**
  (default 20). Full run: ~1111 seeds x 20 ≈ 22k episodes.
* each job: ``make_env → reset → uncover → (F1 gate) → plan on live intermediate → recover``
* if live uncover F1 <= ``--min-uncover-f1`` (default 0), retry uncover for the
  same family (up to ``--max-uncover-retries``)
* default PRS = 3/3; PKL schema matches DC (+ top-level cloth_* for viz)

Soft-body ``saveState``/``restoreState`` is not used.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "assistive-gym-fem"))

from assistive_gym.envs.bu_gnn_util import (  # noqa: E402
    check_grasp_on_cloth,
    get_body_points_from_obs,
    get_covered_status,
    scale_action,
)
from cma_gnn_util import compute_fscore_uncover  # noqa: E402
from line_neighborhood_actions import (  # noqa: E402
    N_LINE_NEIGHBORHOOD_ACTIONS,
    build_tl4_pilot_action_set,
)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_source_manifest(
    manifest_path: Path,
    max_states: int,
    seed: int,
    target_limbs=None,
):
    """Load uncover-source manifest rows.

    ``max_states <= 0`` keeps all matching rows (e.g. full 1111 varaware pool).
    ``target_limbs`` None = all limbs in the manifest.
    """
    allow = None if target_limbs is None else {int(t) for t in target_limbs}
    rows = []
    with open(manifest_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tl = int(rec.get("target_limb", rec.get("target_limb_code", -1)))
            if allow is not None and tl not in allow:
                continue
            rows.append(rec)
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    if max_states > 0:
        rows = rows[: int(max_states)]
    return rows


def resolve_source_pkl(rec: dict, pool_root: Path) -> Path:
    # manifest may store relative path under eval condition
    for key in ("path", "source_path", "original_path"):
        raw = rec.get(key)
        if not raw:
            continue
        p = Path(raw)
        if p.is_file():
            return p
        cand = pool_root / raw
        if cand.is_file():
            return cand
    fname = rec.get("filename")
    if fname:
        hits = list(pool_root.rglob(fname))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Cannot resolve source pkl for record={rec}")


def parse_seed_from_name(name: str) -> int:
    parts = Path(name).name.split("_")
    # tl4_c40_<seed>_pid.... or remote_c_4_<seed>_...
    for i, part in enumerate(parts):
        if part.isdigit() and len(part) >= 8:
            return int(part)
        if part.startswith("c") and i + 1 < len(parts) and parts[i + 1].isdigit():
            # c40_<seed>
            if parts[i + 1].isdigit() and len(parts[i + 1]) >= 8:
                return int(parts[i + 1])
    # fallback: common uncover naming tl*_c*_<seed>_
    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2])
    raise ValueError(f"Cannot parse seed from {name}")


def compute_mode_tags(info, cloth_initial, cloth_intermediate, cloth_final, execute_recover):
    tags = {
        "execute_recover_action": bool(execute_recover),
        "sim_recover_reward": None,
        "sim_recover_f1": None,
    }
    try:
        tags["sim_recover_reward"] = float(info.get("recover_reward", info.get("reward", float("nan"))))
    except Exception:
        pass
    try:
        # best-effort F1 if present in info
        if "f1" in info:
            tags["sim_recover_f1"] = float(info["f1"])
        elif "recover_f1" in info:
            tags["sim_recover_f1"] = float(info["recover_f1"])
    except Exception:
        pass

    ci = np.asarray(cloth_intermediate[1] if isinstance(cloth_intermediate, (list, tuple)) else cloth_intermediate)
    cf = np.asarray(cloth_final[1] if isinstance(cloth_final, (list, tuple)) else cloth_final)
    if ci.ndim == 2 and cf.ndim == 2 and ci.shape == cf.shape:
        disp = np.linalg.norm(cf[:, :2] - ci[:, :2], axis=1)
        tags["mean_xy_disp"] = float(np.mean(disp))
        tags["p90_xy_disp"] = float(np.percentile(disp, 90))
        z_i = float(np.ptp(ci[:, 2])) if ci.shape[1] > 2 else float("nan")
        z_f = float(np.ptp(cf[:, 2])) if cf.shape[1] > 2 else float("nan")
        tags["z_range_initial"] = z_i
        tags["z_range_final"] = z_f
        tags["z_range_delta"] = float(z_f - z_i) if np.isfinite(z_i) and np.isfinite(z_f) else float("nan")
        # crude mode label
        mean_d = tags["mean_xy_disp"]
        if not execute_recover:
            tags["dynamics_mode"] = "null_invalid_grasp"
        elif mean_d < 0.02:
            tags["dynamics_mode"] = "insufficient_displacement"
        elif tags.get("z_range_delta", 0) < -0.01 and mean_d > 0.05:
            tags["dynamics_mode"] = "organized_or_partial_unfolding"
        elif mean_d > 0.08:
            tags["dynamics_mode"] = "large_drag_or_unfolding"
        else:
            tags["dynamics_mode"] = "partial_or_local_motion"
    else:
        tags["dynamics_mode"] = "unknown"
    return tags


N_FAMILY_ACTIONS = int(N_LINE_NEIGHBORHOOD_ACTIONS)


def _safe_fname(*parts: str) -> str:
    raw = "_".join(str(p) for p in parts)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)


def _close_env(env):
    if env is None:
        return
    try:
        env.disconnect()
    except Exception:
        pass
    try:
        env.close()
    except Exception:
        pass


def _body_points_from_env(env, observation, target_limb):
    if isinstance(observation, (list, tuple)):
        human_pose_now = np.reshape(observation[0], (-1, 2))
    else:
        human_pose_now = np.reshape(observation, (-1, 2))
    return np.asarray(
        get_body_points_from_obs(
            human_pose_now,
            target_limb_code=int(target_limb),
            body_info=env.get_human_body_info(),
        ),
        dtype=np.float64,
    )


def _live_uncover_f1(cloth_initial, cloth_intermediate, all_body_points):
    """Target-limb uncover F1 on live meshes (initial -> intermediate)."""
    try:
        ci = np.asarray(
            cloth_initial[1] if isinstance(cloth_initial, (list, tuple)) else cloth_initial,
            dtype=np.float64,
        )
        cm = np.asarray(
            cloth_intermediate[1]
            if isinstance(cloth_intermediate, (list, tuple))
            else cloth_intermediate,
            dtype=np.float64,
        )
        body = np.asarray(all_body_points, dtype=np.float64)
        if ci.ndim != 2 or cm.ndim != 2 or body.ndim != 2 or body.size == 0:
            return float("nan")
        if not np.isfinite(ci).all() or not np.isfinite(cm).all() or not np.isfinite(body).all():
            return float("nan")
        ci_2d = np.delete(ci, 2, axis=1) if ci.shape[1] >= 3 else ci[:, :2]
        cm_2d = np.delete(cm, 2, axis=1) if cm.shape[1] >= 3 else cm[:, :2]
        init_status = get_covered_status(body, ci_2d)
        mid_status = get_covered_status(body, cm_2d)
        return float(compute_fscore_uncover(init_status, mid_status))
    except Exception:
        return float("nan")


def _body_pose_ok(all_body_points, max_abs_xy=2.5):
    """Reject exploded / NaN human meshes that break viz and dynamics."""
    body = np.asarray(all_body_points, dtype=np.float64)
    if body.ndim != 2 or body.shape[0] < 4 or body.shape[1] < 2:
        return False
    if not np.isfinite(body).all():
        return False
    xy = body[:, :2]
    if float(np.max(np.abs(xy))) > float(max_abs_xy):
        return False
    return True


def collect_one_family(job):
    """One parallel worker unit: single (state, family) episode with its own env."""
    try:
        from assistive_gym.learn import make_env
    except ImportError:
        from learn import make_env

    (
        state_idx,
        action_index,
        source_pkl,
        seed,
        uncover_action,
        target_limb,
        source_f1,
        quality_band,
        out_raw,
        run_id,
        uncover_prs,
        recover_prs,
        action_seed,
        env_name,
        min_uncover_f1,
        max_uncover_retries,
    ) = job
    action_index = int(action_index)
    min_f1 = float(min_uncover_f1)
    max_retries = max(1, int(max_uncover_retries))
    uncover_action = np.asarray(uncover_action, dtype=np.float32).reshape(-1)
    if uncover_action.size != 4:
        return {
            "status": "skip",
            "reason": "bad_uncover_action",
            "state_idx": int(state_idx),
            "action_index": action_index,
            "saved": 0,
        }

    coop = "Human" in env_name
    uncover_retry_total = 0
    live_uncover_f1 = float("nan")
    uncover_attempts = 0
    on_grasp = False
    all_body_points = None
    cloth_i = cloth_m = cloth_final = None
    observation = uncover_reward = recover_reward = done = info = None
    execute_recover = False
    recover_action = None
    spec = {
        "action_index": action_index,
        "family": f"slot_{action_index}",
        "slot": f"slot_{action_index}",
    }
    env = None

    def _configure(env_obj):
        env_obj.set_env_variations(
            collect_data=True,
            blanket_pose_var=False,
            high_pose_var=False,
            body_shape_var=False,
        )
        env_obj.set_singulate(True)
        env_obj.set_target_limb_code(int(target_limb))
        env_obj.set_recover(True)
        env_obj.set_seed_val(int(seed))

    try:
        uncover_ok = False
        for uncover_try in range(max_retries):
            uncover_attempts = uncover_try + 1
            _close_env(env)
            env = make_env(env_name, coop=coop, seed=int(seed))
            _configure(env)
            observation = env.reset()
            if hasattr(env, "set_release_sim_steps"):
                env.set_release_sim_steps(post_release_steps=int(uncover_prs))
            cloth_i, cloth_m, execute_uncover = env.uncover_step(uncover_action)
            if not execute_uncover:
                uncover_retry_total += 1
                continue
            try:
                observation = env._get_obs()
            except Exception:
                pass
            all_body_points = _body_points_from_env(env, observation, target_limb)
            if not _body_pose_ok(all_body_points):
                uncover_retry_total += 1
                continue
            live_uncover_f1 = _live_uncover_f1(cloth_i, cloth_m, all_body_points)
            if not np.isfinite(live_uncover_f1) or live_uncover_f1 <= min_f1:
                uncover_retry_total += 1
                continue
            uncover_ok = True
            break

        if not uncover_ok:
            data_collection_info = {
                "action_mode": "line_neighborhood_pilot",
                "run_id": run_id,
                "source_uncover_pkl": str(source_pkl),
                "source_uncover_f1": None if source_f1 is None else float(source_f1),
                "source_quality_band": quality_band,
                "uncover_post_release_steps": int(uncover_prs),
                "recover_post_release_steps": int(recover_prs),
                "target_limb_code": int(target_limb),
                "seed": int(seed),
                "state_idx": int(state_idx),
                "action_index": int(action_index),
                "family": spec.get("family"),
                "slot": spec.get("slot"),
                "execute_uncover_action": False,
                "execute_recover_action": False,
                "dynamics_mode": "uncover_f1_gate_failed",
                "live_uncover_f1": None
                if not np.isfinite(live_uncover_f1)
                else float(live_uncover_f1),
                "min_uncover_f1": float(min_f1),
                "uncover_attempts": int(uncover_attempts),
                "rollout_protocol": "dc_recover_aligned",
                "parallel_unit": "state_family",
            }
            fname = _safe_fname(
                run_id,
                f"tl{int(target_limb)}",
                f"s{state_idx:03d}",
                f"a{action_index:02d}",
                seed,
                "UNCOVERFAIL.pkl",
            )
            with open(Path(out_raw) / fname, "wb") as handle:
                pickle.dump(
                    {
                        "recovering": True,
                        "uncover_action": uncover_action.tolist(),
                        "recover_action": [],
                        "data_collection_info": data_collection_info,
                    },
                    handle,
                )
            return {
                "status": "uncover_fail",
                "state_idx": int(state_idx),
                "action_index": action_index,
                "family": spec.get("family"),
                "saved": 1,
                "live_uncover_f1": live_uncover_f1,
                "uncover_attempts": uncover_attempts,
                "uncover_retry_total": uncover_retry_total,
                "seed": int(seed),
            }

        cloth_i_arr = np.asarray(cloth_m[1], dtype=np.float32)
        cloth_0_arr = np.asarray(cloth_i[1], dtype=np.float32)
        rng = np.random.default_rng(int(action_seed))
        action_set = build_tl4_pilot_action_set(
            cloth_intermediate=cloth_i_arr,
            uncover_action_policy=uncover_action,
            target_limb_code=int(target_limb),
            cloth_initial=cloth_0_arr,
            all_body_points=all_body_points,
            rng=rng,
        )
        spec = dict(action_set[int(action_index)])
        recover_action = np.asarray(spec.get("recover_action"), dtype=np.float32).reshape(-1)
        if recover_action.size != 4:
            raise ValueError(f"bad_recover_action_size:{recover_action.size}")

        _, on_grasp = check_grasp_on_cloth(
            scale_action(recover_action), np.asarray(cloth_m[1], dtype=np.float32)
        )
        if hasattr(env, "set_release_sim_steps"):
            env.set_release_sim_steps(post_release_steps=int(recover_prs))
        cloth_final, execute_recover = env.recover_step(recover_action)
        observation, uncover_reward, recover_reward, done, info = env.get_info()
        try:
            body_after = np.asarray(info.get("all_body_points"), dtype=np.float64)
            if _body_pose_ok(body_after):
                all_body_points = body_after
            elif isinstance(info, dict):
                info = dict(info)
                info["all_body_points"] = all_body_points
                info["body_points_fallback_pre_recover"] = True
        except Exception:
            pass

        mode_tags = compute_mode_tags(
            info if isinstance(info, dict) else {},
            cloth_i,
            cloth_m,
            cloth_final,
            execute_recover,
        )
        try:
            anchor_idx = [int(v) for v in list(getattr(env, "anchor_idx", []) or [])]
        except Exception:
            anchor_idx = []

        data_collection_info = {
            "action_mode": "line_neighborhood_pilot",
            "run_id": run_id,
            "source_uncover_pkl": str(source_pkl),
            "source_uncover_f1": None if source_f1 is None else float(source_f1),
            "source_quality_band": quality_band,
            "uncover_post_release_steps": int(uncover_prs),
            "recover_post_release_steps": int(recover_prs),
            "target_limb_code": int(target_limb),
            "seed": int(seed),
            "collection_seed": int(action_seed) * 100 + int(action_index),
            "state_idx": int(state_idx),
            "action_index": int(action_index),
            "family": spec.get("family"),
            "slot": spec.get("slot"),
            "action_spec": _jsonable({k: v for k, v in spec.items() if k != "debug"}),
            "anchor_idx": anchor_idx,
            "anchor_count": int(len(anchor_idx)),
            "grasp_on_cloth_pre_recover": bool(on_grasp),
            "execute_uncover_action": True,
            "execute_recover_action": bool(execute_recover),
            "generation_fallback": bool(spec.get("fallback") or spec.get("synthesized_hardneg")),
            "live_uncover_f1": float(live_uncover_f1),
            "min_uncover_f1": float(min_f1),
            "uncover_attempts": int(uncover_attempts),
            "rollout_protocol": "dc_recover_aligned",
            "bullet_save_state": False,
            "fresh_env_per_episode": True,
            "parallel_unit": "state_family",
            **mode_tags,
        }
        if isinstance(info, dict):
            info = dict(info)
            info.setdefault("anchor_idx", anchor_idx)
            info.setdefault("anchor_count", int(len(anchor_idx)))
            info.setdefault("cloth_initial", cloth_i)
            info.setdefault("cloth_intermediate", cloth_m)
            info.setdefault("cloth_final", cloth_final)
            if all_body_points is not None:
                info["all_body_points"] = all_body_points

        fname = _safe_fname(
            run_id,
            f"c_{int(target_limb)}",
            seed,
            int(action_seed) * 100 + int(action_index),
            f"s{state_idx:03d}",
            f"a{action_index:02d}",
            spec.get("slot", "act"),
        ) + ".pkl"
        with open(Path(out_raw) / fname, "wb") as handle:
            pickle.dump(
                {
                    "recovering": True,
                    "observation": observation,
                    "info": info,
                    "uncover_action": uncover_action.tolist(),
                    "recover_action": recover_action.tolist(),
                    "data_collection_info": data_collection_info,
                    "cloth_initial": cloth_i,
                    "cloth_intermediate": cloth_m,
                    "cloth_final": cloth_final,
                    "uncover_reward": uncover_reward,
                    "recover_reward": recover_reward,
                },
                handle,
            )
        return {
            "status": "ok",
            "state_idx": int(state_idx),
            "action_index": action_index,
            "family": spec.get("family"),
            "slot": spec.get("slot"),
            "saved": 1,
            "live_uncover_f1": float(live_uncover_f1),
            "uncover_attempts": uncover_attempts,
            "uncover_retry_total": uncover_retry_total,
            "execute_recover_action": bool(execute_recover),
            "dynamics_mode": data_collection_info.get("dynamics_mode"),
            "seed": int(seed),
            "file": fname,
        }
    except Exception as exc:
        data_collection_info = {
            "action_mode": "line_neighborhood_pilot",
            "run_id": run_id,
            "source_uncover_pkl": str(source_pkl),
            "target_limb_code": int(target_limb),
            "seed": int(seed),
            "state_idx": int(state_idx),
            "action_index": int(action_index),
            "family": spec.get("family"),
            "slot": spec.get("slot"),
            "execute_recover_action": False,
            "dynamics_mode": "collect_exception",
            "live_uncover_f1": None
            if not np.isfinite(live_uncover_f1)
            else float(live_uncover_f1),
            "uncover_attempts": int(uncover_attempts),
            "rollout_protocol": "dc_recover_aligned",
            "parallel_unit": "state_family",
            "error": str(exc),
            "action_spec": _jsonable({k: v for k, v in spec.items() if k != "debug"}),
        }
        fname = _safe_fname(
            run_id,
            f"tl{int(target_limb)}",
            f"s{state_idx:03d}",
            f"a{action_index:02d}",
            seed,
            spec.get("slot", "act"),
            "EXC.pkl",
        )
        with open(Path(out_raw) / fname, "wb") as handle:
            pickle.dump(
                {
                    "recovering": True,
                    "uncover_action": uncover_action.tolist(),
                    "recover_action": (
                        []
                        if recover_action is None
                        else np.asarray(recover_action, dtype=np.float32).reshape(-1).tolist()
                    ),
                    "data_collection_info": data_collection_info,
                },
                handle,
            )
        return {
            "status": "exception",
            "state_idx": int(state_idx),
            "action_index": action_index,
            "family": spec.get("family"),
            "saved": 1,
            "error": str(exc),
            "seed": int(seed),
        }
    finally:
        _close_env(env)


def main():
    parser = argparse.ArgumentParser(description="Line-neighborhood recover DC collector")
    parser.add_argument(
        "--pool-root",
        required=True,
        help="recover_source_varaware root (contains cma_evaluations/...)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="source_manifest.jsonl (default: Combined_1k_plus_boost_varaware)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-states",
        type=int,
        default=0,
        help="0 = all manifest seeds (full ~1111); >0 = subsample",
    )
    parser.add_argument(
        "--target-limbs",
        type=int,
        nargs="*",
        default=None,
        help="Optional limb filter, e.g. --target-limbs 4 10 11; default=all",
    )
    parser.add_argument("--state-sample-seed", type=int, default=4)
    parser.add_argument("--run-id", default="line_nb_recover_dc")
    parser.add_argument(
        "--uncover-prs",
        type=int,
        default=3,
        help="uncover post_release_steps (default 3)",
    )
    parser.add_argument(
        "--recover-prs",
        type=int,
        default=3,
        help="recover post_release_steps (default 3)",
    )
    parser.add_argument(
        "--max-actions-per-state",
        type=int,
        default=20,
        help="Families per uncover seed (default 20)",
    )
    parser.add_argument(
        "--min-uncover-f1",
        type=float,
        default=0.0,
        help="Retry uncover if live uncover F1 <= this (default 0.0 rejects exact zeros)",
    )
    parser.add_argument(
        "--max-uncover-retries",
        type=int,
        default=5,
        help="Max reset+uncover attempts per family before UNCOVERFAIL",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=20,
        help="Parallel workers over (state, family) jobs (default 20)",
    )
    parser.add_argument("--env-name", default="RobeReversible-v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    pool_root = Path(args.pool_root).expanduser().resolve()
    manifest = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else pool_root
        / "cma_evaluations"
        / "Combined_1k_plus_boost_varaware"
        / "source_manifest.jsonl"
    )
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_raw = out_dir / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)

    records = load_source_manifest(
        manifest,
        args.max_states,
        args.state_sample_seed,
        target_limbs=args.target_limbs,
    )
    n_actions = max(1, min(N_FAMILY_ACTIONS, int(args.max_actions_per_state)))
    print(
        f"[Manifest] states selected: {len(records)} from {manifest} "
        f"(target_limbs={args.target_limbs or 'all'})"
    )

    jobs = []
    skipped_resume = 0
    for state_idx, rec in enumerate(records):
        source_pkl = resolve_source_pkl(rec, pool_root)
        seed = int(rec.get("seed") or parse_seed_from_name(source_pkl.name))
        target_limb = int(rec.get("target_limb", rec.get("target_limb_code", -1)))
        if target_limb < 0:
            print(f"[Warn] missing target_limb in {source_pkl.name}, skip state {state_idx}")
            continue
        with open(source_pkl, "rb") as handle:
            raw = pickle.load(handle)
        uncover_action = raw.get("uncover_action")
        if uncover_action is None:
            print(f"[Warn] no uncover action in {source_pkl.name}, skip state {state_idx}")
            continue
        uncover_action = np.asarray(uncover_action, dtype=np.float32).reshape(-1).tolist()
        action_seed = int(args.state_sample_seed) * 100000 + state_idx
        for action_index in range(n_actions):
            if args.resume:
                existing = list(
                    out_raw.glob(f"{args.run_id}_*_s{state_idx:03d}_a{action_index:02d}_*.pkl")
                )
                if existing:
                    skipped_resume += 1
                    continue
            jobs.append(
                (
                    state_idx,
                    action_index,
                    str(source_pkl),
                    seed,
                    uncover_action,
                    target_limb,
                    rec.get("f1"),
                    rec.get("quality_band"),
                    str(out_raw),
                    args.run_id,
                    int(args.uncover_prs),
                    int(args.recover_prs),
                    action_seed,
                    args.env_name,
                    float(args.min_uncover_f1),
                    int(args.max_uncover_retries),
                )
            )

    print(
        f"[Jobs] {len(jobs)} family episodes "
        f"(~{len(records)} states x {n_actions} families"
        f"{'' if not args.resume else f', resume_skip={skipped_resume}'}; "
        f"workers={args.num_processes}, prs={args.uncover_prs}/{args.recover_prs}, "
        f"parallel_unit=state_family)"
    )

    results = []
    n_workers = max(1, min(int(args.num_processes), len(jobs))) if jobs else 1
    if n_workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            res = collect_one_family(job)
            results.append(res)
            print(res)
    else:
        import multiprocessing as mp

        with mp.Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(collect_one_family, jobs):
                results.append(res)
                print(res)

    family_counts = Counter()
    status_counts = Counter()
    saved_total = 0
    for res in results:
        saved_total += int(res.get("saved", 0))
        status_counts[str(res.get("status"))] += 1
        fam = res.get("family")
        if fam is not None:
            family_counts[str(fam)] += 1

    mode_counts = Counter()
    for pkl_path in out_raw.glob("*.pkl"):
        try:
            with open(pkl_path, "rb") as handle:
                raw = pickle.load(handle)
            mode = (raw.get("data_collection_info") or {}).get("dynamics_mode")
            if mode:
                mode_counts[str(mode)] += 1
        except Exception:
            pass

    summary = {
        "manifest": str(manifest),
        "pool_root": str(pool_root),
        "output_dir": str(out_dir),
        "states_requested": int(args.max_states),
        "states_selected": len(records),
        "family_jobs": len(jobs),
        "num_processes": int(args.num_processes),
        "workers_used": int(n_workers),
        "parallel_unit": "state_family",
        "saved_total": saved_total,
        "status_counts": dict(status_counts),
        "family_counts": dict(family_counts),
        "dynamics_mode_counts": dict(mode_counts),
        "uncover_prs": int(args.uncover_prs),
        "recover_prs": int(args.recover_prs),
        "min_uncover_f1": float(args.min_uncover_f1),
        "run_id": args.run_id,
        "results": results,
    }
    with open(out_dir / "collection_summary.json", "w") as handle:
        json.dump(_jsonable(summary), handle, indent=2)
    with open(out_dir / "collection_summary.md", "w") as handle:
        handle.write("# Line-Neighborhood Recover DC\n\n")
        handle.write(f"- family_jobs: {len(jobs)}\n")
        handle.write(f"- workers_used: {n_workers}\n")
        handle.write(f"- parallel_unit: state_family\n")
        handle.write(f"- n_families: {n_actions}\n")
        handle.write(f"- saved_total: {saved_total}\n")
        handle.write(f"- status_counts: `{dict(status_counts)}`\n")
        handle.write(f"- family_counts: `{dict(family_counts)}`\n")
        handle.write(f"- dynamics_mode_counts: `{dict(mode_counts)}`\n")
    print(json.dumps({k: summary[k] for k in summary if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
