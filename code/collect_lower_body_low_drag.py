#!/usr/bin/env python3
"""Collect lower-body constrained low-drag unfolding recover data.

Replay varaware uncover states, execute line-heuristic exact/noise recover with
online upper-body exposure cutoff (stop at current EE; no pullback), then accept
from final metrics only:

* accepted_low_drag_unfolding/raw
* dragging_contaminated/raw
* no_effective_unfolding/raw
* grasp_miss|too_short|simulation_failure/raw
* boundary_unsafe_sibling/raw
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "assistive-gym-fem"))

from assistive_gym.envs.bu_gnn_util import (  # noqa: E402
    check_grasp_on_cloth,
    scale_action,
)
from collect_tl4_line_neighborhood_pilot import (  # noqa: E402
    _body_points_from_env,
    _body_pose_ok,
    _close_env,
    _jsonable,
    _live_uncover_f1,
    _safe_fname,
    load_source_manifest,
    parse_seed_from_name,
    resolve_source_pkl,
)
from low_drag_actions import build_low_drag_action_set  # noqa: E402
from low_drag_metrics import (  # noqa: E402
    BUCKET_ACCEPTED,
    BUCKET_BOUNDARY,
    BUCKET_DRAGGING,
    BUCKET_GRASP_MISS,
    BUCKET_NO_UNFOLD,
    BUCKET_SIM_FAIL,
    BUCKET_TOO_SHORT,
    LOWER_BODY_TLS,
    action_length_world,
    bucket_for_result,
    classify_acceptance,
    grasp_to_cloth_edge_distance,
    grasp_to_corner_distance,
    lower_recovery_gain,
    lower_target_points,
    protected_cloth_motion,
    protected_region_node_mask,
    target_region_cloth_motion,
    upper_body_points,
    upper_exposure_delta,
)

ALL_BUCKETS = (
    BUCKET_ACCEPTED,
    BUCKET_DRAGGING,
    BUCKET_NO_UNFOLD,
    BUCKET_GRASP_MISS,
    BUCKET_TOO_SHORT,
    BUCKET_SIM_FAIL,
    BUCKET_BOUNDARY,
)


def _raw_dir(out_root: Path, bucket: str) -> Path:
    return Path(out_root) / bucket / "raw"


def _ensure_bucket_dirs(out_root: Path):
    for b in ALL_BUCKETS:
        _raw_dir(out_root, b).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "manifests").mkdir(parents=True, exist_ok=True)


DEFAULT_LOWER_TLS = list(LOWER_BODY_TLS)


def _human_pose_from_obs(observation):
    if isinstance(observation, (list, tuple)):
        return np.reshape(observation[0], (-1, 2))
    return np.reshape(observation, (-1, 2))


def _write_manifest_line(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_jsonable(rec)) + "\n")


def collect_one_episode(job):
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
        out_root,
        run_id,
        uncover_prs,
        recover_prs,
        action_seed,
        env_name,
        min_uncover_f1,
        max_uncover_retries,
        tau_online,
        tau_final,
        eta_lower_gain,
        monitor_stride,
        n_noise,
        boundary_delta,
        record_traj,
        min_action_length,
        require_lower_gain,
    ) = job

    out_root = Path(out_root)
    uncover_action = np.asarray(uncover_action, dtype=np.float32).reshape(-1)
    if uncover_action.size != 4:
        return {"status": "skip", "reason": "invalid_action", "saved": 0}

    coop = "Human" in env_name
    env = None
    live_uncover_f1 = float("nan")
    uncover_attempts = 0

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
        cloth_i = cloth_m = None
        observation = None
        all_body_points = None
        for uncover_try in range(max(1, int(max_uncover_retries))):
            uncover_attempts = uncover_try + 1
            _close_env(env)
            env = make_env(env_name, coop=coop, seed=int(seed))
            _configure(env)
            observation = env.reset()
            if hasattr(env, "set_release_sim_steps"):
                env.set_release_sim_steps(post_release_steps=int(uncover_prs))
            cloth_i, cloth_m, execute_uncover = env.uncover_step(uncover_action)
            if not execute_uncover:
                continue
            try:
                observation = env._get_obs()
            except Exception:
                pass
            all_body_points = _body_points_from_env(env, observation, target_limb)
            if not _body_pose_ok(all_body_points):
                continue
            live_uncover_f1 = _live_uncover_f1(cloth_i, cloth_m, all_body_points)
            if not np.isfinite(live_uncover_f1) or live_uncover_f1 <= float(min_uncover_f1):
                continue
            uncover_ok = True
            break

        if not uncover_ok:
            return {
                "status": "uncover_fail",
                "state_idx": int(state_idx),
                "action_index": int(action_index),
                "saved": 0,
                "accepted": False,
                "rejection_reason": "simulation_failure",
                "bucket": BUCKET_SIM_FAIL,
            }

        body_info = env.get_human_body_info()
        pose = _human_pose_from_obs(observation)
        upper_pts = upper_body_points(pose, body_info=body_info)
        lower_pts = lower_target_points(pose, int(target_limb), body_info=body_info)

        cloth_mid = np.asarray(cloth_m[1], dtype=np.float32)
        cloth_0 = np.asarray(cloth_i[1], dtype=np.float32)
        rng = np.random.default_rng(int(action_seed) * 100 + int(action_index))
        action_set = build_low_drag_action_set(
            cloth_intermediate=cloth_mid,
            uncover_action_policy=uncover_action,
            target_limb_code=int(target_limb),
            cloth_initial=cloth_0,
            rng=rng,
            n_noise=int(n_noise),
        )
        if int(action_index) >= len(action_set):
            return {"status": "skip", "reason": "bad_action_index", "saved": 0}
        spec = dict(action_set[int(action_index)])
        intended_policy = np.asarray(spec["recover_action"], dtype=np.float32).reshape(-1)
        intended_world = scale_action(intended_policy)
        grasp_xy = intended_world[:2]
        intended_release = intended_world[2:4]
        intended_len = float(spec.get("intended_action_length") or action_length_world(grasp_xy, intended_release))

        e0, _, _ = upper_exposure_delta(upper_pts, cloth_m, cloth_m)
        c0, _, _ = lower_recovery_gain(lower_pts, cloth_m, cloth_m)
        edge_d = grasp_to_cloth_edge_distance(grasp_xy, cloth_m)
        corner_d = grasp_to_corner_distance(grasp_xy, cloth_m)
        prot_mask = protected_region_node_mask(cloth_m, upper_pts)

        _, on_grasp = check_grasp_on_cloth(intended_world, cloth_mid)
        if hasattr(env, "set_release_sim_steps"):
            env.set_release_sim_steps(post_release_steps=int(recover_prs))
        if record_traj and hasattr(env, "set_cloth_trajectory_recording"):
            env.set_cloth_trajectory_recording(True, stride=max(1, int(monitor_stride)), max_frames=400)

        cloth_final, execute_recover, cutoff_meta = env.recover_step_with_exposure_cutoff(
            intended_policy,
            upper_body_points=upper_pts,
            tau_online=float(tau_online),
            monitor_stride=int(monitor_stride),
            record_exposure_timeline=True,
        )
        observation, uncover_reward, recover_reward, done, info = env.get_info()

        actual_policy = np.asarray(getattr(env, "recover_action", intended_policy), dtype=np.float32).reshape(-1)
        actual_world = scale_action(actual_policy)
        actual_release = actual_world[2:4]
        actual_len = action_length_world(actual_world[:2], actual_release)
        cutoff_frac = float(cutoff_meta.get("cutoff_fraction", 1.0))
        cutoff_triggered = bool(cutoff_meta.get("cutoff_triggered"))

        e0b, e_final, upper_delta_final = upper_exposure_delta(upper_pts, cloth_m, cloth_final)
        c0b, c_final, lower_gain = lower_recovery_gain(lower_pts, cloth_m, cloth_final)
        d_prot = protected_cloth_motion(cloth_m, cloth_final, prot_mask)
        d_tgt = target_region_cloth_motion(cloth_m, cloth_final, lower_pts)

        # Also estimate exposure at release from timeline last translate sample if present
        upper_at_release = e0b
        timeline = cutoff_meta.get("exposure_timeline") or []
        if timeline:
            upper_at_release = e0b + int(timeline[-1].get("newly_exposed_upper", 0))

        accepted, reject_reason = classify_acceptance(
            lower_gain=int(lower_gain),
            upper_delta_final=int(upper_delta_final),
            eta_lower_gain=float(eta_lower_gain),
            tau_final=float(tau_final),
            execute_recover=bool(execute_recover) and bool(on_grasp),
            min_actual_length=float(min_action_length),
            actual_length=actual_len,
            require_lower_gain=bool(require_lower_gain),
        )
        if not on_grasp or not execute_recover:
            accepted, reject_reason = False, "grasp_miss"
        accept_label = reject_reason if accepted else ""
        reject_reason = "" if accepted else reject_reason
        bucket = bucket_for_result(accepted, reject_reason or accept_label, "primary")

        pair_id = str(uuid.uuid4())
        traj_payload = None
        if hasattr(env, "_cloth_trajectory_payload"):
            try:
                traj_payload = env._cloth_trajectory_payload()
            except Exception:
                traj_payload = None

        try:
            anchor_idx = [int(v) for v in list(getattr(env, "anchor_idx", []) or [])]
        except Exception:
            anchor_idx = []

        data_collection_info = {
            "action_mode": "lower_body_low_drag_unfold",
            "dataset_name": "lower-body constrained low-drag unfolding dataset",
            "run_id": run_id,
            "source_uncover_pkl": str(source_pkl),
            "source_uncover_f1": None if source_f1 is None else float(source_f1),
            "source_quality_band": quality_band,
            "uncover_post_release_steps": int(uncover_prs),
            "recover_post_release_steps": int(recover_prs),
            "post_release_steps": int(recover_prs),
            "target_limb_code": int(target_limb),
            "seed": int(seed),
            "state_idx": int(state_idx),
            "action_index": int(action_index),
            "action_family": spec.get("family"),
            "family": spec.get("family"),
            "slot": spec.get("slot"),
            "noise_scale": spec.get("noise_scale"),
            "grasp": [float(x) for x in actual_world[:2]],
            "intended_release": [float(x) for x in intended_release],
            "intended_release_xy": [float(x) for x in intended_release],
            "actual_release": [float(x) for x in actual_release],
            "actual_release_xy": [float(x) for x in actual_release],
            "intended_action_length": float(intended_len),
            "actual_action_length": float(actual_len),
            "cutoff_fraction": float(cutoff_frac),
            "cutoff_triggered": bool(cutoff_triggered),
            "cutoff_step": int(cutoff_meta.get("cutoff_step") or cutoff_meta.get("last_safe_step") or 0),
            "last_safe_fraction": cutoff_meta.get("last_safe_fraction"),
            "cutoff_reason": cutoff_meta.get("cutoff_reason"),
            "cutoff_meta": _jsonable(cutoff_meta),
            "ee_pullback": False,
            "grasp_to_cloth_edge_distance": float(edge_d),
            "grasp_to_corner_distance": float(corner_d),
            "upper_exposed_start": int(e0b),
            "upper_exposed_at_release": int(upper_at_release),
            "upper_exposed_final": int(e_final),
            "upper_exposure_delta_final": int(upper_delta_final),
            "lower_metric_start": int(c0b),
            "lower_metric_final": int(c_final),
            "lower_recovery_gain": int(lower_gain),
            "protected_cloth_motion": float(d_prot) if d_prot == d_prot else None,
            "target_region_cloth_motion": float(d_tgt) if d_tgt == d_tgt else None,
            "accepted": bool(accepted),
            "accept_label": accept_label if accepted else None,
            "rejection_reason": reject_reason or None,
            "bucket": bucket,
            "pair_id": pair_id,
            "boundary_role": "primary",
            "live_uncover_f1": float(live_uncover_f1),
            "uncover_attempts": int(uncover_attempts),
            "anchor_idx": anchor_idx,
            "tau_online": float(tau_online),
            "tau_final": float(tau_final),
            "eta_lower_gain": float(eta_lower_gain),
            "require_lower_gain": bool(require_lower_gain),
            "min_action_length": float(min_action_length),
            "monitor_stride": int(monitor_stride),
            "simulation_config": {
                "uncover_prs": int(uncover_prs),
                "recover_prs": int(recover_prs),
                "tau_online": float(tau_online),
                "tau_final": float(tau_final),
                "eta_lower_gain": float(eta_lower_gain),
                "require_lower_gain": bool(require_lower_gain),
                "min_action_length": float(min_action_length),
                "monitor_stride": int(monitor_stride),
                "ee_pullback": False,
                "upper_body_tl": 13,
                "lower_tls": list(DEFAULT_LOWER_TLS),
            },
            "rollout_protocol": "low_drag_cutoff_stop_in_place",
            "action_spec": _jsonable({k: v for k, v in spec.items() if k != "debug"}),
        }

        if isinstance(info, dict):
            info = dict(info)
            info.setdefault("cloth_initial", cloth_i)
            info.setdefault("cloth_intermediate", cloth_m)
            info.setdefault("cloth_final", cloth_final)
            info.setdefault("anchor_idx", anchor_idx)
            if all_body_points is not None:
                info["all_body_points"] = all_body_points

        fname = _safe_fname(
            run_id,
            f"c_{int(target_limb)}",
            seed,
            f"s{int(state_idx):03d}",
            f"a{int(action_index):02d}",
            spec.get("slot", "act"),
            "ok" if accepted else (reject_reason or "rej"),
        ) + ".pkl"
        out_dir = _raw_dir(out_root, bucket)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "recovering": True,
            "observation": observation,
            "info": info,
            "uncover_action": uncover_action.tolist(),
            # Training action MUST be actual stop-in-place release.
            "recover_action": actual_policy.tolist(),
            "data_collection_info": data_collection_info,
            "cloth_initial": cloth_i,
            "cloth_intermediate": cloth_m,
            "cloth_final": cloth_final,
            "uncover_reward": uncover_reward,
            "recover_reward": recover_reward,
            "cloth_trajectory": traj_payload,
        }
        with open(out_dir / fname, "wb") as handle:
            pickle.dump(payload, handle)

        man_rec = {
            "file": fname,
            "bucket": bucket,
            "accepted": bool(accepted),
            "rejection_reason": reject_reason or None,
            "accept_label": accept_label if accepted else None,
            "target_limb_code": int(target_limb),
            "state_idx": int(state_idx),
            "action_index": int(action_index),
            "action_family": spec.get("family"),
            "pair_id": pair_id,
            "boundary_role": "primary",
            "cutoff_triggered": bool(cutoff_triggered),
            "actual_action_length": float(actual_len),
            "cutoff_fraction": float(cutoff_frac),
            "upper_exposure_delta_final": int(upper_delta_final),
            "lower_recovery_gain": int(lower_gain),
            "grasp_to_cloth_edge_distance": float(edge_d),
        }
        _write_manifest_line(out_root / "manifests" / ("%s.jsonl" % bucket), man_rec)
        _write_manifest_line(
            out_root / "manifests" / ("accepted.jsonl" if accepted else "rejected.jsonl"),
            man_rec,
        )

        # Boundary sibling: slightly longer than stop alpha (for viz / boundary eval only).
        boundary_saved = 0
        if cutoff_triggered and float(boundary_delta) > 0:
            alpha_long = min(1.0, float(cutoff_frac) + float(boundary_delta))
            if alpha_long > float(cutoff_frac) + 1e-4:
                g = np.asarray(grasp_xy, dtype=np.float64)
                r_int = np.asarray(intended_release, dtype=np.float64)
                r_long = g + alpha_long * (r_int - g)
                from recover_heuristics import world_to_policy_action

                long_policy = world_to_policy_action(
                    np.array([g[0], g[1], r_long[0], r_long[1]], dtype=np.float64)
                )
                _close_env(env)
                env = make_env(env_name, coop=coop, seed=int(seed))
                _configure(env)
                observation = env.reset()
                if hasattr(env, "set_release_sim_steps"):
                    env.set_release_sim_steps(post_release_steps=int(uncover_prs))
                cloth_i2, cloth_m2, ex_u = env.uncover_step(uncover_action)
                if ex_u:
                    if hasattr(env, "set_release_sim_steps"):
                        env.set_release_sim_steps(post_release_steps=int(recover_prs))
                    # Disable online cutoff so sibling reaches forced_alpha.
                    cf2, ex_r, meta2 = env.recover_step_with_exposure_cutoff(
                        long_policy,
                        upper_body_points=upper_pts,
                        tau_online=1e9,
                        monitor_stride=int(monitor_stride),
                        record_exposure_timeline=False,
                    )
                    obs2, ur2, rr2, done2, info2 = env.get_info()
                    act2 = np.asarray(getattr(env, "recover_action", long_policy), dtype=np.float32)
                    act2_w = scale_action(act2)
                    _, _, ud2 = upper_exposure_delta(upper_pts, cloth_m2, cf2)
                    _, _, lg2 = lower_recovery_gain(lower_pts, cloth_m2, cf2)
                    dci2 = dict(data_collection_info)
                    dci2.update(
                        {
                            "accepted": False,
                            "accept_label": None,
                            "rejection_reason": "boundary_longer_sibling",
                            "bucket": BUCKET_BOUNDARY,
                            "boundary_role": "unsafe_sibling",
                            "pair_id": pair_id,
                            "forced_alpha": float(alpha_long),
                            "upper_exposure_delta_final": int(ud2),
                            "lower_recovery_gain": int(lg2),
                            "cutoff_meta": _jsonable(meta2),
                            "actual_release": act2_w[2:4].tolist(),
                            "actual_release_xy": act2_w[2:4].tolist(),
                            "actual_action_length": float(
                                action_length_world(act2_w[:2], act2_w[2:4])
                            ),
                        }
                    )
                    fname2 = _safe_fname(
                        run_id,
                        f"c_{int(target_limb)}",
                        seed,
                        f"s{int(state_idx):03d}",
                        f"a{int(action_index):02d}",
                        "boundary",
                        f"a{alpha_long:.2f}",
                    ) + ".pkl"
                    bdir = _raw_dir(out_root, BUCKET_BOUNDARY)
                    bdir.mkdir(parents=True, exist_ok=True)
                    with open(bdir / fname2, "wb") as handle:
                        pickle.dump(
                            {
                                "recovering": True,
                                "observation": obs2,
                                "info": info2,
                                "uncover_action": uncover_action.tolist(),
                                "recover_action": act2.tolist(),
                                "data_collection_info": dci2,
                                "cloth_initial": cloth_i2,
                                "cloth_intermediate": cloth_m2,
                                "cloth_final": cf2,
                                "uncover_reward": ur2,
                                "recover_reward": rr2,
                            },
                            handle,
                        )
                    man2 = {
                        "file": fname2,
                        "bucket": BUCKET_BOUNDARY,
                        "accepted": False,
                        "rejection_reason": "boundary_longer_sibling",
                        "pair_id": pair_id,
                        "boundary_role": "unsafe_sibling",
                        "target_limb_code": int(target_limb),
                        "forced_alpha": float(alpha_long),
                    }
                    _write_manifest_line(out_root / "manifests" / ("%s.jsonl" % BUCKET_BOUNDARY), man2)
                    _write_manifest_line(out_root / "manifests" / "rejected.jsonl", man2)
                    boundary_saved = 1

        return {
            "status": "ok",
            "accepted": bool(accepted),
            "rejection_reason": reject_reason or None,
            "accept_label": accept_label if accepted else None,
            "bucket": bucket,
            "state_idx": int(state_idx),
            "action_index": int(action_index),
            "family": spec.get("family"),
            "target_limb_code": int(target_limb),
            "saved": 1 + boundary_saved,
            "file": fname,
            "upper_exposure_delta_final": int(upper_delta_final),
            "lower_recovery_gain": int(lower_gain),
            "actual_action_length": float(actual_len),
            "cutoff_fraction": float(cutoff_frac),
            "cutoff_triggered": bool(cutoff_triggered),
        }
    except Exception as exc:
        return {
            "status": "exception",
            "saved": 0,
            "error": str(exc),
            "state_idx": int(state_idx),
            "action_index": int(action_index),
            "accepted": False,
            "rejection_reason": "simulation_failure",
            "bucket": BUCKET_SIM_FAIL,
        }
    finally:
        _close_env(env)


def main():
    parser = argparse.ArgumentParser(
        description="Lower-body constrained low-drag unfolding collector"
    )
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-states", type=int, default=0, help="0=all")
    parser.add_argument(
        "--target-limbs",
        type=int,
        nargs="*",
        default=DEFAULT_LOWER_TLS,
    )
    parser.add_argument("--state-sample-seed", type=int, default=20260805)
    parser.add_argument("--run-id", default="lowdrag_lb")
    parser.add_argument("--uncover-prs", type=int, default=3)
    parser.add_argument("--recover-prs", type=int, default=3)
    parser.add_argument("--n-noise", type=int, default=3, help="noise actions per state (plus 1 exact)")
    parser.add_argument("--tau-online", type=float, default=2.0)
    parser.add_argument("--tau-final", type=float, default=30.0)
    parser.add_argument("--eta-lower-gain", type=float, default=3.0,
                        help="diagnostic ΔC_lower threshold; only rejects if --require-lower-gain")
    parser.add_argument(
        "--require-lower-gain",
        action="store_true",
        help="Legacy: reject no_effective_unfolding when ΔC_lower < eta (default: off, no-ops accepted)",
    )
    parser.add_argument("--min-action-length", type=float, default=0.02,
                        help="reject too_short if actual length below this (meters)")
    parser.add_argument("--monitor-stride", type=int, default=1,
                        help="check ΔE_upper every N translate steps (1 = every 5mm)")
    parser.add_argument("--boundary-delta", type=float, default=0.1,
                        help="extra alpha for unsafe sibling when cutoff triggers")
    parser.add_argument("--min-uncover-f1", type=float, default=0.0)
    parser.add_argument("--max-uncover-retries", type=int, default=5)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--env-name", default="RobeReversible-v1")
    parser.add_argument("--record-trajectory", action="store_true", default=True)
    parser.add_argument("--no-record-trajectory", action="store_true")
    parser.add_argument("--max-accepted", type=int, default=0,
                        help="Stop scheduling after this many accepted (0=unlimited)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if float(args.tau_online) >= float(args.tau_final):
        raise SystemExit("Require tau_online < tau_final")

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
    _ensure_bucket_dirs(out_dir)

    records = load_source_manifest(
        manifest,
        args.max_states,
        args.state_sample_seed,
        target_limbs=args.target_limbs,
    )
    n_actions = 1 + max(0, int(args.n_noise))
    record_traj = bool(args.record_trajectory) and not bool(args.no_record_trajectory)

    jobs = []
    for state_idx, rec in enumerate(records):
        source_pkl = resolve_source_pkl(rec, pool_root)
        seed = int(rec.get("seed") or parse_seed_from_name(source_pkl.name))
        target_limb = int(rec.get("target_limb", rec.get("target_limb_code", -1)))
        with open(source_pkl, "rb") as handle:
            raw = pickle.load(handle)
        uncover_action = raw.get("uncover_action")
        if uncover_action is None:
            continue
        uncover_action = np.asarray(uncover_action, dtype=np.float32).reshape(-1).tolist()
        action_seed = int(args.state_sample_seed) * 100000 + state_idx
        for action_index in range(n_actions):
            if args.resume:
                pat = f"{args.run_id}_*_s{state_idx:03d}_a{action_index:02d}_*.pkl"
                # NOTE: must materialize glob results — a bare glob iterator is always
                # truthy in bool()/any(), which would skip every job when --resume.
                already = False
                for b in ALL_BUCKETS:
                    if b == BUCKET_BOUNDARY:
                        continue
                    if next(_raw_dir(out_dir, b).glob(pat), None) is not None:
                        already = True
                        break
                if already:
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
                    str(out_dir),
                    args.run_id,
                    int(args.uncover_prs),
                    int(args.recover_prs),
                    action_seed,
                    args.env_name,
                    float(args.min_uncover_f1),
                    int(args.max_uncover_retries),
                    float(args.tau_online),
                    float(args.tau_final),
                    float(args.eta_lower_gain),
                    int(args.monitor_stride),
                    int(args.n_noise),
                    float(args.boundary_delta),
                    record_traj,
                    float(args.min_action_length),
                    bool(args.require_lower_gain),
                )
            )

    print(
        f"[LowDrag] states={len(records)} jobs={len(jobs)} "
        f"actions/state={n_actions} prs={args.uncover_prs}/{args.recover_prs} "
        f"tau={args.tau_online}/{args.tau_final} eta={args.eta_lower_gain} "
        f"require_lower_gain={bool(args.require_lower_gain)} "
        f"stride={args.monitor_stride} Lmin={args.min_action_length} "
        f"(no EE pullback; cutoff≠accept; no-ops accepted by default)"
    )

    results = []
    n_acc = 0
    n_workers = max(1, min(int(args.num_processes), len(jobs))) if jobs else 1
    if n_workers <= 1:
        for job in jobs:
            if args.max_accepted > 0 and n_acc >= int(args.max_accepted):
                break
            res = collect_one_episode(job)
            results.append(res)
            if res.get("accepted"):
                n_acc += 1
            print(res)
    else:
        import multiprocessing as mp

        with mp.Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(collect_one_episode, jobs):
                results.append(res)
                if res.get("accepted"):
                    n_acc += 1
                print(res)
                if args.max_accepted > 0 and n_acc >= int(args.max_accepted):
                    pool.terminate()
                    break

    status_counts = Counter(str(r.get("status")) for r in results)
    accept_counts = Counter()
    reason_counts = Counter()
    bucket_counts = Counter()
    tl_attempt = Counter()
    tl_accept = Counter()
    for r in results:
        if r.get("bucket"):
            bucket_counts[str(r.get("bucket"))] += 1
        if r.get("accepted"):
            accept_counts["accepted"] += 1
            tl_accept[int(r.get("target_limb_code", -1))] += 1
        elif r.get("status") == "ok":
            accept_counts["rejected"] += 1
            reason_counts[str(r.get("rejection_reason") or "unknown")] += 1
        if r.get("target_limb_code") is not None:
            tl_attempt[int(r.get("target_limb_code"))] += 1

    summary = {
        "dataset_name": "lower-body constrained low-drag unfolding dataset",
        "definition": (
            "heuristic/heuristic-noise + lower-body TL + ΔE_upper(final)<=tau_final "
            "+ L_actual>=Lmin; no-ops (low ΔC_lower) accepted by default; "
            "cutoff stop-in-place, no EE pullback"
        ),
        "manifest": str(manifest),
        "pool_root": str(pool_root),
        "output_dir": str(out_dir),
        "buckets": list(ALL_BUCKETS),
        "states_selected": len(records),
        "jobs": len(jobs),
        "results_n": len(results),
        "status_counts": dict(status_counts),
        "accept_counts": dict(accept_counts),
        "bucket_counts": dict(bucket_counts),
        "rejection_reason_counts": dict(reason_counts),
        "per_tl_attempted": dict(tl_attempt),
        "per_tl_accepted": dict(tl_accept),
        "tau_online": float(args.tau_online),
        "tau_final": float(args.tau_final),
        "eta_lower_gain": float(args.eta_lower_gain),
        "require_lower_gain": bool(args.require_lower_gain),
        "min_action_length": float(args.min_action_length),
        "monitor_stride": int(args.monitor_stride),
        "ee_pullback": False,
        "n_noise": int(args.n_noise),
        "uncover_prs": int(args.uncover_prs),
        "recover_prs": int(args.recover_prs),
        "run_id": args.run_id,
        "note": "Thresholds are provisional Stage-0 defaults until Stage-1 viz audit.",
    }
    with open(out_dir / "collection_summary.json", "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
