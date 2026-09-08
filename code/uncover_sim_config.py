"""Shared uncover simulator configuration for collection, replay, and CMA eval."""
from __future__ import annotations

import copy
import inspect
from typing import Any, Dict, Optional


DEFAULT_UNCOVER_SIM_CONFIG = {
    "env": "RobeReversible-v1",
    "singulate_layers": True,
    "recover": False,
    "replay_seed": True,
    "naive": False,
    "action_scale": [0.44, 1.05],
    "grasp_clipping_thres": 0.028,
    "lower_before_release_uncover": True,
    "release_threshold": 0.05,
    "pre_release_steps": 20,
    "post_release_steps": 3,
    "quiet_settle": False,
    "quiet_max_steps": 20,
    "quiet_speed_threshold": 0.15,
    "uncover_release_gravity_boost": False,
    "uncover_release_gravity_z": -39.24,
    "pull_phase_gravity_z": -9.81,
    "default_gravity_z": -9.81,
    "blanket_pose_var": False,
    "high_pose_var": False,
    "body_shape_var": False,
    "collect_data": True,
}


def pilot_uncover_sim_config(
    *,
    lower_before_release_uncover: bool,
    post_release_steps: int = 3,
    collect_data: bool = True,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_UNCOVER_SIM_CONFIG)
    cfg["lower_before_release_uncover"] = bool(lower_before_release_uncover)
    cfg["post_release_steps"] = int(post_release_steps)
    cfg["collect_data"] = bool(collect_data)
    return cfg


def fingerprint_sim_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a stable JSON-serializable fingerprint (subset of keys that affect dynamics)."""
    keys = [
        "env",
        "singulate_layers",
        "recover",
        "replay_seed",
        "lower_before_release_uncover",
        "release_threshold",
        "pre_release_steps",
        "post_release_steps",
        "quiet_settle",
        "quiet_max_steps",
        "quiet_speed_threshold",
        "uncover_release_gravity_boost",
        "uncover_release_gravity_z",
        "pull_phase_gravity_z",
        "default_gravity_z",
        "blanket_pose_var",
        "high_pose_var",
        "body_shape_var",
        "collect_data",
        "grasp_clipping_thres",
    ]
    return {k: cfg.get(k) for k in keys if k in cfg}


def assert_sim_config_match(expected: Dict[str, Any], actual: Dict[str, Any], context: str = "") -> None:
    exp = fingerprint_sim_config(expected)
    act = fingerprint_sim_config(actual)
    mismatches = []
    for key in sorted(set(exp) | set(act)):
        if exp.get(key) != act.get(key):
            mismatches.append(f"{key}: expected={exp.get(key)!r} actual={act.get(key)!r}")
    if mismatches:
        prefix = f"{context}: " if context else ""
        raise ValueError(prefix + "sim_config mismatch: " + "; ".join(mismatches))


def apply_uncover_sim_config(env, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply uncover collection/eval settings to a RobeReversibleEnv instance."""
    cfg = copy.deepcopy(cfg)
    collect_data = bool(cfg.get("collect_data", True))
    env.set_env_variations(
        collect_data=collect_data,
        blanket_pose_var=bool(cfg.get("blanket_pose_var", False)),
        high_pose_var=bool(cfg.get("high_pose_var", False)),
        body_shape_var=bool(cfg.get("body_shape_var", False)),
    )
    env.set_singulate(bool(cfg.get("singulate_layers", True)))
    replay_seed = bool(cfg.get("replay_seed", True))
    recover = bool(cfg.get("recover", False))
    if hasattr(env, "set_recover"):
        sig = inspect.signature(env.set_recover)
        if "replay_seed" in sig.parameters:
            env.set_recover(recover, replay_seed=replay_seed)
        else:
            env.set_recover(recover)
            if hasattr(env, "replay_seed"):
                env.replay_seed = replay_seed
    if hasattr(env, "set_uncover_release_lowering"):
        env.set_uncover_release_lowering(bool(cfg.get("lower_before_release_uncover", True)))
    if hasattr(env, "set_release_sim_steps"):
        env.set_release_sim_steps(
            pre_release_steps=int(cfg.get("pre_release_steps", 20)),
            post_release_steps=int(cfg.get("post_release_steps", 3)),
        )
    if hasattr(env, "set_release_quiet_settle"):
        env.set_release_quiet_settle(
            enabled=bool(cfg.get("quiet_settle", False)),
            speed_threshold=float(cfg.get("quiet_speed_threshold", 0.15)),
            max_steps=int(cfg.get("quiet_max_steps", 20)),
        )
    if hasattr(env, "set_uncover_release_gravity_boost"):
        env.set_uncover_release_gravity_boost(
            enabled=bool(cfg.get("uncover_release_gravity_boost", False)),
            gravity_z=float(cfg.get("uncover_release_gravity_z", -39.24)),
        )
    return fingerprint_sim_config(cfg)


def read_env_sim_fingerprint(env) -> Dict[str, Any]:
    return {
        "env": "RobeReversible-v1",
        "singulate_layers": bool(getattr(env, "singulate_layers", True)),
        "recover": bool(getattr(env, "recover", False)),
        "replay_seed": bool(getattr(env, "replay_seed", False)),
        "lower_before_release_uncover": bool(
            getattr(env, "lower_before_release_uncover", True)
        ),
        "release_threshold": float(getattr(env, "release_threshold", 0.05)),
        "pre_release_steps": int(getattr(env, "pre_release_steps", 20)),
        "post_release_steps": int(getattr(env, "post_release_steps", 50)),
        "quiet_settle": bool(getattr(env, "quiet_settle_enabled", False)),
        "quiet_max_steps": int(getattr(env, "quiet_settle_max_steps", 20)),
        "quiet_speed_threshold": float(getattr(env, "quiet_settle_speed_threshold", 0.15)),
        "uncover_release_gravity_boost": bool(
            getattr(env, "uncover_release_gravity_boost_enabled", False)
        ),
        "uncover_release_gravity_z": float(getattr(env, "uncover_release_gravity_z", -39.24)),
        "pull_phase_gravity_z": -9.81,
        "default_gravity_z": float(getattr(env, "default_gravity_z", -9.81)),
        "blanket_pose_var": bool(getattr(env, "blanket_pose_var", False)),
        "high_pose_var": bool(getattr(env, "high_pose_var", False)),
        "body_shape_var": bool(getattr(env, "body_shape_var", False)),
        "collect_data": bool(getattr(env, "collect_data", False)),
        "grasp_clipping_thres": 0.028,
    }


def build_data_collection_info(
    cfg: Dict[str, Any],
    *,
    seed: Optional[int] = None,
    target_limb_code: Optional[int] = None,
    dynamics_summary: Optional[Dict[str, Any]] = None,
    sample_id: Optional[str] = None,
    lowering_arm: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "sim_config": fingerprint_sim_config(cfg),
        "post_release_steps": int(cfg.get("post_release_steps", 3)),
        "quiet_settle": bool(cfg.get("quiet_settle", False)),
        "quiet_max_steps": int(cfg.get("quiet_max_steps", 20)),
        "quiet_speed_threshold": float(cfg.get("quiet_speed_threshold", 0.15)),
        "lower_before_release_uncover": bool(cfg.get("lower_before_release_uncover", True)),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if target_limb_code is not None:
        payload["target_limb_code"] = int(target_limb_code)
    if dynamics_summary is not None:
        payload["dynamics_summary"] = dynamics_summary
    if sample_id is not None:
        payload["sample_id"] = str(sample_id)
    if lowering_arm is not None:
        payload["lowering_arm"] = str(lowering_arm)
    return payload


def canonical_uncover_info_payload(env, info: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure uncovered cloth is stored consistently in cloth_final for uncover-only runs."""
    out = dict(info)
    if not getattr(env, "recover", False):
        intermediate = out.get("cloth_intermediate")
        final = out.get("cloth_final")
        if (not final or (isinstance(final, list) and len(final) == 0)) and intermediate:
            out["cloth_final"] = intermediate
        if isinstance(out.get("cloth_intermediate"), list) and len(out["cloth_intermediate"]) == 0:
            if out.get("cloth_final"):
                out["cloth_intermediate"] = out["cloth_final"]
    return out
