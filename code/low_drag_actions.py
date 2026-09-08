#!/usr/bin/env python3
"""Line-heuristic exact + joint-noise actions for low-drag unfolding collection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from assistive_gym.envs.bu_gnn_util import scale_action
from line_neighborhood_actions import (
    NOISE_SCALES,
    _blanket_width,
    _perturb_line_action,
    _xy,
    exact_line_action,
)


def build_low_drag_action_set(
    cloth_intermediate: np.ndarray,
    uncover_action_policy: Sequence[float],
    target_limb_code: int,
    cloth_initial: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    n_noise: int = 3,
    noise_scales: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Return 1 exact + n_noise heuristic_noise actions (joint trajectory noise)."""
    rng = rng or np.random.default_rng(0)
    if noise_scales is None:
        # Cycle small/medium for dense sampling (stage2/3); avoid large by default.
        cycle = ("small", "small", "medium")
        noise_scales = [cycle[i % len(cycle)] for i in range(max(0, int(n_noise)))]

    base = exact_line_action(
        cloth_intermediate=cloth_intermediate,
        uncover_action_policy=uncover_action_policy,
        target_limb_code=int(target_limb_code),
        cloth_initial=cloth_initial,
    )
    base["family"] = "heuristic_exact"
    base["slot"] = "heuristic_exact"
    pick = _xy(base["pick_xy"])
    place = _xy(base["place_xy"])
    base["intended_action_length"] = float(np.linalg.norm(place - pick))
    base["grasp_perturbation"] = 0.0
    base["direction_perturbation_deg"] = 0.0

    out = [base]
    blanket_w = _blanket_width(cloth_intermediate)
    for i, scale_name in enumerate(noise_scales[: int(n_noise)]):
        if scale_name not in NOISE_SCALES:
            scale_name = "medium"
        pert = _perturb_line_action(
            base, cloth_intermediate, rng, scale_name, blanket_w
        )
        if pert is None:
            # Fallback: slight length shrink of exact
            pick0 = _xy(base["pick_xy"])
            place0 = _xy(base["place_xy"])
            place1 = pick0 + 0.85 * (place0 - pick0)
            from line_neighborhood_actions import _policy_from_world

            policy = _policy_from_world(pick0, place1)
            pert = {
                "family": "heuristic_noise",
                "slot": "heuristic_noise_%s_fallback" % scale_name,
                "recover_action": policy,
                "action_world": scale_action(policy),
                "pick_xy": pick0,
                "place_xy": place1.astype(np.float32),
                "noise_scale": scale_name,
                "fallback": True,
            }
        else:
            pert["family"] = "heuristic_noise"
            pert["slot"] = "heuristic_noise_%s_%d" % (scale_name, i)
        p = _xy(pert["pick_xy"])
        r = _xy(pert["place_xy"])
        pert["intended_action_length"] = float(np.linalg.norm(r - p))
        pert["grasp_perturbation"] = float(np.linalg.norm(p - pick))
        move0 = place - pick
        move1 = r - p
        if np.linalg.norm(move0) > 1e-8 and np.linalg.norm(move1) > 1e-8:
            u0 = move0 / np.linalg.norm(move0)
            u1 = move1 / np.linalg.norm(move1)
            ang = float(np.degrees(np.arccos(np.clip(np.dot(u0, u1), -1.0, 1.0))))
        else:
            ang = 0.0
        pert["direction_perturbation_deg"] = ang
        out.append(pert)
    return out
