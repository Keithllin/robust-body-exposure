"""Line-heuristic-centered recover action families for unfolding pilot collection.

Samples joint (grasp, release) actions conditioned on intermediate cloth state.
Inverse actions are not used as success centers (rebound-sensitive).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from assistive_gym.envs.bu_gnn_util import check_grasp_on_cloth, scale_action
from recover_heuristics import (
    compute_line_stacking_recover_action_from_states,
    find_cloth_bbox_corners,
    world_to_policy_action,
)

# Blanket nominal width used for relative noise (meters).
DEFAULT_BLANKET_WIDTH = 0.88

# small / medium / large relative scales: (grasp_frac, release_frac, angle_deg)
NOISE_SCALES = {
    "small": (0.01, 0.02, 5.0),
    "medium": (0.03, 0.05, 15.0),
    "large": (0.06, 0.10, 30.0),
}


def _xy(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32).reshape(-1)
    return p[:2].astype(np.float32)


def _blanket_width(cloth_xyz: np.ndarray) -> float:
    xy = np.asarray(cloth_xyz, dtype=np.float32)[:, :2]
    wh = np.ptp(xy, axis=0)
    w = float(max(wh[0], wh[1], 1e-3))
    return w if np.isfinite(w) else DEFAULT_BLANKET_WIDTH


def _policy_from_world(pick_xy: np.ndarray, place_xy: np.ndarray) -> np.ndarray:
    action_world = np.array(
        [float(pick_xy[0]), float(pick_xy[1]), float(place_xy[0]), float(place_xy[1])],
        dtype=np.float32,
    )
    return world_to_policy_action(action_world)


def _on_cloth(policy: np.ndarray, cloth_xyz: np.ndarray) -> bool:
    _, ok = check_grasp_on_cloth(scale_action(policy), np.asarray(cloth_xyz, dtype=np.float32))
    return bool(ok)


def _snap_grasp_to_cloth(pick_xy: np.ndarray, cloth_xyz: np.ndarray) -> Tuple[np.ndarray, int]:
    cloth = np.asarray(cloth_xyz, dtype=np.float32)
    d = np.linalg.norm(cloth[:, :2] - pick_xy[None, :], axis=1)
    idx = int(np.argmin(d))
    return cloth[idx, :2].astype(np.float32), idx


def exact_line_action(
    cloth_intermediate: np.ndarray,
    uncover_action_policy: Sequence[float],
    target_limb_code: int,
    cloth_initial: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    policy, world, pick, place, force, debug = compute_line_stacking_recover_action_from_states(
        cloth_intermediate,
        uncover_action_policy,
        target_limb_code,
        cloth_initial_positions=cloth_initial,
        line_width=0.06,
    )
    return {
        "family": "exact_line",
        "slot": "exact_line",
        "recover_action": np.asarray(policy, dtype=np.float32),
        "action_world": np.asarray(world, dtype=np.float32),
        "pick_xy": _xy(pick),
        "place_xy": _xy(place),
        "force_direction": np.asarray(force, dtype=np.float32)[:2],
        "debug": debug,
    }


def _perturb_line_action(
    base: Dict[str, Any],
    cloth_xyz: np.ndarray,
    rng: np.random.Generator,
    scale_name: str,
    blanket_w: float,
    max_tries: int = 40,
) -> Optional[Dict[str, Any]]:
    g_frac, r_frac, ang_deg = NOISE_SCALES[scale_name]
    pick0 = _xy(base["pick_xy"])
    place0 = _xy(base["place_xy"])
    move0 = place0 - pick0
    dist0 = float(np.linalg.norm(move0))
    if dist0 < 1e-6:
        return None
    theta0 = float(math.atan2(move0[1], move0[0]))

    for _ in range(max_tries):
        pick = pick0 + rng.normal(0.0, g_frac * blanket_w, size=2).astype(np.float32)
        pick, _ = _snap_grasp_to_cloth(pick, cloth_xyz)

        d_theta = math.radians(float(rng.uniform(-ang_deg, ang_deg)))
        # also jitter distance by release fraction
        dist = dist0 * float(1.0 + rng.normal(0.0, r_frac))
        dist = float(np.clip(dist, 0.05 * blanket_w, 0.55 * blanket_w))
        theta = theta0 + d_theta
        place = pick + dist * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        # small isotropic jitter on place endpoint
        place = place + rng.normal(0.0, 0.5 * r_frac * blanket_w, size=2).astype(np.float32)

        policy = _policy_from_world(pick, place)
        if not _on_cloth(policy, cloth_xyz):
            continue
        return {
            "family": f"line_noise_{scale_name}",
            "slot": f"line_noise_{scale_name}",
            "recover_action": policy,
            "action_world": scale_action(policy),
            "pick_xy": pick,
            "place_xy": place.astype(np.float32),
            "force_direction": np.array([math.cos(theta), math.sin(theta)], dtype=np.float32),
            "noise_scale": scale_name,
            "theta_deg": float(math.degrees(theta)),
            "move_dist": dist,
            "base_family": "exact_line",
        }
    return None


def _overlap_ranked_indices(
    cloth_xyz: np.ndarray,
    grid_size: float = 0.05,
    min_count: int = 3,
) -> List[Tuple[float, int, Dict[str, float]]]:
    """Return (score, vertex_idx, meta) sorted descending for overlap/layer candidates."""
    cloth = np.asarray(cloth_xyz, dtype=np.float32)
    xy = cloth[:, :2]
    xy_min = np.min(xy, axis=0)
    cell_idx = np.floor((xy - xy_min[None, :]) / float(grid_size)).astype(np.int64)
    groups: Dict[Tuple[int, int], List[int]] = {}
    for i, cell in enumerate(cell_idx):
        key = (int(cell[0]), int(cell[1]))
        groups.setdefault(key, []).append(i)

    # boundary proximity via AABB
    min_xy = np.min(xy, axis=0)
    max_xy = np.max(xy, axis=0)
    scored: List[Tuple[float, int, Dict[str, float]]] = []
    for indices in groups.values():
        if len(indices) < min_count:
            continue
        local = cloth[indices]
        z = local[:, 2]
        z_range = float(np.max(z) - np.min(z))
        top_local = int(np.argmax(z))
        vidx = int(indices[top_local])
        top_sep = float(z[top_local] - np.median(z))
        p = cloth[vidx, :2]
        edge_prox = float(
            min(
                abs(p[0] - min_xy[0]),
                abs(p[0] - max_xy[0]),
                abs(p[1] - min_xy[1]),
                abs(p[1] - max_xy[1]),
            )
        )
        # higher layering + near boundary preferred
        score = float(z_range * math.log1p(len(indices)) * max(top_sep, 0.0) / (edge_prox + 0.02))
        scored.append(
            (
                score,
                vidx,
                {
                    "local_point_count": float(len(indices)),
                    "local_z_range": z_range,
                    "top_layer_separation": top_sep,
                    "edge_proximity": edge_prox,
                },
            )
        )
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def _line_style_release_from_grasp(
    pick_xy: np.ndarray,
    base_line: Dict[str, Any],
    cloth_initial: Optional[np.ndarray],
    target_limb_code: int,
    uncover_action_policy: Sequence[float],
    rng: np.random.Generator,
    blanket_w: float,
    quadrant_mode: str = "prior",
) -> np.ndarray:
    """Release = grasp + d * u, direction from line heuristic / TL prior."""
    force = _xy(base_line.get("force_direction", [1.0, 0.0]))
    n = float(np.linalg.norm(force))
    if n < 1e-8:
        force = np.array([1.0, 0.0], dtype=np.float32)
    else:
        force = force / n

    base_move = _xy(base_line["place_xy"]) - _xy(base_line["pick_xy"])
    dist0 = float(np.linalg.norm(base_move))
    dist0 = float(np.clip(dist0, 0.08, 0.45))

    # TL4 prior: prefer +x,+y in cloth AABB frame (empirical line corner); soft prior.
    if int(target_limb_code) == 4 and cloth_initial is not None and quadrant_mode == "prior":
        if float(rng.random()) < 0.70:
            corners, _ = find_cloth_bbox_corners(cloth_initial)
            # rank 3 = max_x, max_y for tl4 in prior eval
            target = _xy(corners[min(3, len(corners) - 1)])
            vec = target - pick_xy
            vn = float(np.linalg.norm(vec))
            if vn > 1e-6:
                force = (vec / vn).astype(np.float32)
                dist0 = float(np.clip(vn, 0.08, 0.45))
        elif float(rng.random()) < 0.67:
            # quadrant boundary jitter around line force
            force = force  # keep line force
        # else: keep line force as "other" contrast via later hard negatives

    # small direction jitter
    ang = math.radians(float(rng.uniform(-10.0, 10.0)))
    c, s = math.cos(ang), math.sin(ang)
    force = np.array([c * force[0] - s * force[1], s * force[0] + c * force[1]], dtype=np.float32)
    dist = dist0 * float(1.0 + rng.normal(0.0, 0.05))
    dist = float(np.clip(dist, 0.06, 0.50))
    return (pick_xy + dist * force).astype(np.float32)


def overlap_boundary_line_release(
    cloth_intermediate: np.ndarray,
    cloth_initial: Optional[np.ndarray],
    uncover_action_policy: Sequence[float],
    target_limb_code: int,
    base_line: Dict[str, Any],
    rng: np.random.Generator,
    band: str,
    blanket_w: float,
) -> Optional[Dict[str, Any]]:
    ranked = _overlap_ranked_indices(cloth_intermediate)
    if not ranked:
        # fallback: high-Z near cloth boundary
        cloth = np.asarray(cloth_intermediate, dtype=np.float32)
        xy = cloth[:, :2]
        min_xy, max_xy = np.min(xy, axis=0), np.max(xy, axis=0)
        edge_d = np.minimum.reduce(
            [
                np.abs(xy[:, 0] - min_xy[0]),
                np.abs(xy[:, 0] - max_xy[0]),
                np.abs(xy[:, 1] - min_xy[1]),
                np.abs(xy[:, 1] - max_xy[1]),
            ]
        )
        # prefer high z and near edge
        score = cloth[:, 2] / (edge_d + 0.02)
        order = np.argsort(-score)
        ranked = [(float(score[i]), int(i), {}) for i in order[: max(20, len(order) // 10)]]

    n = len(ranked)
    if band == "top5":
        lo, hi = 0, max(1, int(0.05 * n))
    elif band == "top10":
        lo, hi = 0, max(1, int(0.10 * n))
    elif band == "top10_30":
        lo, hi = max(0, int(0.10 * n)), max(1, int(0.30 * n))
    elif band == "top30_50":
        lo, hi = max(0, int(0.30 * n)), max(1, int(0.50 * n))
    elif band == "near_overlap":
        # take a top candidate then offset 2-5cm outside
        lo, hi = 0, max(1, int(0.20 * n))
    else:  # edge_corner_random
        lo, hi = max(0, int(0.30 * n)), n

    if hi <= lo:
        lo, hi = 0, n
    choice = ranked[int(rng.integers(lo, hi))]
    vidx = int(choice[1])
    pick = np.asarray(cloth_intermediate, dtype=np.float32)[vidx, :2].astype(np.float32)

    if band == "near_overlap":
        # push 2-5 cm away from pick in a random direction (counterfactual-ish grasp locus)
        rad = float(rng.uniform(0.02, 0.05))
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        pick = pick + rad * np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
        pick, vidx = _snap_grasp_to_cloth(pick, cloth_intermediate)

    place = _line_style_release_from_grasp(
        pick,
        base_line,
        cloth_initial,
        target_limb_code,
        uncover_action_policy,
        rng,
        blanket_w,
        quadrant_mode="prior",
    )
    policy = _policy_from_world(pick, place)
    if not _on_cloth(policy, cloth_intermediate):
        return None
    return {
        "family": f"overlap_boundary_{band}",
        "slot": f"overlap_boundary_{band}",
        "recover_action": policy,
        "action_world": scale_action(policy),
        "pick_xy": pick,
        "place_xy": place,
        "grasp_band": band,
        "overlap_meta": choice[2],
        "pick_vertex_idx": vidx,
        "base_family": "line_style_release",
    }


def optimizer_proxy_actions(
    cloth_intermediate: np.ndarray,
    uncover_action_policy: Sequence[float],
    target_limb_code: int,
    cloth_initial: Optional[np.ndarray],
    base_line: Dict[str, Any],
    all_body_points: Optional[np.ndarray],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Pilot stand-ins for optimizer proposals: field-guided + hybrid line/field blend."""
    out: List[Dict[str, Any]] = []
    # 1) field-guided cover action if possible
    try:
        from recover_heuristics import compute_field_guided_recover_action_from_states

        if all_body_points is not None and len(all_body_points) > 0:
            policy, world, pick, place, force, debug = compute_field_guided_recover_action_from_states(
                cloth_intermediate, all_body_points
            )
            if _on_cloth(np.asarray(policy, dtype=np.float32), cloth_intermediate):
                out.append(
                    {
                        "family": "optimizer_proxy_field",
                        "slot": "optimizer_proxy_field",
                        "recover_action": np.asarray(policy, dtype=np.float32),
                        "action_world": np.asarray(world, dtype=np.float32),
                        "pick_xy": _xy(pick),
                        "place_xy": _xy(place),
                        "force_direction": _xy(force),
                        "debug": debug,
                    }
                )
    except Exception as exc:
        out.append({"family": "optimizer_proxy_field", "slot": "optimizer_proxy_field", "error": str(exc)})

    # 2) blend: line grasp + slightly shortened/rotated release (CMA-like local exploit)
    pick = _xy(base_line["pick_xy"])
    place0 = _xy(base_line["place_xy"])
    move = place0 - pick
    dist = float(np.linalg.norm(move))
    theta = math.atan2(move[1], move[0]) + math.radians(float(rng.uniform(-20.0, 20.0)))
    dist2 = float(np.clip(dist * float(rng.uniform(0.7, 1.15)), 0.06, 0.50))
    place = pick + dist2 * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    policy = _policy_from_world(pick, place)
    if _on_cloth(policy, cloth_intermediate):
        out.append(
            {
                "family": "optimizer_proxy_line_local",
                "slot": "optimizer_proxy_line_local",
                "recover_action": policy,
                "action_world": scale_action(policy),
                "pick_xy": pick,
                "place_xy": place,
                "note": "local line exploit proxy until planner-in-the-loop CMA dump exists",
            }
        )
    return [x for x in out if "recover_action" in x][:2]


def hard_negatives(
    cloth_intermediate: np.ndarray,
    base_line: Dict[str, Any],
    cloth_initial: Optional[np.ndarray],
    target_limb_code: int,
    uncover_action_policy: Sequence[float],
    rng: np.random.Generator,
    blanket_w: float,
) -> List[Dict[str, Any]]:
    """Single-variable counterfactuals around the line center."""
    out: List[Dict[str, Any]] = []
    pick_g = _xy(base_line["pick_xy"])
    place_g = _xy(base_line["place_xy"])
    move = place_g - pick_g
    dist = float(np.linalg.norm(move))
    if dist < 1e-6:
        return out
    theta = math.atan2(move[1], move[0])

    # 1) same grasp, reverse release direction
    place_bad = pick_g - move
    policy = _policy_from_world(pick_g, place_bad)
    if _on_cloth(policy, cloth_intermediate):
        out.append(
            {
                "family": "hardneg_fixed_grasp_wrong_dir",
                "slot": "hardneg_fixed_grasp_wrong_dir",
                "recover_action": policy,
                "action_world": scale_action(policy),
                "pick_xy": pick_g,
                "place_xy": place_bad.astype(np.float32),
                "counterfactual": "release_direction",
                "paired_with": "exact_line",
            }
        )

    # 2) same grasp, too-short release
    place_short = pick_g + 0.25 * move
    policy = _policy_from_world(pick_g, place_short)
    if _on_cloth(policy, cloth_intermediate):
        out.append(
            {
                "family": "hardneg_fixed_grasp_short",
                "slot": "hardneg_fixed_grasp_short",
                "recover_action": policy,
                "action_world": scale_action(policy),
                "pick_xy": pick_g,
                "place_xy": place_short.astype(np.float32),
                "counterfactual": "release_distance",
                "paired_with": "exact_line",
            }
        )

    # 3) same release, adjacent non-overlap grasp (2-5cm offset)
    rad = float(rng.uniform(0.02, 0.05))
    ang = float(rng.uniform(0.0, 2.0 * math.pi))
    pick_bad = pick_g + rad * np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
    pick_bad, _ = _snap_grasp_to_cloth(pick_bad, cloth_intermediate)
    policy = _policy_from_world(pick_bad, place_g)
    if _on_cloth(policy, cloth_intermediate):
        out.append(
            {
                "family": "hardneg_fixed_release_adj_grasp",
                "slot": "hardneg_fixed_release_adj_grasp",
                "recover_action": policy,
                "action_world": scale_action(policy),
                "pick_xy": pick_bad,
                "place_xy": place_g,
                "counterfactual": "grasp_adjacent",
                "paired_with": "exact_line",
            }
        )

    return out[:3]


def _fallback_line_like_action(
    cloth_xyz: np.ndarray,
    base: Optional[Dict[str, Any]],
    rng: np.random.Generator,
    blanket_w: float,
    slot: str,
    family: str,
) -> Dict[str, Any]:
    """Last-resort on-cloth action near line center or high-Z vertex."""
    cloth = np.asarray(cloth_xyz, dtype=np.float32)
    if base is not None and "pick_xy" in base and "place_xy" in base:
        pick0 = _xy(base["pick_xy"])
        place0 = _xy(base["place_xy"])
        move = place0 - pick0
        if float(np.linalg.norm(move)) < 1e-6:
            move = np.array([0.0, 0.15], dtype=np.float32)
    else:
        pick0 = cloth[int(np.argmax(cloth[:, 2])), :2].astype(np.float32)
        move = np.array([0.05, 0.15], dtype=np.float32)

    for _ in range(40):
        pick = pick0 + rng.normal(0.0, 0.03 * blanket_w, size=2).astype(np.float32)
        pick, _ = _snap_grasp_to_cloth(pick, cloth)
        scale = float(rng.uniform(0.6, 1.2))
        ang = math.radians(float(rng.uniform(-25.0, 25.0)))
        c, s = math.cos(ang), math.sin(ang)
        rot = np.array([c * move[0] - s * move[1], s * move[0] + c * move[1]], dtype=np.float32)
        place = pick + scale * rot
        policy = _policy_from_world(pick, place)
        if _on_cloth(policy, cloth):
            return {
                "family": family,
                "slot": slot,
                "recover_action": policy,
                "action_world": scale_action(policy),
                "pick_xy": pick,
                "place_xy": place.astype(np.float32),
                "fallback": True,
            }
    # absolute last resort: two nearest high-Z points
    order = np.argsort(-cloth[:, 2])
    pick = cloth[int(order[0]), :2].astype(np.float32)
    place = cloth[int(order[min(10, len(order) - 1)]), :2].astype(np.float32)
    policy = _policy_from_world(pick, place)
    return {
        "family": family,
        "slot": slot,
        "recover_action": policy,
        "action_world": scale_action(policy),
        "pick_xy": pick,
        "place_xy": place,
        "fallback": True,
        "fallback_absolute": True,
    }


N_LINE_NEIGHBORHOOD_ACTIONS = 20


def build_tl4_pilot_action_set(
    cloth_intermediate: np.ndarray,
    uncover_action_policy: Sequence[float],
    target_limb_code: int,
    cloth_initial: Optional[np.ndarray] = None,
    all_body_points: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:
    """Build exactly 20 joint recover actions for one uncover state.

    Schedule (joint g,r — not independent mix-and-match):
      1 exact line
      8 line neighborhood (3x small/medium + 2x large)
      6 overlap/boundary + line-style release
      2 optimizer proxies
      3 paired hard negatives
    """
    rng = rng or np.random.default_rng(0)
    cloth_i = np.asarray(cloth_intermediate, dtype=np.float32)
    cloth_0 = None if cloth_initial is None else np.asarray(cloth_initial, dtype=np.float32)
    blanket_w = _blanket_width(cloth_i)

    # --- slot 0: exact line center ---
    try:
        base = exact_line_action(cloth_i, uncover_action_policy, target_limb_code, cloth_0)
        if not _on_cloth(base["recover_action"], cloth_i):
            # snap grasp onto cloth while keeping line-style release
            pick, _ = _snap_grasp_to_cloth(_xy(base["pick_xy"]), cloth_i)
            place = _xy(base["place_xy"])
            base["pick_xy"] = pick
            base["recover_action"] = _policy_from_world(pick, place)
            base["action_world"] = scale_action(base["recover_action"])
            base["snapped_to_cloth"] = True
    except Exception as exc:
        base = _fallback_line_like_action(
            cloth_i, None, rng, blanket_w, slot="exact_line", family="exact_line"
        )
        base["exact_line_error"] = str(exc)

    if not _on_cloth(base["recover_action"], cloth_i):
        base = _fallback_line_like_action(
            cloth_i, base, rng, blanket_w, slot="exact_line", family="exact_line"
        )

    actions: List[Dict[str, Any]] = [base]

    def _ensure(slot: str, family: str, factory):
        item = None
        for _ in range(8):
            try:
                item = factory()
            except Exception:
                item = None
            if item is not None and "recover_action" in item and _on_cloth(
                np.asarray(item["recover_action"], dtype=np.float32), cloth_i
            ):
                item["slot"] = slot
                item["family"] = item.get("family") or family
                actions.append(item)
                return
        actions.append(
            _fallback_line_like_action(cloth_i, base, rng, blanket_w, slot=slot, family=family)
        )

    # --- line neighborhood: +2 vs pilot (small/medium x3, large x2) ---
    for scale, n_rep in (("small", 3), ("medium", 3), ("large", 2)):
        for k in range(n_rep):
            _ensure(
                f"line_noise_{scale}_{k}",
                f"line_noise_{scale}",
                lambda scale=scale: _perturb_line_action(base, cloth_i, rng, scale, blanket_w),
            )

    # --- overlap/boundary: +2 vs pilot (top5, top30_50) ---
    for band in (
        "top5",
        "top10",
        "top10_30",
        "top30_50",
        "near_overlap",
        "edge_corner_random",
    ):
        _ensure(
            f"overlap_boundary_{band}",
            f"overlap_boundary_{band}",
            lambda band=band: overlap_boundary_line_release(
                cloth_i, cloth_0, uncover_action_policy, target_limb_code, base, rng, band, blanket_w
            ),
        )

    # --- optimizer proxies ---
    opt_items = []
    try:
        opt_items = optimizer_proxy_actions(
            cloth_i, uncover_action_policy, target_limb_code, cloth_0, base, all_body_points, rng
        )
    except Exception:
        opt_items = []
    opt_slots = ["optimizer_proxy_field", "optimizer_proxy_line_local"]
    for i, slot in enumerate(opt_slots):
        if i < len(opt_items) and _on_cloth(np.asarray(opt_items[i]["recover_action"], dtype=np.float32), cloth_i):
            item = dict(opt_items[i])
            item["slot"] = slot
            actions.append(item)
        else:
            actions.append(
                _fallback_line_like_action(cloth_i, base, rng, blanket_w, slot=slot, family=slot)
            )

    # --- hard negatives ---
    hn_items = []
    try:
        hn_items = hard_negatives(
            cloth_i, base, cloth_0, target_limb_code, uncover_action_policy, rng, blanket_w
        )
    except Exception:
        hn_items = []
    hn_slots = [
        "hardneg_fixed_grasp_wrong_dir",
        "hardneg_fixed_grasp_short",
        "hardneg_fixed_release_adj_grasp",
    ]
    for i, slot in enumerate(hn_slots):
        matched = None
        for cand in hn_items:
            if cand.get("slot") == slot or cand.get("family") == slot:
                matched = cand
                break
        if matched is None and i < len(hn_items):
            matched = hn_items[i]
        if matched is not None and _on_cloth(np.asarray(matched["recover_action"], dtype=np.float32), cloth_i):
            item = dict(matched)
            item["slot"] = slot
            actions.append(item)
        else:
            # synthesize a hardneg-like contrast if generator failed
            pick = _xy(base["pick_xy"])
            place = _xy(base["place_xy"])
            move = place - pick
            if slot.endswith("wrong_dir"):
                place_b = pick - move
            elif slot.endswith("short"):
                place_b = pick + 0.25 * move
            else:
                ang = float(rng.uniform(0.0, 2.0 * math.pi))
                pick = pick + 0.03 * np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
                pick, _ = _snap_grasp_to_cloth(pick, cloth_i)
                place_b = place
            policy = _policy_from_world(pick, place_b)
            if not _on_cloth(policy, cloth_i):
                actions.append(
                    _fallback_line_like_action(cloth_i, base, rng, blanket_w, slot=slot, family=slot)
                )
            else:
                actions.append(
                    {
                        "family": slot,
                        "slot": slot,
                        "recover_action": policy,
                        "action_world": scale_action(policy),
                        "pick_xy": pick,
                        "place_xy": _xy(place_b),
                        "synthesized_hardneg": True,
                    }
                )

    # Guarantee exactly 20
    while len(actions) < N_LINE_NEIGHBORHOOD_ACTIONS:
        actions.append(
            _fallback_line_like_action(
                cloth_i,
                base,
                rng,
                blanket_w,
                slot=f"line_noise_pad_{len(actions)}",
                family="line_noise_medium",
            )
        )
    actions = actions[:N_LINE_NEIGHBORHOOD_ACTIONS]
    for i, a in enumerate(actions):
        a["action_index"] = i
        a["blanket_width"] = blanket_w
    return actions
