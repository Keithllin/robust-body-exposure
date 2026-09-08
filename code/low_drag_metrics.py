#!/usr/bin/env python3
"""Metrics for lower-body constrained low-drag unfolding collection."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from assistive_gym.envs.bu_gnn_util import get_body_points_from_obs, get_covered_status

# Upper-body protected region = TL13 target limbs (arms + torso; head excluded by design).
UPPER_BODY_TL = 13
LOWER_BODY_TLS = (4, 5, 10, 11, 12, 14)


def _cloth_xy(cloth) -> np.ndarray:
    arr = np.asarray(
        cloth[1] if isinstance(cloth, (list, tuple)) and len(cloth) > 1 else cloth,
        dtype=np.float64,
    )
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return arr[:, :2]


def _cloth_xyz(cloth) -> np.ndarray:
    arr = np.asarray(
        cloth[1] if isinstance(cloth, (list, tuple)) and len(cloth) > 1 else cloth,
        dtype=np.float64,
    )
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.shape[1] == 2:
        z = np.zeros((arr.shape[0], 1), dtype=np.float64)
        return np.concatenate([arr, z], axis=1)
    return arr[:, :3]


def upper_body_points(human_pose, body_info=None) -> np.ndarray:
    """TL13 body points (is_target=1 on upper limbs)."""
    return np.asarray(
        get_body_points_from_obs(
            np.asarray(human_pose, dtype=np.float64),
            target_limb_code=int(UPPER_BODY_TL),
            body_info=body_info,
        ),
        dtype=np.float64,
    )


def lower_target_points(human_pose, target_limb_code: int, body_info=None) -> np.ndarray:
    """Body points for a lower-body target limb code."""
    tl = int(target_limb_code)
    if tl not in LOWER_BODY_TLS:
        raise ValueError("target_limb_code %s is not a lower-body TL %s" % (tl, LOWER_BODY_TLS))
    return np.asarray(
        get_body_points_from_obs(
            np.asarray(human_pose, dtype=np.float64),
            target_limb_code=tl,
            body_info=body_info,
        ),
        dtype=np.float64,
    )


def coverage_status(points: np.ndarray, cloth) -> np.ndarray:
    return np.asarray(get_covered_status(np.asarray(points, dtype=np.float64), _cloth_xy(cloth)))


def exposed_count(status: np.ndarray, target_only: bool = True) -> int:
    """Count uncovered body points (optionally only is_target==1)."""
    status = np.asarray(status)
    if status.size == 0:
        return 0
    if target_only:
        mask = status[:, 0] == 1
    else:
        mask = status[:, 0] != -1
    if not np.any(mask):
        return 0
    return int(np.sum(~status[mask, 1].astype(bool)))


def covered_count(status: np.ndarray, target_only: bool = True) -> int:
    status = np.asarray(status)
    if status.size == 0:
        return 0
    if target_only:
        mask = status[:, 0] == 1
    else:
        mask = status[:, 0] != -1
    if not np.any(mask):
        return 0
    return int(np.sum(status[mask, 1].astype(bool)))


def newly_exposed_count(status_start: np.ndarray, status_curr: np.ndarray, target_only: bool = True) -> int:
    """Points that were covered at start and uncovered now (target limbs by default)."""
    s0 = np.asarray(status_start)
    s1 = np.asarray(status_curr)
    if s0.shape != s1.shape or s0.size == 0:
        return 0
    if target_only:
        mask = s0[:, 0] == 1
    else:
        mask = s0[:, 0] != -1
    if not np.any(mask):
        return 0
    was_covered = s0[mask, 1].astype(bool)
    now_uncovered = ~s1[mask, 1].astype(bool)
    return int(np.sum(was_covered & now_uncovered))


def upper_exposure_delta(upper_points: np.ndarray, cloth_start, cloth_curr) -> Tuple[int, int, int]:
    """Return (E_start_uncovered, E_curr_uncovered, newly_exposed_from_start)."""
    s0 = coverage_status(upper_points, cloth_start)
    s1 = coverage_status(upper_points, cloth_curr)
    e0 = exposed_count(s0, target_only=True)
    e1 = exposed_count(s1, target_only=True)
    newly = newly_exposed_count(s0, s1, target_only=True)
    return e0, e1, newly


def lower_recovery_gain(lower_points: np.ndarray, cloth_start, cloth_final) -> Tuple[int, int, int]:
    """gain = covered_final - covered_start among target limb points."""
    s0 = coverage_status(lower_points, cloth_start)
    s1 = coverage_status(lower_points, cloth_final)
    c0 = covered_count(s0, target_only=True)
    c1 = covered_count(s1, target_only=True)
    return c0, c1, int(c1 - c0)


def protected_region_node_mask(cloth_start, upper_points: np.ndarray, radius: float = 0.05) -> np.ndarray:
    """Cloth nodes within radius of any upper-body point at recover start."""
    cloth = _cloth_xy(cloth_start)
    body = np.asarray(upper_points, dtype=np.float64)
    if cloth.shape[0] == 0 or body.shape[0] == 0:
        return np.zeros((cloth.shape[0],), dtype=bool)
    # Approximate: node near any upper point
    # Use chunked min-dist to avoid huge memory
    mask = np.zeros((cloth.shape[0],), dtype=bool)
    for i in range(0, cloth.shape[0], 512):
        chunk = cloth[i : i + 512]
        d = np.linalg.norm(chunk[:, None, :] - body[None, :, :2], axis=2)
        mask[i : i + 512] = np.min(d, axis=1) <= float(radius)
    return mask


def protected_cloth_motion(cloth_start, cloth_end, node_mask: np.ndarray) -> float:
    """Mean XY displacement of protected cloth nodes."""
    c0 = _cloth_xy(cloth_start)
    c1 = _cloth_xy(cloth_end)
    if c0.shape != c1.shape or c0.shape[0] == 0:
        return float("nan")
    mask = np.asarray(node_mask, dtype=bool)
    if mask.shape[0] != c0.shape[0] or not np.any(mask):
        return float("nan")
    disp = np.linalg.norm(c1[mask] - c0[mask], axis=1)
    return float(np.mean(disp))


def target_region_cloth_motion(cloth_start, cloth_end, lower_points: np.ndarray, radius: float = 0.05) -> float:
    mask = protected_region_node_mask(cloth_start, lower_points, radius=radius)
    return protected_cloth_motion(cloth_start, cloth_end, mask)


def grasp_to_cloth_edge_distance(grasp_xy: Sequence[float], cloth) -> float:
    """Min distance from grasp to approximate cloth boundary (convex hull edges)."""
    from scipy.spatial import ConvexHull

    xy = _cloth_xy(cloth)
    g = np.asarray(grasp_xy, dtype=np.float64).reshape(-1)[:2]
    if xy.shape[0] < 3 or not np.isfinite(g).all():
        return float("nan")
    try:
        hull = ConvexHull(xy)
        verts = xy[hull.vertices]
    except Exception:
        # Fallback: distance to nearest vertex after removing interior via percentile rim
        d = np.linalg.norm(xy - g[None, :], axis=1)
        return float(np.min(d))
    # Distance to each hull edge segment
    mins = []
    n = len(verts)
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        ab = b - a
        t = np.dot(g - a, ab) / max(np.dot(ab, ab), 1e-12)
        t = float(np.clip(t, 0.0, 1.0))
        proj = a + t * ab
        mins.append(float(np.linalg.norm(g - proj)))
    return float(min(mins)) if mins else float("nan")


def grasp_to_corner_distance(grasp_xy: Sequence[float], cloth) -> float:
    """Distance from grasp to nearest AABB corner of cloth XY."""
    xy = _cloth_xy(cloth)
    g = np.asarray(grasp_xy, dtype=np.float64).reshape(-1)[:2]
    if xy.shape[0] == 0:
        return float("nan")
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    corners = np.array(
        [[mn[0], mn[1]], [mn[0], mx[1]], [mx[0], mn[1]], [mx[0], mx[1]]],
        dtype=np.float64,
    )
    return float(np.min(np.linalg.norm(corners - g[None, :], axis=1)))


def action_length_world(grasp_xy: Sequence[float], release_xy: Sequence[float]) -> float:
    g = np.asarray(grasp_xy, dtype=np.float64).reshape(-1)[:2]
    r = np.asarray(release_xy, dtype=np.float64).reshape(-1)[:2]
    return float(np.linalg.norm(r - g))


# Output bucket names (directory stems under the dataset root).
BUCKET_ACCEPTED = "accepted_low_drag_unfolding"
BUCKET_DRAGGING = "dragging_contaminated"
BUCKET_NO_UNFOLD = "no_effective_unfolding"
BUCKET_GRASP_MISS = "grasp_miss"
BUCKET_TOO_SHORT = "too_short"
BUCKET_SIM_FAIL = "simulation_failure"
BUCKET_BOUNDARY = "boundary_unsafe_sibling"

REJECT_TO_BUCKET = {
    "grasp_miss": BUCKET_GRASP_MISS,
    "too_short": BUCKET_TOO_SHORT,
    "dragging_contaminated": BUCKET_DRAGGING,
    "no_effective_unfolding": BUCKET_NO_UNFOLD,
    "simulation_failure": BUCKET_SIM_FAIL,
    "boundary_longer_sibling": BUCKET_BOUNDARY,
    "invalid_action": BUCKET_SIM_FAIL,
}


def classify_acceptance(
    lower_gain: int,
    upper_delta_final: int,
    eta_lower_gain: float,
    tau_final: float,
    execute_recover: bool,
    min_actual_length: float = 0.02,
    actual_length: Optional[float] = None,
    require_lower_gain: bool = False,
    **_ignored,
) -> Tuple[bool, str]:
    """Return (accepted, rejection_reason).

    Goal = unfolding-not-dragging (not recover success). Default final filters:
    grasp_miss / too_short / dragging_contaminated.

    ``ΔC_lower`` / ``eta_lower_gain`` are diagnostic only unless
    ``require_lower_gain=True`` (legacy no-op reject → no_effective_unfolding).
    """
    if not execute_recover:
        return False, "grasp_miss"
    if actual_length is not None and float(actual_length) < float(min_actual_length):
        return False, "too_short"
    if int(upper_delta_final) > float(tau_final):
        return False, "dragging_contaminated"
    if require_lower_gain and int(lower_gain) < float(eta_lower_gain):
        return False, "no_effective_unfolding"
    # No-op (low ΔC_lower) is still accepted: not-dragging is sufficient.
    return True, "low_drag_unfolding"


def bucket_for_result(accepted: bool, rejection_reason: str, boundary_role: str = "primary") -> str:
    if boundary_role == "unsafe_sibling":
        return BUCKET_BOUNDARY
    if accepted:
        return BUCKET_ACCEPTED
    return REJECT_TO_BUCKET.get(str(rejection_reason or ""), BUCKET_SIM_FAIL)
