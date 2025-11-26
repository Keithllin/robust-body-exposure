"""Reward field utilities for the Robust Body Exposure (RoBE) task.

This module provides helper functions to build spatial reward fields over the
cloth surface, derived from the simulation state. The goal is to expose richer
supervision than a single scalar reward while remaining compatible with
traditional reinforcement learning pipelines.
"""

from typing import Dict, Optional, Tuple

import numpy as np


class ClothState:
    """Container for cloth-level features used by the reward field.

    Implemented as a simple class for Python 3.6 compatibility (avoiding
    dataclasses dependency).
    """

    def __init__(
        self,
        positions: np.ndarray,
        initial_positions: np.ndarray,
        displacement: np.ndarray,
        tension: np.ndarray,
        tension_mean: float,
    ) -> None:
        self.positions = positions
        self.initial_positions = initial_positions
        self.displacement = displacement
        self.tension = tension
        self.tension_mean = tension_mean

    @property
    def tension_std(self) -> float:
        return float(np.std(self.tension)) if self.tension.size else 0.0


DEFAULT_FIELD_WEIGHTS: Dict[str, float] = {
    "coverage": 2.0,
    "nontarget": 1.0,
    "head": 2.0,
    "wrinkle": 0.5,
}


def apply_no_grasp_penalty(reward_scalar: float, is_on_cloth: Optional[bool], penalty: float) -> float:
    """Penalise rewards from actions that failed to grasp the blanket.

    Args:
        reward_scalar: The original scalar reward value.
        is_on_cloth: Whether the commanded grasp touched the blanket.
        penalty: Non-negative penalty to subtract when ``is_on_cloth`` is
            False. If zero or negative, the reward is returned unchanged.

    Returns:
        float: Adjusted reward.
    """

    if penalty <= 0.0 or is_on_cloth is None or bool(is_on_cloth):
        return float(reward_scalar)
    return float(reward_scalar) - float(penalty)


def build_cloth_state(
    initial_positions: np.ndarray,
    final_positions: np.ndarray,
) -> ClothState:
    """Create a :class:`ClothState` from two aligned cloth point clouds.

    Args:
        initial_positions: Array with shape ``(N, 3)`` representing the cloth
            vertex positions before an action is applied.
        final_positions: Array with shape ``(N, 3)`` representing the cloth
            vertex positions after the action.

    Returns:
        ClothState: Structured cloth information including per-vertex
        displacement norms ("tension").
    """

    initial = np.asarray(initial_positions, dtype=np.float32)
    final = np.asarray(final_positions, dtype=np.float32)
    if initial.shape != final.shape:
        raise ValueError(
            "initial_positions and final_positions must share the same shape"
        )

    displacement = final - initial
    tension = np.linalg.norm(displacement, axis=1)
    tension_mean = float(np.mean(tension)) if tension.size else 0.0

    return ClothState(
        positions=final,
        initial_positions=initial,
        displacement=displacement,
        tension=tension,
        tension_mean=tension_mean,
    )


def compute_body_contact_masks(
    cloth_xy: np.ndarray,
    body_points: np.ndarray,
    coverage_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate which cloth vertices interact with target/nontarget regions.

    Notes
    -----
    This helper mirrors the original RoBE reward design only approximately:
    it uses *cloth-centric* distance thresholds to detect contact with target
    and non-target regions. The original environment, however, reasons over
    *body-centric* sample points and measures whether those points remain
    covered relative to the blanket's initial configuration.

    The higher-level :func:`compute_robe_nontarget_head_terms` function builds
    on this by performing RoBE-style point counting on top of the low-level
    cloth/body geometry.

    Args:
        cloth_xy: Array with shape ``(N, 2)`` containing the cloth vertex
            positions projected to the bed plane.
        body_points: Array with shape ``(M, 3)`` where the last channel encodes
            semantic labels (1 = target, 0 = non-target, -1 = head).
        coverage_threshold: Distance (in metres) below which a vertex is
            considered in contact with a body region.

    Returns:
        Tuple of arrays ``(body_mask, target_mask, nontarget_mask, head_mask)``
        all shaped ``(N,)``.
    """

    cloth_xy = np.asarray(cloth_xy, dtype=np.float32)
    body_points = np.asarray(body_points, dtype=np.float32)

    if cloth_xy.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty, empty

    if body_points.size == 0:
        zeros = np.zeros((cloth_xy.shape[0],), dtype=np.float32)
        return zeros, zeros, zeros, zeros

    body_coords = body_points[:, :2]
    semantic = body_points[:, 2]

    def _compute_mask(candidate_points: np.ndarray) -> np.ndarray:
        if candidate_points.size == 0:
            return np.zeros((cloth_xy.shape[0],), dtype=np.float32)
        diff = cloth_xy[:, None, :] - candidate_points[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = (np.min(dists, axis=1) <= coverage_threshold).astype(np.float32)
        return mask

    body_mask = _compute_mask(body_coords)

    target_points = body_coords[semantic == 1]
    target_mask = _compute_mask(target_points)

    nontarget_points = body_coords[semantic == 0]
    nontarget_mask = _compute_mask(nontarget_points)

    head_points = body_coords[semantic == -1]
    head_mask = _compute_mask(head_points)

    return body_mask, target_mask, nontarget_mask, head_mask


def compute_robe_nontarget_head_terms(
    cloth_positions: np.ndarray,
    body_points: np.ndarray,
    *,
    initial_cloth_positions: Optional[np.ndarray] = None,
    coverage_threshold: float = 0.028,
) -> Tuple[np.ndarray, np.ndarray]:
    """RoBE-style non-target/head terms based on body sample points.

    This function mirrors the original reward definitions from
    ``BeddingManipulationEnv``:

    * Non-target: "discourage the robot from uncovering non-target body
      parts". Only body points that were covered in the *initial* blanket
      configuration contribute to this term. Uncovering such a point yields a
      negative contribution; keeping it covered is neutral.
    * Head: "discourage the robot from covering the head". Any head point that
      is covered yields a negative contribution.

    To remain compatible with the reward-field abstraction, the resulting
    scalar signals are distributed uniformly over the corresponding cloth
    vertices that interact with the respective regions.

    Args:
        cloth_positions: Array with shape ``(N, 3)`` giving current cloth
            vertex positions.
        body_points: Array with shape ``(P, 4)`` or ``(P, 3)`` describing
            body sample points. The first three components are XYZ coordinates;
            the last channel (or the third if shape is ``(P, 3)``) encodes
            semantic labels (1 = target, 0 = non-target, -1 = head).
        initial_cloth_positions: Optional array with shape ``(N, 3)`` giving
            the initial cloth configuration. If provided, it is used to
            determine which non-target body points were covered initially,
            matching ``non_target_initially_uncovered`` from the original
            environment. If omitted, all non-target points are treated as
            eligible for uncovering penalties.
        coverage_threshold: Distance threshold (in metres) used for deciding
            whether a body point is covered by the blanket, following the
            original implementation (default 2.8cm).

    Returns:
        Tuple ``(nontarget_term, head_term)`` each shaped ``(N,)`` containing
        per-vertex contributions that approximate the original scalar rewards.
    """

    cloth_positions = np.asarray(cloth_positions, dtype=np.float32)
    if cloth_positions.ndim != 2 or cloth_positions.shape[1] != 3:
        raise ValueError("cloth_positions must have shape (N, 3)")

    body_points = np.asarray(body_points, dtype=np.float32)
    if body_points.size == 0:
        zeros = np.zeros((cloth_positions.shape[0],), dtype=np.float32)
        return zeros, zeros

    if body_points.shape[1] == 4:
        body_xyz = body_points[:, :3]
        semantic = body_points[:, 3]
    elif body_points.shape[1] == 3:
        # If only (x, y, label) were provided, treat z as zero.
        body_xyz = np.zeros((body_points.shape[0], 3), dtype=np.float32)
        body_xyz[:, :2] = body_points[:, :2]
        semantic = body_points[:, 2]
    else:
        raise ValueError("body_points must have shape (P, 3) or (P, 4)")

    cloth_xy = cloth_positions[:, :2]

    # Helper: check whether each body point is covered by any cloth vertex.
    def _body_points_covered(body_xyz_subset: np.ndarray, cloth_xy_ref: np.ndarray) -> np.ndarray:
        if body_xyz_subset.size == 0:
            return np.zeros((0,), dtype=bool)
        body_xy = body_xyz_subset[:, :2]
        diff = cloth_xy_ref[None, :, :] - body_xy[:, None, :]
        dists = np.linalg.norm(diff, axis=2)
        return np.min(dists, axis=1) < coverage_threshold

    # Split body points by semantic label.
    is_target = semantic == 1
    is_nontarget = semantic == 0
    is_head = semantic == -1

    nontarget_pts = body_xyz[is_nontarget]
    head_pts = body_xyz[is_head]
    target_pts = body_xyz[is_target]

    # Determine which non-target points were covered in the initial blanket
    # configuration, if available. This mirrors non_target_initially_uncovered
    # from BeddingManipulationEnv.
    if initial_cloth_positions is not None and nontarget_pts.size:
        initial_cloth_positions = np.asarray(initial_cloth_positions, dtype=np.float32)
        if initial_cloth_positions.shape != cloth_positions.shape:
            raise ValueError("initial_cloth_positions must match cloth_positions shape")
        initial_xy = initial_cloth_positions[:, :2]
        init_cov = _body_points_covered(nontarget_pts, initial_xy)
        effective_nontarget_pts = nontarget_pts[init_cov]
    else:
        effective_nontarget_pts = nontarget_pts

    # If there are no eligible non-target points, both terms are zero.
    n_vertices = cloth_positions.shape[0]
    nontarget_term = np.zeros((n_vertices,), dtype=np.float32)
    head_term = np.zeros((n_vertices,), dtype=np.float32)

    if effective_nontarget_pts.size == 0 and head_pts.size == 0:
        return nontarget_term, head_term

    # Count coverage on current cloth state.
    if effective_nontarget_pts.size:
        covered_now_nt = _body_points_covered(effective_nontarget_pts, cloth_xy)
        total_nt_points = float(effective_nontarget_pts.shape[0])
    else:
        covered_now_nt = np.zeros((0,), dtype=bool)
        total_nt_points = 0.0

    if head_pts.size:
        covered_now_head = _body_points_covered(head_pts, cloth_xy)
        total_head_points = float(head_pts.shape[0])
    else:
        covered_now_head = np.zeros((0,), dtype=bool)
        total_head_points = 0.0

    # For normalisation we reuse the original RoBE heuristic: scale by the
    # number of target points so that non-target penalties are roughly
    # comparable to target exposure rewards.
    total_target_points = float(target_pts.shape[0]) if target_pts.size else 0.0

    # --- Scalar signals in the spirit of BeddingManipulationEnv ---
    scalar_nt = 0.0
    if total_nt_points > 0.0 and total_target_points > 0.0:
        points_uncovered_nt = total_nt_points - float(np.count_nonzero(covered_now_nt))
        scalar_nt = -(points_uncovered_nt / total_target_points) * 100.0

    scalar_head = 0.0
    if total_head_points > 0.0:
        points_covered_head = float(np.count_nonzero(covered_now_head))
        # Original implementation uses double weighting for head points.
        scalar_head = -(points_covered_head / total_head_points) * 200.0

    # --- Distribute scalars back to cloth vertices as per-vertex terms ---
    # We assign a uniform contribution to cloth vertices that are close to
    # non-target / head points respectively. This keeps the aggregate reward
    # approximately aligned with the original scalar definitions while
    # exposing a spatial structure for GNN training and visualisation.

    def _distribute_scalar(body_xyz_subset: np.ndarray, scalar: float) -> np.ndarray:
        term = np.zeros((n_vertices,), dtype=np.float32)
        if body_xyz_subset.size == 0 or scalar == 0.0:
            return term
        body_xy = body_xyz_subset[:, :2]
        diff = cloth_xy[:, None, :] - body_xy[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # A cloth vertex is considered influenced if it is close to any body
        # point of this class.
        influence_mask = np.min(dists, axis=1) < coverage_threshold
        count = float(np.count_nonzero(influence_mask))
        if count == 0.0:
            return term
        term[influence_mask] = scalar / count
        return term

    if scalar_nt != 0.0:
        nontarget_term = _distribute_scalar(effective_nontarget_pts, scalar_nt)

    if scalar_head != 0.0:
        head_term = _distribute_scalar(head_pts, scalar_head)

    return nontarget_term.astype(np.float32), head_term.astype(np.float32)


def compute_reward_field(
    cloth_state: ClothState,
    body_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    nontarget_mask: Optional[np.ndarray] = None,
    head_mask: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
    goal_contact_mask: Optional[np.ndarray] = None,
    coverage_mode: str = "recover",
    normalize_wrinkle: bool = True,
    nontarget_mode: str = "penalize_contact",
    body_points: Optional[np.ndarray] = None,
    diffusion_edge_index: Optional[np.ndarray] = None,
    diffusion_iters: int = 0,
    diffusion_alpha: float = 0.5,
    return_terms: bool = False,
) -> np.ndarray:
    """Compute a spatial reward field across cloth vertices.

    The heuristic blends three intuitive signals:

    * ``coverage`` – reward vertices revealing the desired region.
    * ``nontarget`` – penalise vertices that uncover undesired regions.
    * ``head`` – strongly penalise vertices that cover the head.
    * ``wrinkle`` – encourage low-tension (smooth) cloth configurations.

    Args:
        cloth_state: Structured cloth information produced by
            :func:`build_cloth_state`.
        body_mask: Array of shape ``(N,)`` indicating vertices interacting with
            any body region.
        target_mask: Array of shape ``(N,)`` highlighting vertices near the
            target exposure region.
        nontarget_mask: Optional array of shape ``(N,)`` – if omitted it will be
            inferred as ``body_mask - target_mask`` clamped to ``[0, 1]``.
        head_mask: Optional array of shape ``(N,)`` identifying vertices that
            touch the head.
        weights: Optional dictionary overriding the default contribution of
            each term.
        nontarget_mode: Controls how non-target body regions contribute to the
            reward. Options are:

            * ``"penalize_contact"`` (default) – any cloth contact with
              non-target regions is penalised (legacy behaviour).
            * ``"reward_cover"`` – cloth covering non-target regions is
              rewarded instead of penalised, encouraging the blanket to remain
              over non-target body parts while still respecting head penalties.
            * ``"robe_points"`` – use RoBE-style body reference points and the
              initial blanket configuration to define non-target and head
              terms, closely mirroring ``BeddingManipulationEnv``. This mode
              requires callers to provide ``body_points`` shaped ``(P, 3)`` or
              ``(P, 4)`` (XYZ plus semantic label), and for
              ``cloth_state.initial_positions`` to contain the initial blanket
              configuration.

    Returns:
        np.ndarray: Reward field with shape ``(N,)``.
    """

    weights = {**DEFAULT_FIELD_WEIGHTS, **(weights or {})}

    body_mask = np.asarray(body_mask, dtype=np.float32)
    target_mask = np.asarray(target_mask, dtype=np.float32)

    if nontarget_mask is None:
        nontarget_mask = np.clip(body_mask - target_mask, 0.0, 1.0)
    else:
        nontarget_mask = np.asarray(nontarget_mask, dtype=np.float32)

    if head_mask is None:
        head_mask = np.zeros_like(body_mask)
    else:
        head_mask = np.asarray(head_mask, dtype=np.float32)

    body_points_arr = None
    if body_points is not None:
        body_points_arr = np.asarray(body_points, dtype=np.float32)

    # Coverage/goal consistency term
    target_contact = body_mask * target_mask
    if goal_contact_mask is not None:
        gmask = np.asarray(goal_contact_mask, dtype=np.float32)
        # mask_defined: where goal is specified (>=0). Undefined vertices contribute 0 to coverage term.
        mask_defined = (gmask >= 0.0).astype(np.float32)
        gclipped = np.clip(gmask, 0.0, 1.0) * mask_defined
        current = target_contact * mask_defined
        agreement = 1.0 - np.abs(current - gclipped)  # 1 if matches goal, 0 otherwise
        # Map agreement in [0,1] to [-1,1] so that mismatch yields negative, match yields positive
        coverage_term = (2.0 * agreement - 1.0) * mask_defined
    else:
        mode = coverage_mode.lower()
        if mode == "recover":
            coverage_term = target_contact
        elif mode == "expose":
            coverage_term = -target_contact
        elif mode == "none":
            coverage_term = np.zeros_like(target_mask, dtype=np.float32)
        else:
            raise ValueError(
                f"Unsupported coverage_mode '{coverage_mode}'. Use 'recover', 'expose', or 'none'."
            )

    wrinkle_term = np.zeros_like(coverage_term)
    if cloth_state.tension.size:
        tension_delta = cloth_state.tension - cloth_state.tension_mean
        if normalize_wrinkle:
            tension_std = cloth_state.tension_std
            if tension_std > 1e-6:
                tension_delta = tension_delta / tension_std
        wrinkle_term = -np.abs(tension_delta)

    mode_nt = nontarget_mode.lower()
    if mode_nt == "penalize_contact":
        nontarget_term = -nontarget_mask
        head_term = -head_mask
    elif mode_nt == "reward_cover":
        nontarget_term = 2.0 * nontarget_mask - 1.0
        head_term = -head_mask
    elif mode_nt == "robe_points":
        if body_points_arr is None:
            raise ValueError("body_points must be provided when nontarget_mode='robe_points'")
        robe_nt, robe_head = compute_robe_nontarget_head_terms(
            cloth_state.positions,
            body_points_arr,
            initial_cloth_positions=cloth_state.initial_positions,
        )
        nontarget_term = robe_nt
        head_term = robe_head
    else:
        raise ValueError(
            f"Unsupported nontarget_mode '{nontarget_mode}'. Use 'penalize_contact', 'reward_cover', or 'robe_points'."
        )

    weighted_coverage = weights["coverage"] * coverage_term
    weighted_wrinkle = weights["wrinkle"] * wrinkle_term
    weighted_nontarget = weights["nontarget"] * nontarget_term
    weighted_head = weights["head"] * head_term

    combined = (
        weighted_coverage
        + weighted_wrinkle
        + weighted_nontarget
        + weighted_head
    )

    reward_field = combined

    if diffusion_iters > 0 and diffusion_edge_index is not None:
        reward_field = diffuse_reward_field(
            reward_field,
            diffusion_edge_index,
            alpha=diffusion_alpha,
            num_iters=diffusion_iters,
        )

    if not return_terms:
        return reward_field

    terms = {
        "weighted": {
            "coverage": weighted_coverage,
            "wrinkle": weighted_wrinkle,
            "nontarget": weighted_nontarget,
            "head": weighted_head,
        },
        "raw": {
            "coverage": coverage_term,
            "wrinkle": wrinkle_term,
            "nontarget": nontarget_term,
            "head": head_term,
        },
        "combined_pre_diff": combined,
        "weights": dict(weights),
    }
    if diffusion_iters > 0 and diffusion_edge_index is not None:
        terms["combined_post_diff"] = reward_field

    return reward_field, terms


def diffuse_reward_field(
    reward_field: np.ndarray,
    edge_index: np.ndarray,
    *,
    alpha: float = 0.5,
    num_iters: int = 1,
) -> np.ndarray:
    """Apply simple neighbor averaging diffusion to smooth reward fields.

    Args:
        reward_field: Array of shape ``(N,)``.
        edge_index: Array shaped ``(2, E)`` or ``(E, 2)`` describing directed edges.
        alpha: Mixing factor between original and neighbor mean (0→full neighbor).
        num_iters: Number of diffusion iterations.

    Returns:
        Smoothed reward field with the same shape as the input.
    """

    field = np.asarray(reward_field, dtype=np.float32)
    if field.ndim != 1:
        raise ValueError("reward_field must be a 1D array for diffusion")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    if num_iters <= 0:
        return field

    edges = np.asarray(edge_index)
    if edges.ndim != 2:
        raise ValueError("edge_index must be a 2D array")
    if edges.shape[0] == 2:
        src = edges[0].astype(np.int64)
        dst = edges[1].astype(np.int64)
    elif edges.shape[1] == 2:
        src = edges[:, 0].astype(np.int64)
        dst = edges[:, 1].astype(np.int64)
    else:
        raise ValueError("edge_index must have shape (2, E) or (E, 2)")

    if src.size == 0:
        return field

    n_vertices = field.shape[0]
    if np.max(src) >= n_vertices or np.max(dst) >= n_vertices:
        raise ValueError("edge_index contains vertex indices outside reward_field range")

    smoothed = field.copy()
    for _ in range(num_iters):
        neighbor_sum = np.zeros_like(smoothed)
        counts = np.zeros_like(smoothed)
        np.add.at(neighbor_sum, src, smoothed[dst])
        np.add.at(counts, src, 1)
        # If edges are undirected, we assume the reverse edges are present.
        # Use safe division to avoid runtime warnings for zero-degree nodes.
        mean_neighbor = smoothed.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(neighbor_sum, counts, out=mean_neighbor, where=counts > 0)
        smoothed = alpha * smoothed + (1.0 - alpha) * mean_neighbor

    return smoothed


def aggregate_reward_field(
    reward_field: np.ndarray,
    *,
    mode: str = "mean",
) -> float:
    """Reduce a reward field to a scalar compatible with RL trainers."""

    field = np.asarray(reward_field, dtype=np.float32)
    if field.size == 0:
        return 0.0

    if mode == "mean":
        return float(np.nanmean(field))
    if mode == "sum":
        return float(np.nansum(field))
    if mode == "max":
        return float(np.nanmax(field))

    raise ValueError(f"Unsupported aggregation mode: {mode}")
