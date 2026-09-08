import numpy as np

from assistive_gym.envs.bu_gnn_util import scale_action
from assistive_gym.envs.field_guided_policy import compute_field_guided_action


def world_to_policy_action(action_world, scale=(0.44, 1.05)):
    action_world = np.asarray(action_world, dtype=np.float32)
    scale_vec = np.array([scale[0], scale[1], scale[0], scale[1]], dtype=np.float32)
    return np.clip(action_world / scale_vec, -1.0, 1.0)


def find_cloth_bbox_corners(cloth_positions):
    cloth_positions = np.asarray(cloth_positions, dtype=np.float32)
    xy = cloth_positions[:, :2]
    min_x = float(np.min(xy[:, 0]))
    max_x = float(np.max(xy[:, 0]))
    min_y = float(np.min(xy[:, 1]))
    max_y = float(np.max(xy[:, 1]))
    bbox_targets = np.asarray(
        [
            [min_x, min_y],
            [min_x, max_y],
            [max_x, min_y],
            [max_x, max_y],
        ],
        dtype=np.float32,
    )

    corner_positions = []
    corner_indices = []
    used_indices = set()
    for target_xy in bbox_targets:
        dists = np.linalg.norm(xy - target_xy[None, :], axis=1)
        ordered = np.argsort(dists)
        chosen_idx = int(ordered[0])
        for candidate_idx in ordered:
            if int(candidate_idx) not in used_indices:
                chosen_idx = int(candidate_idx)
                break
        used_indices.add(chosen_idx)
        corner_indices.append(chosen_idx)
        corner_positions.append(np.asarray(cloth_positions[chosen_idx], dtype=np.float32))
    return corner_positions, corner_indices


def compute_field_guided_recover_action_from_states(input_cloth, all_body_points):
    cloth_positions = np.asarray(input_cloth, dtype=np.float32)
    target_center = np.mean(np.asarray(all_body_points, dtype=np.float32), axis=0)
    dist_to_target = np.linalg.norm(cloth_positions - target_center[None, :], axis=1)
    reward_field = -dist_to_target

    field_result = compute_field_guided_action(
        cloth_positions=cloth_positions,
        reward_field=reward_field,
        target_center=target_center,
        task_type='cover',
        threshold=0.0,
        step_size=0.16,
        debug_mode=False,
        p_id=None,
    )
    if field_result is None:
        raise RuntimeError("Field-guided action could not be computed")

    pick_pos, place_pos, force_direction = field_result
    action_world = np.array([pick_pos[0], pick_pos[1], place_pos[0], place_pos[1]], dtype=np.float32)
    action_policy = world_to_policy_action(action_world)
    debug = {
        'target_center': np.asarray(target_center, dtype=np.float32),
    }
    return action_policy, action_world, np.asarray(pick_pos, dtype=np.float32), np.asarray(place_pos, dtype=np.float32), np.asarray(force_direction, dtype=np.float32), debug


def _ensure_xyz_positions(cloth_positions):
    cloth_positions = np.asarray(cloth_positions, dtype=np.float32)
    if cloth_positions.ndim != 2 or cloth_positions.shape[1] not in {2, 3}:
        raise RuntimeError(f'Expected cloth positions with shape (N, 2) or (N, 3), got {cloth_positions.shape}')
    if cloth_positions.shape[1] == 2:
        cloth_positions = np.concatenate(
            [cloth_positions, np.zeros((cloth_positions.shape[0], 1), dtype=cloth_positions.dtype)],
            axis=1,
        )
    return cloth_positions


def compute_bottom_corner_recover_action_from_states(
    cloth_initial_positions,
    cloth_intermediate_positions,
    corner_stack_radius=0.08,
):
    cloth_initial_positions = _ensure_xyz_positions(cloth_initial_positions)
    cloth_intermediate_positions = _ensure_xyz_positions(cloth_intermediate_positions)
    if cloth_initial_positions.shape[0] != cloth_intermediate_positions.shape[0]:
        raise RuntimeError(
            'Initial and intermediate cloth positions must have matching vertex counts: '
            f'{cloth_initial_positions.shape[0]} != {cloth_intermediate_positions.shape[0]}'
        )

    corner_positions, corner_indices = find_cloth_bbox_corners(cloth_initial_positions)
    # In the simulator view used by replay, visual "bottom" corresponds to max-y.
    bottom_corner_ranks = [1, 3]
    bottom_corner_indices = [int(corner_indices[rank]) for rank in bottom_corner_ranks]
    bottom_initial_positions = [np.asarray(corner_positions[rank], dtype=np.float32) for rank in bottom_corner_ranks]
    bottom_intermediate_positions = [
        np.asarray(cloth_intermediate_positions[corner_idx], dtype=np.float32)
        for corner_idx in bottom_corner_indices
    ]

    xy = cloth_intermediate_positions[:, :2]
    stack_scores = []
    local_point_counts = []
    local_z_ranges = []
    displacements = []
    for initial_pos, intermediate_pos in zip(bottom_initial_positions, bottom_intermediate_positions):
        dists = np.linalg.norm(xy - intermediate_pos[:2][None, :], axis=1)
        local_idx = np.where(dists <= float(corner_stack_radius))[0]
        if local_idx.size == 0:
            local_z_range = np.inf
            local_count = 0
            stack_score = np.inf
        else:
            local_z = cloth_intermediate_positions[local_idx, 2]
            local_z_range = float(np.max(local_z) - np.min(local_z))
            local_count = int(local_idx.size)
            stack_score = float(local_z_range + 0.02 * local_count)
        stack_scores.append(stack_score)
        local_point_counts.append(local_count)
        local_z_ranges.append(local_z_range)
        displacements.append(float(np.linalg.norm(intermediate_pos[:2] - initial_pos[:2])))

    intermediate_midpoint = np.mean(np.asarray(bottom_intermediate_positions, dtype=np.float32), axis=0)
    initial_midpoint = np.mean(np.asarray(bottom_initial_positions, dtype=np.float32), axis=0)
    midpoint_dists = np.linalg.norm(cloth_intermediate_positions[:, :2] - intermediate_midpoint[:2][None, :], axis=1)
    midpoint_vertex_idx = int(np.argmin(midpoint_dists))
    pick_pos = np.asarray(cloth_intermediate_positions[midpoint_vertex_idx], dtype=np.float32)
    place_pos = np.asarray(cloth_initial_positions[midpoint_vertex_idx], dtype=np.float32)
    chosen_local_idx = -1
    selection_reason = 'bottom_edge_midpoint'
    move_vec = place_pos - pick_pos
    move_vec[2] = 0.0
    move_norm = float(np.linalg.norm(move_vec))
    force_direction = move_vec / move_norm if move_norm > 1e-8 else np.zeros(3, dtype=np.float32)
    action_world = np.array([pick_pos[0], pick_pos[1], place_pos[0], place_pos[1]], dtype=np.float32)
    action_policy = world_to_policy_action(action_world)

    debug = {
        'corner_source': 'initial_upper_y_as_visual_bottom',
        'corner_stack_radius': float(corner_stack_radius),
        'bottom_corner_indices': [int(idx) for idx in bottom_corner_indices],
        'bottom_corner_initial_positions': [np.asarray(pos, dtype=np.float32) for pos in bottom_initial_positions],
        'bottom_corner_intermediate_positions': [np.asarray(pos, dtype=np.float32) for pos in bottom_intermediate_positions],
        'bottom_corner_stack_scores': [float(score) for score in stack_scores],
        'bottom_corner_local_point_counts': [int(count) for count in local_point_counts],
        'bottom_corner_local_z_ranges': [float(z_range) for z_range in local_z_ranges],
        'bottom_corner_displacements': [float(disp) for disp in displacements],
        'chosen_bottom_corner_index': -1,
        'chosen_bottom_corner_rank': int(chosen_local_idx),
        'bottom_edge_initial_midpoint': np.asarray(initial_midpoint, dtype=np.float32),
        'bottom_edge_intermediate_midpoint': np.asarray(intermediate_midpoint, dtype=np.float32),
        'bottom_edge_midpoint_vertex_index': int(midpoint_vertex_idx),
        'bottom_edge_midpoint_vertex_distance': float(midpoint_dists[midpoint_vertex_idx]),
        'bottom_edge_midpoint_pick_vertex': np.asarray(pick_pos, dtype=np.float32),
        'bottom_edge_midpoint_place_vertex_initial': np.asarray(place_pos, dtype=np.float32),
        'selection_reason': selection_reason,
    }
    return action_policy, action_world, pick_pos, place_pos, np.asarray(force_direction, dtype=np.float32), debug


def compute_line_stacking_recover_action_from_states(
    cloth_intermediate_positions,
    uncover_action_policy,
    target_limb_code,
    cloth_initial_positions=None,
    line_width=0.06,
):
    cloth_positions = np.asarray(cloth_intermediate_positions, dtype=np.float32)
    if cloth_positions.shape[1] == 2:
        cloth_positions = np.concatenate(
            [cloth_positions, np.zeros((cloth_positions.shape[0], 1), dtype=cloth_positions.dtype)],
            axis=1,
        )

    cloth_initial_positions = None if cloth_initial_positions is None else np.asarray(cloth_initial_positions, dtype=np.float32)
    if cloth_initial_positions is not None and cloth_initial_positions.shape[1] == 2:
        cloth_initial_positions = np.concatenate(
            [cloth_initial_positions, np.zeros((cloth_initial_positions.shape[0], 1), dtype=cloth_initial_positions.dtype)],
            axis=1,
        )

    uncover_action_world = scale_action(np.asarray(uncover_action_policy, dtype=np.float32))
    line_start = np.array([uncover_action_world[0], uncover_action_world[1]], dtype=np.float32)
    line_end = np.array([uncover_action_world[2], uncover_action_world[3]], dtype=np.float32)
    line_vec = line_end - line_start
    line_len_sq = float(np.dot(line_vec, line_vec))
    if line_len_sq < 1e-8:
        raise RuntimeError('Uncover action is degenerate; cannot compute line recover action')

    xy = cloth_positions[:, :2]
    rel = xy - line_start[None, :]
    proj = np.clip(np.sum(rel * line_vec[None, :], axis=1) / line_len_sq, 0.0, 1.0)
    closest = line_start[None, :] + proj[:, None] * line_vec[None, :]
    dist_to_line = np.linalg.norm(xy - closest, axis=1)
    candidate_idx = np.where(dist_to_line <= float(line_width))[0]
    if candidate_idx.size == 0:
        candidate_idx = np.arange(cloth_positions.shape[0])

    local_heights = cloth_positions[candidate_idx, 2]
    worst_idx = candidate_idx[int(np.argmax(local_heights))]
    pick_pos = cloth_positions[worst_idx]
    uncover_start_3d = np.array([uncover_action_world[0], uncover_action_world[1], pick_pos[2]], dtype=np.float32)

    use_line_place_fallback = int(target_limb_code) in {12, 13, 14, 15}
    if use_line_place_fallback:
        corner_positions = []
        corner_indices = []
        chosen_corner_local_idx = -1
        place_pos = np.array([uncover_action_world[0], uncover_action_world[1], pick_pos[2]], dtype=np.float32)
        corner_source = 'uncover_start_fallback'
    elif cloth_initial_positions is None:
        corner_positions = []
        corner_indices = []
        chosen_corner_local_idx = -1
        place_pos = np.array([uncover_action_world[0], uncover_action_world[1], pick_pos[2]], dtype=np.float32)
        corner_source = 'uncover_start_no_initial'
    else:
        corner_positions, corner_indices = find_cloth_bbox_corners(cloth_initial_positions)
        corner_dists = np.asarray(
            [np.linalg.norm(np.asarray(corner[:2], dtype=np.float32) - uncover_start_3d[:2]) for corner in corner_positions],
            dtype=np.float32,
        )
        chosen_corner_local_idx = int(np.argmin(corner_dists))
        place_pos = np.asarray(corner_positions[chosen_corner_local_idx], dtype=np.float32)
        corner_source = 'cloth_initial'

    move_vec = place_pos - pick_pos
    move_vec[2] = 0.0
    move_norm = float(np.linalg.norm(move_vec))
    force_direction = move_vec / move_norm if move_norm > 1e-8 else np.zeros(3, dtype=np.float32)
    action_world = np.array([pick_pos[0], pick_pos[1], place_pos[0], place_pos[1]], dtype=np.float32)
    action_policy = world_to_policy_action(action_world)
    debug = {
        'line_start': line_start,
        'line_end': line_end,
        'dist_to_line': dist_to_line,
        'candidate_count': int(candidate_idx.size),
        'line_width': float(line_width),
        'uncover_start': uncover_start_3d,
        'corner_source': corner_source,
        'corner_indices': [int(idx) for idx in corner_indices],
        'corner_positions': [np.asarray(corner, dtype=np.float32) for corner in corner_positions],
        'chosen_corner_index': int(corner_indices[chosen_corner_local_idx]) if chosen_corner_local_idx >= 0 else -1,
        'chosen_corner_rank': int(chosen_corner_local_idx),
    }
    return action_policy, action_world, pick_pos, place_pos, np.asarray(force_direction, dtype=np.float32), debug
