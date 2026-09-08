import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str((THIS_DIR / '../assistive-gym-fem').resolve()))

from assistive_gym.learn import make_env
from assistive_gym.envs.bu_gnn_util import check_grasp_on_cloth, get_body_points_from_obs, scale_action
from recover_heuristics import (
    compute_bottom_corner_recover_action_from_states,
    compute_line_stacking_recover_action_from_states,
    world_to_policy_action,
)


def bool_arg(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if s in {'0', 'false', 'f', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {v}')


def load_pkl(path: Path):
    with path.open('rb') as handle:
        return pickle.load(handle)


def resolve_pkl(query: str, search_root: Path):
    candidate = Path(query).expanduser()
    if candidate.exists():
        return candidate.resolve()

    name = query if query.endswith('.pkl') else f'{query}.pkl'
    matches = sorted(search_root.rglob(name))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise RuntimeError(f'Multiple PKLs matched {query}: ' + ', '.join(str(m) for m in matches[:10]))
    raise FileNotFoundError(f'Could not find pkl: {query}')


def parse_seed_from_filename(path: Path):
    parts = path.stem.split('_')
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except Exception:
            return None
    return None


def parse_tl_from_filename(path: Path):
    first = path.stem.split('_')[0]
    if first.startswith('tl'):
        try:
            return int(first.replace('tl', ''))
        except Exception:
            return None
    return None


def invert_action_policy(action_policy):
    action_policy = np.asarray(action_policy, dtype=np.float32)
    if action_policy.shape[0] != 4:
        raise RuntimeError(f'Expected 4D action to invert, got shape={action_policy.shape}')
    return np.array([action_policy[2], action_policy[3], action_policy[0], action_policy[1]], dtype=np.float32)


def add_marker(env, pos, text, color):
    pos = np.asarray(pos, dtype=np.float32)
    d = 0.025
    p.addUserDebugText(text, pos + np.array([0, 0, 0.03], dtype=np.float32), textColorRGB=color, textSize=1.3, physicsClientId=env.id)
    p.addUserDebugLine(pos - np.array([d, 0, 0]), pos + np.array([d, 0, 0]), color, lineWidth=3, physicsClientId=env.id)
    p.addUserDebugLine(pos - np.array([0, d, 0]), pos + np.array([0, d, 0]), color, lineWidth=3, physicsClientId=env.id)


def has_mesh_state(mesh_state):
    return isinstance(mesh_state, (list, tuple)) and len(mesh_state) > 1


def configure_debug_camera(env):
    p.resetDebugVisualizerCamera(
        cameraDistance=1.8,
        cameraYaw=120,
        cameraPitch=-35,
        cameraTargetPosition=[0.0, 0.0, 0.95],
        physicsClientId=env.id,
    )


def print_stage_summary(stage_name, env, recover_action_world=None):
    print('=' * 80)
    if has_mesh_state(getattr(env, 'cloth_final', None)):
        _, uncover_reward, recover_reward, _, info = env.get_info()
        print(f'[{stage_name}] uncover_reward={uncover_reward} recover_reward={recover_reward}')
        print(f'[{stage_name}] grasp_on_cloth_uncover={info.get("grasp_on_cloth_uncover")} grasp_on_cloth_recover={info.get("grasp_on_cloth_recover")}')
    else:
        print(f'[{stage_name}] cloth_final not available yet; recover has not been executed.')
        print(
            f'[{stage_name}] grasp_on_cloth_uncover={getattr(env, "execute_uncover_action", None)} '
            f'grasp_on_cloth_recover={getattr(env, "execute_recover_action", None)}'
        )
        print(
            f'[{stage_name}] cloth_ready initial={has_mesh_state(getattr(env, "cloth_initial", None))} '
            f'intermediate={has_mesh_state(getattr(env, "cloth_intermediate", None))} '
            f'final={has_mesh_state(getattr(env, "cloth_final", None))}'
        )
    if recover_action_world is not None:
        _, is_on_cloth = check_grasp_on_cloth(recover_action_world, np.asarray(env.cloth_intermediate[1]))
        print(f'[{stage_name}] recover_precheck_on_cloth={bool(is_on_cloth)} recover_action_world={recover_action_world.tolist()}')
    print('=' * 80)


def maybe_wait(prompt, enabled=True):
    if enabled:
        input(prompt)


def drain_mouse_events(settle_polls=5, sleep_s=0.01):
    empty_polls = 0
    while empty_polls < settle_polls:
        events = p.getMouseEvents()
        if len(events) == 0:
            empty_polls += 1
        else:
            empty_polls = 0
        time.sleep(sleep_s)


def wait_for_left_button_release(settle_polls=5, sleep_s=0.01):
    released_polls = 0
    while released_polls < settle_polls:
        events = p.getMouseEvents()
        left_button_down = False
        for event in events:
            if len(event) < 5:
                continue
            _, _, _, button_idx, button_state = event[:5]
            if button_idx == 0 and int(button_state) != 0:
                left_button_down = True
                break
        if left_button_down:
            released_polls = 0
        else:
            released_polls += 1
        time.sleep(sleep_s)


def poll_click_point(env, prompt):
    print(prompt)
    print('Left-click in the PyBullet window. Press Ctrl+C to cancel.')
    drain_mouse_events()
    last_event = None
    while True:
        events = p.getMouseEvents()
        for event in events:
            if len(event) < 5:
                continue
            event_type, mouse_x, mouse_y, button_idx, button_state = event[:5]
            if button_idx != 0:
                continue
            if int(button_state) == 0:
                continue
            signature = (event_type, mouse_x, mouse_y, button_idx, button_state)
            if signature == last_event:
                continue
            last_event = signature
            ray_from, ray_to, _ = env.getRayFromTo(mouse_x, mouse_y)
            hit = p.rayTest(ray_from, ray_to, physicsClientId=env.id)[0]
            if hit[0] < 0:
                continue
            hit_pos = np.asarray(hit[3], dtype=np.float32)
            print(f'  clicked world xyz={hit_pos.tolist()}')
            wait_for_left_button_release()
            drain_mouse_events()
            return hit_pos
        time.sleep(0.01)


def run_manual_recover(env, pause_before_pick=True):
    maybe_wait('Press Enter to start interactive pick/place selection...', enabled=pause_before_pick)
    pick_pos = poll_click_point(env, 'Select PICK point on the uncovered cloth.')
    add_marker(env, pick_pos, 'Pick', [1, 0, 0])
    place_pos = poll_click_point(env, 'Select PLACE point for the cloth.')
    add_marker(env, place_pos, 'Place', [0, 1, 0])
    p.addUserDebugLine(pick_pos, place_pos, [0, 1, 0], lineWidth=4, physicsClientId=env.id)

    action_world = np.array([pick_pos[0], pick_pos[1], place_pos[0], place_pos[1]], dtype=np.float32)
    action_policy = world_to_policy_action(action_world)
    print(f'[Interactive] action_world={action_world.tolist()}')
    print(f'[Interactive] action_policy={action_policy.tolist()}')

    env.recover_step(action_policy)
    print_stage_summary('InteractiveRecover', env, recover_action_world=action_world)
    return action_policy, action_world


def replay_saved_recover(env, recover_action_policy):
    recover_action_policy = np.asarray(recover_action_policy, dtype=np.float32)
    recover_action_world = scale_action(recover_action_policy)
    add_marker(env, recover_action_world[0:2].tolist() + [0.58], 'SavedPick', [1, 0, 0])
    add_marker(env, recover_action_world[2:4].tolist() + [0.58], 'SavedPlace', [0, 1, 0])
    p.addUserDebugLine(
        np.array([recover_action_world[0], recover_action_world[1], 0.58], dtype=np.float32),
        np.array([recover_action_world[2], recover_action_world[3], 0.58], dtype=np.float32),
        [0, 1, 0],
        lineWidth=4,
        physicsClientId=env.id,
    )
    env.recover_step(recover_action_policy)
    print_stage_summary('SavedRecover', env, recover_action_world=recover_action_world)
    return recover_action_policy, recover_action_world


def replay_inverse_uncover_recover(env, uncover_action_policy):
    inverse_action_policy = invert_action_policy(uncover_action_policy)
    inverse_action_world = scale_action(inverse_action_policy)
    pick_marker = np.array([inverse_action_world[0], inverse_action_world[1], 0.58], dtype=np.float32)
    place_marker = np.array([inverse_action_world[2], inverse_action_world[3], 0.58], dtype=np.float32)
    add_marker(env, pick_marker, 'InvPick', [1, 0.5, 0])
    add_marker(env, place_marker, 'InvPlace', [0, 0.8, 1])
    p.addUserDebugLine(pick_marker, place_marker, [1, 0.8, 0], lineWidth=4, physicsClientId=env.id)
    print(f'[InverseUncover] action_policy={inverse_action_policy.tolist()}')
    print(f'[InverseUncover] action_world={inverse_action_world.tolist()}')
    env.recover_step(inverse_action_policy)
    print_stage_summary('InverseUncoverRecover', env, recover_action_world=inverse_action_world)
    return inverse_action_policy, inverse_action_world


def compute_line_stacking_recover_action(env, uncover_action_policy, line_width=0.06):
    return compute_line_stacking_recover_action_from_states(
        cloth_intermediate_positions=np.asarray(env.cloth_intermediate[1], dtype=np.float32),
        uncover_action_policy=uncover_action_policy,
        target_limb_code=int(getattr(env, "target_limb_code", -1)),
        cloth_initial_positions=np.asarray(env.cloth_initial[1], dtype=np.float32),
        line_width=line_width,
    )


def replay_field_recover(env, uncover_action_policy):
    field_action_policy, field_action_world, pick_pos, place_pos, force_direction, debug = compute_line_stacking_recover_action(
        env,
        uncover_action_policy,
    )
    pick_marker = np.array([pick_pos[0], pick_pos[1], pick_pos[2]], dtype=np.float32)
    place_marker = np.array([place_pos[0], place_pos[1], place_pos[2]], dtype=np.float32)
    add_marker(env, pick_marker, 'FieldPick', [1, 0.3, 0])
    add_marker(env, place_marker, 'FieldPlace', [0.2, 0.9, 0.2])
    p.addUserDebugLine(pick_marker, place_marker, [0.1, 0.9, 0.1], lineWidth=4, physicsClientId=env.id)
    p.addUserDebugLine(
        np.array([debug['line_start'][0], debug['line_start'][1], pick_pos[2]], dtype=np.float32),
        np.array([debug['line_end'][0], debug['line_end'][1], pick_pos[2]], dtype=np.float32),
        [0.9, 0.6, 0.1],
        lineWidth=2,
        physicsClientId=env.id,
    )
    print(f'[FieldWarmStart] action_policy={field_action_policy.tolist()}')
    print(f'[FieldWarmStart] action_world={field_action_world.tolist()}')
    print(f'[FieldWarmStart] force_direction={np.asarray(force_direction, dtype=np.float32).tolist()}')
    print(
        '[FieldWarmStart] '
        f'line_width={debug["line_width"]:.3f} candidate_count={debug["candidate_count"]} '
        f'uncover_start={debug["line_start"].tolist()} uncover_end={debug["line_end"].tolist()}'
    )
    if debug["chosen_corner_rank"] >= 0:
        print(
            '[FieldWarmStart] '
            f'corner_source={debug["corner_source"]} '
            f'chosen_corner_index={debug["chosen_corner_index"]} '
            f'chosen_corner_pos={np.asarray(debug["corner_positions"][debug["chosen_corner_rank"]], dtype=np.float32).tolist()}'
        )
    else:
        print(
            '[FieldWarmStart] '
            f'corner_source={debug["corner_source"]} '
            f'place_pos={place_pos.tolist()}'
        )
    env.recover_step(field_action_policy)
    print_stage_summary('FieldRecover', env, recover_action_world=field_action_world)
    return field_action_policy, field_action_world


def compute_bottom_corner_recover_action(env, corner_stack_radius=0.08):
    return compute_bottom_corner_recover_action_from_states(
        cloth_initial_positions=np.asarray(env.cloth_initial[1], dtype=np.float32),
        cloth_intermediate_positions=np.asarray(env.cloth_intermediate[1], dtype=np.float32),
        corner_stack_radius=corner_stack_radius,
    )


def replay_bottom_corner_recover(env, show_corner_candidates=False):
    action_policy, action_world, pick_pos, place_pos, force_direction, debug = compute_bottom_corner_recover_action(env)
    pick_marker = np.asarray(pick_pos, dtype=np.float32)
    place_marker = np.asarray(place_pos, dtype=np.float32)
    add_marker(env, pick_marker, 'EdgeMidPick', [1, 0.15, 0.15])
    add_marker(env, place_marker, 'EdgeMidPlace', [0.1, 0.8, 1.0])
    p.addUserDebugLine(pick_marker, place_marker, [0.1, 0.8, 1.0], lineWidth=4, physicsClientId=env.id)

    for rank, (initial_pos, intermediate_pos, stack_score) in enumerate(
        zip(
            debug['bottom_corner_initial_positions'],
            debug['bottom_corner_intermediate_positions'],
            debug['bottom_corner_stack_scores'],
        )
    ):
        initial_marker = np.asarray(initial_pos, dtype=np.float32)
        intermediate_marker = np.asarray(intermediate_pos, dtype=np.float32)
        if show_corner_candidates:
            add_marker(env, initial_marker, f'B{rank}Init', [0.2, 0.5, 1.0])
            add_marker(env, intermediate_marker, f'B{rank}Now', [1.0, 0.7, 0.1])
            p.addUserDebugLine(
                initial_marker,
                intermediate_marker,
                [0.8, 0.8, 0.1],
                lineWidth=2,
                physicsClientId=env.id,
            )
        print(
            f'[BottomCorner] candidate_rank={rank} '
            f'index={debug["bottom_corner_indices"][rank]} '
            f'stack_score={stack_score:.4f} '
            f'local_count={debug["bottom_corner_local_point_counts"][rank]} '
            f'z_range={debug["bottom_corner_local_z_ranges"][rank]:.4f} '
            f'displacement={debug["bottom_corner_displacements"][rank]:.4f}'
        )

    print(f'[BottomCorner] chosen_corner_rank={debug["chosen_bottom_corner_rank"]}')
    print(f'[BottomCorner] chosen_corner_index={debug["chosen_bottom_corner_index"]}')
    print(f'[BottomCorner] initial_pos={np.asarray(debug["bottom_edge_initial_midpoint"], dtype=np.float32).tolist()}')
    print(f'[BottomCorner] intermediate_pos={np.asarray(debug["bottom_edge_intermediate_midpoint"], dtype=np.float32).tolist()}')
    print(f'[BottomCorner] stack_scores={debug["bottom_corner_stack_scores"]}')
    print(f'[BottomCorner] selection_reason={debug["selection_reason"]} corner_stack_radius={debug["corner_stack_radius"]:.3f}')
    print(
        f'[BottomCorner] midpoint_vertex_index={debug["bottom_edge_midpoint_vertex_index"]} '
        f'midpoint_vertex_distance={debug["bottom_edge_midpoint_vertex_distance"]:.4f}'
    )
    print(f'[BottomCorner] force_direction={np.asarray(force_direction, dtype=np.float32).tolist()}')
    print(f'[BottomCorner] action_policy={action_policy.tolist()}')
    print(f'[BottomCorner] action_world={action_world.tolist()}')

    env.recover_step(action_policy)
    print_stage_summary('BottomCornerRecover', env, recover_action_world=action_world)
    return action_policy, action_world


def main():
    parser = argparse.ArgumentParser(description='Replay a specific recover rollout in GUI and optionally select a manual pick/place after uncover.')
    parser.add_argument('--pkl', required=True, help='Path to a raw rollout pkl, or just its filename/stem to search for')
    parser.add_argument('--search-root', default=str(THIS_DIR), help='Root directory used when --pkl is not a direct path')
    parser.add_argument('--env-name', default='RobeReversible-v1')
    parser.add_argument('--blanket-pose-var', type=bool_arg, default=False)
    parser.add_argument('--high-pose-var', type=bool_arg, default=False)
    parser.add_argument('--body-shape-var', type=bool_arg, default=False)
    parser.add_argument('--singulate-layers', type=bool_arg, default=True)
    parser.add_argument('--pause-after-reset', type=bool_arg, default=True)
    parser.add_argument('--pause-after-uncover', type=bool_arg, default=True)
    parser.add_argument('--show-corner-candidates', type=bool_arg, default=False)
    parser.add_argument(
        '--mode',
        choices=['saved', 'interactive', 'inverse-uncover', 'field', 'bottom-corner', 'prompt', 'uncover-only'],
        default='prompt',
    )
    args = parser.parse_args()

    pkl_path = resolve_pkl(args.pkl, Path(args.search_root).expanduser().resolve())
    raw = load_pkl(pkl_path)

    seed = int(raw.get('seed', parse_seed_from_filename(pkl_path)))
    target_limb_code = int(raw.get('target_limb_code', parse_tl_from_filename(pkl_path)))
    uncover_action = np.asarray(raw.get('uncover_action', []), dtype=np.float32)
    recover_action = np.asarray(raw.get('recover_action', []), dtype=np.float32)
    if uncover_action.shape[0] != 4:
        raise RuntimeError(f'Invalid uncover action in {pkl_path}')

    env = make_env(args.env_name, coop=False, seed=seed)
    env.render()
    env.set_env_variations(
        collect_data=False,
        blanket_pose_var=args.blanket_pose_var,
        high_pose_var=args.high_pose_var,
        body_shape_var=args.body_shape_var,
    )
    env.set_singulate(args.singulate_layers)
    env.set_target_limb_code(target_limb_code)
    env.set_recover(True)
    env.set_seed_val(seed)

    try:
        env.reset()
        env.set_release_sim_steps(post_release_steps=5)

        configure_debug_camera(env)
        print(f'[Replay] pkl={pkl_path}')
        print(f'[Replay] seed={seed} target_limb_code={target_limb_code}')
        print(f'[Replay] uncover_action_policy={uncover_action.tolist()}')
        print(f'[Replay] recover_action_policy={recover_action.tolist() if recover_action.shape[0] == 4 else None}')
        maybe_wait('Press Enter to execute uncover in GUI...', enabled=args.pause_after_reset)

        env.uncover_step(uncover_action)
        print_stage_summary('AfterUncover', env)
        maybe_wait('Inspect the uncovered state, then press Enter to continue...', enabled=args.pause_after_uncover)

        chosen_mode = args.mode
        if chosen_mode == 'prompt':
            print('Choose next step: [s]aved recover / [i]nteractive pick-place / [v] inverse-uncover / [f]ield warm-start / [b]ottom-corner / [u]ncover-only')
            choice = input('Mode [s/i/v/f/b/u]: ').strip().lower()
            chosen_mode = {
                's': 'saved',
                'i': 'interactive',
                'v': 'inverse-uncover',
                'f': 'field',
                'b': 'bottom-corner',
                'u': 'uncover-only',
            }.get(choice, 'saved')

        if chosen_mode == 'saved':
            if recover_action.shape[0] != 4:
                raise RuntimeError('Selected mode=saved but recover_action is missing in the pkl')
            replay_saved_recover(env, recover_action)
        elif chosen_mode == 'interactive':
            run_manual_recover(env, pause_before_pick=False)
        elif chosen_mode == 'inverse-uncover':
            replay_inverse_uncover_recover(env, uncover_action)
        elif chosen_mode == 'field':
            replay_field_recover(env, uncover_action)
        elif chosen_mode == 'bottom-corner':
            replay_bottom_corner_recover(env, show_corner_candidates=args.show_corner_candidates)
        elif chosen_mode == 'uncover-only':
            print('[Replay] Leaving the environment at the uncovered state. No recover executed.')
        else:
            raise ValueError(f'Unsupported mode: {chosen_mode}')

        print('GUI remains open. Press Enter in this terminal to close the environment.')
        input()
    finally:
        try:
            env.disconnect()
        except Exception:
            try:
                env.close()
            except Exception:
                pass


if __name__ == '__main__':
    main()
