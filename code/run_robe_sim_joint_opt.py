import argparse
import json
import os.path as osp
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '../assistive-gym-fem')


torch = None
check_grasp_on_cloth = None
get_body_points_from_obs = None
get_covered_status = None
get_recovering_reward = None
get_uncovering_reward = None
scale_action = None
make_env = None
Runtime_Graph = None
compute_fscore_uncover = None
compute_fscore_recover = None
save_data_to_pickle = None
set_x0_for_cmaes = None
GNN_Manager = None
compute_field_guided_action = None


target_limb_list = [2, 4, 5, 8, 10, 11, 12, 13, 14, 15]

all_graph_configs = {
    '2D': {'filt_drape': False, 'rot_drape': True, 'use_3D': False, 'use_disp': True},
    '3D': {'filt_drape': False, 'rot_drape': False, 'use_3D': True, 'use_disp': True},
}

all_env_vars = {
    'standard': {'blanket_var': False, 'high_pose_var': False, 'body_shape_var': False},
    'body_shape_var': {'blanket_var': False, 'high_pose_var': False, 'body_shape_var': True},
    'pose_var': {'blanket_var': False, 'high_pose_var': True, 'body_shape_var': False},
    'blanket_var': {'blanket_var': True, 'high_pose_var': False, 'body_shape_var': False},
    'combo_var': {'blanket_var': True, 'high_pose_var': True, 'body_shape_var': True},
}

INVALID_REWARD = -1e9
SCREENED_REWARD_BASE = -1e8
REWARD_EPS = 1e-6
ACTION_DEDUP_EPS = 0.05
BASELINE_FILENAME_RE = re.compile(r'^tl(?P<tl>\d+)_c(?P<idx>\d+)_(?P<seed>\d+)_pid(?P<pid>\d+)\.pkl$')


def load_runtime_dependencies():
    global torch
    global check_grasp_on_cloth
    global get_body_points_from_obs
    global get_covered_status
    global get_recovering_reward
    global get_uncovering_reward
    global scale_action
    global make_env
    global Runtime_Graph
    global compute_fscore_uncover
    global compute_fscore_recover
    global save_data_to_pickle
    global set_x0_for_cmaes
    global GNN_Manager
    global compute_field_guided_action

    if Runtime_Graph is not None:
        return

    import torch as torch_module
    from assistive_gym.envs.bu_gnn_util import (
        check_grasp_on_cloth as check_grasp_on_cloth_fn,
        get_body_points_from_obs as get_body_points_from_obs_fn,
        get_covered_status as get_covered_status_fn,
        get_recovering_reward as get_recovering_reward_fn,
        get_uncovering_reward as get_uncovering_reward_fn,
        scale_action as scale_action_fn,
    )
    from assistive_gym.envs.field_guided_policy import compute_field_guided_action as compute_field_guided_action_fn
    from assistive_gym.learn import make_env as make_env_fn
    from build_runtime_graph import Runtime_Graph as Runtime_Graph_cls
    from cma_gnn_util import (
        compute_fscore_uncover as compute_fscore_uncover_fn,
        compute_fscore_recover as compute_fscore_recover_fn,
        save_data_to_pickle as save_data_to_pickle_fn,
        set_x0_for_cmaes as set_x0_for_cmaes_fn,
    )
    from gnn_manager import GNN_Manager as GNN_Manager_cls

    torch = torch_module
    check_grasp_on_cloth = check_grasp_on_cloth_fn
    get_body_points_from_obs = get_body_points_from_obs_fn
    get_covered_status = get_covered_status_fn
    get_recovering_reward = get_recovering_reward_fn
    get_uncovering_reward = get_uncovering_reward_fn
    scale_action = scale_action_fn
    make_env = make_env_fn
    Runtime_Graph = Runtime_Graph_cls
    compute_fscore_uncover = compute_fscore_uncover_fn
    compute_fscore_recover = compute_fscore_recover_fn
    save_data_to_pickle = save_data_to_pickle_fn
    set_x0_for_cmaes = set_x0_for_cmaes_fn
    GNN_Manager = GNN_Manager_cls
    compute_field_guided_action = compute_field_guided_action_fn


def para_to_action(para):
    return np.array([para['x_i'], para['y_i'], para['x_f'], para['y_f']], dtype=np.float32)


def action_to_para(action_policy):
    action_policy = np.asarray(action_policy, dtype=np.float32)
    return {
        'x_i': float(action_policy[0]),
        'y_i': float(action_policy[1]),
        'x_f': float(action_policy[2]),
        'y_f': float(action_policy[3]),
    }


def build_search_space(step_size):
    return {
        'x_i': np.arange(-1, 1 + step_size, step_size),
        'y_i': np.arange(-1, 1 + step_size, step_size),
        'x_f': np.arange(-1, 1 + step_size, step_size),
        'y_f': np.arange(-1, 1 + step_size, step_size),
    }


def ensure_2d(cloth):
    cloth = np.asarray(cloth)
    if cloth.shape[1] == 2:
        return cloth
    return np.delete(cloth, 2, axis=1)


def ensure_3d(cloth, reference_3d=None):
    cloth = np.asarray(cloth)
    if cloth.shape[1] == 3:
        return cloth
    if reference_3d is not None:
        ref = np.asarray(reference_3d)
        if ref.shape[0] != cloth.shape[0]:
            zeros = np.zeros((cloth.shape[0], 1), dtype=cloth.dtype)
            return np.concatenate([cloth, zeros], axis=1)
        cloth_3d = np.array(ref, copy=True)
        cloth_3d[:, :2] = cloth[:, :2]
        return cloth_3d
    zeros = np.zeros((cloth.shape[0], 1), dtype=cloth.dtype)
    return np.concatenate([cloth, zeros], axis=1)


def world_to_policy_action(action_world, scale=(0.44, 1.05)):
    action_world = np.asarray(action_world, dtype=np.float32)
    scale_vec = np.array(list(scale) * 2, dtype=np.float32)
    action_policy = action_world / scale_vec
    return np.clip(action_policy, -1.0, 1.0)


def safe_mean(values, default=np.nan):
    if len(values) == 0:
        return float(default)
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def has_finite_reward(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def action_distance_inf(action_a, action_b):
    action_a = np.asarray(action_a, dtype=np.float32)
    action_b = np.asarray(action_b, dtype=np.float32)
    return float(np.max(np.abs(action_a - action_b)))


def dedupe_action_samples(samples, action_key, eps=ACTION_DEDUP_EPS):
    deduped = []
    for sample in samples:
        action = np.asarray(sample[action_key], dtype=np.float32)
        if any(action_distance_inf(action, existing[action_key]) < eps for existing in deduped):
            continue
        deduped.append(sample)
    return deduped


def parse_baseline_filename(path_obj):
    match = BASELINE_FILENAME_RE.match(path_obj.name)
    if match is None:
        return None
    return {
        'target_limb_code': int(match.group('tl')),
        'eval_idx': int(match.group('idx')),
        'seed': int(match.group('seed')),
        'pid': int(match.group('pid')),
    }


def load_baseline_action_pool(raw_dir):
    raw_path = Path(raw_dir)
    pool = {
        'raw_dir': str(raw_path),
        'available': False,
        'by_limb': {},
        'stats': {
            'num_files': 0,
            'num_loaded': 0,
            'num_skipped_bad_name': 0,
            'num_skipped_read_error': 0,
            'num_skipped_invalid_fields': 0,
        },
    }
    if not raw_path.exists() or not raw_path.is_dir():
        pool['stats']['missing_dir'] = True
        return pool

    samples_by_limb = {tl: [] for tl in target_limb_list}
    files = sorted(raw_path.glob('*.pkl'))
    pool['stats']['num_files'] = len(files)

    for path_obj in files:
        meta = parse_baseline_filename(path_obj)
        if meta is None:
            pool['stats']['num_skipped_bad_name'] += 1
            continue
        try:
            with open(path_obj, 'rb') as handle:
                raw_data = pickle.load(handle)
        except Exception:
            pool['stats']['num_skipped_read_error'] += 1
            continue

        tl = int(raw_data.get('target_limb_code', meta['target_limb_code']))
        uncover_action = np.asarray(raw_data.get('uncover_action', []), dtype=np.float32)
        recover_action = np.asarray(raw_data.get('recover_action', []), dtype=np.float32)
        cma_info = raw_data.get('cma_info', {})
        best_reward = cma_info.get('best_reward', cma_info.get('pred_recover_reward', np.nan))

        if (
            tl not in target_limb_list or
            uncover_action.shape != (4,) or
            recover_action.shape != (4,) or
            not has_finite_reward(best_reward)
        ):
            pool['stats']['num_skipped_invalid_fields'] += 1
            continue

        sample = {
            'path': str(path_obj),
            'filename': path_obj.name,
            'seed': int(meta['seed']),
            'eval_idx': int(meta['eval_idx']),
            'target_limb_code': tl,
            'uncover_action': np.clip(uncover_action.astype(np.float32), -1.0, 1.0),
            'recover_action': np.clip(recover_action.astype(np.float32), -1.0, 1.0),
            'best_reward': float(best_reward),
        }
        samples_by_limb[tl].append(sample)
        pool['stats']['num_loaded'] += 1

    for tl, samples in samples_by_limb.items():
        samples.sort(key=lambda sample: sample['best_reward'], reverse=True)
    pool['available'] = pool['stats']['num_loaded'] > 0
    pool['by_limb'] = samples_by_limb
    return pool


def summarize_baseline_limb_pool(pool, target_limb_code):
    if not pool or not pool.get('available', False):
        return {
            'available': False,
            'num_samples': 0,
            'num_unique_uncover': 0,
            'num_recover_candidates': 0,
        }
    limb_samples = list(pool['by_limb'].get(int(target_limb_code), []))
    unique_uncover = dedupe_action_samples(limb_samples, 'uncover_action')
    return {
        'available': len(limb_samples) > 0,
        'num_samples': int(len(limb_samples)),
        'num_unique_uncover': int(len(unique_uncover)),
        'num_recover_candidates': int(len(limb_samples)),
    }


def select_outer_baseline_seed_samples(pool, target_limb_code, num_seeds, selection_mode, rng):
    if pool is None or not pool.get('available', False) or num_seeds <= 0:
        return []
    limb_samples = list(pool['by_limb'].get(int(target_limb_code), []))
    if len(limb_samples) == 0:
        return []
    unique_samples = dedupe_action_samples(limb_samples, 'uncover_action')
    if selection_mode == 'random':
        order = rng.permutation(len(unique_samples))
        unique_samples = [unique_samples[idx] for idx in order]
    elif selection_mode == 'best_reward':
        unique_samples.sort(key=lambda sample: sample['best_reward'], reverse=True)
    else:
        raise ValueError(f'Unsupported baseline seed selection mode: {selection_mode}')
    return unique_samples[:max(0, int(num_seeds))]


def collect_baseline_recover_candidates(
    pool,
    target_limb_code,
    topk,
    paired_sample=None,
    include_pool=True,
):
    candidates = []

    def append_unique(action_policy, source, sample):
        action_policy = np.clip(np.asarray(action_policy, dtype=np.float32), -1.0, 1.0)
        if action_policy.shape != (4,):
            return
        if any(action_distance_inf(action_policy, existing['action_policy']) < ACTION_DEDUP_EPS for existing in candidates):
            return
        candidates.append({
            'action_policy': action_policy,
            'source': source,
            'sample': sample,
        })

    if paired_sample is not None:
        append_unique(paired_sample['recover_action'], 'paired_baseline', paired_sample)

    if include_pool and pool is not None and pool.get('available', False) and topk > 0:
        limb_samples = list(pool['by_limb'].get(int(target_limb_code), []))
        for sample in limb_samples[:max(0, int(topk))]:
            append_unique(sample['recover_action'], 'limb_baseline_pool', sample)

    return candidates


def is_valid_cloth(cloth, expected_cols=None):
    try:
        cloth = np.asarray(cloth, dtype=np.float32)
    except Exception:
        return False
    if cloth.ndim != 2 or cloth.shape[0] == 0:
        return False
    if expected_cols is not None and cloth.shape[1] != expected_cols:
        return False
    return bool(np.all(np.isfinite(cloth)))


def is_valid_reward(value):
    return isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value)


def compute_rollout_f1_metrics(all_body_points, cloth_initial, cloth_intermediate, cloth_final):
    load_runtime_dependencies()
    try:
        cloth_initial_2d = ensure_2d(np.asarray(cloth_initial, dtype=np.float32))
        cloth_intermediate_2d = ensure_2d(np.asarray(cloth_intermediate, dtype=np.float32))
        cloth_final_2d = ensure_2d(np.asarray(cloth_final, dtype=np.float32))
        initial_status = get_covered_status(all_body_points, cloth_initial_2d)
        intermediate_status = get_covered_status(all_body_points, cloth_intermediate_2d)
        final_status = get_covered_status(all_body_points, cloth_final_2d)
        uncover_f1 = float(compute_fscore_uncover(initial_status, intermediate_status))
        recover_f1 = float(compute_fscore_recover(initial_status, intermediate_status, final_status, False))
        return {
            'uncover_f1': uncover_f1,
            'recover_f1': recover_f1,
            'initial_status': initial_status,
            'intermediate_status': intermediate_status,
            'final_status': final_status,
        }
    except Exception:
        return {
            'uncover_f1': float('nan'),
            'recover_f1': float('nan'),
            'initial_status': None,
            'intermediate_status': None,
            'final_status': None,
        }


def to_serializable(value, seen=None):
    if seen is None:
        seen = set()

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)

    obj_id = id(value)
    recursive = isinstance(value, (dict, list, tuple, set, np.ndarray))
    if recursive:
        if obj_id in seen:
            return '<cycle>'
        seen.add(obj_id)

    try:
        if isinstance(value, np.ndarray):
            return to_serializable(value.tolist(), seen)
        if isinstance(value, dict):
            return {str(key): to_serializable(val, seen) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [to_serializable(item, seen) for item in value]
        if hasattr(value, 'item') and callable(value.item):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)
    finally:
        if recursive:
            seen.remove(obj_id)


def build_graph_reference(cloth_input, graph_config, graph_root, description, use_3d_override=None):
    load_runtime_dependencies()
    use_3d = graph_config['use_3D'] if use_3d_override is None else use_3d_override
    graph = Runtime_Graph(
        root=graph_root,
        description=description,
        voxel_size=0.05,
        edge_threshold=0.06,
        action_to_all=True,
        cloth_initial=cloth_input,
        filter_draping=graph_config['filt_drape'],
        rot_draping=graph_config['rot_drape'],
        use_3D=use_3d,
    )
    return np.asarray(graph.initial_blanket_state), graph


def predict_cloth_state(action_policy, cloth_input, graph_config, model, device, graph_root, description):
    load_runtime_dependencies()
    use_3d = graph_config['use_3D']
    use_disp = graph_config['use_disp']

    action_policy = np.asarray(action_policy, dtype=np.float32)
    action_world = scale_action(action_policy)
    cloth_input = np.asarray(cloth_input)
    _, is_on_cloth = check_grasp_on_cloth(action_world, cloth_input)

    cloth_initial_graph, graph = build_graph_reference(
        cloth_input=cloth_input,
        graph_config=graph_config,
        graph_root=graph_root,
        description=description,
    )
    graph_initial_3d = cloth_initial_graph if use_3d else build_graph_reference(
        cloth_input=cloth_input,
        graph_config=graph_config,
        graph_root=graph_root,
        description=f'{description}_3dref',
        use_3d_override=True,
    )[0]

    if is_on_cloth:
        data = graph.build_graph(action_policy)
        data = data.to(device).to_dict()
        batch = data['batch']
        batch_num = np.max(batch.data.cpu().numpy()) + 1
        global_vec = torch.zeros(int(batch_num), 0, dtype=torch.float32, device=device)
        data['u'] = global_vec
        pred = model(data)['target'].detach().cpu().numpy()
        if use_disp:
            pred = cloth_initial_graph + pred
    else:
        pred = np.copy(cloth_initial_graph)

    pred_2d = ensure_2d(pred)
    graph_initial_2d = ensure_2d(cloth_initial_graph)
    pred_3d = ensure_3d(pred, reference_3d=graph_initial_3d)
    return {
        'pred': pred,
        'pred_2d': pred_2d,
        'pred_3d': pred_3d,
        'graph_initial': cloth_initial_graph,
        'graph_initial_2d': graph_initial_2d,
        'graph_initial_3d': graph_initial_3d,
        'is_on_cloth': bool(is_on_cloth),
        'action_world': action_world,
        'action_policy': action_policy,
    }


def build_warm_start_candidates(reference_action_policy, input_cloth, all_body_points, strategy, task_type='cover'):
    load_runtime_dependencies()
    reference_action_policy = np.asarray(reference_action_policy, dtype=np.float32)
    reverse_action = np.array(
        [reference_action_policy[2], reference_action_policy[3], reference_action_policy[0], reference_action_policy[1]],
        dtype=np.float32,
    )
    reverse_dict = action_to_para(reverse_action)

    if strategy == 'none':
        return []

    warm_starts = []
    if strategy in ('reverse', 'hybrid'):
        warm_starts.append({'para': reverse_dict, 'source': 'reverse'})

    if strategy in ('field', 'hybrid'):
        try:
            cloth_positions = np.asarray(input_cloth, dtype=np.float32)
            target_center = np.mean(np.asarray(all_body_points, dtype=np.float32), axis=0)
            dist_to_target = np.linalg.norm(cloth_positions - target_center[None, :], axis=1)
            reward_field = -dist_to_target
            field_result = compute_field_guided_action(
                cloth_positions=cloth_positions,
                reward_field=reward_field,
                target_center=target_center,
                task_type=task_type,
                threshold=0.0,
                step_size=0.16,
                debug_mode=False,
                p_id=None,
            )
            if field_result is not None:
                pick_pos, place_pos, _ = field_result
                action_world = np.array([pick_pos[0], pick_pos[1], place_pos[0], place_pos[1]], dtype=np.float32)
                warm_starts.append({'para': action_to_para(world_to_policy_action(action_world)), 'source': 'field'})
        except Exception as exc:
            print(f"[WarmStart] Field-guided warm start failed, using fallback: {exc}")

    if len(warm_starts) == 0:
        warm_starts.append({'para': reverse_dict, 'source': 'reverse'})
    return warm_starts


def evaluate_uncover_candidate(
    rollout_idx,
    eval_idx,
    action_policy,
    initial_cloth_3d,
    all_body_points,
    initial_covered_status,
    uncover_model,
    graph_config,
    device,
    graph_root,
):
    load_runtime_dependencies()
    action_policy = np.clip(np.asarray(action_policy, dtype=np.float32), -1.0, 1.0)
    detail = {
        'eval_idx': int(eval_idx),
        'action_policy': action_policy.copy(),
        'action_world': scale_action(action_policy).copy(),
        'is_on_cloth': False,
        'reward': float(INVALID_REWARD),
        'f1': 0.0,
        'status': 'invalid_uncover',
        'pred_intermediate_3d': np.asarray(initial_cloth_3d, dtype=np.float32).copy(),
        'covered_status': None,
    }

    try:
        pred_info = predict_cloth_state(
            action_policy=action_policy,
            cloth_input=initial_cloth_3d,
            graph_config=graph_config,
            model=uncover_model,
            device=device,
            graph_root=graph_root,
            description=f'rollout_{rollout_idx}_eval_{eval_idx}_uncover',
        )
    except Exception as exc:
        detail['status'] = f'predict_exception:{type(exc).__name__}'
        return detail

    detail['action_world'] = pred_info['action_world'].copy()
    detail['is_on_cloth'] = bool(pred_info['is_on_cloth'])
    detail['pred_intermediate_3d'] = np.asarray(pred_info['pred_3d']).copy()

    if not pred_info['is_on_cloth']:
        detail['status'] = 'off_cloth_uncover'
        return detail

    if not is_valid_cloth(pred_info['pred_2d'], expected_cols=2) or not is_valid_cloth(pred_info['pred_3d'], expected_cols=3):
        detail['status'] = 'invalid_uncover_prediction'
        return detail

    try:
        reward, uncover_status = get_uncovering_reward(
            pred_info['action_world'],
            all_body_points,
            pred_info['graph_initial_2d'],
            pred_info['pred_2d'],
        )
        uncover_f1 = compute_fscore_uncover(initial_covered_status, uncover_status)
    except Exception as exc:
        detail['status'] = f'uncover_reward_exception:{type(exc).__name__}'
        return detail

    if not is_valid_reward(reward) or not np.isfinite(uncover_f1):
        detail['status'] = 'invalid_uncover_reward'
        return detail

    detail.update({
        'reward': float(reward),
        'f1': float(uncover_f1),
        'status': 'valid_uncover',
        'covered_status': uncover_status,
    })
    return detail


def evaluate_recover_candidate(
    rollout_idx,
    eval_idx,
    action_policy,
    intermediate_cloth_3d,
    original_initial_cloth_graph,
    initial_covered_status,
    all_body_points,
    recover_model,
    graph_config,
    device,
    graph_root,
    description_suffix='recover',
):
    load_runtime_dependencies()
    action_policy = np.clip(np.asarray(action_policy, dtype=np.float32), -1.0, 1.0)
    detail = {
        'eval_idx': int(eval_idx),
        'action_policy': action_policy.copy(),
        'action_world': scale_action(action_policy).copy(),
        'is_on_cloth': False,
        'reward': float(INVALID_REWARD),
        'f1': float('nan'),
        'status': 'invalid_recover',
        'pred_final_3d': np.asarray(intermediate_cloth_3d, dtype=np.float32).copy(),
        'covered_status': None,
    }

    try:
        pred_info = predict_cloth_state(
            action_policy=action_policy,
            cloth_input=intermediate_cloth_3d,
            graph_config=graph_config,
            model=recover_model,
            device=device,
            graph_root=graph_root,
            description=f'rollout_{rollout_idx}_eval_{eval_idx}_{description_suffix}',
        )
    except Exception as exc:
        detail['status'] = f'predict_exception:{type(exc).__name__}'
        return detail

    detail['action_world'] = pred_info['action_world'].copy()
    detail['is_on_cloth'] = bool(pred_info['is_on_cloth'])
    detail['pred_final_3d'] = np.asarray(pred_info['pred_3d']).copy()

    if not pred_info['is_on_cloth']:
        detail['status'] = 'off_cloth_recover'
        return detail

    if not is_valid_cloth(pred_info['pred_2d'], expected_cols=2) or not is_valid_cloth(pred_info['pred_3d'], expected_cols=3):
        detail['status'] = 'invalid_recover_prediction'
        return detail

    try:
        reward, recover_status = get_recovering_reward(
            pred_info['action_world'],
            all_body_points,
            ensure_2d(original_initial_cloth_graph),
            pred_info['graph_initial_2d'],
            pred_info['pred_2d'],
        )
        intermediate_status = get_covered_status(all_body_points, pred_info['graph_initial_2d'])
        recover_f1 = compute_fscore_recover(initial_covered_status, intermediate_status, recover_status, False)
    except Exception as exc:
        detail['status'] = f'recover_reward_exception:{type(exc).__name__}'
        detail['exception_message'] = str(exc)
        return detail

    if not is_valid_reward(reward) or not np.isfinite(recover_f1):
        detail['status'] = 'invalid_recover_reward'
        return detail

    detail.update({
        'reward': float(reward),
        'f1': float(recover_f1),
        'status': 'valid_recover',
        'covered_status': recover_status,
    })
    return detail


def split_joint_action(action_joint):
    action_joint = np.asarray(action_joint, dtype=np.float32)
    return action_joint[:4], action_joint[4:8]


def build_joint_x0(target_limb_code):
    load_runtime_dependencies()
    uncover_x0 = np.asarray(set_x0_for_cmaes(target_limb_code), dtype=np.float32)
    recover_x0 = np.asarray([uncover_x0[2], uncover_x0[3], uncover_x0[0], uncover_x0[1]], dtype=np.float32)
    return np.concatenate([uncover_x0, recover_x0], axis=0)


def optimize_recover_given_uncover(
    rollout_idx,
    outer_eval_idx,
    uncover_action_policy,
    intermediate_cloth_3d,
    original_initial_cloth_graph,
    initial_covered_status,
    all_body_points,
    recover_model,
    graph_config,
    device,
    max_fevals,
    graph_root,
    search_method,
    warm_start_strategy,
    feasible_only_best,
    popsize,
    sigma,
    baseline_recover_candidates=None,
):
    import cma
    import gradient_free_optimizers as gfo

    warm_start_entries = []
    if baseline_recover_candidates is None:
        baseline_recover_candidates = []
    for candidate in baseline_recover_candidates:
        if isinstance(candidate, dict):
            action_policy = candidate['action_policy']
            source = candidate.get('source', 'baseline')
            sample = candidate.get('sample')
        else:
            action_policy = candidate
            source = 'baseline'
            sample = None
        warm_start_entries.append({
            'para': action_to_para(action_policy),
            'source': source,
            'sample': sample,
        })

    warm_start_entries.extend(build_warm_start_candidates(
        reference_action_policy=uncover_action_policy,
        input_cloth=intermediate_cloth_3d,
        all_body_points=all_body_points,
        strategy=warm_start_strategy,
        task_type='cover',
    ))
    deduped_warm_start_entries = []
    for entry in warm_start_entries:
        action_policy = para_to_action(entry['para'])
        if any(action_distance_inf(action_policy, para_to_action(existing['para'])) < ACTION_DEDUP_EPS for existing in deduped_warm_start_entries):
            continue
        deduped_warm_start_entries.append(entry)
    warm_start_entries = deduped_warm_start_entries
    warm_starts = [entry['para'] for entry in warm_start_entries]
    search_space = build_search_space(step_size=0.01)
    trace = []
    best_reward_history = []
    feasible_tracker = {'found': False, 'best_reward': -np.inf, 'best_para': None}
    best_detail = None
    eval_counter = 0

    def identify_action_source(action_policy):
        action_policy = np.asarray(action_policy, dtype=np.float32)
        for entry in warm_start_entries:
            if action_distance_inf(action_policy, para_to_action(entry['para'])) < ACTION_DEDUP_EPS:
                return entry['source']
        return 'search'

    def record_summary(detail):
        trace.append({
            'eval_idx': int(detail['eval_idx']),
            'reward': float(detail['reward']),
            'f1': float(detail.get('f1', np.nan)),
            'status': detail['status'],
            'is_on_cloth': bool(detail['is_on_cloth']),
            'action_policy': detail['action_policy'].copy(),
            'source': detail.get('source', 'search'),
            'baseline_filename': detail.get('baseline_filename'),
            'exception_message': detail.get('exception_message'),
        })

    def evaluate_action(action_policy, eval_idx, description_suffix='recover'):
        nonlocal best_detail
        detail = evaluate_recover_candidate(
            rollout_idx=rollout_idx,
            eval_idx=eval_idx,
            action_policy=action_policy,
            intermediate_cloth_3d=intermediate_cloth_3d,
            original_initial_cloth_graph=original_initial_cloth_graph,
            initial_covered_status=initial_covered_status,
            all_body_points=all_body_points,
            recover_model=recover_model,
            graph_config=graph_config,
            device=device,
            graph_root=graph_root,
            description_suffix=description_suffix,
        )
        detail['source'] = identify_action_source(detail['action_policy'])
        detail['baseline_filename'] = None
        for entry in warm_start_entries:
            if action_distance_inf(detail['action_policy'], para_to_action(entry['para'])) < ACTION_DEDUP_EPS:
                sample = entry.get('sample')
                if sample is not None:
                    detail['baseline_filename'] = sample.get('filename')
                break
        record_summary(detail)
        if best_detail is None or detail['reward'] > best_detail['reward'] + REWARD_EPS:
            best_detail = detail
            best_reward_history.append(float(detail['reward']))
        if detail['is_on_cloth'] and detail['reward'] > feasible_tracker['best_reward'] + REWARD_EPS:
            feasible_tracker['found'] = True
            feasible_tracker['best_reward'] = float(detail['reward'])
            feasible_tracker['best_para'] = action_to_para(detail['action_policy'])
        return detail

    def objective_random(para):
        nonlocal eval_counter
        eval_counter += 1
        return float(evaluate_action(para_to_action(para), eval_counter)['reward'])

    if search_method == 'random':
        optimizer = gfo.RandomSearchOptimizer(
            search_space,
            initialize={'grid': 4, 'random': 10, 'vertices': 4, 'warm_start': warm_starts},
        )
        optimizer.search(objective_random, n_iter=max_fevals, verbosity=False)
        best_para = optimizer.best_para
        if feasible_only_best and feasible_tracker['found']:
            best_para = feasible_tracker['best_para']
        selected_action = para_to_action(best_para)
        best_fevals = int(max_fevals)
        best_iterations = int(max_fevals)
    elif search_method == 'cma':
        if len(warm_starts) > 0:
            x0_dict = warm_starts[0]
            x0 = np.array([x0_dict['x_i'], x0_dict['y_i'], x0_dict['x_f'], x0_dict['y_f']], dtype=np.float32)
        else:
            x0 = np.zeros(4, dtype=np.float32)
        opts = cma.CMAOptions({
            'verb_disp': 0,
            'popsize': int(popsize),
            'maxfevals': int(max_fevals),
            'tolfun': 1e-11,
            'tolflatfitness': 20,
            'tolfunhist': 1e-20,
            'bounds': [[-1] * 4, [1] * 4],
        })
        es = cma.CMAEvolutionStrategy(x0.tolist(), float(sigma), opts)
        while not es.stop() and es.countevals < max_fevals:
            xs = es.ask()
            costs = []
            for x in xs:
                eval_counter += 1
                detail = evaluate_action(np.asarray(x, dtype=np.float32), eval_counter)
                costs.append(float(-detail['reward']))
            es.tell(xs, costs)
        if feasible_only_best and feasible_tracker['found']:
            best_para = feasible_tracker['best_para']
            selected_action = para_to_action(best_para)
        else:
            selected_action = np.clip(np.asarray(es.result.xbest, dtype=np.float32), -1.0, 1.0)
        best_fevals = int(es.result.evaluations)
        best_iterations = int(es.countiter)
    else:
        raise ValueError(f'Unsupported recover search method: {search_method}')

    final_detail = evaluate_recover_candidate(
        rollout_idx=rollout_idx,
        eval_idx=eval_counter + 1,
        action_policy=selected_action,
        intermediate_cloth_3d=intermediate_cloth_3d,
        original_initial_cloth_graph=original_initial_cloth_graph,
        initial_covered_status=initial_covered_status,
        all_body_points=all_body_points,
        recover_model=recover_model,
        graph_config=graph_config,
        device=device,
        graph_root=graph_root,
        description_suffix='recover_selected',
    )

    if feasible_only_best and feasible_tracker['found'] and final_detail['reward'] + REWARD_EPS < feasible_tracker['best_reward']:
        final_detail = evaluate_recover_candidate(
            rollout_idx=rollout_idx,
            eval_idx=eval_counter + 2,
            action_policy=para_to_action(feasible_tracker['best_para']),
            intermediate_cloth_3d=intermediate_cloth_3d,
            original_initial_cloth_graph=original_initial_cloth_graph,
            initial_covered_status=initial_covered_status,
            all_body_points=all_body_points,
            recover_model=recover_model,
            graph_config=graph_config,
            device=device,
            graph_root=graph_root,
            description_suffix='recover_feasible_selected',
        )

    final_detail['search_diagnostics'] = {
        'search_method': search_method,
        'max_fevals': int(max_fevals),
        'num_evals': int(eval_counter),
        'best_fevals': int(best_fevals),
        'best_iterations': int(best_iterations),
        'num_on_cloth': int(sum(1 for record in trace if record['is_on_cloth'])),
        'on_cloth_ratio': float(sum(1 for record in trace if record['is_on_cloth']) / max(1, len(trace))),
        'best_reward': float(final_detail['reward']),
        'best_f1': float(final_detail.get('f1', np.nan)),
        'mean_reward_all': safe_mean([record['reward'] for record in trace]),
        'mean_reward_on_cloth': safe_mean([record['reward'] for record in trace if record['is_on_cloth']]),
        'mean_f1_all': safe_mean([record['f1'] for record in trace if np.isfinite(record['f1'])]),
        'mean_f1_on_cloth': safe_mean([record['f1'] for record in trace if record['is_on_cloth'] and np.isfinite(record['f1'])]),
        'feasible_found': bool(feasible_tracker['found']),
        'selected_from_feasible_tracker': bool(feasible_only_best and feasible_tracker['found']),
        'num_warm_starts': int(len(warm_starts)),
        'num_baseline_recover_candidates': int(len(baseline_recover_candidates)),
        'warm_start_sources': [entry['source'] for entry in warm_start_entries],
        'best_reward_history': best_reward_history,
        'trace': trace,
    }
    return final_detail


def evaluate_joint_candidate(
    rollout_idx,
    eval_idx,
    action_joint,
    initial_cloth_3d,
    original_initial_cloth_graph,
    all_body_points,
    initial_covered_status,
    uncover_model,
    recover_model,
    graph_config,
    device,
    graph_root,
    uncover_f1_threshold,
    uncover_weight,
    recover_weight,
):
    uncover_action_policy, recover_action_policy = split_joint_action(action_joint)
    uncover_detail = evaluate_uncover_candidate(
        rollout_idx=rollout_idx,
        eval_idx=eval_idx,
        action_policy=uncover_action_policy,
        initial_cloth_3d=initial_cloth_3d,
        all_body_points=all_body_points,
        initial_covered_status=initial_covered_status,
        uncover_model=uncover_model,
        graph_config=graph_config,
        device=device,
        graph_root=graph_root,
    )

    detail = {
        'eval_idx': int(eval_idx),
        'uncover_action_policy': uncover_action_policy.copy(),
        'recover_action_policy': recover_action_policy.copy(),
        'uncover_action_world': uncover_detail['action_world'].copy(),
        'recover_action_world': scale_action(recover_action_policy).copy(),
        'uncover_is_on_cloth': bool(uncover_detail['is_on_cloth']),
        'recover_is_on_cloth': False,
        'joint_reward': float(INVALID_REWARD),
        'joint_cost': float(-INVALID_REWARD),
        'uncover_reward': float(uncover_detail['reward']),
        'recover_reward': float(INVALID_REWARD),
        'uncover_f1': float(uncover_detail['f1']),
        'recover_f1': float('nan'),
        'pred_intermediate_3d': np.asarray(uncover_detail['pred_intermediate_3d']).copy(),
        'pred_final_3d': np.asarray(uncover_detail['pred_intermediate_3d']).copy(),
        'status': uncover_detail['status'],
    }

    if uncover_detail['status'] != 'valid_uncover':
        return detail

    if uncover_detail['f1'] < uncover_f1_threshold:
        joint_reward = SCREENED_REWARD_BASE + float(uncover_detail['f1'])
        detail.update({
            'status': 'below_f1_threshold',
            'joint_reward': float(joint_reward),
            'joint_cost': float(-joint_reward),
        })
        return detail

    recover_detail = evaluate_recover_candidate(
        rollout_idx=rollout_idx,
        eval_idx=eval_idx,
        action_policy=recover_action_policy,
        intermediate_cloth_3d=uncover_detail['pred_intermediate_3d'],
        original_initial_cloth_graph=original_initial_cloth_graph,
        initial_covered_status=initial_covered_status,
        all_body_points=all_body_points,
        recover_model=recover_model,
        graph_config=graph_config,
        device=device,
        graph_root=graph_root,
        description_suffix='recover_joint',
    )
    detail['recover_action_world'] = recover_detail['action_world'].copy()
    detail['recover_is_on_cloth'] = bool(recover_detail['is_on_cloth'])
    detail['pred_final_3d'] = np.asarray(recover_detail['pred_final_3d']).copy()

    if recover_detail['status'] != 'valid_recover':
        detail['status'] = recover_detail['status']
        return detail

    joint_reward = float(uncover_weight * uncover_detail['reward'] + recover_weight * recover_detail['reward'])
    detail.update({
        'status': 'feasible',
        'recover_reward': float(recover_detail['reward']),
        'recover_f1': float(recover_detail['f1']),
        'joint_reward': float(joint_reward),
        'joint_cost': float(-joint_reward),
    })
    return detail


def optimize_joint_actions_coupled(
    rollout_idx,
    env,
    human_pose,
    target_limb_code,
    uncover_model,
    recover_model,
    graph_config,
    device,
    max_fevals,
    graph_root,
    uncover_f1_threshold,
    uncover_weight,
    recover_weight,
    popsize,
    sigma,
):
    import cma

    initial_cloth_3d = np.asarray(env.get_cloth_state())
    original_initial_cloth_graph, _ = build_graph_reference(
        cloth_input=initial_cloth_3d,
        graph_config=graph_config,
        graph_root=graph_root,
        description=f'rollout_{rollout_idx}_initial_graph',
    )
    body_info = env.get_human_body_info()
    all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)
    initial_covered_status = get_covered_status(all_body_points, ensure_2d(original_initial_cloth_graph))

    best_detail = None
    eval_counter = 0
    trace = []
    generation_summaries = []

    opts = cma.CMAOptions({
        'verb_disp': 0,
        'popsize': int(popsize),
        'maxfevals': int(max_fevals),
        'tolfun': 1e-11,
        'tolflatfitness': 20,
        'tolfunhist': 1e-20,
        'bounds': [[-1] * 8, [1] * 8],
    })
    es = cma.CMAEvolutionStrategy(build_joint_x0(target_limb_code).tolist(), float(sigma), opts)

    generation_idx = 0
    while not es.stop() and es.countevals < max_fevals:
        generation_idx += 1
        generation_start_eval = eval_counter
        joint_actions = es.ask()
        costs = []
        for action_joint in joint_actions:
            eval_counter += 1
            detail = evaluate_joint_candidate(
                rollout_idx=rollout_idx,
                eval_idx=eval_counter,
                action_joint=np.clip(np.asarray(action_joint, dtype=np.float32), -1.0, 1.0),
                initial_cloth_3d=initial_cloth_3d,
                original_initial_cloth_graph=original_initial_cloth_graph,
                all_body_points=all_body_points,
                initial_covered_status=initial_covered_status,
                uncover_model=uncover_model,
                recover_model=recover_model,
                graph_config=graph_config,
                device=device,
                graph_root=graph_root,
                uncover_f1_threshold=uncover_f1_threshold,
                uncover_weight=uncover_weight,
                recover_weight=recover_weight,
            )
            trace.append(detail)
            costs.append(float(detail['joint_cost']))
            if best_detail is None or detail['joint_reward'] > best_detail['joint_reward'] + REWARD_EPS:
                best_detail = {
                    key: (np.asarray(val).copy() if isinstance(val, np.ndarray) else val)
                    for key, val in detail.items()
                }
        es.tell(joint_actions, costs)

        generation_records = trace[generation_start_eval:eval_counter]
        feasible_records = [record for record in generation_records if record['status'] == 'feasible']
        generation_summary = {
            'generation_idx': int(generation_idx),
            'eval_start_idx': int(generation_start_eval + 1),
            'eval_end_idx': int(eval_counter),
            'num_evals': int(len(generation_records)),
            'num_feasible': int(len(feasible_records)),
            'feasible_ratio': float(len(feasible_records) / max(1, len(generation_records))),
            'uncover_on_cloth_ratio': float(sum(1 for record in generation_records if record['uncover_is_on_cloth']) / max(1, len(generation_records))),
            'recover_on_cloth_ratio_feasible_uncover': float(
                sum(1 for record in generation_records if record['uncover_is_on_cloth'] and record['recover_is_on_cloth']) /
                max(1, sum(1 for record in generation_records if record['uncover_is_on_cloth']))
            ),
            'mean_joint_reward': safe_mean([record['joint_reward'] for record in generation_records]),
            'best_joint_reward_gen': float(max(record['joint_reward'] for record in generation_records)),
            'best_joint_reward_so_far': float(best_detail['joint_reward']) if best_detail is not None else float(INVALID_REWARD),
            'mean_uncover_reward': safe_mean([record['uncover_reward'] for record in generation_records]),
            'mean_recover_reward_feasible': safe_mean([record['recover_reward'] for record in feasible_records]),
            'mean_uncover_f1': safe_mean([record['uncover_f1'] for record in generation_records]),
            'mean_recover_f1_feasible': safe_mean([record['recover_f1'] for record in feasible_records if np.isfinite(record['recover_f1'])]),
            'best_uncover_f1_gen': float(max(record['uncover_f1'] for record in generation_records)),
            'best_uncover_f1_so_far': float(best_detail['uncover_f1']) if best_detail is not None else 0.0,
            'sigma': float(es.sigma),
        }
        generation_summaries.append(generation_summary)
        print(
            f"  Coupled CMA gen {generation_idx:02d}: evals {generation_summary['eval_start_idx']}-{generation_summary['eval_end_idx']} "
            f"best_so_far={generation_summary['best_joint_reward_so_far']:.2f} "
            f"gen_best={generation_summary['best_joint_reward_gen']:.2f} "
            f"feasible={generation_summary['num_feasible']}/{generation_summary['num_evals']} "
            f"mean_f1={generation_summary['mean_uncover_f1']:.3f} "
            f"mean_r_f1={generation_summary['mean_recover_f1_feasible']:.3f} "
            f"sigma={generation_summary['sigma']:.4f}"
        )

    if best_detail is None:
        raise RuntimeError('Coupled joint CMA did not evaluate any candidates.')

    diagnostics = {
        'num_joint_evals': int(eval_counter),
        'num_generations': int(len(generation_summaries)),
        'num_feasible_joint': int(sum(1 for record in trace if record['status'] == 'feasible')),
        'joint_feasible_ratio': float(sum(1 for record in trace if record['status'] == 'feasible') / max(1, len(trace))),
        'uncover_on_cloth_ratio': float(sum(1 for record in trace if record['uncover_is_on_cloth']) / max(1, len(trace))),
        'recover_on_cloth_ratio': float(sum(1 for record in trace if record['recover_is_on_cloth']) / max(1, len(trace))),
            'best_eval_idx': int(best_detail['eval_idx']),
            'trace': trace,
            'generation_summaries': generation_summaries,
            'stop_reason': es.stop(),
            'final_countevals': int(es.countevals),
    }
    return best_detail, diagnostics


def optimize_uncover_sequential_joint(
    rollout_idx,
    env,
    human_pose,
    target_limb_code,
    uncover_model,
    recover_model,
    graph_config,
    device,
    uncover_max_fevals,
    recover_max_fevals,
    graph_root,
    uncover_weight,
    recover_weight,
    popsize,
    sigma,
    recover_search_method,
    recover_warm_start_strategy,
    recover_feasible_only_best,
    screen_uncover_f1_threshold,
    outer_init_source,
    baseline_action_pool,
    baseline_seed_selection,
    outer_baseline_seeds,
    outer_random_seeds,
    inner_include_baseline_recover,
    inner_baseline_recover_topk,
    rng,
):
    import cma

    initial_cloth_3d = np.asarray(env.get_cloth_state())
    original_initial_cloth_graph, _ = build_graph_reference(
        cloth_input=initial_cloth_3d,
        graph_config=graph_config,
        graph_root=graph_root,
        description=f'rollout_{rollout_idx}_initial_graph',
    )
    body_info = env.get_human_body_info()
    all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)
    initial_covered_status = get_covered_status(all_body_points, ensure_2d(original_initial_cloth_graph))

    best_detail = None
    eval_counter = 0
    outer_trace = []
    generation_summaries = []
    if rng is None:
        rng = np.random.RandomState(0)
    baseline_pool_summary = summarize_baseline_limb_pool(baseline_action_pool, target_limb_code)
    init_metadata = {
        'outer_init_source': str(outer_init_source),
        'baseline_limb_summary': baseline_pool_summary,
        'baseline_outer_seed_count': 0,
        'random_outer_seed_count': 0,
        'baseline_recover_candidate_count': 0,
        'generation0_seed_details': [],
        'fallback_to_heuristic': False,
    }

    def objective(action_policy, seed_metadata=None):
        nonlocal best_detail, eval_counter
        eval_counter += 1
        uncover_detail = evaluate_uncover_candidate(
            rollout_idx=rollout_idx,
            eval_idx=eval_counter,
            action_policy=action_policy,
            initial_cloth_3d=initial_cloth_3d,
            all_body_points=all_body_points,
            initial_covered_status=initial_covered_status,
            uncover_model=uncover_model,
            graph_config=graph_config,
            device=device,
            graph_root=graph_root,
        )

        record = {
            'eval_idx': int(eval_counter),
            'status': uncover_detail['status'],
            'joint_reward': float(INVALID_REWARD),
            'uncover_reward': float(uncover_detail['reward']),
            'uncover_f1': float(uncover_detail['f1']),
            'recover_reward': float(INVALID_REWARD),
            'recover_f1': float('nan'),
            'is_on_cloth': bool(uncover_detail['is_on_cloth']),
            'recover_success': False,
            'init_source': None if seed_metadata is None else seed_metadata.get('source'),
            'uncover_status': uncover_detail['status'],
            'recover_status': None,
            'baseline_filename': None if seed_metadata is None or seed_metadata.get('baseline_sample') is None else seed_metadata['baseline_sample']['filename'],
            'baseline_reward': None if seed_metadata is None or seed_metadata.get('baseline_sample') is None else float(seed_metadata['baseline_sample']['best_reward']),
            'exception_message': None,
        }
        candidate_detail = {
            'best_outer_eval_idx': int(eval_counter),
            'uncover_action_policy': uncover_detail['action_policy'].copy(),
            'uncover_action_world': uncover_detail['action_world'].copy(),
            'pred_intermediate_3d': np.asarray(uncover_detail['pred_intermediate_3d']).copy(),
            'pred_uncover_reward': float(uncover_detail['reward']),
            'pred_uncover_f1': float(uncover_detail['f1']),
            'pred_uncover_status': uncover_detail['covered_status'],
            'recover_detail': None,
            'joint_reward': float(INVALID_REWARD),
            'status': uncover_detail['status'],
            'init_source': None if seed_metadata is None else seed_metadata.get('source'),
            'baseline_sample': None if seed_metadata is None else seed_metadata.get('baseline_sample'),
        }

        if uncover_detail['status'] != 'valid_uncover':
            outer_trace.append(record)
            if best_detail is None or candidate_detail['joint_reward'] > best_detail['joint_reward'] + REWARD_EPS:
                best_detail = candidate_detail
            return float(INVALID_REWARD)

        if screen_uncover_f1_threshold is not None and uncover_detail['f1'] < screen_uncover_f1_threshold:
            screened_reward = SCREENED_REWARD_BASE + float(uncover_detail['f1'])
            record.update({
                'status': 'screened_by_uncover_f1',
                'joint_reward': float(screened_reward),
            })
            candidate_detail['joint_reward'] = float(screened_reward)
            candidate_detail['status'] = 'screened_by_uncover_f1'
            outer_trace.append(record)
            if best_detail is None or candidate_detail['joint_reward'] > best_detail['joint_reward'] + REWARD_EPS:
                best_detail = candidate_detail
            return float(screened_reward)

        recover_detail = optimize_recover_given_uncover(
            rollout_idx=rollout_idx,
            outer_eval_idx=eval_counter,
            uncover_action_policy=uncover_detail['action_policy'],
            intermediate_cloth_3d=uncover_detail['pred_intermediate_3d'],
            original_initial_cloth_graph=original_initial_cloth_graph,
            initial_covered_status=initial_covered_status,
            all_body_points=all_body_points,
            recover_model=recover_model,
            graph_config=graph_config,
            device=device,
            max_fevals=recover_max_fevals,
            graph_root=graph_root,
            search_method=recover_search_method,
            warm_start_strategy=recover_warm_start_strategy,
            feasible_only_best=recover_feasible_only_best,
            popsize=popsize,
            sigma=sigma,
            baseline_recover_candidates=collect_baseline_recover_candidates(
                pool=baseline_action_pool,
                target_limb_code=target_limb_code,
                topk=inner_baseline_recover_topk if inner_include_baseline_recover else 0,
                paired_sample=None if seed_metadata is None else seed_metadata.get('baseline_sample'),
                include_pool=inner_include_baseline_recover,
            ),
        )
        init_metadata['baseline_recover_candidate_count'] = max(
            int(init_metadata['baseline_recover_candidate_count']),
            int(recover_detail['search_diagnostics'].get('num_baseline_recover_candidates', 0)),
        )

        if recover_detail['status'] == 'valid_recover':
            joint_reward = float(uncover_weight * uncover_detail['reward'] + recover_weight * recover_detail['reward'])
            record.update({
                'status': 'feasible',
                'joint_reward': joint_reward,
                'recover_reward': float(recover_detail['reward']),
                'recover_f1': float(recover_detail['f1']),
                'recover_success': True,
                'recover_status': recover_detail['status'],
                'exception_message': recover_detail.get('exception_message'),
                'inner_num_evals': int(recover_detail['search_diagnostics']['num_evals']),
                'inner_on_cloth_ratio': float(recover_detail['search_diagnostics']['on_cloth_ratio']),
                'inner_best_reward': float(recover_detail['search_diagnostics']['best_reward']),
            })
        else:
            joint_reward = float(INVALID_REWARD)
            record.update({
                'status': recover_detail['status'],
                'joint_reward': joint_reward,
                'recover_reward': float(recover_detail['reward']),
                'recover_f1': float(recover_detail.get('f1', np.nan)),
                'recover_success': False,
                'recover_status': recover_detail['status'],
                'exception_message': recover_detail.get('exception_message'),
                'inner_num_evals': int(recover_detail['search_diagnostics']['num_evals']),
                'inner_on_cloth_ratio': float(recover_detail['search_diagnostics']['on_cloth_ratio']),
                'inner_best_reward': float(recover_detail['search_diagnostics']['best_reward']),
            })

        candidate_detail.update({
            'recover_detail': recover_detail,
            'joint_reward': float(joint_reward),
            'status': record['status'],
        })
        outer_trace.append(record)
        if best_detail is None or joint_reward > best_detail['joint_reward'] + REWARD_EPS:
            best_detail = candidate_detail
        return float(joint_reward)

    x0 = np.asarray(set_x0_for_cmaes(target_limb_code), dtype=np.float32)
    opts = cma.CMAOptions({
        'verb_disp': 0,
        'popsize': int(popsize),
        'maxfevals': int(uncover_max_fevals),
        'tolfun': 1e-11,
        'tolflatfitness': 20,
        'tolfunhist': 1e-20,
        'bounds': [[-1] * 4, [1] * 4],
    })
    es = cma.CMAEvolutionStrategy(x0.tolist(), float(sigma), opts)

    generation_idx = 0
    if outer_init_source == 'baseline_limb_mixed':
        requested_baseline = max(0, int(outer_baseline_seeds))
        requested_random = max(0, int(outer_random_seeds))
        if requested_baseline + requested_random <= 0:
            requested_random = int(popsize)
        baseline_seed_samples = select_outer_baseline_seed_samples(
            pool=baseline_action_pool,
            target_limb_code=target_limb_code,
            num_seeds=requested_baseline,
            selection_mode=baseline_seed_selection,
            rng=rng,
        )
        seed_actions = []
        seed_metadata = []
        for sample in baseline_seed_samples:
            seed_actions.append(np.asarray(sample['uncover_action'], dtype=np.float32))
            seed_metadata.append({
                'source': 'baseline',
                'baseline_sample': sample,
            })

        max_seed_count = int(popsize)
        remaining_slots = max(0, max_seed_count - len(seed_actions))
        requested_random = max(requested_random, remaining_slots)
        random_seed_count = min(max_seed_count - len(seed_actions), requested_random)
        for _ in range(random_seed_count):
            seed_actions.append(rng.uniform(-1.0, 1.0, size=4).astype(np.float32))
            seed_metadata.append({
                'source': 'random',
                'baseline_sample': None,
            })
        while len(seed_actions) < max_seed_count:
            seed_actions.append(rng.uniform(-1.0, 1.0, size=4).astype(np.float32))
            seed_metadata.append({
                'source': 'random',
                'baseline_sample': None,
            })

        init_metadata['baseline_outer_seed_count'] = int(sum(1 for item in seed_metadata if item['source'] == 'baseline'))
        init_metadata['random_outer_seed_count'] = int(sum(1 for item in seed_metadata if item['source'] == 'random'))
        init_metadata['generation0_seed_details'] = [
            {
                'source': item['source'],
                'action_policy': np.asarray(action, dtype=np.float32).copy(),
                'baseline_filename': None if item['baseline_sample'] is None else item['baseline_sample']['filename'],
                'baseline_reward': None if item['baseline_sample'] is None else float(item['baseline_sample']['best_reward']),
                'baseline_seed': None if item['baseline_sample'] is None else int(item['baseline_sample']['seed']),
            }
            for action, item in zip(seed_actions, seed_metadata)
        ]

        if init_metadata['baseline_outer_seed_count'] == 0:
            init_metadata['fallback_to_heuristic'] = True
            print(
                f"  Sequential init fallback to heuristic for TL {target_limb_code}: "
                f"no usable baseline uncover seeds in {baseline_pool_summary['num_samples']} limb samples"
            )
        else:
            print(
                f"  Sequential init TL {target_limb_code}: source=baseline_limb_mixed "
                f"baseline_outer={init_metadata['baseline_outer_seed_count']} "
                f"random_outer={init_metadata['random_outer_seed_count']} "
                f"limb_samples={baseline_pool_summary['num_samples']} "
                f"unique_uncover={baseline_pool_summary['num_unique_uncover']}"
            )

        generation_idx += 1
        eval_start = eval_counter
        costs = []
        asked_actions = es.ask()
        if len(asked_actions) != len(seed_actions):
            raise RuntimeError(
                f'Seeded init population size mismatch: ask() returned {len(asked_actions)} actions, '
                f'but constructed {len(seed_actions)} seed actions.'
            )
        for action, metadata in zip(seed_actions, seed_metadata):
            reward = objective(np.asarray(action, dtype=np.float32), seed_metadata=metadata)
            costs.append(float(-reward))
        es.tell([np.asarray(action, dtype=np.float32) for action in seed_actions], costs)

        generation_records = outer_trace[eval_start:eval_counter]
        feasible_records = [record for record in generation_records if record['recover_success']]
        gen_summary = {
            'generation_idx': int(generation_idx),
            'generation_type': 'seeded_init',
            'eval_start_idx': int(eval_start + 1),
            'eval_end_idx': int(eval_counter),
            'num_evals': int(len(generation_records)),
            'num_feasible': int(len(feasible_records)),
            'feasible_ratio': float(len(feasible_records) / max(1, len(generation_records))),
            'on_cloth_ratio': float(sum(1 for record in generation_records if record['is_on_cloth']) / max(1, len(generation_records))),
            'mean_joint_reward': safe_mean([record['joint_reward'] for record in generation_records]),
            'best_joint_reward_gen': float(max(record['joint_reward'] for record in generation_records)),
            'best_joint_reward_so_far': float(best_detail['joint_reward']) if best_detail is not None else float(INVALID_REWARD),
            'mean_uncover_f1': safe_mean([record['uncover_f1'] for record in generation_records]),
            'best_uncover_f1_gen': float(max(record['uncover_f1'] for record in generation_records)),
            'best_uncover_f1_so_far': float(best_detail['pred_uncover_f1']) if best_detail is not None else 0.0,
            'mean_recover_reward_feasible': safe_mean([record['recover_reward'] for record in feasible_records]),
            'mean_recover_f1_feasible': safe_mean([record['recover_f1'] for record in feasible_records if np.isfinite(record['recover_f1'])]),
            'total_inner_evals': int(sum(record.get('inner_num_evals', 0) for record in generation_records)),
            'sigma': float(es.sigma),
            'baseline_outer_seed_count': int(init_metadata['baseline_outer_seed_count']),
            'random_outer_seed_count': int(init_metadata['random_outer_seed_count']),
        }
        generation_summaries.append(gen_summary)
        print(
            f"  Sequential outer CMA gen {generation_idx:02d} [seeded]: evals {gen_summary['eval_start_idx']}-{gen_summary['eval_end_idx']} "
            f"best_so_far={gen_summary['best_joint_reward_so_far']:.2f} "
            f"gen_best={gen_summary['best_joint_reward_gen']:.2f} "
            f"feasible={gen_summary['num_feasible']}/{gen_summary['num_evals']} "
            f"mean_f1={gen_summary['mean_uncover_f1']:.3f} "
            f"mean_r_f1={gen_summary['mean_recover_f1_feasible']:.3f} "
            f"inner_total={gen_summary['total_inner_evals']} sigma={gen_summary['sigma']:.4f}"
        )
        for record in generation_records:
            print(
                f"    eval {record['eval_idx']:02d} src={record.get('init_source') or 'cma'} "
                f"status={record['status']} uncover={record.get('uncover_status')} recover={record.get('recover_status')} "
                f"u_r={record['uncover_reward']:.2f} u_f1={record['uncover_f1']:.3f} "
                f"r_r={record['recover_reward']:.2f} r_f1={record['recover_f1']:.3f} "
                f"base={record.get('baseline_filename')} "
                f"exc={record.get('exception_message')}"
            )
        print(f"    seeded_init_cma_stop={es.stop()} countevals={es.countevals}")

    while not es.stop() and es.countevals < uncover_max_fevals:
        generation_idx += 1
        eval_start = eval_counter
        actions = es.ask()
        costs = []
        for action in actions:
            reward = objective(np.asarray(action, dtype=np.float32))
            costs.append(float(-reward))
        es.tell(actions, costs)

        generation_records = outer_trace[eval_start:eval_counter]
        feasible_records = [record for record in generation_records if record['recover_success']]
        gen_summary = {
            'generation_idx': int(generation_idx),
            'generation_type': 'cma',
            'eval_start_idx': int(eval_start + 1),
            'eval_end_idx': int(eval_counter),
            'num_evals': int(len(generation_records)),
            'num_feasible': int(len(feasible_records)),
            'feasible_ratio': float(len(feasible_records) / max(1, len(generation_records))),
            'on_cloth_ratio': float(sum(1 for record in generation_records if record['is_on_cloth']) / max(1, len(generation_records))),
            'mean_joint_reward': safe_mean([record['joint_reward'] for record in generation_records]),
            'best_joint_reward_gen': float(max(record['joint_reward'] for record in generation_records)),
            'best_joint_reward_so_far': float(best_detail['joint_reward']) if best_detail is not None else float(INVALID_REWARD),
            'mean_uncover_f1': safe_mean([record['uncover_f1'] for record in generation_records]),
            'best_uncover_f1_gen': float(max(record['uncover_f1'] for record in generation_records)),
            'best_uncover_f1_so_far': float(best_detail['pred_uncover_f1']) if best_detail is not None else 0.0,
            'mean_recover_reward_feasible': safe_mean([record['recover_reward'] for record in feasible_records]),
            'mean_recover_f1_feasible': safe_mean([record['recover_f1'] for record in feasible_records if np.isfinite(record['recover_f1'])]),
            'total_inner_evals': int(sum(record.get('inner_num_evals', 0) for record in generation_records)),
            'sigma': float(es.sigma),
        }
        generation_summaries.append(gen_summary)
        print(
            f"  Sequential outer CMA gen {generation_idx:02d}: evals {gen_summary['eval_start_idx']}-{gen_summary['eval_end_idx']} "
            f"best_so_far={gen_summary['best_joint_reward_so_far']:.2f} "
            f"gen_best={gen_summary['best_joint_reward_gen']:.2f} "
            f"feasible={gen_summary['num_feasible']}/{gen_summary['num_evals']} "
            f"mean_f1={gen_summary['mean_uncover_f1']:.3f} "
            f"mean_r_f1={gen_summary['mean_recover_f1_feasible']:.3f} "
            f"inner_total={gen_summary['total_inner_evals']} sigma={gen_summary['sigma']:.4f}"
        )

    if best_detail is None:
        raise RuntimeError('Sequential joint optimizer did not evaluate any uncover candidates.')

    diagnostics = {
        'num_outer_evals': int(eval_counter),
        'num_generations': int(len(generation_summaries)),
        'num_feasible_outer': int(sum(1 for record in outer_trace if record['recover_success'])),
        'outer_feasible_ratio': float(sum(1 for record in outer_trace if record['recover_success']) / max(1, len(outer_trace))),
        'outer_on_cloth_ratio': float(sum(1 for record in outer_trace if record['is_on_cloth']) / max(1, len(outer_trace))),
        'best_outer_eval_idx': int(best_detail['best_outer_eval_idx']),
        'total_inner_evals': int(sum(record.get('inner_num_evals', 0) for record in outer_trace)),
        'outer_init_source': str(outer_init_source),
        'baseline_outer_seed_count': int(init_metadata['baseline_outer_seed_count']),
        'random_outer_seed_count': int(init_metadata['random_outer_seed_count']),
        'baseline_recover_candidate_count': int(init_metadata['baseline_recover_candidate_count']),
        'generation0_seed_details': init_metadata['generation0_seed_details'],
        'fallback_to_heuristic': bool(init_metadata['fallback_to_heuristic']),
        'baseline_limb_summary': init_metadata['baseline_limb_summary'],
        'stop_reason': es.stop(),
        'final_countevals': int(es.countevals),
        'trace': outer_trace,
        'generation_summaries': generation_summaries,
    }
    return best_detail, diagnostics


def execute_rollout_sim(env, uncover_action_policy, recover_action_policy=None, execute_recover=True):
    sim_start = time.time()
    env.uncover_step(uncover_action_policy)
    if execute_recover and recover_action_policy is not None:
        env.recover_step(recover_action_policy)
    else:
        env.recover_action = np.zeros(4, dtype=np.float32)
        env.execute_recover_action = False
        env.cloth_final = env.cloth_intermediate
    observation, uncover_reward_sim, recover_reward_sim, done, info = env.get_info()
    sim_time = time.time() - sim_start
    return {
        'observation': observation,
        'uncover reward': uncover_reward_sim,
        'recover_reward': recover_reward_sim,
        'done': done,
        'info': info,
    }, sim_time


def build_failed_sim_payload(env, target_limb_code, fail_reason):
    cloth_state = ensure_3d(np.asarray(env.get_cloth_state(), dtype=np.float32))
    cloth_tuple = [None, cloth_state.copy()]
    info = {
        'recovering': True,
        'cloth_initial': cloth_tuple,
        'cloth_intermediate': [None, cloth_state.copy()],
        'cloth_final': [None, cloth_state.copy()],
        'RBG_human': getattr(env, 'human_no_occlusion_RGB', None),
        'depth_human': getattr(env, 'human_no_occlusion_depth', None),
        'uncovered_status_sim': None,
        'recovered_status_sim': None,
        'target_limb_code': int(target_limb_code),
        'human_body_info': env.get_human_body_info(),
        'gender': getattr(getattr(env, 'human', None), 'gender', None),
        'grasp_on_cloth_uncover': False,
        'grasp_on_cloth_recover': False,
        'sim_skipped_reason': fail_reason,
    }
    return {
        'observation': None,
        'uncover reward': float(INVALID_REWARD),
        'recover_reward': float(INVALID_REWARD),
        'done': False,
        'info': info,
    }, 0.0


def run_single_rollout(rollout_idx, args, uncover_model, recover_model, graph_config, env_variations, device, output_dir, rng, baseline_action_pool):
    load_runtime_dependencies()
    target_limb_code = args.target_limb_code if args.target_limb_code is not None else int(rng.choice(target_limb_list))
    seed = int(rng.randint(0, 2**31 - 1))

    env = make_env('RobeReversible-v1', coop=False, seed=seed)
    env.set_env_variations(
        collect_data=False,
        blanket_pose_var=env_variations['blanket_var'],
        high_pose_var=env_variations['high_pose_var'],
        body_shape_var=env_variations['body_shape_var'],
    )
    env.set_target_limb_code(target_limb_code)
    env.set_recover(True)
    env.set_singulate(True)
    env.set_seed_val(seed)

    human_pose = env.reset()
    human_pose = np.reshape(human_pose, (-1, 2))

    start_time = time.time()
    if args.optimization_mode == 'coupled':
        best_detail, search_diagnostics = optimize_joint_actions_coupled(
            rollout_idx=rollout_idx,
            env=env,
            human_pose=human_pose,
            target_limb_code=target_limb_code,
            uncover_model=uncover_model,
            recover_model=recover_model,
            graph_config=graph_config,
            device=device,
            max_fevals=args.max_fevals,
            graph_root=output_dir,
            uncover_f1_threshold=args.uncover_f1_threshold,
            uncover_weight=args.uncover_weight,
            recover_weight=args.recover_weight,
            popsize=args.popsize,
            sigma=args.sigma,
        )
        recover_action_policy = best_detail['recover_action_policy']
        recover_search_diagnostics = None
        uncover_search_diagnostics = None
    else:
        uncover_max_fevals = args.uncover_max_fevals if args.uncover_max_fevals is not None else args.max_fevals
        recover_max_fevals = args.recover_max_fevals if args.recover_max_fevals is not None else args.max_fevals
        best_detail, uncover_search_diagnostics = optimize_uncover_sequential_joint(
            rollout_idx=rollout_idx,
            env=env,
            human_pose=human_pose,
            target_limb_code=target_limb_code,
            uncover_model=uncover_model,
            recover_model=recover_model,
            graph_config=graph_config,
            device=device,
            uncover_max_fevals=uncover_max_fevals,
            recover_max_fevals=recover_max_fevals,
            graph_root=output_dir,
            uncover_weight=args.uncover_weight,
            recover_weight=args.recover_weight,
            popsize=args.popsize,
            sigma=args.sigma,
            recover_search_method=args.recover_search_method,
            recover_warm_start_strategy=args.recover_warm_start_strategy,
            recover_feasible_only_best=args.recover_feasible_only_best,
            screen_uncover_f1_threshold=args.screen_uncover_f1_threshold,
            outer_init_source=args.outer_init_source,
            baseline_action_pool=baseline_action_pool,
            baseline_seed_selection=args.baseline_seed_selection,
            outer_baseline_seeds=args.outer_baseline_seeds,
            outer_random_seeds=args.outer_random_seeds,
            inner_include_baseline_recover=args.inner_include_baseline_recover,
            inner_baseline_recover_topk=args.inner_baseline_recover_topk,
            rng=rng,
        )
        recover_action_policy = None if best_detail['recover_detail'] is None else best_detail['recover_detail']['action_policy']
        recover_search_diagnostics = None if best_detail['recover_detail'] is None else best_detail['recover_detail']['search_diagnostics']
        search_diagnostics = None
    optimizer_time = time.time() - start_time

    execute_recover = False
    if args.optimization_mode == 'coupled':
        execute_recover = bool(best_detail['recover_is_on_cloth']) and best_detail['status'] == 'feasible'
        pred_recover_reward = float(best_detail['recover_reward'])
        pred_final = ensure_2d(best_detail['pred_final_3d'])
        best_eval_idx = int(search_diagnostics['best_eval_idx'])
        no_valid_candidate = not np.isfinite(best_detail['joint_reward']) or best_detail['joint_reward'] <= INVALID_REWARD / 2.0
        sim_skip_reason = None if not no_valid_candidate else f"no_valid_candidate:{best_detail['status']}"
    else:
        execute_recover = best_detail['recover_detail'] is not None and best_detail['recover_detail']['status'] == 'valid_recover'
        pred_recover_reward = float(best_detail['recover_detail']['reward']) if best_detail['recover_detail'] is not None else float(INVALID_REWARD)
        pred_final = ensure_2d(best_detail['recover_detail']['pred_final_3d']) if best_detail['recover_detail'] is not None else ensure_2d(best_detail['pred_intermediate_3d'])
        best_eval_idx = int(uncover_search_diagnostics['best_outer_eval_idx'])
        no_valid_candidate = (
            not np.isfinite(best_detail['joint_reward']) or
            best_detail['joint_reward'] <= INVALID_REWARD / 2.0 or
            int(uncover_search_diagnostics['num_feasible_outer']) == 0
        )
        sim_skip_reason = None if not no_valid_candidate else f"no_valid_candidate:{best_detail['status']}"

    if sim_skip_reason is None:
        sim_info, sim_time = execute_rollout_sim(
            env=env,
            uncover_action_policy=best_detail['uncover_action_policy'],
            recover_action_policy=recover_action_policy,
            execute_recover=execute_recover,
        )
        sim_f1_metrics = compute_rollout_f1_metrics(
            all_body_points=all_body_points,
            cloth_initial=np.asarray(sim_info['info']['cloth_initial'][1]),
            cloth_intermediate=np.asarray(sim_info['info']['cloth_intermediate'][1]),
            cloth_final=np.asarray(sim_info['info']['cloth_final'][1]),
        )
    else:
        print(f"  Skipping final sim for rollout {rollout_idx}: {sim_skip_reason}")
        sim_info, sim_time = build_failed_sim_payload(env, target_limb_code, sim_skip_reason)
        sim_f1_metrics = {
            'uncover_f1': float('nan'),
            'recover_f1': float('nan'),
            'initial_status': None,
            'intermediate_status': None,
            'final_status': None,
        }
    pred_recover_f1 = (
        float(best_detail['recover_detail']['f1'])
        if args.optimization_mode == 'sequential' and best_detail['recover_detail'] is not None
        else float(best_detail.get('recover_f1', np.nan))
    )

    cma_info = {
        'optimization_mode': args.optimization_mode,
        'optimizer_time': optimizer_time,
        'sim_time': sim_time,
        'max_fevals': int(args.max_fevals),
        'uncover_weight': float(args.uncover_weight),
        'recover_weight': float(args.recover_weight),
        'pred_joint_reward': float(best_detail['joint_reward']),
        'pred_uncover_reward': float(best_detail['pred_uncover_reward'] if args.optimization_mode == 'sequential' else best_detail['uncover_reward']),
        'pred_recover_reward': float(pred_recover_reward),
        'pred_uncover_f1': float(best_detail['pred_uncover_f1'] if args.optimization_mode == 'sequential' else best_detail['uncover_f1']),
        'pred_recover_f1': float(pred_recover_f1),
        'sim_uncover_f1': float(sim_f1_metrics['uncover_f1']),
        'sim_recover_f1': float(sim_f1_metrics['recover_f1']),
        'uncover_action_world': best_detail['uncover_action_world'].tolist(),
        'recover_action_world': scale_action(recover_action_policy).tolist() if recover_action_policy is not None else [],
        'best_pred': pred_final,
        'best_eval_idx': int(best_eval_idx),
        'execute_recover_in_sim': bool(execute_recover),
    }
    if args.optimization_mode == 'coupled':
        cma_info.update({
            'search_method': 'joint_cma',
            'joint_num_generations': int(search_diagnostics['num_generations']),
            'joint_num_evals': int(search_diagnostics['num_joint_evals']),
            'joint_feasible_ratio': float(search_diagnostics['joint_feasible_ratio']),
        })
    else:
        cma_info.update({
            'search_method': f'sequential_outer_cma_inner_{args.recover_search_method}',
            'uncover_max_fevals': int(args.uncover_max_fevals if args.uncover_max_fevals is not None else args.max_fevals),
            'recover_max_fevals': int(args.recover_max_fevals if args.recover_max_fevals is not None else args.max_fevals),
            'outer_num_generations': int(uncover_search_diagnostics['num_generations']),
            'outer_num_evals': int(uncover_search_diagnostics['num_outer_evals']),
            'outer_feasible_ratio': float(uncover_search_diagnostics['outer_feasible_ratio']),
            'outer_init_source': uncover_search_diagnostics['outer_init_source'],
            'baseline_raw_dir': args.baseline_raw_dir,
            'baseline_outer_seed_count': int(uncover_search_diagnostics['baseline_outer_seed_count']),
            'random_outer_seed_count': int(uncover_search_diagnostics['random_outer_seed_count']),
            'baseline_recover_candidate_count': int(uncover_search_diagnostics['baseline_recover_candidate_count']),
            'inner_num_evals_best': int(recover_search_diagnostics['num_evals']) if recover_search_diagnostics is not None else 0,
            'inner_on_cloth_ratio_best': float(recover_search_diagnostics['on_cloth_ratio']) if recover_search_diagnostics is not None else 0.0,
        })

    diagnostics_payload = {
        'rollout_idx': int(rollout_idx),
        'seed': int(seed),
        'target_limb_code': int(target_limb_code),
        'optimization_mode': args.optimization_mode,
        'best_pred_joint_reward': float(best_detail['joint_reward']),
        'best_pred_uncover_reward': float(best_detail['pred_uncover_reward'] if args.optimization_mode == 'sequential' else best_detail['uncover_reward']),
        'best_pred_recover_reward': float(pred_recover_reward),
        'best_pred_uncover_f1': float(best_detail['pred_uncover_f1'] if args.optimization_mode == 'sequential' else best_detail['uncover_f1']),
        'best_pred_recover_f1': float(pred_recover_f1),
        'sim_uncover_reward': float(sim_info['uncover reward']),
        'sim_recover_reward': float(sim_info['recover_reward']),
        'sim_uncover_f1': float(sim_f1_metrics['uncover_f1']),
        'sim_recover_f1': float(sim_f1_metrics['recover_f1']),
        'pred_sim_recover_gap': float(pred_recover_reward - sim_info['recover_reward']),
        'baseline_raw_dir': args.baseline_raw_dir,
        'sim_skip_reason': sim_skip_reason,
    }
    if args.optimization_mode == 'coupled':
        diagnostics_payload['joint_search_diagnostics'] = search_diagnostics
    else:
        diagnostics_payload['best_pair_summary'] = {
            'best_outer_eval_idx': int(best_detail['best_outer_eval_idx']),
            'best_uncover_action_policy': best_detail['uncover_action_policy'],
            'best_recover_action_policy': [] if recover_action_policy is None else np.asarray(recover_action_policy, dtype=np.float32),
            'status': best_detail['status'],
            'init_source': best_detail.get('init_source'),
            'baseline_filename': None if best_detail.get('baseline_sample') is None else best_detail['baseline_sample']['filename'],
        }
        diagnostics_payload['uncover_search_diagnostics'] = uncover_search_diagnostics
        diagnostics_payload['recover_search_diagnostics'] = recover_search_diagnostics

    diagnostics_path = save_rollout_diagnostics(output_dir, rollout_idx, diagnostics_payload)

    save_data_to_pickle(
        rollout_idx,
        seed,
        True,
        best_detail['uncover_action_policy'],
        [] if recover_action_policy is None else recover_action_policy,
        human_pose,
        target_limb_code,
        sim_info,
        cma_info,
        output_dir,
    )

    env.disconnect()
    result = {
        'optimization_mode': args.optimization_mode,
        'seed': seed,
        'target_limb_code': target_limb_code,
        'pred_joint_reward': float(best_detail['joint_reward']),
        'pred_uncover_reward': float(best_detail['pred_uncover_reward'] if args.optimization_mode == 'sequential' else best_detail['uncover_reward']),
        'pred_recover_reward': float(pred_recover_reward),
        'sim_uncover_reward': float(sim_info['uncover reward']),
        'sim_recover_reward': float(sim_info['recover_reward']),
        'pred_recover_f1': float(pred_recover_f1),
        'sim_uncover_f1': float(sim_f1_metrics['uncover_f1']),
        'sim_recover_f1': float(sim_f1_metrics['recover_f1']),
        'pred_sim_gap': float(pred_recover_reward - sim_info['recover_reward']),
        'pred_uncover_f1': float(best_detail['pred_uncover_f1'] if args.optimization_mode == 'sequential' else best_detail['uncover_f1']),
        'best_eval_idx': int(best_eval_idx),
        'diagnostics_path': diagnostics_path,
        'optimizer_time': optimizer_time,
        'sim_time': sim_time,
        'sim_skip_reason': sim_skip_reason,
    }
    if args.optimization_mode == 'coupled':
        result.update({
            'joint_feasible_ratio': float(search_diagnostics['joint_feasible_ratio']),
            'joint_num_generations': int(search_diagnostics['num_generations']),
            'joint_num_evals': int(search_diagnostics['num_joint_evals']),
        })
    else:
        result.update({
            'outer_feasible_ratio': float(uncover_search_diagnostics['outer_feasible_ratio']),
            'outer_num_generations': int(uncover_search_diagnostics['num_generations']),
            'outer_num_evals': int(uncover_search_diagnostics['num_outer_evals']),
            'outer_init_source': uncover_search_diagnostics['outer_init_source'],
            'baseline_outer_seed_count': int(uncover_search_diagnostics['baseline_outer_seed_count']),
            'random_outer_seed_count': int(uncover_search_diagnostics['random_outer_seed_count']),
            'inner_num_evals_best': int(recover_search_diagnostics['num_evals']) if recover_search_diagnostics is not None else 0,
            'inner_total_evals': int(uncover_search_diagnostics['total_inner_evals']),
        })
    return result


def save_rollout_diagnostics(output_dir, rollout_idx, diagnostics):
    diag_dir = Path(output_dir) / 'diagnostics'
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_path = diag_dir / f'rollout_{rollout_idx:04d}_joint_convergence.json'
    with open(diag_path, 'w', encoding='utf-8') as handle:
        json.dump(to_serializable(diagnostics), handle, indent=2)
    return str(diag_path)


def summarize_results(results, optimization_mode):
    pred_joint = np.array([r['pred_joint_reward'] for r in results], dtype=np.float32)
    pred_uncover = np.array([r['pred_uncover_reward'] for r in results], dtype=np.float32)
    pred_recover = np.array([r['pred_recover_reward'] for r in results], dtype=np.float32)
    sim_uncover = np.array([r['sim_uncover_reward'] for r in results], dtype=np.float32)
    sim_recover = np.array([r['sim_recover_reward'] for r in results], dtype=np.float32)
    pred_sim_gap = np.array([r['pred_sim_gap'] for r in results], dtype=np.float32)
    uncover_f1 = np.array([r['pred_uncover_f1'] for r in results], dtype=np.float32)
    pred_recover_f1 = np.array([r['pred_recover_f1'] for r in results], dtype=np.float32)
    sim_uncover_f1 = np.array([r['sim_uncover_f1'] for r in results], dtype=np.float32)
    sim_recover_f1 = np.array([r['sim_recover_f1'] for r in results], dtype=np.float32)

    print('=' * 80)
    print(f'Completed {len(results)} {optimization_mode} rollouts')
    print(f'Pred joint reward mean/std: {pred_joint.mean():.2f} / {pred_joint.std():.2f}')
    print(f'Pred uncover reward mean/std: {pred_uncover.mean():.2f} / {pred_uncover.std():.2f}')
    print(f'Pred recover reward mean/std: {pred_recover.mean():.2f} / {pred_recover.std():.2f}')
    print(f'Sim uncover reward mean/std: {sim_uncover.mean():.2f} / {sim_uncover.std():.2f}')
    print(f'Sim recover reward mean/std: {sim_recover.mean():.2f} / {sim_recover.std():.2f}')
    print(f'Pred-sim recover gap mean/std: {pred_sim_gap.mean():.2f} / {pred_sim_gap.std():.2f}')
    print(f'Pred uncover F1 mean/std: {uncover_f1.mean():.3f} / {uncover_f1.std():.3f}')
    print(f'Pred recover F1 mean/std: {pred_recover_f1.mean():.3f} / {pred_recover_f1.std():.3f}')
    print(f'Sim uncover F1 mean/std: {sim_uncover_f1.mean():.3f} / {sim_uncover_f1.std():.3f}')
    print(f'Sim recover F1 mean/std: {sim_recover_f1.mean():.3f} / {sim_recover_f1.std():.3f}')
    if optimization_mode == 'coupled':
        feasible_ratio = np.array([r['joint_feasible_ratio'] for r in results], dtype=np.float32)
        num_evals = np.array([r['joint_num_evals'] for r in results], dtype=np.float32)
        print(f'Joint feasible ratio mean/std: {feasible_ratio.mean():.3f} / {feasible_ratio.std():.3f}')
        print(f'Joint evals mean/std: {num_evals.mean():.2f} / {num_evals.std():.2f}')
    else:
        feasible_ratio = np.array([r['outer_feasible_ratio'] for r in results], dtype=np.float32)
        outer_evals = np.array([r['outer_num_evals'] for r in results], dtype=np.float32)
        inner_evals = np.array([r['inner_total_evals'] for r in results], dtype=np.float32)
        baseline_seed_counts = np.array([r['baseline_outer_seed_count'] for r in results], dtype=np.float32)
        random_seed_counts = np.array([r['random_outer_seed_count'] for r in results], dtype=np.float32)
        print(f'Outer feasible ratio mean/std: {feasible_ratio.mean():.3f} / {feasible_ratio.std():.3f}')
        print(f'Outer evals mean/std: {outer_evals.mean():.2f} / {outer_evals.std():.2f}')
        print(f'Inner total evals mean/std: {inner_evals.mean():.2f} / {inner_evals.std():.2f}')
        print(f'Baseline outer seed count mean/std: {baseline_seed_counts.mean():.2f} / {baseline_seed_counts.std():.2f}')
        print(f'Random outer seed count mean/std: {random_seed_counts.mean():.2f} / {random_seed_counts.std():.2f}')
    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(description='Coupled 8D CMA baseline or sequential 4D-4D nested joint optimization')
    parser.add_argument('--uncover-model-path', type=str,
                        default='/mnt/data/MudkipUsersSu2025/kpputhuveetil/git/robe/robust-body-exposure_unstable/trained_models/FINAL_MODELS/Recover/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Uncover_10000_states_New_Grasp_16000_epochs=250_batch=50_workers=4_1705905825')
    parser.add_argument('--recover-model-path', type=str,
                        default='/mnt/data/MudkipUsersSu2025/kpputhuveetil/git/robe/robust-body-exposure_unstable/trained_models/FINAL_MODELS/Recover/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Recover_Data_100_seeds_30000_states_30000_epochs=250_batch=50_workers=4_1705986655')
    parser.add_argument('--optimization-mode', type=str, default='sequential', choices=['coupled', 'sequential'])
    parser.add_argument('--graph-config', default='2D', choices=list(all_graph_configs.keys()))
    parser.add_argument('--env-var', default='standard', choices=list(all_env_vars.keys()))
    parser.add_argument('--max-fevals', type=int, default=80)
    parser.add_argument('--uncover-max-fevals', type=int, default=None)
    parser.add_argument('--recover-max-fevals', type=int, default=None)
    parser.add_argument('--num-rollouts', type=int, default=500)
    parser.add_argument('--uncover-f1-threshold', type=float, default=0.745, help='Only used by coupled mode.')
    parser.add_argument('--screen-uncover-f1-threshold', type=float, default=None, help='Optional sequential-mode screen; disabled by default.')
    parser.add_argument('--uncover-weight', type=float, default=1.0)
    parser.add_argument('--recover-weight', type=float, default=1.0)
    parser.add_argument('--popsize', type=int, default=8)
    parser.add_argument('--sigma', type=float, default=0.2)
    parser.add_argument('--recover-search-method', type=str, default='cma', choices=['random', 'cma'])
    parser.add_argument('--recover-warm-start-strategy', type=str, default='reverse', choices=['none', 'reverse', 'field', 'hybrid'])
    parser.add_argument('--recover-feasible-only-best', action='store_true', help='Sequential mode: only pick recover best among on-cloth candidates when possible.')
    parser.add_argument('--outer-init-source', type=str, default='heuristic', choices=['heuristic', 'baseline_limb_mixed'])
    parser.add_argument('--baseline-raw-dir', type=str, default=None, help='Optional recover baseline raw dir used for sequential baseline-mixed initialization.')
    parser.add_argument('--outer-baseline-seeds', type=int, default=4)
    parser.add_argument('--outer-random-seeds', type=int, default=4)
    parser.add_argument('--baseline-seed-selection', type=str, default='random', choices=['best_reward', 'random'])
    parser.add_argument('--inner-include-baseline-recover', dest='inner_include_baseline_recover', action='store_true', help='Sequential mode: include same-limb baseline recover actions in inner warm-start candidates.')
    parser.add_argument('--no-inner-include-baseline-recover', dest='inner_include_baseline_recover', action='store_false', help='Sequential mode: disable same-limb baseline recover warm-start candidates.')
    parser.add_argument('--inner-baseline-recover-topk', type=int, default=1)
    parser.add_argument('--arg-seed', type=int, default=0)
    parser.add_argument('--target-limb-code', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.set_defaults(inner_include_baseline_recover=True)
    args = parser.parse_args()

    if args.uncover_model_path is None or args.recover_model_path is None:
        raise ValueError('Both --uncover-model-path and --recover-model-path are required.')

    load_runtime_dependencies()

    graph_config = all_graph_configs[args.graph_config]
    env_variations = all_env_vars[args.env_var]
    device = 'cpu'

    uncover_manager = GNN_Manager(device)
    uncover_manager.load_model_from_checkpoint(args.uncover_model_path)
    uncover_manager.model.to(torch.device('cpu'))
    uncover_manager.model.eval()

    recover_manager = GNN_Manager(device)
    recover_manager.load_model_from_checkpoint(args.recover_model_path)
    recover_manager.model.to(torch.device('cpu'))
    recover_manager.model.eval()

    if args.optimization_mode == 'coupled':
        default_dir = (
            f'coupled_joint_cma_f{args.max_fevals}_thr{args.uncover_f1_threshold}_'
            f'u{args.uncover_weight}_r{args.recover_weight}'
        )
    else:
        uncover_max = args.uncover_max_fevals if args.uncover_max_fevals is not None else args.max_fevals
        recover_max = args.recover_max_fevals if args.recover_max_fevals is not None else args.max_fevals
        default_dir = (
            f'sequential_outer{uncover_max}_inner{recover_max}_{args.recover_search_method}_'
            f'{args.recover_warm_start_strategy}_u{args.uncover_weight}_r{args.recover_weight}'
        )
        if args.outer_init_source != 'heuristic':
            default_dir += (
                f'_init-{args.outer_init_source}_{args.baseline_seed_selection}'
                f'_b{args.outer_baseline_seeds}_rand{args.outer_random_seeds}'
            )
            if args.inner_include_baseline_recover:
                default_dir += f'_innerb{args.inner_baseline_recover_topk}'
    output_dir = args.output_dir if args.output_dir is not None else osp.join(
        args.recover_model_path,
        'joint_evaluations',
        default_dir,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(output_dir)

    baseline_action_pool = None
    if args.optimization_mode == 'sequential' and args.outer_init_source == 'baseline_limb_mixed':
        if args.baseline_raw_dir is None:
            raise ValueError('--baseline-raw-dir is required when --outer-init-source=baseline_limb_mixed')
        baseline_action_pool = load_baseline_action_pool(args.baseline_raw_dir)
        print(
            f"Loaded baseline pool from {args.baseline_raw_dir}: "
            f"loaded={baseline_action_pool['stats']['num_loaded']} files={baseline_action_pool['stats']['num_files']} "
            f"bad_name={baseline_action_pool['stats']['num_skipped_bad_name']} "
            f"read_error={baseline_action_pool['stats']['num_skipped_read_error']} "
            f"invalid={baseline_action_pool['stats']['num_skipped_invalid_fields']}"
        )

    rng = np.random.RandomState(args.arg_seed)
    results = []
    for rollout_idx in range(args.num_rollouts):
        result = run_single_rollout(
            rollout_idx=rollout_idx,
            args=args,
            uncover_model=uncover_manager.model,
            recover_model=recover_manager.model,
            graph_config=graph_config,
            env_variations=env_variations,
            device=device,
            output_dir=output_dir,
            rng=rng,
            baseline_action_pool=baseline_action_pool,
        )
        results.append(result)
        if args.optimization_mode == 'coupled':
            print(
                f"[{rollout_idx + 1}/{args.num_rollouts}] mode=coupled seed={result['seed']} tl={result['target_limb_code']} "
                f"pred_joint={result['pred_joint_reward']:.2f} pred_u={result['pred_uncover_reward']:.2f} "
                f"pred_r={result['pred_recover_reward']:.2f} sim_r={result['sim_recover_reward']:.2f} "
                f"u_f1={result['pred_uncover_f1']:.3f}/{result['sim_uncover_f1']:.3f} "
                f"r_f1={result['pred_recover_f1']:.3f}/{result['sim_recover_f1']:.3f} "
                f"best_eval={result['best_eval_idx']} "
                f"joint={result['joint_num_evals']}evals opt={result['optimizer_time']:.2f}s sim={result['sim_time']:.2f}s"
            )
        else:
            print(
                f"[{rollout_idx + 1}/{args.num_rollouts}] mode=sequential seed={result['seed']} tl={result['target_limb_code']} "
                f"pred_joint={result['pred_joint_reward']:.2f} pred_u={result['pred_uncover_reward']:.2f} "
                f"pred_r={result['pred_recover_reward']:.2f} sim_r={result['sim_recover_reward']:.2f} "
                f"u_f1={result['pred_uncover_f1']:.3f}/{result['sim_uncover_f1']:.3f} "
                f"r_f1={result['pred_recover_f1']:.3f}/{result['sim_recover_f1']:.3f} "
                f"init={result['outer_init_source']} bseed={result['baseline_outer_seed_count']} rseed={result['random_outer_seed_count']} "
                f"best_outer={result['best_eval_idx']} "
                f"outer={result['outer_num_evals']}evals inner_total={result['inner_total_evals']} "
                f"opt={result['optimizer_time']:.2f}s sim={result['sim_time']:.2f}s"
            )

    summarize_results(results, args.optimization_mode)


if __name__ == '__main__':
    main()
