#%%
import math
import sys
import time
import argparse
import configparser
import json
import pickle
import re
from collections import defaultdict

from torch import multiprocessing

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "assistive-gym-fem"))
sys.path.insert(0, str(_REPO_ROOT / "code"))
import os.path as osp

import random
import cma
import numpy as np
import torch
from assistive_gym.envs.bu_gnn_util import *
from assistive_gym.learn import make_env
from build_runtime_graph import Runtime_Graph
# import assistive_gym.envs.bu_gnn_util
from cma_gnn_util import *
from gnn_manager import GNN_Manager
from uncover_sim_config import apply_uncover_sim_config, pilot_uncover_sim_config, read_env_sim_fingerprint
from gym.utils import seeding
from tqdm import tqdm
import glob

#%%
recover = False
test = False
model_path_uncover = str(
    _REPO_ROOT
    / 'trained_models/FINAL_MODELS/standard_2D_not_subsampled_epochs=250_batch=100_workers=4_1687986938'
)

seed_path = str(
    _REPO_ROOT
    / 'trained_models/FINAL_MODELS/Uncover/TL_All__Uncover_Model_10k_Old_Grasp_10000_epochs=250_batch=100_workers=4_1706810068/cma_evaluations/TL_All_Uncover_Evals_500_states_1706823382593/raw'
)
filenames_iterated = None

eval_dir_name = 'cma_evaluations'
search_dir = osp.join(model_path_uncover, eval_dir_name)

eval_conditions = ['TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Uncover_Evals_Train_1000_states']

x0 = []
target_limb_list =  [2, 4, 5, 8, 10, 11, 12, 13, 14, 15]

#* parameters for the graph representation of the cloth fed as inpurecoveringt to the model
all_graph_configs = {
        '2D':{'filt_drape':False,  'rot_drape':True, 'use_3D':False, 'use_disp':True},
        '3D':{'filt_drape':False,  'rot_drape':False, 'use_3D':True, 'use_disp':True}
    }
#* parameters for which enviornmental variations to use in simulation
all_env_vars = {
        'standard':{'blanket_var':False, 'high_pose_var':False, 'body_shape_var':False},   # standard
        'body_shape_var':{'blanket_var':False, 'high_pose_var':False, 'body_shape_var':True},    # body shape var
        'pose_var':{'blanket_var':False, 'high_pose_var':True, 'body_shape_var':False},    # high pose var
        'blanket_var':{'blanket_var':True, 'high_pose_var':False, 'body_shape_var':False},    # blanket var
        'combo_var':{'blanket_var':True, 'high_pose_var':True, 'body_shape_var':True}       # combo var
    }


def parse_voxel_size(value):
    if value is None:
        return np.nan
    text = str(value).strip().lower()
    if text in {'nan', 'none', 'no', 'false'}:
        return np.nan
    return float(value)


def format_voxel_tag(voxel_size):
    if np.isnan(voxel_size):
        return 'nan'
    return f'{float(voxel_size):g}'


def load_model_dataset_settings(checkpoint):
    config = configparser.ConfigParser()
    config_path = osp.join(checkpoint, 'config.ini')
    settings = {}
    if not osp.exists(config_path):
        return settings
    config.read(config_path)
    dataset_cfg = config['Dataset'] if 'Dataset' in config else {}
    if 'voxel_size' in dataset_cfg:
        settings['voxel_size'] = parse_voxel_size(dataset_cfg['voxel_size'])
    if 'edge_threshold' in dataset_cfg:
        try:
            settings['edge_threshold'] = float(dataset_cfg['edge_threshold'])
        except ValueError:
            pass
    if 'edge_mode' in dataset_cfg:
        settings['edge_mode'] = dataset_cfg['edge_mode']
    return settings


def resolve_runtime_graph_config(graph_config, checkpoint, voxel_size_arg='auto', edge_threshold_arg='auto'):
    runtime = dict(graph_config)
    model_settings = load_model_dataset_settings(checkpoint)
    edge_mode = model_settings.get('edge_mode', runtime.get('edge_mode', 'radius'))

    if str(voxel_size_arg).strip().lower() == 'auto':
        default_voxel_size = np.nan if edge_mode == 'mesh' else 0.05
        runtime['voxel_size'] = model_settings.get('voxel_size', default_voxel_size)
    else:
        runtime['voxel_size'] = parse_voxel_size(voxel_size_arg)

    if str(edge_threshold_arg).strip().lower() == 'auto':
        runtime['edge_threshold'] = model_settings.get('edge_threshold', 0.06)
    else:
        runtime['edge_threshold'] = float(edge_threshold_arg)

    runtime['edge_mode'] = edge_mode
    return runtime


def grasp_on_cloth(action, cloth_initial_raw):
    dist, is_on_cloth = check_grasp_on_cloth(action, np.array(cloth_initial_raw))
    return is_on_cloth

def normalized_xy_occupancy_entropy(cloth_final_2D, cloth_initial_2D, grid_size=0.05):
    cloth_final_2D = np.asarray(cloth_final_2D, dtype=np.float32)
    cloth_initial_2D = np.asarray(cloth_initial_2D, dtype=np.float32)
    xy_stack = np.vstack([cloth_initial_2D[:, :2], cloth_final_2D[:, :2]])
    xy_min = np.min(xy_stack, axis=0) - float(grid_size)
    xy_max = np.max(xy_stack, axis=0) + float(grid_size)
    shape = np.maximum(np.ceil((xy_max - xy_min) / float(grid_size)).astype(np.int64) + 1, 1)
    cell_idx = np.floor((cloth_final_2D[:, :2] - xy_min) / float(grid_size)).astype(np.int64)
    valid = np.all((cell_idx >= 0) & (cell_idx < shape), axis=1)
    flat_idx = cell_idx[valid, 0] * shape[1] + cell_idx[valid, 1]
    counts = np.bincount(flat_idx, minlength=int(shape[0] * shape[1]))
    counts = counts[counts > 0].astype(np.float64)
    total_cells = int(shape[0] * shape[1])
    if counts.size == 0 or total_cells <= 1:
        return 0.0
    probs = counts / np.sum(counts)
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / math.log(float(total_cells))


def cost_function(action, all_body_points, first_cloth, cloth_initial_raw, graph, model, device, use_disp, use_3D, entropy_weight=0.0, entropy_grid_size=0.05):
    action_policy = np.asarray(action, dtype=np.float32)
    action_world = scale_action(action_policy)
    cloth_initial = graph.initial_blanket_state
    is_on_cloth= grasp_on_cloth(action_world, cloth_initial_raw)

    if is_on_cloth:
        # Runtime_Graph applies the same policy-to-world scaling as BMDataset.
        data = graph.build_graph(action_policy)

        data = data.to(device).to_dict()
        batch = data['batch']
        batch_num = np.max(batch.data.cpu().numpy()) + 1
        global_size = 0
        global_vec = torch.zeros(int(batch_num), global_size, dtype=torch.float32, device=device)
        data['u'] = global_vec
        pred = model(data)['target'].detach().numpy()

        if use_disp:
            pred = cloth_initial + pred
    else:
        pred = np.copy(cloth_initial)


    if use_3D:
        cloth_initial_2D = np.delete(cloth_initial, 2, axis = 1)
        pred_2D = np.delete(pred, 2, axis = 1)
        cost, covered_status, reward_raw, xy_entropy, overlap_penalty = get_cost(
            action_world, all_body_points, first_cloth, cloth_initial_2D, pred_2D, entropy_weight, entropy_grid_size)
    else:
        cost, covered_status, reward_raw, xy_entropy, overlap_penalty = get_cost(
            action_world, all_body_points, first_cloth, cloth_initial, pred, entropy_weight, entropy_grid_size)

    return [cost, pred, covered_status, is_on_cloth, reward_raw, xy_entropy, overlap_penalty]

def get_cost(action, all_body_points, first_cloth, cloth_initial_2D, cloth_final_2D, entropy_weight=0.0, entropy_grid_size=0.05):
    if recover:
        reward, covered_status = get_recovering_reward(action, all_body_points, first_cloth, cloth_initial_2D, cloth_final_2D)
        xy_entropy = float('nan')
        overlap_penalty = 0.0
    else:
        reward, covered_status = get_uncovering_reward(action, all_body_points, cloth_initial_2D, cloth_final_2D)
        xy_entropy = normalized_xy_occupancy_entropy(cloth_final_2D, cloth_initial_2D, entropy_grid_size)
        overlap_penalty = 1.0 - xy_entropy

    cost = -reward + float(entropy_weight) * overlap_penalty
    return cost, covered_status, float(reward), float(xy_entropy), float(overlap_penalty)

def counter_callback(output):
    global counter
    counter += 1
    print(
        f"{counter} - Trial Completed: Objective Reward: {output[1]:.2f}, "
        f"Pred Uncover Reward: {output[7]:.2f}, Pred XY Entropy: {output[8]:.3f}, "
        f"Sim Reward: {output[2]:.2f}, TL: {output[5]}, GoC: {output[6]}")

def find(seed):
    for eval_condition in eval_conditions:
        path = Path(model_path_uncover +'/' + eval_dir_name + '/' + eval_condition + '/raw/')
        filenames = path.glob('*.pkl')
        for f in filenames:
            if str(seed) in f.name:
                return f
    #%%
def load_eval_set(eval_set_path, target_limb_filter=None):
    if not eval_set_path:
        return None
    with open(Path(eval_set_path).expanduser().resolve(), 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    records = payload.get('records', []) if isinstance(payload, dict) else payload
    allowed = None if not target_limb_filter else {int(code.strip()) for code in str(target_limb_filter).split(',') if code.strip()}
    normalized = []
    for idx, record in enumerate(records):
        target_limb_code = int(record['target_limb_code'])
        if allowed is not None and target_limb_code not in allowed:
            continue
        normalized.append({
            'eval_id': record.get('eval_id', f'eval_{idx:04d}'),
            'seed': int(record['seed']),
            'target_limb_code': target_limb_code,
        })
        source_uncover_pkl = (
            record.get('source_uncover_pkl')
            or record.get('source_pkl')
            or record.get('pkl')
        )
        if source_uncover_pkl:
            normalized[-1]['source_uncover_pkl'] = str(
                Path(source_uncover_pkl).expanduser().resolve()
            )
        if 'uncover_cma_region' in record:
            normalized[-1]['uncover_cma_region'] = record['uncover_cma_region']
    return normalized


def _saved_result_path(iter_data_dir, idx, eval_record):
    """Find a valid saved fixed-eval result without scanning the whole raw dir."""
    raw_dir = Path(iter_data_dir) / 'raw'
    if not raw_dir.is_dir():
        return None
    target_limb_code = int(eval_record['target_limb_code'])
    seed = int(eval_record['seed'])
    pattern = f'tl{target_limb_code}_c{idx}_{seed}_pid*.pkl'
    candidates = sorted(raw_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    expected_eval_id = str(eval_record.get('eval_id', ''))
    for path in reversed(candidates):
        try:
            with open(path, 'rb') as handle:
                payload = pickle.load(handle)
            cma_info = payload.get('cma_info', {}) if isinstance(payload, dict) else {}
            if isinstance(cma_info, dict) and str(cma_info.get('eval_id', '')) == expected_eval_id:
                return path
        except Exception:
            # Ignore stale or partially written files.  Results are written atomically.
            continue
    # Fallback for results created by an older runner whose filename index may
    # differ after resume; this path is only used when the fast pattern misses.
    for path in raw_dir.glob('*.pkl'):
        try:
            with open(path, 'rb') as handle:
                payload = pickle.load(handle)
            cma_info = payload.get('cma_info', {}) if isinstance(payload, dict) else {}
            if isinstance(cma_info, dict) and str(cma_info.get('eval_id', '')) == expected_eval_id:
                return path
        except Exception:
            continue
    return None


def _load_saved_result(path, eval_record):
    """Reconstruct the legacy runner return tuple from a saved PKL."""
    with open(path, 'rb') as handle:
        payload = pickle.load(handle)
    cma_info = payload.get('cma_info', {})
    sim_info = payload.get('sim_info', {})
    seed = int(eval_record['seed'])
    target_limb_code = int(eval_record['target_limb_code'])
    best_reward = _safe_float(cma_info.get('best_reward', np.nan))
    uncover_reward = _safe_float(sim_info.get('uncover_reward', np.nan))
    recover_reward = _safe_float(sim_info.get('recover_reward', np.nan))
    best_time = _safe_float(cma_info.get('best_time', np.nan))
    best_is_on_cloth = bool(cma_info.get('feasible_found', False))
    pred_uncover_reward = _safe_float(cma_info.get('best_pred_uncover_reward_raw', np.nan))
    pred_xy_entropy = _safe_float(cma_info.get('best_pred_xy_entropy', np.nan))
    return (
        seed, best_reward, uncover_reward, recover_reward, best_time,
        target_limb_code, best_is_on_cloth, pred_uncover_reward, pred_xy_entropy,
    )


def _safe_float(value, default=np.nan):
    """Convert scalar-like legacy PKL values, including NumPy/list wrappers."""
    try:
        array = np.asarray(value)
        if array.size != 1:
            return float(default)
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return float(default)


def _write_watchdog_report(iter_data_dir, failures):
    if not failures:
        return
    report_path = Path(iter_data_dir) / 'watchdog_failures.json'
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(failures, handle, indent=2)
        handle.write('\n')
    print(f'[Watchdog] Failure report: {report_path}')


ACTION_POLICY_SCALE = np.array([0.44, 1.05, 0.44, 1.05], dtype=np.float64)


def _clipped_policy_interval(world_low, world_high, scale, min_width=1e-3):
    low = float(world_low) / float(scale)
    high = float(world_high) / float(scale)
    if low > high:
        low, high = high, low

    low = max(-1.0, low)
    high = min(1.0, high)
    if low > high:
        center = float(np.clip((low + high) / 2.0, -1.0, 1.0))
        low = max(-1.0, center - min_width / 2.0)
        high = min(1.0, center + min_width / 2.0)
    elif high - low < min_width:
        center = (low + high) / 2.0
        low = max(-1.0, center - min_width / 2.0)
        high = min(1.0, center + min_width / 2.0)
    return low, high


def build_uncover_cma_bounds(all_body_points, region, cloth_points=None):
    lower = np.full(4, -1.0, dtype=np.float64)
    upper = np.full(4, 1.0, dtype=np.float64)
    if not region:
        return lower, upper, None

    region_type = region.get('type')
    edge = region.get('edge')
    applies_to = region.get('applies_to', 'pick')
    bbox_source = region.get('bbox_source', 'human')
    if region_type not in ('human_bbox_edge_band', 'bbox_edge_band'):
        raise ValueError(f"Unsupported uncover_cma_region type: {region_type}")
    if edge != 'bottom_max_y':
        raise ValueError(f"Unsupported bbox edge: {edge}")
    if applies_to != 'pick':
        raise ValueError(f"Unsupported uncover_cma_region applies_to: {applies_to}")
    if bbox_source not in ('human', 'cloth', 'blanket'):
        raise ValueError(f"Unsupported uncover_cma_region bbox_source: {bbox_source}")

    band = float(region.get('band', 0.08))
    inner_margin = float(region.get('inner_margin', band))
    outer_margin = float(region.get('outer_margin', band))
    x_margin = float(region.get('x_margin', 0.03))
    x_mode = region.get('x_mode', 'bbox')
    if inner_margin < 0:
        raise ValueError("uncover_cma_region inner_margin must be non-negative")
    if outer_margin < 0:
        raise ValueError("uncover_cma_region outer_margin must be non-negative")
    if inner_margin == 0 and outer_margin == 0:
        raise ValueError("uncover_cma_region must keep a non-zero y interval")
    if x_margin < 0:
        raise ValueError("uncover_cma_region x_margin must be non-negative")
    if x_mode not in ('bbox', 'full'):
        raise ValueError(f"Unsupported uncover_cma_region x_mode: {x_mode}")

    if bbox_source == 'human':
        bbox_xy = np.asarray(all_body_points, dtype=np.float64)[:, :2]
    else:
        if cloth_points is None:
            raise ValueError("Cannot build cloth bbox bounds without cloth points")
        bbox_xy = np.asarray(cloth_points, dtype=np.float64)[:, :2]
    bbox_xy = bbox_xy[np.all(np.isfinite(bbox_xy), axis=1)]
    if bbox_xy.size == 0:
        raise ValueError(f"Cannot build {bbox_source} bbox bounds from empty points")

    bbox_min = np.min(bbox_xy, axis=0)
    bbox_max = np.max(bbox_xy, axis=0)
    if x_mode == 'bbox':
        pick_x = _clipped_policy_interval(
            bbox_min[0] - x_margin,
            bbox_max[0] + x_margin,
            ACTION_POLICY_SCALE[0])
        lower[0], upper[0] = pick_x
    pick_y = _clipped_policy_interval(
        bbox_max[1] - inner_margin,
        bbox_max[1] + outer_margin,
        ACTION_POLICY_SCALE[1])

    lower[1], upper[1] = pick_y
    applied = {
        'region': region,
        'bbox_source': bbox_source,
        'bbox_world_xy': {
            'min': bbox_min.astype(float).tolist(),
            'max': bbox_max.astype(float).tolist(),
        },
        'bounds_policy': {
            'lower': lower.astype(float).tolist(),
            'upper': upper.astype(float).tolist(),
        },
    }
    return lower, upper, applied


def get_eval_region_tag(eval_records):
    if not eval_records:
        return ''
    regions = [record.get('uncover_cma_region') for record in eval_records if record.get('uncover_cma_region')]
    if not regions:
        return ''
    region = regions[0]
    if region.get('type') in ('human_bbox_edge_band', 'bbox_edge_band') and region.get('edge') == 'bottom_max_y':
        source_tag = 'Cloth' if region.get('bbox_source') in ('cloth', 'blanket') else 'BBox'
        if 'inner_margin' in region or 'outer_margin' in region:
            inner_margin = float(region.get('inner_margin', 0.0))
            outer_margin = float(region.get('outer_margin', 0.0))
            x_mode = region.get('x_mode', 'bbox')
            x_tag = 'XFull' if x_mode == 'full' else f"XMargin{float(region.get('x_margin', 0.03)):g}"
            return f'_{source_tag}BottomPickYInner{inner_margin:g}_Outer{outer_margin:g}_{x_tag}'
        band = float(region.get('band', 0.08))
        x_margin = float(region.get('x_margin', 0.03))
        return f'_{source_tag}BottomPickBand{band:g}_XMargin{x_margin:g}'
    return '_ConstrainedCMA'


DEFAULT_TRAIN_ACTION_DIR = str(
    _REPO_ROOT
    / 'DATASETS/Uncover_Data/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_'
      'Uncover_Data_10000_states_New_Grasp/raw'
)


def load_training_action_bounds(train_dir, low_pct=1.0, high_pct=99.0):
    by_tl = defaultdict(list)
    for p in Path(train_dir).glob('*.pkl'):
        match = re.match(r'c_(\d+)_', p.name)
        if not match:
            continue
        with open(p, 'rb') as handle:
            raw_data = pickle.load(handle)
        by_tl[int(match.group(1))].append(np.asarray(raw_data['uncover_action'], dtype=np.float64))

    bounds_by_tl = {}
    for target_limb_code, actions in sorted(by_tl.items()):
        action_array = np.stack(actions)
        bounds_by_tl[target_limb_code] = {
            'lower': np.percentile(action_array, low_pct, axis=0).astype(np.float64),
            'upper': np.percentile(action_array, high_pct, axis=0).astype(np.float64),
            'count': int(action_array.shape[0]),
            'low_pct': float(low_pct),
            'high_pct': float(high_pct),
        }
    return bounds_by_tl


def intersect_action_bounds(lower_bounds, upper_bounds, train_lower, train_upper, dims):
    lower_bounds = np.asarray(lower_bounds, dtype=np.float64).copy()
    upper_bounds = np.asarray(upper_bounds, dtype=np.float64).copy()
    train_lower = np.asarray(train_lower, dtype=np.float64)
    train_upper = np.asarray(train_upper, dtype=np.float64)

    for dim in dims:
        low = max(float(lower_bounds[dim]), float(train_lower[dim]))
        high = min(float(upper_bounds[dim]), float(train_upper[dim]))
        if low >= high:
            center = 0.5 * (float(train_lower[dim]) + float(train_upper[dim]))
            width = max(float(train_upper[dim]) - float(train_lower[dim]), 1e-3)
            low = max(-1.0, center - width / 2.0)
            high = min(1.0, center + width / 2.0)
        lower_bounds[dim] = low
        upper_bounds[dim] = high
    return lower_bounds, upper_bounds


def get_action_bounds_tag(action_bounds_mode, train_percentile_low, train_percentile_high, train_action_pick_only):
    if action_bounds_mode == 'full':
        return '_ActFull'
    pick_tag = 'Pick' if train_action_pick_only else 'All'
    return f'_ActTrainP{train_percentile_low:g}-{train_percentile_high:g}{pick_tag}'


def gnn_cma(seed, target, env_name, idx, model, device, target_limb_code, iter_data_dir, graph_config, env_var, max_fevals,
            entropy_weight=0.0, entropy_grid_size=0.05, eval_id=None, fixed_eval=False, uncover_cma_region=None,
            source_uncover_pkl=None, post_release_steps=3, feasible_only_best=False, action_bounds_mode='full',
            training_action_bounds=None, train_action_pick_only=True,
            lower_before_release_uncover=True, cma_seed_base=0,
            uncover_pick_policy='legacy', optimize_place_only=False):
    uncover_pick_policy = str(uncover_pick_policy or 'legacy')
    optimize_place_only = bool(optimize_place_only)
    if optimize_place_only:
        raise ValueError(
            'place-only generic geometry is not included in the paper release.'
        )
    if optimize_place_only and recover:
        raise ValueError('--optimize-place-only is only supported for uncover (recover=False)')

    use_disp = graph_config['use_disp']
    filter_draping = graph_config['filt_drape']
    rot_draping = graph_config['rot_drape']
    use_3D = graph_config['use_3D']
    edge_mode = graph_config.get('edge_mode', 'radius')
    graph_voxel_size = graph_config.get('voxel_size', np.nan if edge_mode == 'mesh' else 0.05)
    graph_edge_threshold = graph_config.get('edge_threshold', 0.06)

    coop = 'Human' in env_name

    if target_limb_code is None:
        target_limb_code = random.sample(target_limb_list, 1)[0]

    target_limb_code = target

    # seed = seeding.create_seed()

    # choose uncovered state from test set to recover from
    if recover:
        if source_uncover_pkl:
            random_file = Path(source_uncover_pkl).expanduser().resolve()
            if not random_file.is_file():
                raise FileNotFoundError(
                    f'Fixed evaluation source PKL does not exist: {random_file}'
                )
        else:
            random_file = np.random.choice(
                list(
                    (
                        Path(model_path_uncover)
                        / eval_dir_name
                        / eval_conditions[0]
                        / 'raw'
                    ).iterdir()
                )
            )
        seed = int(random_file.name.split('_')[2])
        target_limb_code = int(random_file.name.split('_')[0].replace('tl', ''))
        # seed = 16050922050611383883
        # seed_path = find(seed) # for recovering from one state
        # seed_path = open(seed_path, 'rb')
        seed_path = open(random_file, 'rb')
        raw_data = pickle.load(seed_path)

    env = make_env(env_name, coop=coop, seed=seed)

    sim_cfg = pilot_uncover_sim_config(
        lower_before_release_uncover=bool(lower_before_release_uncover),
        post_release_steps=int(post_release_steps),
        collect_data=False,
    )
    sim_cfg['blanket_pose_var'] = bool(env_var['blanket_var'])
    sim_cfg['high_pose_var'] = bool(env_var['high_pose_var'])
    sim_cfg['body_shape_var'] = bool(env_var['body_shape_var'])
    apply_uncover_sim_config(env, sim_cfg)
    env.set_target_limb_code(target_limb_code)
    env.set_seed_val(seed)

    done = False
    human_pose = env.reset()
    human_pose = np.reshape(human_pose, (-1,2))

    if recover:
        cloth_initial_dc = np.delete(np.array(raw_data['info']['cloth_initial'][1]), 2, axis=1)
        cloth_intermediate_dc = raw_data['info']['cloth_final'][1]
        input_cloth = cloth_intermediate_dc
        uncover_action = raw_data['uncover_action']
    else:
        cloth_initial_dc = []
        input_cloth = env.get_cloth_state()

    env.set_target_limb_code(target_limb_code)
    pop_size = 8

    body_info = env.get_human_body_info()
    all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)

    graph = Runtime_Graph(
        root = iter_data_dir,
        description=f"iter_{iter}_processed",
        voxel_size=graph_voxel_size,
        edge_threshold=graph_edge_threshold,
        action_to_all=True,
        cloth_initial=input_cloth,
        filter_draping=filter_draping,
        rot_draping=rot_draping,
        use_3D=use_3D,
        edge_mode=edge_mode)

    # * set variables to initialize CMA-ES
    cma_seed = int(cma_seed_base) + (abs(hash(str(eval_id))) % 1_000_000 if eval_id is not None else int(idx))
    opts = cma.CMAOptions({'verb_disp': 1, 'popsize': pop_size, 'maxfevals': max_fevals, 'tolfun': 1e-11, 'tolflatfitness': 20, 'tolfunhist': 1e-20, 'seed': cma_seed}) # , 'tolfun': 10, 'max_fevals': 500
    lower_bounds, upper_bounds, applied_cma_region = build_uncover_cma_bounds(
        all_body_points,
        None if recover else uncover_cma_region,
        None if recover else input_cloth)
    applied_train_bounds = None
    if action_bounds_mode == 'train-percentile':
        if training_action_bounds is None or target_limb_code not in training_action_bounds:
            raise ValueError(
                f'No training action bounds available for target limb {target_limb_code} '
                f'(mode={action_bounds_mode}).')
        train_bounds = training_action_bounds[target_limb_code]
        dims = [0, 1] if train_action_pick_only else list(range(4))
        lower_bounds, upper_bounds = intersect_action_bounds(
            lower_bounds,
            upper_bounds,
            train_bounds['lower'],
            train_bounds['upper'],
            dims=dims)
        applied_train_bounds = {
            'mode': action_bounds_mode,
            'pick_only': bool(train_action_pick_only),
            'dims': dims,
            'train_count': int(train_bounds['count']),
            'train_lower': train_bounds['lower'].astype(float).tolist(),
            'train_upper': train_bounds['upper'].astype(float).tolist(),
            'bounds_lower': lower_bounds.astype(float).tolist(),
            'bounds_upper': upper_bounds.astype(float).tolist(),
        }
    # Generic-geometry fixed pick + place warm-start (uncover place-only mode).
    generic_geometry_meta = None
    pick_fixed = None
    place_warm_start = None
    if optimize_place_only:
        raise ValueError(
            'place-only generic geometry is not included in the paper release; '
            'use the legacy four-dimensional CMA contract.'
        )

    if optimize_place_only:
        cma_lower = np.asarray(lower_bounds[2:4], dtype=np.float64)
        cma_upper = np.asarray(upper_bounds[2:4], dtype=np.float64)
    else:
        cma_lower = np.asarray(lower_bounds, dtype=np.float64)
        cma_upper = np.asarray(upper_bounds, dtype=np.float64)
    cma_stds = np.maximum((cma_upper - cma_lower) / 2.0, 1e-3)
    opts.set('bounds', [cma_lower.tolist(), cma_upper.tolist()])
    opts.set('CMA_stds', cma_stds)

    # initialize x0
    global x0
    if optimize_place_only:
        x0 = np.clip(np.asarray(place_warm_start, dtype=np.float64), cma_lower, cma_upper).tolist()
    else:
        x0 = set_x0_for_cmaes(target_limb_code)
        if recover:
            x0 = [*uncover_action[2:4], *uncover_action[0:2]]
        x0 = np.clip(np.asarray(x0, dtype=np.float64), cma_lower, cma_upper).tolist()

    sigma0 = 0.2
    reward_threshold = 95

    total_fevals = 0

    fevals = 0
    iterations = 0
    t0 = time.time()

    # * initialize CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_cost = None
    best_pred_uncover_reward_raw = -np.inf
    best_action = None
    feasible_tracker = {
        'found': False,
        'best_cost': None,
    }
    while True:
        if fixed_eval:
            if total_fevals >= max_fevals:
                break
        elif es.stop():
            break
        iterations += 1
        fevals += pop_size
        total_fevals += pop_size

        # ! this will be your recovering actions
        asked = es.ask()
        if optimize_place_only:
            actions_cma = asked
            actions = [assemble_fixed_pick_action(pick_fixed, place) for place in actions_cma]
        else:
            actions_cma = asked
            actions = asked
        output = [
            cost_function(
                x, all_body_points, cloth_initial_dc, input_cloth, graph, model, device, use_disp, use_3D,
                entropy_weight=entropy_weight, entropy_grid_size=entropy_grid_size)
            for x in actions
        ]
        t1 = time.time()
        output = [list(x) for x in zip(*output)]
        costs = output[0]
        preds = output[1]
        covered_status = output[2]
        is_on_cloth = output[3]
        raw_rewards = output[4]
        xy_entropies = output[5]
        overlap_penalties = output[6]
        es.tell(actions_cma, costs)

        def _accept_candidate(ind):
            nonlocal best_cost, best_reward, best_action, best_pred, best_covered_status
            nonlocal best_time, best_fevals, best_iterations, best_is_on_cloth
            nonlocal best_pred_uncover_reward_raw, best_pred_xy_entropy, best_pred_overlap_penalty
            best_cost = costs[ind]
            best_reward = -best_cost
            best_action = actions[ind]
            best_pred = preds[ind]
            best_covered_status = covered_status[ind]
            best_time = t1 - t0
            best_fevals = fevals
            best_iterations = iterations
            best_is_on_cloth = is_on_cloth[ind]
            best_pred_uncover_reward_raw = raw_rewards[ind]
            best_pred_xy_entropy = xy_entropies[ind]
            best_pred_overlap_penalty = overlap_penalties[ind]

        # Track best among on-cloth candidates (used when --feasible-only-best).
        for ind, on_cloth in enumerate(is_on_cloth):
            if not on_cloth:
                continue
            if (feasible_tracker['best_cost'] is None) or (costs[ind] < feasible_tracker['best_cost']):
                feasible_tracker.update({
                    'found': True,
                    'best_cost': costs[ind],
                    'best_reward': -costs[ind],
                    'best_action': actions[ind],
                    'best_pred': preds[ind],
                    'best_covered_status': covered_status[ind],
                    'best_time': t1 - t0,
                    'best_fevals': fevals,
                    'best_iterations': iterations,
                    'best_is_on_cloth': True,
                    'best_pred_uncover_reward_raw': raw_rewards[ind],
                    'best_pred_xy_entropy': xy_entropies[ind],
                    'best_pred_overlap_penalty': overlap_penalties[ind],
                })

        if (best_cost is None) or (np.min(costs) < best_cost):
            _accept_candidate(int(np.argmin(costs)))

        stop_reward = best_pred_uncover_reward_raw
        if feasible_only_best and feasible_tracker['found']:
            stop_reward = feasible_tracker['best_pred_uncover_reward_raw']
        if not fixed_eval and stop_reward >= reward_threshold:
            break

    if feasible_only_best and feasible_tracker['found']:
        best_cost = feasible_tracker['best_cost']
        best_reward = feasible_tracker['best_reward']
        best_action = feasible_tracker['best_action']
        best_pred = feasible_tracker['best_pred']
        best_covered_status = feasible_tracker['best_covered_status']
        best_time = feasible_tracker['best_time']
        best_fevals = feasible_tracker['best_fevals']
        best_iterations = feasible_tracker['best_iterations']
        best_is_on_cloth = feasible_tracker['best_is_on_cloth']
        best_pred_uncover_reward_raw = feasible_tracker['best_pred_uncover_reward_raw']
        best_pred_xy_entropy = feasible_tracker['best_pred_xy_entropy']
        best_pred_overlap_penalty = feasible_tracker['best_pred_overlap_penalty']

    if best_action is None:
        # Degenerate empty CMA loop — fall back to warm start / legacy x0.
        if optimize_place_only:
            best_action = assemble_fixed_pick_action(pick_fixed, place_warm_start)
        else:
            best_action = np.asarray(x0, dtype=np.float64)
        best_cost = float('inf')
        best_reward = -best_cost
        best_pred = None
        best_covered_status = None
        best_time = time.time() - t0
        best_fevals = 0
        best_iterations = 0
        best_is_on_cloth = False
        best_pred_uncover_reward_raw = -np.inf
        best_pred_xy_entropy = 0.0
        best_pred_overlap_penalty = 0.0
    else:
        best_action = np.asarray(best_action, dtype=np.float64)

    if optimize_place_only and pick_fixed is not None:
        # Hard-lock pick in the recorded / executed action.
        best_action = assemble_fixed_pick_action(pick_fixed, best_action[2:4])

    if recover:
        cloth_initial_sim, cloth_intermediate_sim, execute_recover_action = env.uncover_step(uncover_action)
        cloth_final_sim, execute_recover_action = env.recover_step(best_action) # if recovering recover action is predicted by the model
    else:
        cloth_initial_sim, cloth_intermediate_sim, execute_uncover_action = env.uncover_step(best_action)
        cloth_final_sim, execute_recover_action = env.recover_step([]) # if not recovering, don't need to provide an action

    observation, uncover_reward, recover_reward, done, info = env.get_info()

    sim_info = {'observation':observation, 'uncover_reward':uncover_reward, 'recover_reward':recover_reward, 'done':done, 'info':info}
    cma_info = {'best_cost':best_cost, 'best_reward':best_reward, 'best_pred':best_pred, 'best_time':best_time,
                'best_covered_status':best_covered_status, 'best_fevals':best_fevals, 'best_iterations':best_iterations,
                'eval_id':eval_id, 'source_uncover_pkl':source_uncover_pkl,
                'entropy_weight':float(entropy_weight), 'entropy_grid_size':float(entropy_grid_size),
                'best_pred_xy_entropy':float(best_pred_xy_entropy), 'best_pred_overlap_penalty':float(best_pred_overlap_penalty),
                'best_pred_uncover_reward_raw':float(best_pred_uncover_reward_raw),
                'best_pred_regularized_cost':float(best_cost),
                'uncover_cma_region': uncover_cma_region,
                'applied_cma_region': applied_cma_region,
                'cma_bounds_lower': lower_bounds.astype(float).tolist(),
                'cma_bounds_upper': upper_bounds.astype(float).tolist(),
                'post_release_steps': int(post_release_steps),
                'feasible_only_best': bool(feasible_only_best),
                'feasible_found': bool(feasible_tracker['found']),
                'runtime_voxel_size': None if np.isnan(graph_voxel_size) else float(graph_voxel_size),
                'runtime_edge_threshold': float(graph_edge_threshold),
                'runtime_edge_mode': edge_mode,
                'max_fevals': int(max_fevals),
                'action_bounds_mode': action_bounds_mode,
                'applied_train_bounds': applied_train_bounds,
                'lower_before_release_uncover': bool(lower_before_release_uncover),
                'cma_seed': int(cma_seed),
                'cma_seed_base': int(cma_seed_base),
                'uncover_pick_policy': uncover_pick_policy,
                'optimize_place_only': bool(optimize_place_only),
                'cma_optimize_dims': [2, 3] if optimize_place_only else [0, 1, 2, 3],
                'generic_geometry': generic_geometry_meta,
                'sim_config': read_env_sim_fingerprint(env)}

    if recover:
        save_data_to_pickle(
            idx,
            seed,
            recover,
            uncover_action,
            best_action,
            human_pose,
            target_limb_code,
            sim_info,
            cma_info,
            iter_data_dir)
    else:
        save_data_to_pickle(
            idx,
            seed,
            recover,
            best_action,
            [],
            human_pose,
            target_limb_code,
            sim_info,
            cma_info,
            iter_data_dir)

    return (
        seed, best_reward, uncover_reward, recover_reward, best_time, target_limb_code, best_is_on_cloth,
        best_pred_uncover_reward_raw, best_pred_xy_entropy)


def evaluate_dyn_model(env_name, target_limb_code, trials, model, iter_data_dir, device, num_processes, graph_config, env_variations, max_fevals,
                       entropy_weight=0.0, entropy_grid_size=0.05, eval_records=None, idx_offset=0, post_release_steps=3,
                       feasible_only_best=False, action_bounds_mode='full', training_action_bounds=None,
                       train_action_pick_only=True, lower_before_release_uncover=True, cma_seed_base=0,
                       uncover_pick_policy='legacy', optimize_place_only=False,
                       trial_timeout=900.0, max_retries=2):
    """Run one evaluation batch without allowing a simulator worker to block forever.

    Fixed-eval jobs are retried by eval_id.  Any PKL atomically written before a
    timed-out pool is terminated is retained and skipped on the retry.
    """
    if eval_records is None:
        # Keep legacy random-rollout mode compatible, but bound result.get().
        result_objs = []
        pool = multiprocessing.Pool(processes=num_processes)
        normal_completion = False
        try:
            for i in range(trials):
                idx = idx_offset + i
                filename = next(filenames_iterated)
                target = int(filename.name.split('_')[0][2:])
                seed = int(filename.name.split('_')[2])
                result_objs.append(pool.apply_async(
                    gnn_cma,
                    args=(seed, target, env_name, idx, model, device, target_limb_code, iter_data_dir, graph_config, env_variations,
                          max_fevals, entropy_weight, entropy_grid_size, None, False, None, None,
                          post_release_steps, feasible_only_best, action_bounds_mode, training_action_bounds,
                          train_action_pick_only, lower_before_release_uncover, cma_seed_base,
                          uncover_pick_policy, optimize_place_only),
                    callback=counter_callback))
            results = [result.get(timeout=trial_timeout) for result in result_objs]
            normal_completion = True
        finally:
            if normal_completion:
                pool.close()
            else:
                pool.terminate()
            pool.join()

        results_array = np.array(results)
        pred_sim_reward_error = abs(results_array[:,2] - results_array[:,1])
        return list(results_array[:,1]), list(results_array[:,2]), list(pred_sim_reward_error)

    all_jobs = []
    pending_jobs = []
    for i in range(trials):
        idx = idx_offset + i
        eval_record = eval_records[idx]
        job = (idx, eval_record)
        all_jobs.append(job)
        saved = _saved_result_path(iter_data_dir, idx, eval_record)
        if saved is None:
            pending_jobs.append(job)
        else:
            print(f'[Watchdog] Reusing completed case idx={idx}: {saved.name}')

    failures = []
    worker_count = max(1, min(num_processes, len(pending_jobs))) if pending_jobs else 1

    for attempt in range(max_retries + 1):
        if not pending_jobs:
            break

        print(
            f'[Watchdog] attempt={attempt + 1}/{max_retries + 1}, '
            f'pending={len(pending_jobs)}, workers={worker_count}, '
            f'timeout={trial_timeout:.0f}s')
        pool = None
        normal_completion = False
        timed_out = False
        submitted = []
        try:
            pool = multiprocessing.Pool(processes=worker_count)
            for idx, eval_record in pending_jobs:
                target = int(eval_record['target_limb_code'])
                seed = int(eval_record['seed'])
                eval_id = eval_record['eval_id']
                uncover_cma_region = eval_record.get('uncover_cma_region')
                source_uncover_pkl = eval_record.get('source_uncover_pkl')
                result = pool.apply_async(
                    gnn_cma,
                    args=(seed, target, env_name, idx, model, device, target_limb_code, iter_data_dir, graph_config, env_variations,
                          max_fevals, entropy_weight, entropy_grid_size, eval_id, True, uncover_cma_region,
                          source_uncover_pkl, post_release_steps, feasible_only_best, action_bounds_mode, training_action_bounds,
                          train_action_pick_only, lower_before_release_uncover, cma_seed_base,
                          uncover_pick_policy, optimize_place_only),
                    callback=counter_callback)
                submitted.append((idx, eval_record, result))

            active = list(submitted)
            deadline = time.monotonic() + float(trial_timeout)
            while active:
                next_active = []
                for idx, eval_record, result in active:
                    if not result.ready():
                        next_active.append((idx, eval_record, result))
                        continue
                    try:
                        result.get(timeout=0)
                    except Exception as exc:
                        failures.append({
                            'idx': int(idx),
                            'eval_id': eval_record.get('eval_id'),
                            'type': 'worker_exception',
                            'message': repr(exc),
                            'attempt': int(attempt + 1),
                        })
                active = next_active
                if not active:
                    normal_completion = True
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    for idx, eval_record, _ in active:
                        failures.append({
                            'idx': int(idx),
                            'eval_id': eval_record.get('eval_id'),
                            'type': 'timeout',
                            'timeout_seconds': float(trial_timeout),
                            'attempt': int(attempt + 1),
                        })
                    print(f'[Watchdog] timeout: {len(active)} worker(s) still active; terminating pool')
                    break
                time.sleep(0.5)
        finally:
            if pool is not None:
                if normal_completion and not timed_out:
                    pool.close()
                else:
                    pool.terminate()
                pool.join()

        pending_jobs = [
            job for job in all_jobs
            if _saved_result_path(iter_data_dir, job[0], job[1]) is None
        ]
        completed_count = len(all_jobs) - len(pending_jobs)
        print(f'[Watchdog] saved={completed_count}/{len(all_jobs)} in this batch')

        if not pending_jobs:
            break
        if attempt >= max_retries:
            _write_watchdog_report(iter_data_dir, failures)
            unresolved = [
                {'idx': int(idx), 'eval_id': record.get('eval_id'), 'seed': int(record['seed'])}
                for idx, record in pending_jobs
            ]
            raise RuntimeError(
                f'Watchdog exhausted retries; {len(unresolved)} case(s) have no valid PKL: {unresolved}'
            )

        worker_count = max(1, min(max(1, worker_count // 2), len(pending_jobs)))
        print(f'[Watchdog] retrying unresolved cases with workers={worker_count}')

    _write_watchdog_report(iter_data_dir, failures)
    result_rows = []
    for idx, eval_record in all_jobs:
        saved = _saved_result_path(iter_data_dir, idx, eval_record)
        if saved is None:
            raise RuntimeError(
                f'Missing valid result after watchdog: idx={idx}, eval_id={eval_record.get("eval_id")}'
            )
        result_rows.append(_load_saved_result(saved, eval_record))

    results_array = np.array(result_rows)
    pred_sim_reward_error = abs(results_array[:,2] - results_array[:,1])
    return list(results_array[:,1]), list(results_array[:,2]), list(pred_sim_reward_error)


#%%

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    trained_models_dir = './trained_models/FINAL_MODELS'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval-multiple-models', type=bool, default=False)
    parser.add_argument('--model-path', type=str, default = 'Uncover/TL_All__Uncover_Model_10k_New_Grasp_10000_epochs=250_batch=100_workers=4_1706814845')
    parser.add_argument('--graph-config', type=str, default='2D')
    parser.add_argument('--env-var', type=str, default='standard')
    parser.add_argument('--max-fevals', type=int, default=300)
    parser.add_argument('--num-rollouts', type=int, default=500)
    parser.add_argument('--arg_seed', type=int, default=0)
    parser.add_argument('--eval-set', type=str, default='',
                        help='Fixed evaluation-set JSON; seed and target limb are reused while actions are re-optimized.')
    parser.add_argument('--recover', action='store_true',
                        help='Run Recover CMA from the fixed Uncover source PKLs.')
    parser.add_argument('--resume-eval-dir', type=str, default='',
                        help='Existing fixed-eval output directory. Skip eval_ids already present in raw/ and append only missing records.')
    parser.add_argument('--target-limb-filter', type=str, default='',
                        help='Comma-separated target limb codes to keep from --eval-set, for example 12,14,15.')
    parser.add_argument('--entropy-weight', type=float, default=0.0,
                        help='Weight applied to predicted XY overlap penalty during uncover optimization.')
    parser.add_argument('--entropy-grid-size', type=float, default=0.05,
                        help='XY occupancy grid cell size used for normalized entropy.')
    parser.add_argument('--num-processes', type=int, default=1,
                        help='Workers for fixed eval-set runs; keep low because each worker loads a simulation.')
    parser.add_argument('--trial-timeout', type=float, default=900.0,
                        help='Watchdog timeout in seconds for one fixed-eval batch before terminating stuck workers.')
    parser.add_argument('--max-retries', type=int, default=2,
                        help='Maximum retries for unresolved fixed-eval cases after a timeout or worker exception.')
    parser.add_argument('--post-release-steps', type=int, default=3,
                        help='Post-release settle steps for uncover/recover sim (env default is 50 if unset).')
    parser.add_argument('--feasible-only-best', action='store_true',
                        help='Select best action only among candidates with is_on_cloth=True when any exist.')
    parser.add_argument('--voxel-size', default='auto',
                        help='Runtime graph voxel size. Use auto to read model config.ini; nan disables subsampling.')
    parser.add_argument('--edge-threshold', default='auto',
                        help='Runtime graph edge threshold. Use auto to read model config.ini.')
    parser.add_argument('--action-bounds-mode', choices=['full', 'train-percentile'], default='full',
                        help='CMA action bounds: full [-1,1] or intersect with training-action percentiles.')
    parser.add_argument('--train-action-dir', default=DEFAULT_TRAIN_ACTION_DIR,
                        help='Training raw PKL directory used for train-percentile action bounds.')
    parser.add_argument('--train-percentile-low', type=float, default=1.0,
                        help='Lower training-action percentile when --action-bounds-mode train-percentile.')
    parser.add_argument('--train-percentile-high', type=float, default=99.0,
                        help='Upper training-action percentile when --action-bounds-mode train-percentile.')
    parser.add_argument('--train-action-pick-only', action='store_true', default=True,
                        help='When using train-percentile bounds, constrain only pick dims [x0,y0].')
    parser.add_argument('--train-action-all-dims', action='store_true',
                        help='When set, train-percentile bounds apply to all 4 action dims.')
    parser.add_argument('--lower-before-release', dest='lower_before_release', action='store_true', default=True,
                        help='Enable active effector lowering before uncover release (default: on).')
    parser.add_argument('--no-lower-before-release', dest='lower_before_release', action='store_false',
                        help='Disable active effector lowering before uncover release.')
    parser.add_argument('--cma-seed-base', type=int, default=0,
                        help='Base seed for deterministic CMA optimization; paired crossed runs should reuse it.')
    parser.add_argument('--checkpoint-name', default='model_best_heldout.pth',
                        help='Checkpoint filename inside model checkpoints/ directory.')
    parser.add_argument('--lowering-tag', default='',
                        help='Optional tag appended to eval output directory, e.g. LowerOn or LowerOff.')
    parser.add_argument(
        '--uncover-pick-policy',
        choices=['legacy'],
        default='legacy',
        help='Use the paper release four-dimensional CMA action contract.',
    )
    args = parser.parse_args()
    recover = bool(args.recover)

    if args.entropy_grid_size <= 0:
        raise ValueError('--entropy-grid-size must be positive.')
    if args.post_release_steps < 0:
        raise ValueError('--post-release-steps must be >= 0.')
    if args.trial_timeout <= 0:
        raise ValueError('--trial-timeout must be positive.')
    if args.max_retries < 0:
        raise ValueError('--max-retries must be non-negative.')
    if not 0 <= args.train_percentile_low < args.train_percentile_high <= 100:
        raise ValueError('Training percentiles must satisfy 0 <= low < high <= 100.')
    train_action_pick_only = not args.train_action_all_dims
    print(f'[Release Steps] post_release_steps={args.post_release_steps}')
    print(f'[Feasible Only] feasible_only_best={args.feasible_only_best}')
    print(f'[Watchdog] trial_timeout={args.trial_timeout:.0f}s, max_retries={args.max_retries}')
    print(
        f'[Pick Policy] uncover_pick_policy={args.uncover_pick_policy}')
    print(
        f'[Action Bounds] mode={args.action_bounds_mode}, '
        f'pick_only={train_action_pick_only}, '
        f'percentiles=({args.train_percentile_low:g}, {args.train_percentile_high:g})')

    training_action_bounds = None
    if args.action_bounds_mode == 'train-percentile':
        training_action_bounds = load_training_action_bounds(
            args.train_action_dir,
            low_pct=args.train_percentile_low,
            high_pct=args.train_percentile_high)
        print(
            f'[Action Bounds] loaded training bounds for {len(training_action_bounds)} limbs from {args.train_action_dir}')

    eval_records = load_eval_set(args.eval_set, args.target_limb_filter) if args.eval_set else None
    resume_eval_dir = None
    resume_existing_count = 0
    if args.resume_eval_dir:
        if eval_records is None:
            raise ValueError('--resume-eval-dir requires --eval-set.')
        resume_eval_dir = Path(args.resume_eval_dir).expanduser().resolve()
        raw_dir = resume_eval_dir / 'raw'
        if not raw_dir.exists():
            raise FileNotFoundError(f'Resume raw directory does not exist: {raw_dir}')
        existing_eval_ids = set()
        for pickle_path in raw_dir.glob('*.pkl'):
            try:
                with open(pickle_path, 'rb') as handle:
                    saved = pickle.load(handle)
                eval_id = saved.get('cma_info', {}).get('eval_id')
                if eval_id:
                    existing_eval_ids.add(str(eval_id))
            except Exception as exc:
                print(f'[Resume] Ignoring unreadable PKL {pickle_path.name}: {exc}')
        resume_existing_count = len(existing_eval_ids)
        eval_records = [
            record for record in eval_records
            if str(record['eval_id']) not in existing_eval_ids
        ]
        print(
            f'[Resume] dir={resume_eval_dir}, existing={resume_existing_count}, '
            f'remaining={len(eval_records)}')
        if not eval_records:
            print('ALL EVALS ALREADY COMPLETE')
            raise SystemExit(0)
    if eval_records is not None and not eval_records:
        raise ValueError('No evaluation records remain after applying --target-limb-filter.')
    if eval_records is not None:
        print(
            f"Fixed eval-set mode: {len(eval_records)} rollouts, limbs={args.target_limb_filter or 'all'}, "
            f"entropy_weight={args.entropy_weight:g}, entropy_grid_size={args.entropy_grid_size:g}")
    else:
        filenames = list(Path(seed_path).glob('*.pkl'))
        filenames_iterated = iter(filenames)
        if not filenames:
            raise RuntimeError(f'No legacy input PKLs found in {seed_path}. Use --eval-set for fixed-seed evaluation.')

    if not args.eval_multiple_models:
        loop_data = [{
            'model': args.model_path,
            'graph_config':args.graph_config,
            'env_var':args.env_var,
            'max_fevals':args.max_fevals
        }]



    env_name = "RobeReversible-v1"

    target_limb_code = None

    recover_string = 'Uncover_Evals'
    if recover:
        recover_string = 'Recover_Evals'

    test_string = 'Train'
    if recover:
        test_string = ''

    for i in range(len(loop_data)):
        data = loop_data[i]
        checkpoint= osp.join(trained_models_dir, data['model'])
        env_var = data['env_var']
        env_variations = all_env_vars[env_var]
        graph_config = resolve_runtime_graph_config(
            all_graph_configs[data['graph_config']],
            checkpoint,
            voxel_size_arg=args.voxel_size,
            edge_threshold_arg=args.edge_threshold)
        max_fevals = data['max_fevals']
        print(
            f"[Graph Runtime] voxel_size={format_voxel_tag(graph_config['voxel_size'])}, "
            f"edge_threshold={graph_config['edge_threshold']}, edge_mode={graph_config.get('edge_mode', 'radius')}")

        num_rollouts = len(eval_records) if eval_records is not None else args.num_rollouts
        entropy_tag = f'_OverlapEntropy_w{args.entropy_weight:g}_g{args.entropy_grid_size:g}'
        region_tag = get_eval_region_tag(eval_records)
        feasible_tag = '_FeasibleOnly' if args.feasible_only_best else ''
        voxel_tag = f'_Vox{format_voxel_tag(graph_config["voxel_size"])}'
        feval_tag = f'_Fmax{max_fevals}'
        action_bounds_tag = get_action_bounds_tag(
            args.action_bounds_mode,
            args.train_percentile_low,
            args.train_percentile_high,
            train_action_pick_only)
        limb_tag = ''
        if eval_records is not None:
            limb_tag = f'_TL{(args.target_limb_filter or "all").replace(",", "-")}_FixedEval'
        lowering_tag = f'_{args.lowering_tag}' if args.lowering_tag else ''
        place_only_tag = ''
        if resume_eval_dir is not None:
            data_dir = str(resume_eval_dir)
        else:
            data_dir = osp.join(
                checkpoint,
                f'cma_evaluations/TL_All_{recover_string}_{num_rollouts}_states{limb_tag}{lowering_tag}{place_only_tag}{entropy_tag}{region_tag}{feasible_tag}{voxel_tag}{feval_tag}{action_bounds_tag}_{int(time.time()*1000)}')
        Path(data_dir).mkdir(parents=True, exist_ok=True)

        device = 'cpu'
        gnn_manager = GNN_Manager(device)
        checkpoint_path = osp.join(checkpoint, 'checkpoints', args.checkpoint_name)
        if osp.exists(checkpoint_path):
            gnn_manager.load_model_from_checkpoint(checkpoint)
            state = torch.load(checkpoint_path, map_location='cpu')
            gnn_manager.model.load_state_dict(state['model'])
            print(f'Loaded checkpoint override: {args.checkpoint_name}')
        else:
            gnn_manager.load_model_from_checkpoint(checkpoint)
        gnn_manager.model.to(torch.device('cpu'))
        gnn_manager.model.share_memory()
        gnn_manager.model.eval()

        counter = 0
        all_results = []

        if eval_records is not None:
            num_processes = max(1, min(args.num_processes, len(eval_records)))
            iterations = math.ceil(len(eval_records) / num_processes)
            for iter in tqdm(range(iterations)):
                idx_offset = iter * num_processes
                trials = min(num_processes, len(eval_records) - idx_offset)
                cma_reward, sim_reward, pred_sim_reward_error = evaluate_dyn_model(
                    env_name=env_name,
                    target_limb_code=target_limb_code,
                    trials=trials,
                    model=gnn_manager.model,
                    iter_data_dir=data_dir,
                    device=device,
                    num_processes=trials,
                    graph_config=graph_config,
                    env_variations=env_variations,
                    max_fevals=max_fevals,
                    entropy_weight=args.entropy_weight,
                    entropy_grid_size=args.entropy_grid_size,
                    eval_records=eval_records,
                    idx_offset=idx_offset,
                    post_release_steps=args.post_release_steps,
                    feasible_only_best=args.feasible_only_best,
                    action_bounds_mode=args.action_bounds_mode,
                    training_action_bounds=training_action_bounds,
                    train_action_pick_only=train_action_pick_only,
                    lower_before_release_uncover=args.lower_before_release,
                    cma_seed_base=args.cma_seed_base,
                    uncover_pick_policy=args.uncover_pick_policy,
                    optimize_place_only=False,
                    trial_timeout=args.trial_timeout,
                    max_retries=args.max_retries)
        else:
            num_processes = multiprocessing.cpu_count() - 1
            num_processes = trials = 100 if args.num_rollouts >= num_processes else args.num_rollouts
            iterations = round(args.num_rollouts/num_processes)

            for iter in tqdm(range(iterations)):
                cma_reward, sim_reward, pred_sim_reward_error = evaluate_dyn_model(
                    env_name=env_name,
                    target_limb_code=target_limb_code,
                    trials=trials,
                    model=gnn_manager.model,
                    iter_data_dir=data_dir,
                    device=device,
                    num_processes=num_processes,
                    graph_config=graph_config,
                    env_variations=env_variations,
                    max_fevals=max_fevals,
                    entropy_weight=args.entropy_weight,
                    entropy_grid_size=args.entropy_grid_size,
                    post_release_steps=args.post_release_steps,
                    feasible_only_best=args.feasible_only_best,
                    action_bounds_mode=args.action_bounds_mode,
                    training_action_bounds=training_action_bounds,
                    train_action_pick_only=train_action_pick_only,
                    lower_before_release_uncover=args.lower_before_release,
                    cma_seed_base=args.cma_seed_base,
                    uncover_pick_policy=args.uncover_pick_policy,
                    optimize_place_only=False,
                    trial_timeout=args.trial_timeout,
                    max_retries=args.max_retries)

    print("ALL EVALS COMPLETE")
