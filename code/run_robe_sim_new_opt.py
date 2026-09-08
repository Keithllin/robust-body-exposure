#%%
import math
import sys
import time
import argparse
import json
import pickle

from torch import multiprocessing

import os.path as osp
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / 'assistive-gym-fem'))
sys.path.insert(0, str(_REPO_ROOT / 'code'))

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
from gym.utils import seeding
from tqdm import tqdm
import glob

try:
    import gradient_free_optimizers as gfo
except ModuleNotFoundError:  # CMA mode does not require this optional package.
    gfo = None
from recover_heuristics import (
    compute_field_guided_recover_action_from_states,
    compute_line_stacking_recover_action_from_states,
)


#%%
recover = True
test = False
model_path_uncover = str(
    _REPO_ROOT
    / 'trained_models/FINAL_MODELS/Recover/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Uncover_10000_states_New_Grasp_16000_epochs=250_batch=50_workers=4_1705905825'
)

eval_dir_name = 'cma_evaluations'
search_dir = osp.join(model_path_uncover, eval_dir_name)

eval_conditions = ['TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Uncover_Evals_Train_500_states']
data_path = osp.join(model_path_uncover, eval_dir_name, eval_conditions[0], 'raw/')

filenames = list(Path(data_path).glob('*.pkl'))

x0 = []
target_limb_list = [2, 4, 5, 8, 10, 11, 12, 13, 14, 15]

#* parameters for the graph representation of the cloth fed as inpurecoveringt to the model
all_graph_configs = {
        # Default 2D recover graph: radius connectivity, matching radius-trained models.
        '2D':{
            'filt_drape': False,
            'rot_drape': True,
            'use_3D': False,
            'use_disp': True,
            'edge_mode': 'radius',
            'edge_threshold': 0.04,
            'voxel_size': float('nan'),
        },
        # Voxelized radius graph used by the voxelxyz05 recover checkpoints.
        '2D_voxelxyz05':{
            'filt_drape': False,
            'rot_drape': True,
            'use_3D': False,
            'use_disp': True,
            'edge_mode': 'radius',
            'edge_threshold': 0.06,
            'voxel_size': 0.05,
        },
        # Explicit mesh variant for checkpoints trained with GT blanket edges.
        '2D_mesh':{
            'filt_drape': False,
            'rot_drape': True,
            'use_3D': False,
            'use_disp': True,
            'edge_mode': 'mesh',
            'edge_threshold': 0.04,
            'voxel_size': float('nan'),
        },
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

#%%
def para_to_action(para):
    action = np.array([para['x_i'], para['y_i'], para['x_f'], para['y_f']])
    return action


def world_to_policy_action(action_world, scale=[0.44, 1.05]):
    """Inverse of scale_action() for 4D actions [x_i, y_i, x_f, y_f]."""
    action_world = np.asarray(action_world, dtype=np.float32)
    scale_vec = np.array(scale * 2, dtype=np.float32)
    action_policy = action_world / scale_vec
    action_policy = np.clip(action_policy, -1.0, 1.0)
    return action_policy


def _rotation_matrix(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.asarray([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
    ], dtype=np.float32)


def _undo_draping_rotation(graph_state, raw_reference, enabled):
    """Convert an Uncover graph-space prediction back to raw cloth space."""
    state = np.asarray(graph_state, dtype=np.float32).copy()
    reference = np.asarray(raw_reference, dtype=np.float32)
    if not enabled or len(state) != len(reference):
        return state

    right_forward = _rotation_matrix([0.0, 1.0, 0.0], -np.pi / 2.0)
    left_forward = _rotation_matrix([0.0, 1.0, 0.0], np.pi / 2.0)
    right_inverse = right_forward.T
    left_inverse = left_forward.T
    right_center = np.asarray([0.44, 0.0, 0.58], dtype=np.float32)
    left_center = np.asarray([-0.44, 0.0, 0.58], dtype=np.float32)

    movable = reference[:, 2] < 0.575
    right_mask = movable & (reference[:, 0] > 0.0)
    left_mask = movable & (reference[:, 0] < 0.0)
    state[right_mask] = (
        (state[right_mask] - right_center) @ right_inverse.T + right_center
    )
    state[left_mask] = (
        (state[left_mask] - left_center) @ left_inverse.T + left_center
    )
    return state


def _apply_draping_rotation(raw_state, enabled):
    """Apply the same draping transform used by ``Runtime_Graph``."""
    state = np.asarray(raw_state, dtype=np.float32).copy()
    if not enabled:
        return state

    right_rotation = _rotation_matrix([0.0, 1.0, 0.0], -np.pi / 2.0)
    left_rotation = _rotation_matrix([0.0, 1.0, 0.0], np.pi / 2.0)
    right_center = np.asarray([0.44, 0.0, 0.58], dtype=np.float32)
    left_center = np.asarray([-0.44, 0.0, 0.58], dtype=np.float32)

    movable = state[:, 2] < 0.575
    right_mask = movable & (state[:, 0] > 0.0)
    left_mask = movable & (state[:, 0] < 0.0)
    state[right_mask] = (
        (state[right_mask] - right_center) @ right_rotation.T + right_center
    )
    state[left_mask] = (
        (state[left_mask] - left_center) @ left_rotation.T + left_center
    )
    return state


def _prediction_intermediate_from_raw(raw_data, raw_initial_state, graph_config):
    """Load the fixed Uncover prediction saved in a W200 evaluation PKL.

    ``run_robe_sim.py`` stores the 2-D prediction in graph coordinates under
    ``cma_info['best_pred']``.  Recover training used the corresponding raw
    3-D state, so restore z from the raw initial cloth and undo the shared
    draping rotation before constructing the Recover graph.
    """
    cma_info = raw_data.get('cma_info', {})
    if not isinstance(cma_info, dict) or cma_info.get('best_pred') is None:
        raise ValueError(
            'recover-state-source=uncover_prediction requires '
            "cma_info['best_pred'] in the fixed Uncover PKL"
        )

    reference = np.asarray(raw_initial_state, dtype=np.float32)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError(
            'Expected raw initial cloth with shape (N, 3), got '
            f'{reference.shape}'
        )
    use_rotation = bool(graph_config.get('rot_drape', False))
    graph_reference = _apply_draping_rotation(reference, enabled=use_rotation)
    prediction = np.asarray(cma_info['best_pred'], dtype=np.float32)
    if prediction.ndim != 2 or prediction.shape[0] != reference.shape[0]:
        raise ValueError(
            'Uncover best_pred shape must match raw initial cloth: '
            f'pred={prediction.shape}, reference={reference.shape}'
        )
    if prediction.shape[1] == 2:
        # best_pred is in graph coordinates.  Its missing z coordinate must
        # come from the graph-space initial cloth, not the unrotated raw
        # state; otherwise inverse draping rotates an inconsistent 3-D point.
        prediction_graph_3d = graph_reference.copy()
        prediction_graph_3d[:, :2] = prediction
    elif prediction.shape[1] == 3:
        prediction_graph_3d = prediction.copy()
    else:
        raise ValueError(
            'Uncover best_pred must have 2 or 3 columns, got '
            f'{prediction.shape}'
        )
    if not np.all(np.isfinite(prediction_graph_3d)):
        raise ValueError('Uncover best_pred contains non-finite values')
    return _undo_draping_rotation(
        prediction_graph_3d,
        reference,
        enabled=use_rotation,
    )


def build_warm_start_candidates(uncover_action, input_cloth, all_body_points, strategy, target_limb_code, cloth_initial_positions=None):
    """
    Build warm-start candidates in policy space for gradient_free_optimizers.
    strategy: none | reverse | field | line | hybrid
    """
    warm_starts = []

    # Baseline warm-start: reverse uncover action
    reverse_action = np.array([*uncover_action[2:4], *uncover_action[0:2]], dtype=np.float32)
    reverse_dict = {'x_i': float(reverse_action[0]), 'y_i': float(reverse_action[1]), 'x_f': float(reverse_action[2]), 'y_f': float(reverse_action[3])}

    if strategy in ['reverse', 'hybrid']:
        warm_starts.append(reverse_dict)

    if strategy in ['field', 'hybrid']:
        try:
            field_action, action_world, _, _, _, _ = compute_field_guided_recover_action_from_states(
                input_cloth=input_cloth,
                all_body_points=all_body_points,
            )
            if field_action is not None:
                field_action = world_to_policy_action(action_world)
                field_dict = {
                    'x_i': float(field_action[0]),
                    'y_i': float(field_action[1]),
                    'x_f': float(field_action[2]),
                    'y_f': float(field_action[3]),
                }
                warm_starts.append(field_dict)
        except Exception as e:
            print(f"[WarmStart] Field-guided warm start failed, fallback used: {e}")

    if strategy in ['line', 'hybrid']:
        try:
            line_action, action_world, _, _, _, _ = compute_line_stacking_recover_action_from_states(
                cloth_intermediate_positions=input_cloth,
                uncover_action_policy=uncover_action,
                target_limb_code=target_limb_code,
                cloth_initial_positions=cloth_initial_positions,
                line_width=0.06,
            )
            if line_action is not None:
                line_action = world_to_policy_action(action_world)
                line_dict = {
                    'x_i': float(line_action[0]),
                    'y_i': float(line_action[1]),
                    'x_f': float(line_action[2]),
                    'y_f': float(line_action[3]),
                }
                warm_starts.append(line_dict)
        except Exception as e:
            print(f"[WarmStart] Line-guided warm start failed, fallback used: {e}")

    # If strategy is none, keep empty warm-start list
    if strategy == 'none':
        return []

    # Otherwise guarantee at least one candidate
    if len(warm_starts) == 0:
        warm_starts.append(reverse_dict)

    return warm_starts

def grasp_on_cloth(action, cloth_initial_raw):
    dist, is_on_cloth = check_grasp_on_cloth(action, np.array(cloth_initial_raw))
    return is_on_cloth

def cost_function(action, all_body_points, first_cloth, cloth_initial_raw, graph, model, device, use_disp, use_3D):
    action = scale_action(action)
    cloth_initial = graph.initial_blanket_state
    is_on_cloth = grasp_on_cloth(action, cloth_initial_raw)

    if is_on_cloth:
        data = graph.build_graph(action)
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
        cost, covered_status = get_cost(action, all_body_points, first_cloth, cloth_initial_2D, pred_2D)
    else:
        cost, covered_status = get_cost(action, all_body_points, first_cloth, cloth_initial, pred)

    return [cost, pred, covered_status, is_on_cloth]

def get_cost(action, all_body_points, first_cloth, cloth_initial_2D, cloth_final_2D):
    if recover:
        reward, covered_status = get_recovering_reward(action, all_body_points, first_cloth, cloth_initial_2D, cloth_final_2D)
    else:
        reward, covered_status = get_uncovering_reward(action, all_body_points, cloth_initial_2D, cloth_final_2D)

    cost = -reward

    return cost, covered_status

def counter_callback(output):
    global counter
    counter += 1
    print(f"{counter} - Trial Completed: Best Pred Reward:{output[1]:.2f}, Sim Reward: {output[3]:.2f}, Opt Time: {output[4]/60:.2f}min, Sim Time: {output[7]/60:.2f}min, TL: {output[5]}, GoC: {output[6]}")

def find(seed):
    for eval_condition in eval_conditions:
        path = Path(model_path_uncover +'/' + eval_dir_name + '/' + eval_condition + '/raw/')
        filenames = path.glob('*.pkl')
        for f in filenames:
            if str(seed) in f.name:
                return f
    #%%
def load_eval_set(eval_set_path):
    if not eval_set_path:
        return None
    path = Path(eval_set_path).expanduser().resolve()
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        records = payload.get('records', [])
    else:
        records = payload
    normalized = []
    for idx, record in enumerate(records):
        source_pkl = record.get('source_uncover_pkl') or record.get('pkl') or record.get('source_pkl')
        if not source_pkl:
            raise ValueError(f'eval-set record {idx} is missing source_uncover_pkl')
        normalized.append({
            'eval_id': record.get('eval_id', f'eval_{idx:04d}'),
            'seed': int(record['seed']),
            'target_limb_code': int(record['target_limb_code']),
            'source_uncover_pkl': str(Path(source_pkl).expanduser().resolve()),
            'sim_uncover_f1': float(record.get('sim_uncover_f1', record.get('uncover_f1', np.nan))),
        })
    return normalized


def optimizer(env_name, idx, eval_record, model, device, target_limb_code, iter_data_dir, graph_config, env_var, max_fevals, warm_start_strategy, search_method, feasible_only_best, recover_state_source='gt_intermediate'):
    use_disp = graph_config['use_disp']
    filter_draping = graph_config['filt_drape']
    rot_draping = graph_config['rot_drape']
    use_3D = graph_config['use_3D']
    edge_mode = graph_config.get('edge_mode', 'radius')
    graph_voxel_size = graph_config.get('voxel_size', np.nan if edge_mode == 'mesh' else 0.05)

    coop = 'Human' in env_name

    if target_limb_code is None:
        target_limb_code = random.sample(target_limb_list, 1)[0]

    seed = seeding.create_seed()

    # choose uncovered state from test set to recover from
    if recover:
        if eval_record is not None:
            random_file = Path(eval_record['source_uncover_pkl']).expanduser().resolve()
            seed = int(eval_record['seed'])
            target_limb_code = int(eval_record['target_limb_code'])
        else:
            random_file = np.random.choice(list((Path(model_path_uncover)/eval_dir_name/eval_conditions[0]/'raw').iterdir()))
            seed = int(random_file.name.split('_')[2])
            target_limb_code = int(random_file.name.split('_')[0].replace('tl', ''))
        with open(random_file, 'rb') as seed_path:
            raw_data = pickle.load(seed_path)

    env = make_env(env_name, coop=coop, seed=seed)

    env.set_env_variations(
        collect_data = False,
        blanket_pose_var = env_var['blanket_var'],
        high_pose_var = env_var['high_pose_var'],
        body_shape_var = env_var['body_shape_var'])

    env.set_singulate(True)
    env.set_target_limb_code(target_limb_code)
    env.set_recover(recover)
    env.set_seed_val(seed)

    done = False
    human_pose = env.reset()
    env.set_release_sim_steps(post_release_steps=3)
    human_pose = np.reshape(human_pose, (-1,2))
    uncover_sim_time = 0.0

    if recover:
        uncover_action = raw_data['uncover_action']

        # IMPORTANT: run uncover first and optimize on the exact simulated intermediate cloth.
        # The simulator always executes the action so the final Recover action
        # is evaluated on the physical GT state.  The graph input can be either
        # that GT state or the fixed Uncover GNN prediction, selected by the
        # explicit recover_state_source argument.
        t_uncover_start = time.time()
        cloth_initial_sim, cloth_intermediate_sim, execute_uncover_action = env.uncover_step(uncover_action)
        uncover_sim_time = time.time() - t_uncover_start

        cloth_initial_dc = np.delete(np.array(cloth_initial_sim[1]), 2, axis=1)
        gt_intermediate_cloth = np.asarray(cloth_intermediate_sim[1], dtype=np.float32)
        input_cloth = gt_intermediate_cloth
        if recover_state_source == 'uncover_prediction':
            input_cloth = _prediction_intermediate_from_raw(
                raw_data=raw_data,
                raw_initial_state=np.asarray(cloth_initial_sim[1], dtype=np.float32),
                graph_config=graph_config,
            )
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
        edge_threshold=graph_config.get('edge_threshold', 0.06),
        action_to_all=True,
        cloth_initial=input_cloth,
        filter_draping=filter_draping,
        rot_draping=rot_draping,
        use_3D=use_3D,
        edge_mode=edge_mode)

    # This is the exact cloth state used as graph nodes by the Recover GNN.
    # Runtime_Graph has already applied draping rotation, optional filtering,
    # and voxel representative selection at this point.  Keep the original
    # full-resolution input separately because it is still useful for
    # diagnostics and simulator-side bookkeeping.
    recover_graph_node_state = np.asarray(
        graph.initial_blanket_state,
        dtype=np.float32,
    ).copy()

    # Build warm-start candidates (reverse / field / hybrid)
    warm_starts = []
    if recover:
        cloth_initial_positions = np.asarray(cloth_initial_sim[1], dtype=np.float32)
        warm_starts = build_warm_start_candidates(
            uncover_action=uncover_action,
            input_cloth=input_cloth,
            all_body_points=all_body_points,
            strategy=warm_start_strategy,
            target_limb_code=target_limb_code,
            cloth_initial_positions=cloth_initial_positions,
        )

    pass_through = {'pass_through': [all_body_points, cloth_initial_dc, input_cloth, graph, model, device, use_disp, use_3D]}
    feasible_tracker = {
        'found': False,
        'best_reward': -np.inf,
        'best_para': None,
    }

    def cost_function_helper(para):
        all_body_points, cloth_initial_dc, input_cloth, graph, model, device, use_disp, use_3D = pass_through['pass_through']
        action = para_to_action(para)
        cost, pred, covered_status, is_on_cloth = cost_function(action, all_body_points, cloth_initial_dc, input_cloth, graph, model, device, use_disp, use_3D)
        reward = -cost

        if is_on_cloth and reward > feasible_tracker['best_reward']:
            feasible_tracker['found'] = True
            feasible_tracker['best_reward'] = float(reward)
            feasible_tracker['best_para'] = {
                'x_i': float(para['x_i']),
                'y_i': float(para['y_i']),
                'x_f': float(para['x_f']),
                'y_f': float(para['y_f']),
            }

        return reward

    step_size = 0.01
    search_space = {
        "x_i": np.arange(-1, 1+step_size, step_size),
        "y_i": np.arange(-1, 1+step_size, step_size),
        "x_f": np.arange(-1, 1+step_size, step_size),
        "y_f": np.arange(-1, 1+step_size, step_size),
        }

    t0 = time.time()
    if search_method == 'random':
        if gfo is None:
            raise RuntimeError(
                "RandomSearch requires the optional "
                "'gradient-free-optimizers' package."
            )
        opt = gfo.RandomSearchOptimizer(search_space, initialize={"grid": 4, "random": 10, "vertices": 4, "warm_start": warm_starts})
        opt.search(
            cost_function_helper,
            n_iter=max_fevals,
            verbosity=False)
        best_para = opt.best_para
        if feasible_only_best and feasible_tracker['found']:
            best_para = feasible_tracker['best_para']
        best_action = para_to_action(best_para)
        best_fevals = max_fevals
        best_iterations = max_fevals
    elif search_method == 'cma':
        if len(warm_starts) > 0:
            x0_dict = warm_starts[0]
            x0_vec = np.array([x0_dict['x_i'], x0_dict['y_i'], x0_dict['x_f'], x0_dict['y_f']], dtype=np.float32)
        else:
            x0_vec = np.zeros(4, dtype=np.float32)

        opts = cma.CMAOptions({
            'verb_disp': 0,
            'popsize': pop_size,
            'maxfevals': max_fevals,
            'tolfun': 1e-11,
            'tolflatfitness': 20,
            'tolfunhist': 1e-20,
            'bounds': [[-1]*4, [1]*4],
        })

        es = cma.CMAEvolutionStrategy(x0_vec.tolist(), 0.1, opts)

        while not es.stop() and es.countevals < max_fevals:
            xs = es.ask()
            costs = []
            for x in xs:
                para = {'x_i': float(x[0]), 'y_i': float(x[1]), 'x_f': float(x[2]), 'y_f': float(x[3])}
                reward = cost_function_helper(para)
                costs.append(-reward)  # CMA-ES minimizes objective
            es.tell(xs, costs)

        if feasible_only_best and feasible_tracker['found']:
            best_para = feasible_tracker['best_para']
            best_x = np.asarray([best_para['x_i'], best_para['y_i'], best_para['x_f'], best_para['y_f']], dtype=np.float32)
        else:
            best_x = np.asarray(es.result.xbest, dtype=np.float32)
        best_action = np.clip(best_x, -1.0, 1.0)
        best_fevals = int(es.result.evaluations)
        best_iterations = int(es.countiter)
    else:
        raise ValueError(f"Unsupported search method: {search_method}")

    t1 = time.time()
    optimizer_time = t1-t0  # Time for optimization only

    best_cost, best_pred, best_covered_status, best_is_on_cloth = cost_function(best_action, all_body_points, cloth_initial_dc, input_cloth, graph, model, device, use_disp, use_3D)
    best_reward = -best_cost

    # Start timing simulation execution
    t_sim_start = time.time()
    if recover:
        cloth_final_sim, execute_recover_action = env.recover_step(best_action) # if recovering recover action is predicted by the model
    else:
        cloth_initial_sim, cloth_intermediate_sim, execute_uncover_action = env.uncover_step(best_action)
        cloth_final_sim, execute_recover_action = env.recover_step([]) # if not recovering, don't need to provide an action

    observation, uncover_reward, recover_reward, done, info = env.get_info()
    t_sim_end = time.time()
    sim_time = (t_sim_end - t_sim_start) + uncover_sim_time  # Keep uncover + recover simulation time

    sim_grasp_on_cloth = info.get('grasp_on_cloth_recover', None) if recover else info.get('grasp_on_cloth_uncover', None)

    sim_info = {'observation':observation, 'uncover reward':uncover_reward, 'recover_reward':recover_reward, 'done':done, 'info':info}
    cma_info = {'best_cost':best_cost, 'best_reward':best_reward, 'best_pred':best_pred, 'optimizer_time':optimizer_time, 'sim_time':sim_time,
                'best_covered_status':best_covered_status, 'best_fevals':best_fevals, 'best_iterations':best_iterations,
                'search_method':search_method, 'model_grasp_on_cloth':bool(best_is_on_cloth), 'sim_grasp_on_cloth':sim_grasp_on_cloth,
                'feasible_only_best': bool(feasible_only_best), 'feasible_found': bool(feasible_tracker['found']),
                'recover_state_source': str(recover_state_source),
                # Keep the exact state used to build the Recover graph so
                # downstream figures can show the prediction graph rather
                # than silently substituting the simulator GT intermediate.
                'recover_graph_input_state': (
                    np.asarray(input_cloth, dtype=np.float32).copy()
                    if recover else None
                ),
                # Unlike recover_graph_input_state, this is post-draping and
                # post-voxelization (when enabled), i.e. the actual node
                # coordinate array consumed by the GNN and aligned with
                # cma_info['best_pred'].
                'recover_graph_node_state': (
                    recover_graph_node_state if recover else None
                ),
                'recover_graph_node_count': (
                    int(len(recover_graph_node_state)) if recover else None
                ),
                'recover_graph_voxel_size': (
                    float(graph_voxel_size)
                    if recover and np.isfinite(graph_voxel_size)
                    else None
                ),
                'recover_graph_edge_mode': str(edge_mode) if recover else None,
                'recover_graph_rot_drape': bool(rot_draping) if recover else None,
                'recover_graph_filter_drape': bool(filter_draping) if recover else None,
            }

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

    return seed, best_reward, uncover_reward, recover_reward, optimizer_time, target_limb_code, best_is_on_cloth, sim_time


def _saved_result_path(iter_data_dir, idx, eval_record):
    """Return a valid saved result for one fixed-eval case, if one exists."""
    if eval_record is None:
        return None
    target_limb_code = int(eval_record['target_limb_code'])
    seed = int(eval_record['seed'])
    raw_dir = Path(iter_data_dir) / 'raw'
    pattern = f'tl{target_limb_code}_c{idx}_{seed}_pid*.pkl'
    candidates = sorted(raw_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    for path in reversed(candidates):
        try:
            with open(path, 'rb') as handle:
                payload = pickle.load(handle)
            if isinstance(payload, dict) and isinstance(payload.get('cma_info'), dict):
                return path
        except Exception:
            # Atomic writes should prevent this; ignore any stale partial file.
            continue
    return None


def _safe_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_saved_result(path, eval_record):
    """Reconstruct optimizer return fields from an already saved PKL."""
    with open(path, 'rb') as handle:
        payload = pickle.load(handle)
    cma_info = payload.get('cma_info', {})
    sim_info = payload.get('sim_info', {})
    seed = int(eval_record['seed'])
    target_limb_code = int(eval_record['target_limb_code'])
    best_reward = _safe_float(cma_info.get('best_reward'))
    uncover_reward = _safe_float(sim_info.get('uncover reward'))
    recover_reward = _safe_float(sim_info.get('recover_reward'))
    optimizer_time = _safe_float(cma_info.get('optimizer_time'))
    sim_time = _safe_float(cma_info.get('sim_time'))
    best_is_on_cloth = bool(cma_info.get('model_grasp_on_cloth', False))
    return (
        seed,
        best_reward,
        uncover_reward,
        recover_reward,
        optimizer_time,
        target_limb_code,
        best_is_on_cloth,
        sim_time,
    )


def _write_watchdog_report(iter_data_dir, failures):
    if not failures:
        return
    report_path = Path(iter_data_dir) / 'watchdog_failures.json'
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(failures, handle, indent=2)
        handle.write('\n')
    print(f'[Watchdog] Failure report: {report_path}')


def evaluate_dyn_model(
    env_name,
    target_limb_code,
    trials,
    idx_offset,
    eval_records,
    model,
    iter_data_dir,
    device,
    num_processes,
    graph_config,
    env_variations,
    max_fevals,
    warm_start_strategy,
    search_method,
    feasible_only_best,
    recover_state_source='gt_intermediate',
    trial_timeout=900.0,
    max_retries=2,
):
    """Run one batch with a process watchdog and resumable retries.

    The old implementation called ``result.get()`` without a timeout.  A
    simulator worker that blocked inside an environment step therefore held
    the whole 400-case run forever.  Fixed evals are retried by case: files
    already written before a worker is terminated are retained and skipped.
    """
    if eval_records is None:
        # Keep legacy random-rollout mode compatible, but never wait forever.
        result_objs = []
        pool = multiprocessing.Pool(processes=num_processes)
        normal_completion = False
        try:
            for local_idx in range(trials):
                idx = idx_offset + local_idx
                result_objs.append(pool.apply_async(
                    optimizer,
                    args=(env_name, idx, None, model, device, target_limb_code,
                          iter_data_dir, graph_config, env_variations, max_fevals,
                          warm_start_strategy, search_method, feasible_only_best,
                          recover_state_source),
                    callback=counter_callback,
                ))
            results = [result.get(timeout=trial_timeout) for result in result_objs]
            normal_completion = True
        finally:
            if normal_completion:
                pool.close()
            else:
                pool.terminate()
            pool.join()
        results_array = np.array(results)
        pred_sim_reward_error = abs(results_array[:, 3] - results_array[:, 1])
        return (
            list(results_array[:, 1]),
            list(results_array[:, 3]),
            list(pred_sim_reward_error),
            list(results_array[:, 4]),
            list(results_array[:, 7]),
        )

    all_jobs = []
    pending_jobs = []
    for local_idx in range(trials):
        idx = idx_offset + local_idx
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
            f'timeout={trial_timeout:.0f}s'
        )
        pool = None
        normal_completion = False
        timed_out = False
        submitted = []
        try:
            pool = multiprocessing.Pool(processes=worker_count)
            for idx, eval_record in pending_jobs:
                result = pool.apply_async(
                    optimizer,
                    args=(env_name, idx, eval_record, model, device, target_limb_code,
                          iter_data_dir, graph_config, env_variations, max_fevals,
                          warm_start_strategy, search_method, feasible_only_best,
                          recover_state_source),
                )
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

        # A lower worker count is safer on a retry when the first attempt was
        # caused by simulator/CPU contention.
        worker_count = max(1, min(worker_count // 2, len(pending_jobs)))
        print(f'[Watchdog] retrying unresolved cases with workers={worker_count}')

    _write_watchdog_report(iter_data_dir, failures)
    result_rows = []
    for idx, eval_record in all_jobs:
        saved = _saved_result_path(iter_data_dir, idx, eval_record)
        if saved is None:
            raise RuntimeError(f'Missing valid result after watchdog: idx={idx}, eval_id={eval_record.get("eval_id")}')
        result_rows.append(_load_saved_result(saved, eval_record))

    results_array = np.array(result_rows)
    pred_sim_reward_error = abs(results_array[:, 3] - results_array[:, 1])
    return (
        list(results_array[:, 1]),
        list(results_array[:, 3]),
        list(pred_sim_reward_error),
        list(results_array[:, 4]),
        list(results_array[:, 7]),
    )


#%%

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval-multiple-models', type=bool, default=False)
    parser.add_argument('--model-path', type=str)
    parser.add_argument(
        '--models-dir',
        type=str,
        default=str(_REPO_ROOT / 'trained_models/FINAL_MODELS'),
        help='Root directory containing the model checkpoint directories.',
    )
    parser.add_argument('--graph-config', type=str)
    parser.add_argument('--env-var', type=str)
    parser.add_argument('--max-fevals', type=int, default=300)
    parser.add_argument('--num-rollouts', type=int, default=500)
    parser.add_argument('--arg_seed', type=int, default=0)
    parser.add_argument('--warm-start-strategy', type=str, default='line', choices=['none', 'reverse', 'field', 'line', 'hybrid'])
    parser.add_argument('--search-method', type=str, default='cma', choices=['random', 'cma'])
    parser.add_argument('--feasible-only-best', action='store_true', help='Select best action only among candidates with is_on_cloth=True')
    parser.add_argument('--recover-state-source', type=str, default='gt_intermediate',
                        choices=['gt_intermediate', 'uncover_prediction'],
                        help='State used as Recover graph input. The prediction mode reads cma_info[best_pred] from each fixed Uncover PKL.')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--num-processes', type=int, default=4, help='Number of parallel worker processes. Lower this if the machine runs out of memory.')
    parser.add_argument('--eval-set', type=str, default='', help='Path to a JSON manifest of fixed decoupled evaluation cases.')
    parser.add_argument('--trial-timeout', type=float, default=900.0,
                        help='Watchdog timeout in seconds for one parallel batch. Fixed-eval jobs are retried after timeout.')
    parser.add_argument('--max-retries', type=int, default=2,
                        help='Maximum retries for unresolved fixed-eval cases after a worker timeout/exception.')
    args = parser.parse_args()

    if args.trial_timeout <= 0:
        raise ValueError('--trial-timeout must be positive')
    if args.max_retries < 0:
        raise ValueError('--max-retries must be non-negative')

    if not args.eval_multiple_models:
        loop_data = [{
            'model': args.model_path,
            'graph_config':args.graph_config,
            'env_var':args.env_var,
            'max_fevals':args.max_fevals
        }]

    env_name = "RobeReversible-v1"

    target_limb_code = None
    eval_records = load_eval_set(args.eval_set) if args.eval_set else None
    if eval_records is not None and len(eval_records) == 0:
        raise ValueError('eval-set is empty')

    recover_string = 'Uncover_Evals'
    if recover:
        recover_string = 'Recover_Evals'

    test_string = 'Test'
    if recover:
        test_string = ''


    for i in range(len(loop_data)):
        data = loop_data[i]
        checkpoint = osp.join(args.models_dir, data['model'])
        env_var = data['env_var']
        env_variations = all_env_vars[env_var]
        graph_config = all_graph_configs[data['graph_config']]
        max_fevals = data['max_fevals']

        method_tag = 'RandomSearch' if args.search_method == 'random' else 'CMAES'
        warm_tag = f"WS_{args.warm_start_strategy}"
        total_rollouts = len(eval_records) if eval_records is not None else args.num_rollouts
        default_dir_name = f'TL_{target_limb_list}_{recover_string}_{test_string}_{total_rollouts}_states_{method_tag}_{warm_tag}_Opt'
        data_dir = args.output_dir if args.output_dir is not None else osp.join(checkpoint, f'cma_evaluations/{default_dir_name}')
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        print(data_dir)
        device = 'cpu'
        gnn_manager = GNN_Manager(device)
        gnn_manager.load_model_from_checkpoint(checkpoint)
        gnn_manager.model.to(torch.device('cpu'))
        gnn_manager.model.share_memory()
        gnn_manager.model.eval()

        counter = 0
        all_results = []
        all_opt_times = []
        all_sim_times = []
        start_wall_clock = time.time()

        max_auto_processes = max(1, multiprocessing.cpu_count() - 1)
        num_processes = max(1, min(args.num_processes, max_auto_processes, total_rollouts))
        iterations = math.ceil(total_rollouts / num_processes)
        print(f"Parallel workers: {num_processes} (requested={args.num_processes}, cpu_limit={max_auto_processes}, rollouts={total_rollouts})")
        print(f"Watchdog: trial_timeout={args.trial_timeout:.0f}s, max_retries={args.max_retries}")
        if eval_records is not None:
            print(f"Using fixed eval-set: {args.eval_set} ({len(eval_records)} cases)")

        for iter in tqdm(range(iterations)):
            batch_rollouts = min(num_processes, total_rollouts - (iter * num_processes))
            cma_reward, sim_reward, pred_sim_reward_error, opt_times, sim_times = evaluate_dyn_model(
                env_name=env_name,
                target_limb_code = target_limb_code,
                trials = batch_rollouts,
                idx_offset = iter * num_processes,
                eval_records = eval_records,
                model = gnn_manager.model,
                iter_data_dir = data_dir,
                device = device,
                num_processes = num_processes,
                graph_config = graph_config,
                env_variations = env_variations,
                max_fevals=max_fevals,
                warm_start_strategy=args.warm_start_strategy,
                search_method=args.search_method,
                feasible_only_best=args.feasible_only_best,
                recover_state_source=args.recover_state_source,
                trial_timeout=args.trial_timeout,
                max_retries=args.max_retries)
            
            all_opt_times.extend(opt_times)
            all_sim_times.extend(sim_times)
            
        end_wall_clock = time.time()
        wall_clock_duration = end_wall_clock - start_wall_clock

        # Print summary statistics
        print("\n" + "="*80)
        print("ALL EVALS COMPLETE - TIME SUMMARY")
        print("="*80)
        print(f"Total Rollouts: {len(all_opt_times)}")
        print(f"Actual Wall-Clock Time: {wall_clock_duration/60:.2f} min ({wall_clock_duration/3600:.2f} hrs)")
        
        print(f"\nOptimizer Time (Cumulative):")
        print(f"  Total Task: {sum(all_opt_times)/60:.2f} min ({sum(all_opt_times)/3600:.2f} hrs)")
        print(f"  Average:    {np.mean(all_opt_times):.2f} sec")
        print(f"  Min/Max:    {np.min(all_opt_times):.2f}s / {np.max(all_opt_times):.2f}s")
        
        print(f"\nSimulation Time (Cumulative):")
        print(f"  Total Task: {sum(all_sim_times)/60:.2f} min ({sum(all_sim_times)/3600:.2f} hrs)")
        print(f"  Average:    {np.mean(all_sim_times):.2f} sec")
        print(f"  Min/Max:    {np.min(all_sim_times):.2f}s / {np.max(all_sim_times):.2f}s")
        
        print(f"\nExecution Efficiency:")
        print(f"  Parallel Speedup: {(sum(all_opt_times) + sum(all_sim_times)) / wall_clock_duration:.1f}x")
        print(f"  Optimizer vs Sim: {sum(all_opt_times)/sum(all_sim_times):.2f} ratio")
        print("="*80)
