#%%
import math
import sys
import time
import argparse

from torch import multiprocessing

sys.path.insert(0, './assistive-gym-fem')
import os.path as osp
from pathlib import Path

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

import gradient_free_optimizers as gfo


#%%
recover = True
test = False
model_path_uncover = '/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/FINAL_MODELS/standard_2D_not_subsampled_epochs=250_batch=100_workers=4_1687986938'

eval_dir_name = 'cma_evaluations'
search_dir = osp.join(model_path_uncover, eval_dir_name)

eval_conditions = ['TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Uncover_Evals_Test_1000_states']

x0 = []
target_limb_list = [2, 4, 5, 8, 10, 11, 12, 13, 14, 15]

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

#%%
def para_to_action(para):
    action = np.array([para['x_i'], para['y_i'], para['x_f'], para['y_f']])
    return action

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
    print(f"{counter} - Trial Completed: CMA-ES Best Reward:{output[1]:.2f}, Sim Reward: {output[3]:.2f}, CMA Time: {output[4]/60:.2f}, TL: {output[5]}, GoC: {output[6]}")

def find(seed):
    for eval_condition in eval_conditions:
        path = Path(model_path_uncover +'/' + eval_dir_name + '/' + eval_condition + '/raw/')
        filenames = path.glob('*.pkl')
        for f in filenames:
            if str(seed) in f.name:
                return f
    #%%
def optimizer(env_name, idx, model, device, target_limb_code, iter_data_dir, graph_config, env_var, max_fevals):
    use_disp = graph_config['use_disp']
    filter_draping = graph_config['filt_drape']
    rot_draping = graph_config['rot_drape']
    use_3D = graph_config['use_3D']

    coop = 'Human' in env_name

    if target_limb_code is None:
        target_limb_code = random.sample(target_limb_list, 1)[0]

    seed = seeding.create_seed()

    env = make_env(env_name, coop=coop, seed=seed)

    env.set_env_variations(
        collect_data = False,
        blanket_pose_var = env_var['blanket_var'],
        high_pose_var = env_var['high_pose_var'],
        body_shape_var = env_var['body_shape_var'])

    env.set_target_limb_code(target_limb_code)
    env.set_recover(recover)
    env.set_seed_val(seed)

    done = False
    human_pose = env.reset()
    human_pose = np.reshape(human_pose, (-1,2))

    env.set_target_limb_code(target_limb_code)
    pop_size = 8

    body_info = env.get_human_body_info()
    all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)

    # Uncovering Optimization
    uncover_input_cloth = env.get_cloth_state()

    uncover_graph = Runtime_Graph(
        root = iter_data_dir,
        description=f"iter_{iter}_processed",
        voxel_size=0.05,
        edge_threshold=0.06,
        action_to_all=True,
        cloth_initial=uncover_input_cloth,
        filter_draping=filter_draping,
        rot_draping=rot_draping,
        use_3D=use_3D)

    pass_through = {'pass_through': [all_body_points, uncover_input_cloth, uncover_input_cloth, uncover_model, device, use_disp, use_3D]}

    def cost_function_helper(para):
        all_body_points, cloth_initial_dc, input_cloth, graph, model, device, use_disp, use_3D = pass_through['pass_through']
        action = para_to_action(para)
        cost, pred, covered_status, is_on_cloth = cost_function(action, all_body_points, cloth_initial_dc, input_cloth, graph, model, device, use_disp, use_3D)

        return -cost

    if recover:
        x0 = [*uncover_action[2:4], *uncover_action[0:2]]

    x0 = {'x_i':x0[0], 'y_i':x0[1], 'x_f':x0[2], 'y_f':x0[3]}

    step_size = 0.01
    search_space = {
        "x_i": np.arange(-1, 1+step_size, step_size),
        "y_i": np.arange(-1, 1+step_size, step_size),
        "x_f": np.arange(-1, 1+step_size, step_size),
        "y_f": np.arange(-1, 1+step_size, step_size),
        }

    #Predict 8-dimensional vector for actions
    opt = gfo.RandomSearchOptimizer(search_space, initialize={"grid": 4, "random": 10, "vertices": 8, "warm_start": [x0]})
    t0 = time.time()
    opt.search(
        cost_function_helper,
        n_iter=500,
        verbosity=False)

    t1 = time.time()
    best_para = opt.best_para

    best_action = para_to_action(best_para)
    best_time = t1-t0
    best_fevals = None
    best_iterations = None

    best_cost, best_pred, best_covered_status, best_is_on_cloth = cost_function(best_action, all_body_points, cloth_initial_dc, input_cloth, graph, model, device, use_disp, use_3D)
    best_reward = -best_cost

    recover_input_cloth = env.get_cloth_state()

    recover_graph = Runtime_Graph(
        root = iter_data_dir,
        description=f"iter_{iter}_processed",
        voxel_size=0.05,
        edge_threshold=0.06,
        action_to_all=True,
        cloth_initial=uncover_input_cloth,
        filter_draping=filter_draping,
        rot_draping=rot_draping,
        use_3D=use_3D)
        
    if recover:
        cloth_initial_sim, cloth_intermediate_sim, execute_recover_action = env.uncover_step(uncover_action)
        cloth_final_sim, execute_recover_action = env.recover_step(best_action) # if recovering recover action is predicted by the model
    else:
        cloth_initial_sim, cloth_intermediate_sim, execute_uncover_action = env.uncover_step(best_action)
        cloth_final_sim, execute_recover_action = env.recover_step([]) # if not recovering, don't need to provide an action

    observation, uncover_reward, recover_reward, done, info = env.get_info()

    sim_info = {'observation':observation, 'uncover reward':uncover_reward, 'recover_reward':recover_reward, 'done':done, 'info':info}
    cma_info = {'best_cost':best_cost, 'best_reward':best_reward, 'best_pred':best_pred, 'best_time':best_time,
                'best_covered_status':best_covered_status, 'best_fevals':best_fevals, 'best_iterations':best_iterations}

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

    return seed, best_reward, uncover_reward, recover_reward, best_time, target_limb_code, best_is_on_cloth


def evaluate_dyn_model(env_name, target_limb_code, trials, model, iter_data_dir, device, num_processes, graph_config, env_variations, max_fevals):

    result_objs = []
    # ! Why doing trials/num_processes? equals 1
    for j in tqdm(range(math.ceil(trials/num_processes))):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(num_processes):
                idx = i+(j*num_processes)
                result = pool.apply_async(optimizer, args = (env_name, idx, model, device, target_limb_code, iter_data_dir, graph_config, env_variations, max_fevals), callback=counter_callback)
                result_objs.append(result)

            results = [result.get() for result in result_objs]
            all_results.append(results)

    results_array = np.array(results)
    pred_sim_reward_error = abs(results_array[:,2] - results_array[:,1])

    return list(results_array[:,1]), list(results_array[:,2]), list(pred_sim_reward_error)


#%%

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    trained_models_dir = './trained_models/FINAL_MODELS'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval-multiple-models', type=bool, default=False)
    parser.add_argument('--uncover_model-path', type=str)
    parser.add_argument('--recover_model-path', type=str)
    parser.add_argument('--graph-config', type=str)
    parser.add_argument('--env-var', type=str)
    parser.add_argument('--max-fevals', type=int, default=300)
    parser.add_argument('--num-rollouts', type=int, default=500)
    parser.add_argument('--arg_seed', type=int, default=0)
    args = parser.parse_args()

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

    test_string = 'Test'
    if recover:
        test_string = ''


    for i in range(len(loop_data)):
        data = loop_data[i]
        checkpoint= osp.join(trained_models_dir, data['model'])
        env_var = data['env_var']
        env_variations = all_env_vars[env_var]
        graph_config = all_graph_configs[data['graph_config']]
        max_fevals = data['max_fevals']

        data_dir = osp.join(checkpoint, f'cma_evaluations/TL_{target_limb_list}_{recover_string}_{test_string}_{args.num_rollouts}_states_RandomSearch_Opt')
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

        num_processes = multiprocessing.cpu_count() - 1

        num_processes = trials = 100 if args.num_rollouts >= num_processes else args.num_rollouts
        iterations = round(args.num_rollouts/num_processes)

        for iter in tqdm(range(iterations)):
            cma_reward, sim_reward, pred_sim_reward_error = evaluate_dyn_model(
                env_name=env_name,
                target_limb_code = target_limb_code,
                trials = trials,
                model = gnn_manager.model,
                iter_data_dir = data_dir,
                device = device,
                num_processes = num_processes,
                graph_config = graph_config,
                env_variations = env_variations,
                max_fevals=max_fevals)

    print("ALL EVALS COMPLETE")