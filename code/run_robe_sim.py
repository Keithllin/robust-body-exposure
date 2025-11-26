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

#%%
recover = False
test = False
model_path_uncover = '/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/FINAL_MODELS/standard_2D_not_subsampled_epochs=250_batch=100_workers=4_1687986938'

seed_path = '/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/FINAL_MODELS/Uncover/TL_All__Uncover_Model_10k_Old_Grasp_10000_epochs=250_batch=100_workers=4_1706810068/cma_evaluations/TL_All_Uncover_Evals_500_states_1706823382593/raw'
filenames = list(Path(seed_path).glob('*.pkl'))
filenames_iterated = iter(filenames)

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

def grasp_on_cloth(action, cloth_initial_raw):
    dist, is_on_cloth = check_grasp_on_cloth(action, np.array(cloth_initial_raw))
    return is_on_cloth

def cost_function(action, all_body_points, first_cloth, cloth_initial_raw, graph, model, device, use_disp, use_3D):
    action = scale_action(action)
    cloth_initial = graph.initial_blanket_state
    is_on_cloth= grasp_on_cloth(action, cloth_initial_raw)

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
    print(f"{counter} - Trial Completed: CMA-ES Best Reward:{output[1]:.2f}), Sim Reward: {output[2]:.2f}, TL: {output[4]}, GoC: {output[5]}")

def find(seed):
    for eval_condition in eval_conditions:
        path = Path(model_path_uncover +'/' + eval_dir_name + '/' + eval_condition + '/raw/')
        filenames = path.glob('*.pkl')
        for f in filenames:
            if str(seed) in f.name:
                return f
    #%%
def gnn_cma(seed, target, env_name, idx, model, device, target_limb_code, iter_data_dir, graph_config, env_var, max_fevals):
    use_disp = graph_config['use_disp']
    filter_draping = graph_config['filt_drape']
    rot_draping = graph_config['rot_drape']
    use_3D = graph_config['use_3D']

    coop = 'Human' in env_name

    if target_limb_code is None:
        target_limb_code = random.sample(target_limb_list, 1)[0]

    target_limb_code = target

    # seed = seeding.create_seed()

    # choose uncovered state from test set to recover from
    if recover:
        random_file = np.random.choice(list((Path(model_path_uncover)/eval_dir_name/eval_conditions[0]/'raw').iterdir())) # currently used for a test set
        seed = int(random_file.name.split('_')[2])
        target_limb_code = int(random_file.name.split('_')[0].replace('tl', ''))
        # seed = 16050922050611383883
        # seed_path = find(seed) # for recovering from one state
        # seed_path = open(seed_path, 'rb')
        seed_path = open(random_file, 'rb')
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
        voxel_size=0.05,
        edge_threshold=0.06,
        action_to_all=True,
        cloth_initial=input_cloth,
        filter_draping=filter_draping,
        rot_draping=rot_draping,
        use_3D=use_3D)

    # * set variables to initialize CMA-ES
    opts = cma.CMAOptions({'verb_disp': 1, 'popsize': pop_size, 'maxfevals': max_fevals, 'tolfun': 1e-11, 'tolflatfitness': 20, 'tolfunhist': 1e-20}) # , 'tolfun': 10, 'max_fevals': 500
    bounds = np.array([1]*4)
    opts.set('bounds', [[-1]*4, bounds])
    opts.set('CMA_stds', bounds)

    # initialize x0 cma with the naive reversal action
    global x0
    x0 = set_x0_for_cmaes(target_limb_code)
    if recover:
        x0 = [*uncover_action[2:4], *uncover_action[0:2]]

    sigma0 = 0.2
    reward_threshold = 95

    total_fevals = 0

    fevals = 0
    iterations = 0
    t0 = time.time()

    # * initialize CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_cost = None
    while not es.stop():
        iterations += 1
        fevals += pop_size
        total_fevals += pop_size

        # ! this will be your recovering actions
        actions = es.ask()
        output = [cost_function(x, all_body_points, cloth_initial_dc, input_cloth, graph, model, device, use_disp, use_3D) for x in actions]
        t1 = time.time()
        output = [list(x) for x in zip(*output)]
        costs = output[0]
        preds = output[1]
        covered_status = output[2]
        is_on_cloth = output[3]
        es.tell(actions, costs)

        if (best_cost is None) or (np.min(costs) < best_cost):
            best_cost = np.min(costs)
            best_cost_ind = np.argmin(costs)
            best_reward = -best_cost
            best_action = actions[best_cost_ind]
            best_pred = preds[best_cost_ind]
            best_covered_status = covered_status[best_cost_ind]
            best_time = t1 - t0
            best_fevals = fevals
            best_iterations = iterations
            best_is_on_cloth = is_on_cloth[best_cost_ind]
        if best_reward >= reward_threshold:
            break

    if recover:
        cloth_initial_sim, cloth_intermediate_sim, execute_recover_action = env.uncover_step(uncover_action)
        cloth_final_sim, execute_recover_action = env.recover_step(best_action) # if recovering recover action is predicted by the model
    else:
        cloth_initial_sim, cloth_intermediate_sim, execute_uncover_action = env.uncover_step(best_action)
        cloth_final_sim, execute_recover_action = env.recover_step([]) # if not recovering, don't need to provide an action

    observation, uncover_reward, recover_reward, done, info = env.get_info()

    sim_info = {'observation':observation, 'uncover_reward':uncover_reward, 'recover_reward':recover_reward, 'done':done, 'info':info}
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
    for j in tqdm(range(math.ceil(trials/num_processes))):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(num_processes):
                filename = next(filenames_iterated)
                idx = i+(j*num_processes)
                target = int(filename.name.split('_')[0][2:])
                seed = int(filename.name.split('_')[2])
                result = pool.apply_async(gnn_cma, args = (seed, target, env_name, idx, model, device, target_limb_code, iter_data_dir, graph_config, env_variations, max_fevals), callback=counter_callback)
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
    parser.add_argument('--model-path', type=str, default = 'Uncover/TL_All__Uncover_Model_10k_New_Grasp_10000_epochs=250_batch=100_workers=4_1706814845')
    parser.add_argument('--graph-config', type=str, default='2D')
    parser.add_argument('--env-var', type=str, default='standard')
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

    test_string = 'Train'
    if recover:
        test_string = ''

    for i in range(len(loop_data)):
        data = loop_data[i]
        checkpoint= osp.join(trained_models_dir, data['model'])
        env_var = data['env_var']
        env_variations = all_env_vars[env_var]
        graph_config = all_graph_configs[data['graph_config']]
        max_fevals = data['max_fevals']

        data_dir = osp.join(checkpoint, f'cma_evaluations/TL_All_{recover_string}_{args.num_rollouts}_states_{int(time.time()*1000)}')
        Path(data_dir).mkdir(parents=True, exist_ok=True)

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