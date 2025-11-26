#%%
import math
import sys
import time

# import multiprocessing
from torch import multiprocessing

sys.path.insert(0, '/home/kpputhuveetil/git/vBM-GNNdev/assistive-gym-fem')
import os.path as osp
from pathlib import Path

import cma
import numpy as np
import torch
from assistive_gym.envs.bu_gnn_util import *
from assistive_gym.learn import make_env
from build_bm_graph import BM_Graph
# import assistive_gym.envs.bu_gnn_util
from cma_gnn_util import *
from gnn_train_test_new import GNN_Train_Test
from gym.utils import seeding
from tqdm import tqdm

import gradient_free_optimizers as gfo


#%%
#! NEED TO ADD TO CODE
def para_to_action(para):
    action = np.array([para['x_i'], para['y_i'], para['x_f'], para['y_f']])
    return action

# check if grasp is on the cloth BEFORE subsampling! cloth_initial_raw is pre subsampling
# ! maybe there is already a util function for this? check_grasp_on_cloth
def grasp_on_cloth(action, cloth_initial_raw):
    dist, is_on_cloth = check_grasp_on_cloth(action, np.array(cloth_initial_raw))
    return is_on_cloth

#! THIS CAN BE THE SAME AS WHAT WAS USED FOR CMA
def cost_function(action, all_body_points, cloth_initial_raw, graph, model, device, use_disp, use_3D):
    action = scale_action(action)
    cloth_initial = graph.initial_blanket_state
    # if not grasp_on_cloth(action, cloth_initial_raw):
    #     return [0, cloth_initial, -1, None]
    is_on_cloth= grasp_on_cloth(action, cloth_initial_raw)
    if is_on_cloth:
        data = graph.build_graph(action)

        data = data.to(device).to_dict()
        batch = data['batch']
        batch_num = np.max(batch.data.cpu().numpy()) + 1
        # batch_num = np.max(batch.data.detach().cpu().numpy()) + 1    # version used for gpu, not cpu only for this script
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
        # print('predicted', pred[0:10])
        cost, covered_status = get_cost(action, all_body_points, cloth_initial_2D, pred_2D)
    else:
        cost, covered_status = get_cost(action, all_body_points, cloth_initial, pred)

    return [cost, pred, covered_status, is_on_cloth]

#! THIS CAN BE THE SAME AS WHAT WAS USED FOR CMA
def get_cost(action, all_body_points, cloth_initial_2D, cloth_final_2D):
    reward, covered_status = get_reward(action, all_body_points, cloth_initial_2D, cloth_final_2D)
    cost = -reward
    return cost, covered_status

#! THIS CAN BE THE SAME AS WHAT WAS USED FOR CMA
def counter_callback(output):
    global counter
    counter += 1
    print(f"{counter} - Trial Completed: Optimization Best Reward:{output[1]:.2f}, Sim Reward: {output[2]:.2f}, CMA Time: {output[3]/60:.2f}, TL: {output[4]}, GoC: {output[5]}")

    # print(f"{counter} - Trial Completed: {output[0]}, Worker: {output[2]}, Filename: {output[1]}")
    # print(f"Trial Completed: {output[0]}, Worker: {os.getpid()}, Filename: {output[1]}")

#%%

#! MAIN PART THAT IS DIFFERENT - THIS IS EQUIVANENT TO GNN_CMA
def gnn_bayseian(env_name, idx, model, device, target_limb_code, iter_data_dir, params, env_var):

    use_disp = params['use_disp']
    filter_draping = params['filt_drape']
    rot_draping = params['rot_drape']
    use_3D = params['use_3D']

    coop = 'Human' in env_name
    seed = seeding.create_seed()
    env = make_env(env_name, coop=coop, seed=seed)

    env.set_env_variations(
        collect_data = False,
        blanket_pose_var = env_var['blanket_var'],
        high_pose_var = env_var['high_pose_var'],
        body_shape_var = env_var['body_shape_var'])
    done = False
    # #env.render())
    human_pose = env.reset()
    human_pose = np.reshape(human_pose, (-1,2))
    if target_limb_code is None:
        target_limb_code = randomize_target_limbs()
    cloth_initial_raw = env.get_cloth_state()
    env.set_target_limb_code(target_limb_code)
    pop_size = 8

    body_info = env.get_human_body_info()
    all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)

    graph = BM_Graph(
        root = iter_data_dir,
        description=f"iter_{iter}_processed",
        voxel_size=0.05,
        edge_threshold=0.06,
        action_to_all=True,
        cloth_initial=cloth_initial_raw,
        filter_draping=filter_draping,
        rot_draping=rot_draping,
        use_3D=use_3D)

    #! WE NEED TO CONVERT X0 (as a list) to X0 as a parameter dictionary
    #! this if-else chain is replaced by reverse action
    if target_limb_code in [0, 1, 2]:       # Top Right
        x0 = [0.5, -0.4, 0, 0]
        # x0 = [0.5, -0.4, -0.5, -0.5]
    elif target_limb_code in [3, 4, 5]:     # Bottom Right
        x0 = [0.5, 0.5, 0, 0]
    elif target_limb_code in [6, 7, 8]:     # Top Left
        x0 = [-0.5, -0.4, 0, 0]
        # x0 = [-0.5, -0.4, 0.5, -0.5]
    elif target_limb_code in [9, 10, 11]:   # Bottom Left
        x0 = [-0.5, 0.5, 0, 0]
    else:
        x0 = [0, 0, 0, -0.5]
    x0 = {'x_i':x0[0], 'y_i':x0[1], 'x_f':x0[2], 'y_f':x0[3]}

    #! should contain all arguments that go to the cost function
    pass_through = {'pass_through': [all_body_points, cloth_initial_raw, graph, model, device, use_disp, use_3D]}

    #! NEED TO BRING THIS INTO NEW CODE, MAKE SURE YOUR ARE UNPACKING PASS THROUGH AND FEEDING IT INTO YOUR COST FUNCTION APPROPRIATELY
    def cost_function_helper(para):
        all_body_points, cloth_initial_raw, graph, model, device, use_disp, use_3D = pass_through['pass_through']
        action = para_to_action(para)
        cost, pred, covered_status, is_on_cloth = cost_function(action, all_body_points, cloth_initial_raw, graph, model, device, use_disp, use_3D)

        # output = {'output': [pred, covered_status, is_on_cloth]}
        return -cost

    #! BRING THIS INTO NEW CODE, no changes needed
    step_size = 0.01
    search_space = {
        "x_i": np.arange(-1, 1+step_size, step_size),
        "y_i": np.arange(-1, 1+step_size, step_size),
        "x_f": np.arange(-1, 1+step_size, step_size),
        "y_f": np.arange(-1, 1+step_size, step_size),
    }

    #! REPLACE CMA SETUP WITH THIS
    # opt = gfo.RandomSearchOptimizer(search_space, initialize={"grid": 4, "random": 10, "vertices": 4, "warm_start": [x0]})
    opt = gfo.ParallelTemperingOptimizer(search_space, initialize={"grid": 4, "random": 10, "vertices": 4, "warm_start": [x0]})
    t0 = time.time()
    opt.search(
        cost_function_helper,
        n_iter=500,
        verbosity=False)
    t1 = time.time()
    best_para = opt.best_para
    # best_score = opt.best_score(cost_function_helper)

    best_action = para_to_action(best_para)
    best_time = t1-t0
    best_fevals = None
    best_iterations = None

    best_cost, best_pred, best_covered_status, best_is_on_cloth = cost_function(best_action, all_body_points, cloth_initial_raw, graph, model, device, use_disp, use_3D)
    best_reward = -best_cost

    observation, env_reward, done, info = env.step(best_action)
    # print(info.keys())
    #! DO THIS HOWEVER YOU HAVE IT IN YOUR CODE ALREADY
    # return cloth_initial, best_pred, all_body_points, best_covered_status, info
    sim_info = {'observation':observation, 'reward':env_reward, 'done':done, 'info':info}
    cma_info = {
        'best_cost':best_cost, 'best_reward':best_reward, 'best_pred':best_pred, 'best_time':best_time,
        'best_covered_status':best_covered_status, 'best_fevals':best_fevals, 'best_iterations':best_iterations}

    # only save data is error is greater than some threshold, 15
    save_data_to_pickle(
        idx,
        seed,
        best_action,
        human_pose,
        target_limb_code,
        sim_info,
        cma_info,
        iter_data_dir)
    # save_dataset(idx, graph, best_data, sim_info, best_action, human_pose, best_covered_status)
    return seed, best_reward, env_reward, best_time, target_limb_code, best_is_on_cloth


def load_model_for_eval(checkpoint):
    pass

def load_model_for_update():
    pass

def evaluate_dyn_model(env_name, target_limb_code, trials, model, iter_data_dir, device, num_processes, params, env_variations):

    result_objs = []
    for j in tqdm(range(math.ceil(trials/num_processes))):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(num_processes):
                idx = i+(j*num_processes)
                result = pool.apply_async(gnn_bayseian, args = (env_name, idx, model, device, target_limb_code, iter_data_dir, params, env_variations), callback=counter_callback)
                result_objs.append(result)

            results = [result.get() for result in result_objs]
            all_results.append(results)

    results_array = np.array(results)
    pred_sim_reward_error = abs(results_array[:,2] - results_array[:,1])

    return list(results_array[:,1]), list(results_array[:,2]), list(pred_sim_reward_error)


def update_dyn_model(model):
    pass

#%%
#! START MAIN
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    trained_models_dir = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/FINAL_MODELS'

    params = {
        '2D_rot':{'filt_drape':False,  'rot_drape':True, 'use_3D':False, 'use_disp':True},
        '3D':{'filt_drape':False,  'rot_drape':False, 'use_3D':True, 'use_disp':True}
    }

    all_env_vars = {
        'standard':{'blanket_var':False, 'high_pose_var':False, 'body_shape_var':False},   # standard
        'body_shape_var':{'blanket_var':False, 'high_pose_var':False, 'body_shape_var':True},    # body shape var
        'pose_var':{'blanket_var':False, 'high_pose_var':True, 'body_shape_var':False},    # high pose var
        'blanket_var':{'blanket_var':True, 'high_pose_var':False, 'body_shape_var':False},    # blanket var
        'combo_var':{'blanket_var':True, 'high_pose_var':True, 'body_shape_var':True}       # combo var
    }

    loop_data = [
        {'model': 'standard_2D_10k_epochs=250_batch=100_workers=4_1668718872', 'params': '2D_rot', 'env_var': 'standard'},
    ]

    env_name = "BodiesUncoveredGNN-v1"
    target_limb_code = None

    for i in range(len(loop_data)):
        data = loop_data[i]
        checkpoint= osp.join(trained_models_dir, data['model'])
        env_var = data['env_var']
        env_variations = all_env_vars[env_var]
        param = params[data['params']]


        data_dir = osp.join(checkpoint, f'parallel_temp_evaluations/{env_var}')
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        print(data_dir)

        device = 'cpu'
        gnn_train_test = GNN_Train_Test(device)
        gnn_train_test.load_model_from_checkpoint(checkpoint)
        gnn_train_test.model.to(torch.device('cpu'))
        gnn_train_test.model.share_memory()
        gnn_train_test.model.eval()

        counter = 0
        all_results = []

        # reserve one cpu to keep working while collecting data
        num_processes = multiprocessing.cpu_count() - 1

        iterations = 10

        num_processes = trials = 50
        k_largest = int(trials/2)

        # iterations = trials = num_processes = 1

        for iter in tqdm(range(iterations)):

            cma_reward, sim_reward, pred_sim_reward_error = evaluate_dyn_model(
                env_name=env_name,
                target_limb_code = target_limb_code,
                trials = trials,
                model = gnn_train_test.model,
                iter_data_dir = data_dir,
                device = device,
                num_processes = num_processes,
                params = param,
                env_variations = env_variations)

    print("ALL EVALS COMPLETE")