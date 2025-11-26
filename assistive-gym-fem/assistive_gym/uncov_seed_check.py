import gym, sys, argparse, multiprocessing, time, os, math
from gym.utils import seeding
from learn import make_env
import numpy as np
import pickle, pathlib
import os.path as osp
import random
import glob
sys.path.insert(0, './assistive-gym-fem')
sys.path.insert(0, './code')
from assistive_gym.envs.bu_gnn_util import *
from cma_gnn_util import *
import pull_random_seeds
from pathlib import Path
import re
#2:50 , 4:28 5: 41 8: 51, 10:58 , 11:25, 12:145, 13:177, 14:140, 15:132
uncover_model_path = '/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/FINAL_MODELS/standard_2D_not_subsampled_epochs=250_batch=100_workers=4_1687986938'
eval_dir_name = 'cma_evaluations'
threshold = 0.745

eval_condition = 'TL_[5]_Uncover_Evals_Train_5000_states'
data_path = osp.join(uncover_model_path, eval_dir_name, eval_condition, 'raw/')
filenames = list(Path(data_path).glob('*.pkl'))
parsed_file_dir = Path('TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Uncover_Evals_Train_New_Grasp')

output_path = Path(osp.join(uncover_model_path, eval_dir_name, parsed_file_dir, 'raw/'))
output_path.mkdir(exist_ok=True, parents=True)

def rollout(env_name, i, name, seed, raw_data):
    # set up env
    # coop = 'Human' in 'RobeReversible-v1'
    # env = make_env('RobeReversible-v1', coop=coop, seed=seed)
    # env.set_env_variations(
    #     collect_data = True,
    #     blanket_pose_var = False,
    #     high_pose_var = False,
    #     body_shape_var = False)

    # recover = False
    # env.set_singulate(True)
    # env.set_target_limb_code(target)
    # env.set_recover(recover)
    # env.set_seed_val(seed)

    # done = False
    # observation = env.reset()
    pid = os.getpid()


    # run uncovering step
    # cloth_initial_sim, cloth_intermediate_sim, execute_uncover_action = env.uncover_step(uncover_action)

    # recover_action = [0,0,0,0] # not used

    # # run recovering step (recover is false though so does not actually recover)
    # cloth_final_sim, execute_recover_action = env.recover_step(recover_action)
    # observation, uncover_reward, recover_reward, done, info = env.get_info()

    # if not recover:
    #     recover_action = []

    target  = raw_data['target_limb_code']
    human_pose = raw_data['human_pose']
    cma_info = raw_data['cma_info']

    # body_info = info['human_body_info']
    body_info = raw_data['info']['human_body_info']


    all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target, body_info=body_info)
    # initial_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_initial_sim[1]), 2, axis = 1))
    # sim_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_final_sim[1]), 2, axis = 1))

    initial_covered_status = get_covered_status(all_body_points, np.delete(np.array(raw_data['sim_info']['info']['cloth_initial'][1]), 2, axis = 1))
    sim_covered_status = get_covered_status(all_body_points, np.delete(np.array(raw_data['sim_info']['info']['cloth_final'][1]), 2, axis = 1))

    # compute f-score
    sim_fscore = compute_fscore_uncover(initial_covered_status, sim_covered_status)

    saved = False

    # if fscore is greater than the threshold, add uncovered state to new directory
    if sim_fscore >= threshold:
        saved = True
        # sim_info = {'observation':observation, 'uncover reward':uncover_reward, 'recover_reward':recover_reward, 'done':done, 'info':info}

        with open(output_path/name, 'wb') as f:
            pickle.dump(raw_data, f)

    output = [i, seed, sim_fscore, pid, saved]
    # del env
    return output

def counter_callback(output):
    global counter
    counter += 1
    print(f"{counter} - Trial Completed: {output[0]}, Process: {output[3]}, Seed: {output[1]}, Fscore: {output[2]}, Saved: {output[4]}")

if __name__ == "__main__":
    counter = 0
    num_processes = 100
    trials = len(filenames)
    counter = 0

    result_objs = []
    for j in range(math.ceil(trials/num_processes)):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(num_processes):
                with open(filenames[i+i*j], 'rb') as f:
                    raw_data  = pickle.load(f)
                    seed = int(filenames[i+i*j].name.split('_')[2])

                result = pool.apply_async(rollout, args = ('RobeReversible-v1', i, filenames[i+i*j].name, seed, raw_data), callback=counter_callback)
                result_objs.append(result)

            results = [result.get() for result in result_objs]
