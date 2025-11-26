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

uncover_model_path = '/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/FINAL_MODELS/standard_2D_not_subsampled_epochs=250_batch=100_workers=4_1687986938'
eval_dir_name = 'cma_evaluations'
threshold = 0.745

eval_condition = 'TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Uncover_Evals_Train_1000_states'
data_path = osp.join(uncover_model_path, eval_dir_name, eval_condition, 'raw/')
filenames = Path(data_path).glob('*.pkl')
parsed_file_dir = Path('TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Uncover_Evals_Train_New_Grasp')

output_path = Path(osp.join(uncover_model_path, eval_dir_name, parsed_file_dir, 'raw/'))
output_path.mkdir(exist_ok=True, parents=True)

# iterate through uncovered states
for filename in filenames:
    with open(filename, 'rb') as f:
        raw_data  = pickle.load(f)

        seed = int(filename.name.split('_')[2])
        target  = raw_data['target_limb_code']
        uncover_action = raw_data['uncover_action']
        human_pose = raw_data['human_pose']


    # set up env
    coop = 'Human' in 'RobeReversible-v1'
    env = make_env('RobeReversible-v1', coop=coop, seed=seed)
    env.set_env_variations(
        collect_data = True,
        blanket_pose_var = False,
        high_pose_var = False,
        body_shape_var = False)

    recover = False
    env.set_singulate(True)
    env.set_target_limb_code(target)
    env.set_recover(recover)
    env.set_seed_val(seed)

    done = False
    observation = env.reset()
    pid = os.getpid()


    # run uncovering step
    cloth_initial_sim, cloth_intermediate_sim, execute_uncover_action = env.uncover_step(uncover_action)

    recover_action = [0,0,0,0] # not used

    # run recovering step (recover is false though so does not actually recover)
    cloth_final_sim, execute_recover_action = env.recover_step(recover_action)
    observation, uncover_reward, recover_reward, done, info = env.get_info()

    if not recover:
        recover_action = []

    body_info = info['human_body_info']

    all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target, body_info=body_info)
    initial_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_initial_sim[1]), 2, axis = 1))
    sim_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_final_sim[1]), 2, axis = 1))

    # compute f-score
    sim_fscore = compute_fscore_uncover(initial_covered_status, sim_covered_status)

    # if fscore is greater than the threshold, add uncovered state to new directory
    if sim_fscore >= threshold:
        sim_info = {'observation':observation, 'uncover reward':uncover_reward, 'recover_reward':recover_reward, 'done':done, 'info':info}
        cma_info = raw_data['cma_info']

        with open(output_path/filename.name, 'wb') as f:
            pickle.dump({
            "recovering" : False,
            "uncover_action":uncover_action,
            "recover_action" : recover_action,
            "human_pose":human_pose,
            'target_limb_code':target,
            'sim_info':sim_info,
            'cma_info':cma_info,
            'observation':[sim_info['observation']],
            'info':sim_info['info']}, f)

