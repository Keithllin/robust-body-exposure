#%%

import glob
import os.path as osp
import pickle
import sys
from curses import raw

import matplotlib.pyplot as plt
import numpy as np
import argparse
import math


sys.path.insert(0, './assistive-gym-fem')
from assistive_gym.envs.bu_gnn_util import *
from cma_gnn_util import *
from tabulate import tabulate

model_dir = './trained_models/FINAL_MODELS/'

parser = argparse.ArgumentParser()
parser.add_argument('--arg_model', type=str, default='Uncover/TL_All__Uncover_Model_10k_New_Grasp_10000_epochs=250_batch=100_workers=4_1706814845')
parser.add_argument('--tlc', type=int, default=4)

args = parser.parse_args()

model = args.arg_model
model_path = osp.join(model_dir, model)
eval_dir_name = 'cma_evaluations'

eval_conditions = ['TL_All_Uncover_Evals_500_states_1706828964151']
parsed_file_dir = Path('TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Uncover_Evals_Train_1000_states')
# eval_conditions = ['TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Recover_Evals__1000_states_RandomSearch_Opt']
target_limbs = [2, 4, 5, 8, 10, 11, 12, 13, 14, 15]
counts = [0] * len(target_limbs)
count = dict(zip(target_limbs, counts))

target_limb_distribution = [0] * len(target_limbs)
threshold = 0.745

for eval_condition in eval_conditions:
    data_path = osp.join(model_path, eval_dir_name, eval_condition, 'raw/')
    filenames = Path(data_path).glob('*.pkl')
    output_path = Path(osp.join(model_path, eval_dir_name, parsed_file_dir, 'raw/'))
    output_path.mkdir(exist_ok=True, parents=True)
    num_targets = 16

    targ_data_reward = [[[] for _ in range(num_targets)], [[] for _ in range(num_targets)]]

    targ_data_fscore = [[[] for _ in range(num_targets)], [[] for _ in range(num_targets)]]

    failed_grasps = np.zeros(num_targets)

    ##%%
    count_ng = 0
    seed = 0

    for filename in filenames:
        with open(filename, 'rb') as f:

            #Parse filename to get the seed out
            seed = filename.name.split('_')[2]

            raw_data = pickle.load(f)
            target_limb_code = int(filename.name.split('_')[0][2:])
            plt.close()

            recover = False #raw_data['recovering']

            cma_reward = raw_data['cma_info']['best_reward']

            if recover:
                sim_reward = raw_data['sim_info']['recover_reward']
            else:
                if 'uncover_reward' in raw_data['sim_info']:
                    sim_reward = raw_data['sim_info']['uncover_reward']
                else:
                    sim_reward = raw_data['sim_info']['uncover reward']

            targ_data_reward[0][target_limb_code].append(cma_reward)
            targ_data_reward[1][target_limb_code].append(sim_reward)
            cloth_initial = raw_data['sim_info']['info']['cloth_initial'][1]

            if recover:
                cloth_intermediate = raw_data['sim_info']['info']['cloth_intermediate'][1]

            cloth_final = raw_data['sim_info']['info']['cloth_final'][1]
            pred = raw_data['cma_info']['best_pred']

            human_pose = raw_data['human_pose']
            body_info = raw_data['info']['human_body_info']
            count_ng += 1

            all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)
            initial_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_initial), 2, axis = 1))
            if recover:
                intermediate_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_intermediate), 2, axis = 1))

            cma_covered_status = get_covered_status(all_body_points, pred)
            sim_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_final), 2, axis = 1))

            # cma_covered_status = raw_data['cma_info']['best_covered_status']
            # sim_covered_status = raw_data['sim_info']['info']['covered_status_sim']

            if recover:
                sim_fscore = compute_fscore_recover(initial_covered_status, intermediate_covered_status, sim_covered_status, False)
                cma_fscore  = compute_fscore_recover(initial_covered_status, intermediate_covered_status, cma_covered_status, False)
            else:
                sim_fscore = compute_fscore_uncover(initial_covered_status, sim_covered_status)
                cma_fscore = compute_fscore_uncover(initial_covered_status, cma_covered_status)

            # if sim_fscore >= threshold:
            #     with open(output_path/filename.name, 'wb') as f:
            #         pickle.dump(raw_data, f)

            count[target_limb_code] += 1

            target_index = target_limbs.index(target_limb_code)

            if recover:
                if not raw_data['sim_info']['info']['grasp_on_cloth_recover']:
                    failed_grasps[target_limb_code] += 1
                    targ_data_fscore[0][target_limb_code].append(cma_fscore)
            else:
                if not raw_data['sim_info']['info']['grasp_on_cloth_uncover']:
                    failed_grasps[target_limb_code] += 1
                    targ_data_fscore[0][target_limb_code].append(cma_fscore)

            if not np.isnan(sim_fscore):
                targ_data_fscore[1][target_limb_code].append(sim_fscore)
            if not np.isnan(cma_fscore) and not math.isnan(cma_fscore):
                targ_data_fscore[0][target_limb_code].append(cma_fscore)

            cleanedList = [x for x in targ_data_fscore[0][target_limb_code] if str(x) != 'nan']
            targ_data_fscore[0][target_limb_code] = cleanedList

            cleanedList = [x for x in targ_data_fscore[1][target_limb_code] if str(x) != 'nan']
            targ_data_fscore[1][target_limb_code] = cleanedList
    reward_means = [[],[]]
    reward_stds = [[],[]]
    fscore_means = [[],[]]
    fscore_stds = [[],[]]
    samples = []
    rewards = [0] * 10
    counts = [0] * 10
    targets = [2, 4, 5, 8, 10, 11, 12, 13, 14, 15]
    for target in range(num_targets):
        if target in [2, 4, 5, 8, 10, 11, 12, 13, 14, 15]:

            samples.append(len(targ_data_reward[0][target]))

            reward_means[0].append(np.mean(targ_data_reward[0][target]))
            reward_stds[0].append(np.std(targ_data_reward[0][target]))
            reward_means[1].append(np.mean(targ_data_reward[1][target]))
            reward_stds[1].append(np.std(targ_data_reward[1][target]))

            fscore_means[0].append(np.mean(targ_data_fscore[0][target]))
            fscore_stds[0].append(np.std(targ_data_fscore[0][target]))
            fscore_means[1].append(np.mean(targ_data_fscore[1][target]))
            fscore_stds[1].append(np.std(targ_data_fscore[1][target]))
        else:
            samples.append(len(targ_data_reward[0][target]))

            reward_means[0].append(0)
            reward_stds[0].append(0)
            reward_means[1].append(0)
            reward_stds[1].append(0)

            fscore_means[0].append(0)
            fscore_stds[0].append(0)
            fscore_means[1].append(0)
            fscore_stds[1].append(0)

    prop_failed = failed_grasps/np.array(samples)

    target_names_full = [
        '', '', 'R Arm',
        '', 'R L. Leg', 'R Leg',
        '', '', 'L Arm',
        '', 'L L. Leg', 'L Leg',
        'B L. Legs', 'Upper Body',
        'Lower Body', 'Whole Body']

    print(tabulate([[i, target_names_full[i] ,fscore_means[0][i], fscore_means[1][i], 100*prop_failed[i]] for i in range(num_targets)], headers=['TL#', 'Target', 'opt', 'sim', '% failed grasp']))

    #%%
    target_names = [
        'R Arm', 'R L. Leg', 'R Leg',
        'L Arm', 'L L. Leg', 'L Leg',
        'B L. Legs', 'Upper Body',
        'Lower Body', 'Whole Body']
    T = [2, 4, 5, 8, 10, 11, 12, 13, 14, 15]
    import pandas as pd
    # table_info = [[target_names[i], round(fscore_means[1][T[i]], 2), round(fscore_stds[1][T[i]],2), round(100*prop_failed[T[i]], 2)] for i in range(len(T))]
    # # table = pd.DataFrame([[target_names],
    # #                       [round(fscore_means[1][10[:]], 2)],
    # #                       [round(fscore_stds[1][10[:]],2)],
    # #                       [round(100*prop_failed[10[:]], 2)]])
    # # print(table)
    # # table_info = [[target_[i], round(fscore_means[1][T[i]], 2), round(fscore_stds[1][T[i]],2), round(100*prop_failed[T[i]], 2)] for i in range(len(T))]

    # print(tabulate([[fscore_means[0][i], fscore_means[1][i]] for i in range(num_targets)], headers=['cma', 'sim']))

    # table = pd.DataFrame(data=[[target_names[i], fscore_means[0][t], fscore_means[1][t]] for i, t in enumerate(T)], columns=['Target Name', 'CMA', 'SIM'])

    # table.to_latex("F-SCORES.tex")
    # print(table)
    # print(tabulate(table_info, headers=['target', 'sim,', 'stds', '% null']))
    reward_diff = list(abs(np.array(reward_means[0])-np.array(reward_means[1])))
    fscore_diff = list(abs(np.array(fscore_means[0])-np.array(fscore_means[1])))

    for i, t in enumerate(T):
        print(target_names[i], reward_means[1][t])

    all_cma_reward = sum(targ_data_reward[0], [])
    all_sim_reward = sum(targ_data_reward[1], [])

    all_cma_fscore = sum(targ_data_fscore[0], [])
    all_sim_fscore = sum(targ_data_fscore[1], [])

    print('optmizer reward mean:', np.mean(all_cma_reward))
    # print('cma std:', np.std(all_cma))
    print('sim reward mean:', np.mean(all_sim_reward))
    # print('sim std:', np.std(all_sim))
    print('cma fscore:', round(np.mean(all_cma_fscore), 2))
    # print('cma std:', np.std(all_cma))
    print('sim fscore:', round(np.mean(all_sim_fscore), 2))
    print('sim fscore std:', round(np.std(all_sim_fscore), 2))
    print(f'Overall % Null Grasp: {np.sum(failed_grasps)}/{count_ng} = {np.sum(failed_grasps)/count_ng * 100}')
#% # %%
# print("Counts: ", count)

# %%
