#%%
import glob
import os.path as osp
import pickle
import sys
from pathlib import Path
import argparse
import time
import numpy as np
from cma_gnn_util import *

sys.path.insert(0, '/home/kpputhuveetil/git/robe/robust-body-exposure/assistive-gym-fem')
import matplotlib.pyplot as plt
from assistive_gym.envs.bu_gnn_util import *

# parser = argparse.ArgumentParser()
# parser.add_argument('--arg_model', type=str, default='tl_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_100_states_1k_1000_epochs=250_batch=50_workers=4_1700540415')
# # parser.add_argument('--arg_model', type=str, default='standard_2D_not_subsampled_epochs=250_batch=100_workers=4_1687986938')

# args = parser.parse_args()

model_dir = '/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/FINAL_MODELS/standard_2D_not_subsampled_epochs=250_batch=100_workers=4_1687986938/cma_evaluations/TL_[4]_Recover_Evals__20_states_RandomSearch_Opt'
# model = args.arg_model
# model_path = osp.join(model_dir, model)
# eval_dir_name = 'cma_evaluations'

eval_conditions = ['TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Recover_Evals__1000_states_RandomSearch_Opt'] #[next(f for f in (Path(model_path)/eval_dir_name).iterdir() if f.name.startswith("standard")).name]

threshold = 0.745

image_dir = osp.join(model_dir, 'images/')

Path(image_dir).mkdir(parents=True, exist_ok=True)

for eval_condition in eval_conditions:
    data_path = osp.join(model_dir,'raw/')
    filenames = Path(data_path).glob('*.pkl')

    count = 0
    ##%%
    for i, path in enumerate(filenames):
        with open(path, 'rb') as f:
            filename = path.name
            seed = filename.split('_')[2]
            raw_data = pickle.load(f)

            fig_id = filename.split('_')[2]
            tl = filename.split('_')[0]

            if len(raw_data['sim_info']['info']) > 2:
                target_limb_code = raw_data['target_limb_code']
                human_pose = raw_data['human_pose']
                body_info = raw_data['sim_info']['info']['human_body_info']
                best_total_cost, uncov_data, recov_data = raw_data['cma_info']
                uncov_cost, uncov_pred, covered_status, is_on_cloth = uncov_data
                best_recov_action, best_recov_cost, pred, best_covered_status, best_is_on_cloth = recov_data
                # pred = raw_data['cma_info']['best_pred']

                recover = raw_data['recovering']

                all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)

                cloth_initial = np.array(raw_data['info']['cloth_initial'][1])

                if recover:
                    cloth_intermediate = np.array(raw_data['info']['cloth_intermediate'][1])

                cloth_final = np.array(raw_data['info']['cloth_final'][1])

                initial_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_initial), 2, axis = 1))

                if recover:
                    intermediate_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_intermediate), 2, axis = 1))

                # cma_covered_status = raw_data['cma_info']['best_covered_status']
                # sim_covered_status = raw_data['info']["covered_status_sim"]

                cma_covered_status = get_covered_status(all_body_points, pred)
                sim_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_final), 2, axis = 1))
                info = raw_data['sim_info']['info']

                cma_reward = -best_total_cost
                sim_reward = raw_data['sim_info']['recover_reward']

                if recover:
                    sim_fscore, sim_info_fscore = compute_fscore_recover(initial_covered_status, intermediate_covered_status, sim_covered_status, True)
                    cma_fscore, cma_info_fscore = compute_fscore_recover(initial_covered_status, intermediate_covered_status, cma_covered_status, True)
                else:
                    sim_fscore = compute_fscore_uncover(initial_covered_status, sim_covered_status)
                    cma_fscore = compute_fscore_uncover(initial_covered_status, cma_covered_status)

                uncover_action = scale_action(raw_data['uncover_action'])

                if recover:
                    recover_action = scale_action(best_recov_action)

                if recover:
                    sim_reward, sim_reward_info = get_recovering_reward(recover_action, all_body_points, np.delete(np.array(cloth_initial),2,axis=1),
                                                                                                np.delete(np.array(cloth_intermediate),2,axis=1),
                                                                                                np.delete(np.array(cloth_final),2,axis=1), True)
                    cma_reward, cma_reward_info = get_recovering_reward(recover_action, all_body_points, np.delete(np.array(cloth_initial),2,axis=1),
                                                                                np.delete(np.array(cloth_intermediate),2,axis=1),
                                                                            pred, True)
                else:
                    sim_reward = get_uncovering_reward(uncover_action, all_body_points, np.delete(np.array(cloth_initial),2,axis=1),
                                                                                                np.delete(np.array(cloth_final),2,axis=1))
                    cma_reward = get_uncovering_reward(uncover_action, all_body_points, np.delete(np.array(cloth_initial),2,axis=1),
                                                                            pred)

                if not recover:
                    if sim_fscore >= threshold:
                        fig = generate_figure_uncover(
                            sim_reward,
                            cma_reward,
                            target_limb_code,
                            uncover_action,
                            body_info,
                            all_body_points,
                            cloth_initial,
                            final_cloths = [cloth_final, pred],
                            initial_covered_status = [initial_covered_status],
                            covered_statuses = [sim_covered_status, cma_covered_status] ,
                            fscores = [sim_fscore, cma_fscore],
                            plot_initial=True, compare_subplots=False)

                        img_file = f'{target_limb_code}_{seed}_f-score={sim_fscore}.png'

                        fig.write_image(osp.join(image_dir, img_file))
                else:
                    fig = generate_figure_recover(
                            sim_info_fscore,
                            cma_info_fscore,
                            sim_reward,
                            sim_reward_info,
                            cma_reward,
                            cma_reward_info,
                            target_limb_code,
                            uncover_action,
                            recover_action,
                            body_info,
                            all_body_points,
                            cloth_initial,
                            final_cloths = [cloth_final, pred],
                            cloth_intermediate=cloth_intermediate,
                            initial_covered_status = [initial_covered_status, intermediate_covered_status],
                            covered_statuses = [sim_covered_status, cma_covered_status] ,
                            fscores = [sim_fscore, cma_fscore],
                            plot_initial=True, compare_subplots=False)

                    img_file = f'{target_limb_code}_{seed}_f-score={sim_fscore}.png'

                    fig.write_image(osp.join(image_dir, img_file))

# %%
