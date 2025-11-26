#%%
import glob
import os.path as osp
import pickle
import sys
from pathlib import Path

import numpy as np
from cma_gnn_util import *

sys.path.insert(0, '/home/kpputhuveetil/git/robe/robust-body-exposure/assistive-gym-fem')
import matplotlib.pyplot as plt
from assistive_gym.envs.bu_gnn_util import *
import time

model_path = '/home/kpputhuveetil/git/robe/robust-body-exposure/DATASETS/Recover_Data/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Recover_Data_100_seeds_30000_states/raw' #'TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Recover_Data_100_seeds_10000_states_lowerbody/raw'
dataset = 'TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Recover_Data_100_seeds_30000_states'
image_path = '/home/kpputhuveetil/git/robe/robust-body-exposure/DATASETS/Recover_Data'
image_dir = osp.join(image_path, 'images', dataset)

Path(image_dir).mkdir(parents=True, exist_ok=True)

filenames = Path(model_path).glob('*.pkl')

##%%
for i, filename in enumerate(filenames):
    with open(filename, 'rb') as f:

        seed = filename.name.split('/')[-1].split('_')[2]
        raw_data = pickle.load(f)

        if 'cloth_initial' not in raw_data['info']:
            continue
        tl = filename.name.split('/')[-1]
        fig_id = filename.name.split('_')[2]
        tl = filename.name.split('_')[0]

        cloth_initial = np.array(raw_data['info']['cloth_initial'][1])
        cloth_intermediate = np.array(raw_data['info']['cloth_intermediate'][1])
        cloth_final = np.array(raw_data['info']['cloth_final'][1])
        uncover_action = scale_action(raw_data['uncover_action'])
        recover_action = scale_action(raw_data['recover_action'])

        # target_limb_code = [raw_data_original['target_limb_code']]
        # human_pose = raw_data_original['human_pose']
        # body_info = raw_data_original['sim_info']['info']['human_body_info']
        # all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)

        target_limb_code = filename.name.split('_')[1]
        body_info = []
        all_body_points = []

        fig = generate_figure_data_collection(
            target_limb_code,
            uncover_action,
            recover_action,
            body_info,
            all_body_points,
            cloth_initial,
            cloth_intermediate,
            cloth_final)

        # fig = generate_figure(
        #     action,
        #     all_body_points,
        #     cloth_initial,
        #     cloth_intermediate,
        #     cloth_final,
        #     plot_initial=True, compare_subplots=True)

        img_file = f'{target_limb_code}_{seed}_{time.time()}.png'
        # fig.show()
        fig.write_image(osp.join(image_dir, img_file))

        # fig.write_image(osp.join(tl_dir, img_file))
        # fig.write_image(osp.join(model_path, eval_dir_name, eval_condition, 'eval_imgs', img_file))



 # %%
