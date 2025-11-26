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
import re
import shutil
sys.path.insert(0, '/home/kpputhuveetil/git/robe/robust-body-exposure/assistive-gym-fem')
import matplotlib.pyplot as plt
from assistive_gym.envs.bu_gnn_util import *
from tqdm import tqdm

data_dir = Path('/home/kpputhuveetil/git/robe/robust-body-exposure/DATASETS/Recover_Data/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Recover_Data_100_seeds_30000_states3/raw')
data_dir.mkdir(parents=True, exist_ok=True)

dataset_path = '/home/kpputhuveetil/git/robe/robust-body-exposure/DATASETS/Recover_Data/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Recover_Data_100_seeds_30000_states/raw'
filenames_recover_old = list(Path(dataset_path).glob('*.pkl'))

dataset_path_new = '/home/kpputhuveetil/git/robe/robust-body-exposure/DATASETS/Recover_Data/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Recover_Data_100_seeds_30000_states2/raw'
filenames_recover_new = list(Path(dataset_path_new).glob('*.pkl'))

dataset_path_uncover = '/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/FINAL_MODELS/standard_2D_not_subsampled_epochs=250_batch=100_workers=4_1687986938/cma_evaluations/TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Uncover_Evals_Train_New_Grasp_Parsed/raw'
filenames_uncover = list(Path(dataset_path_uncover).glob('*.pkl'))

dataset_path_new = '/home/kpputhuveetil/git/robe/robust-body-exposure/DATASETS/Recover_Data/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Recover_Data_100_seeds_30000_states3/raw'
filenames_recover_new = list(Path(dataset_path_new).glob('*.pkl'))


seeds = {}

for filename in filenames_uncover:
    target = int(filename.name.split('_')[0][2:])
    seed = int(filename.name.split('_')[2])
    if target not in seeds:
        seeds[target] = set()
    seeds[target].add(seed)

seeds2 = {}
for filename in filenames_recover_new:
    target = int(filename.name.split('_')[1])
    seed = int(filename.name.split('_')[2])
    if target not in seeds2:
        seeds2[target] = set()
    seeds2[target].add(seed)


print(seeds[8] - seeds2[8])

to_transfer = {}

for filename in filenames_recover_new:
    target = int(filename.name.split('_')[1])
    seed = int(filename.name.split('_')[2])

    if (target, seed) not in to_transfer:
        to_transfer[(target, seed)] = set()

    if len(to_transfer[(target, seed)]) < 30:
        to_transfer[(target, seed)].add(filename)

count = 0
count1 = 0
count2 = 0
for (target, seed), paths in to_transfer.items():
    count += len(paths)
    if len(paths) < 30:
        # print(target, seed, len(paths))
        count1 += 1
    elif len(paths) > 30:
        raise Exception((target, seed))

for filename in filenames_recover_old:
    target = int(filename.name.split('_')[1])
    seed = int(filename.name.split('_')[2])

    if (target, seed) not in to_transfer:
        to_transfer[(target, seed)] = set()

    if len(to_transfer[(target, seed)]) < 30:
        to_transfer[(target, seed)].add(filename)

counts = {}

for (target, seed) in to_transfer:
    if target not in counts:
        counts[target] = 0
    counts[target] += 1

for target in counts:
    print(target, counts[target])

# for paths in tqdm(to_transfer.values()):
#     for filename in paths:
#         name = filename.name
#         shutil.copy2(filename, data_dir/name)
#         count2 += 1
