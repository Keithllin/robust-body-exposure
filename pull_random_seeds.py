import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import re

def random_seeds(num_seeds, target_limb_list):
    data_dir = '/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/FINAL_MODELS/standard_2D_not_subsampled_epochs=250_batch=100_workers=4_1687986938/cma_evaluations/'
    eval_dir = 'TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Uncover_Evals_Train_New_Grasp/raw'
    dir = Path(data_dir) / eval_dir
    filenames = dir.glob('*.pkl')

    parsed_file_dir = Path('TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Uncover_Evals_Train_New_Grasp2')

    output_path = Path(osp.join(data_dir, parsed_file_dir, 'raw/'))
    output_path.mkdir(exist_ok=True, parents=True)

    seed_list = []
    seed_dict = {}

    for filename in filenames:
        target_limb, seed = map(int, re.findall(r'tl(\d+)_c\d+_(\d+)', filename.name)[0])

        if target_limb not in target_limb_list:
            continue

        if target_limb not in seed_dict:
            seed_dict[target_limb] = []

        seed_dict[target_limb].append(seed)

    for seeds in seed_dict.values():
        seed_list += random.sample(seeds, num_seeds)

    return seed_list
#     print(seed_list)

if __name__ == "__main__":
    x = random_seeds(10, [2, 4, 5, 8, 10, 11, 12, 13, 14, 15])
    print(len(x), x)