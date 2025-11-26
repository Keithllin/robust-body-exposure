import glob
import os
import os.path as osp
import pickle
import sys
from curses import raw
from pathlib import Path

import matplotlib.pyplot as plt

original_data_dir = Path("/home/kpputhuveetil/git/robe/robust-body-exposure/DATASETS/Recover_Data/Test3/raw")
parsed_data_dir = Path('/home/kpputhuveetil/git/robe/robust-body-exposure/DATASETS/Recover_Data/Test0/raw')#TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Recover_Data_100_seeds_10000_states_extralowerbody/raw')
parsed_data_dir.mkdir(exist_ok=True, parents=True)

filenames = Path(original_data_dir).glob('*.pkl')
lower_body = [4, 5, 10, 11, 13, 14]
for filename in filenames:
        with open(filename, 'rb') as f:
            target = int(filename.name.split('_')[1])
            raw_data = pickle.load(f)
            if not Path(parsed_data_dir/filename.name).exists():
                with open(parsed_data_dir/filename.name, 'wb') as f:
                    pickle.dump(raw_data, f)
