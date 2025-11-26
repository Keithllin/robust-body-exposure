#%%
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os.path as osp

file_dir = '/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/FINAL_MODELS/tl_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_100_states_1k_lowerbody_16000_epochs=250_batch=50_workers=4_1702371732/cma_evaluations/TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Recover_Evals__500_states_RandomSearch_Opt/raw'
filename = 'tl4_c68_18237961019803109524_pid29064.pkl'
raw_data = pickle.load(open(osp.join(file_dir, filename), 'rb'))
# action = raw_data['action']
# %%
