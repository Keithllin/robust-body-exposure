#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from assistive_gym.envs.bu_gnn_util import *



# raw_data = pickle.load(open('/home/kpputhuveetil/git/robe/robust-body-exposure/test_data_initial2.pkl', 'rb'))
data_dir = '/home/kpputhuveetil/git/robe/robust-body-exposure/DATASETS/Leg_Recovered_States_4_2918277101486668407/raw'
file = 'c99_4_2918277101486668407_pid129743.pkl'

raw_data = pickle.load(open(osp.join(data_dir, file), 'rb'))

# cloth_inital1, cloth_inter1, cloth_final1, action1,  = raw_data
# plt.scatter(np.array(cloth_inter1[1])[:, 0], np.array(cloth_inter1[1])[:,1])

# cloth_inital1, cloth_inter1, cloth_final1, action1,  = raw_data_work
# plt.scatter(np.array(cloth_inter1[1])[:, 0], np.array(cloth_inter1[1])[:,1])

# cloth_initial, action = raw_data
# cloth_initial = np.array(cloth_initial[1])
# plt.scatter(cloth_initial[:, 0], cloth_initial[:,1]+1)
# plt.scatter(cloth_initial2[:, 0], cloth_initial2[:,1])
# plt.scatter(action[0], action[1])

# parent_dir = '/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/FINAL_MODELS/standard_2D_not_subsampled_epochs=250_batch=100_workers=4_1687986938/cma_evaluations/standard_300_1693853952/raw'
# filename = 'tl13_c0_4368610321526353822_pid17277.pkl'
# raw_data_seed = pickle.load(open(osp.join(parent_dir, filename), 'rb'))
# cloth_initial_seed = np.array(raw_data_seed['sim_info']['info']['cloth_initial'][1])
# plt.scatter(cloth_initial_seed[:, 0], cloth_initial_seed[:,1])


# %%
clipping_thres = 0.028
action = [-0.79550357, -0.34619188,  0.29715349, -0.69629126]
grasp_loc = np.array([action[0]*0.44, action[1]*1.05])
print(grasp_loc)
dist = np.linalg.norm(cloth_initial[:, 0:2] - grasp_loc, axis=1)
# * if no points on the blanket are within 2.8 cm of the grasp location, clip
is_on_cloth = np.any(np.array(dist) < clipping_thres)
is_on_cloth


# %%
# raw_data2 = pickle.load(open('/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/FINAL_MODELS/standard_2D_not_subsampled_epochs=250_batch=100_workers=4_1687986938/cma_evaluations/all_targets_1k_standard_300_1688146883/raw/tl13_c0_18178056716270716263_pid30862.pkl', 'rb'))
raw_data2 = pickle.load(open('/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/FINAL_MODELS/standard_2D_not_subsampled_epochs=250_batch=100_workers=4_1687986938/cma_evaluations/multiple_standard_300_1689000438/raw/tl10_c0_10408118586461862978_pid18825.pkl', 'rb'))
cloth_initial2 = np.array(raw_data2['sim_info']['info']['cloth_initial'][1])


# %%
data_dir2 = '/home/kpputhuveetil/git/robe/robust-body-exposure/'
# test1 = 'test_data_intermediate.pkl'
# test2 = 'test_data_intermediate.pkl2'
# test3 = 'test_data_intermediate.pkl3'
# test_no_recover_action = 'test_data_intermediate.pkl4'

test1 = 'not_working_test_data1695496547.88999.pkl'
test2 = 'not_working_test_data1695496313.1759553.pkl'
test3 = 'not_working_test_data_intermediate3.pkl'
test_no_recover_action = 'not_working_test_data_intermediate4.pkl'

raw_data1 = pickle.load(open(osp.join(data_dir2, test1), 'rb'))
cloth_inital1, cloth_inter1, cloth_final1, action1,  = raw_data1
plt.scatter(np.array(cloth_inter1[1])[:, 0], np.array(cloth_inter1[1])[:,1])

raw_data2 = pickle.load(open(osp.join(data_dir2, test2), 'rb'))
cloth_inital2, cloth_inter2, cloth_final2, action2,  = raw_data2
plt.scatter(np.array(cloth_inter2[1])[:, 0], np.array(cloth_inter2[1])[:,1])

# raw_data3 = pickle.load(open(osp.join(data_dir2, test3), 'rb'))
# cloth_inital3, cloth_inter3, cloth_final3, action3,  = raw_data3
# plt.scatter(np.array(cloth_inter3[1])[:, 0], np.array(cloth_inter3[1])[:,1])

# raw_data_no_recover = pickle.load(open(osp.join(data_dir2, test_no_recover_action), 'rb'))
# cloth_inital4, cloth_inter4, cloth_final4, action4,  = raw_data_no_recover
# plt.scatter(np.array(cloth_inter4[1])[:, 0], np.array(cloth_inter4[1])[:,1])
# #


# %%
