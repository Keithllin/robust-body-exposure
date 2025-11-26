import sys, argparse, multiprocessing, time, os, math
import numpy as np
import pickle, pathlib
import os.path as osp
import random
import glob
# import pull_random_seeds
from pathlib import Path

uncover_model_path = '/home/kpputhuveetil/git/robe/robust-body-exposure/trained_models/FINAL_MODELS/standard_2D_not_subsampled_epochs=250_batch=100_workers=4_1687986938'
eval_dir_name = 'cma_evaluations'
threshold = 0.745

search_dir = osp.join(uncover_model_path, eval_dir_name)

eval_conditions = ['TL_[2, 4, 5, 8, 10, 11, 12, 13, 14, 15]_Uncover_Evals_Train_New_Grasp_Parsed']

data_path = osp.join(uncover_model_path, eval_dir_name, eval_conditions[0], 'raw/')
filenames = list(Path(data_path).glob('*.pkl'))

def sample_action(env):
    return env.action_space.sample()

def set_seed():
    seed = random.sample(seed_list, 1)[0]
    seed_list.remove(seed)
    return seed

def find(seed):
    for eval_condition in eval_conditions:
        path = Path(uncover_model_path +'/' + eval_dir_name + '/' + eval_condition + '/raw/')
        filenames = path.glob('*.pkl')
        for f in filenames:
            seed_f = f.name.split('_')[2]
            if seed_f == str(seed):
                return f
    raise Exception(f"Could not find seed file: {repr(seed)}")

def gnn_data_collect(env_name, i, filename, seed, raw_data):
    import gym
    from gym.utils import seeding
    from learn import make_env
    from assistive_gym.envs.bu_gnn_util import scale_action, check_grasp_on_cloth

    coop = 'Human' in env_name
    seed_path = ''
    target = 0
    on_grasp = False
    uncover_action = []
    recover_action = []
    filename = []
    recover = True

    # set up seed
    if recover:
        # seed = set_seed()
        #seed_path = open(find(seed), 'rb')
        #raw_data = pickle.load(seed_path)
        target = raw_data['target_limb_code']
        uncover_action = raw_data['uncover_action']
        cloth_initial_dc = np.array(raw_data['info']['cloth_initial'][1])
        cloth_intermediate_dc = np.array(raw_data['info']['cloth_final'][1])
    else:
        seed = seeding.create_seed()

    # create environment
    env = make_env(env_name, coop=coop, seed=seed)
    env.set_env_variations(
        collect_data = True,
        blanket_pose_var = False,
        high_pose_var = False,
        body_shape_var = False)

    env.set_singulate(True)
    env.set_target_limb_code(target)
    env.set_recover(recover)
    env.set_seed_val(seed)

    done = False
    observation = env.reset()
    pid = os.getpid()

    # set up actions
    if not recover:
        uncover_action = sample_action(env)

    cloth_initial_sim, cloth_intermediate_sim, execute_uncover_action = env.uncover_step(uncover_action)

    if not execute_uncover_action:
        return [i, filename, pid]

    if recover:
        while not on_grasp:
            recover_action = sample_action(env)
            _, on_grasp = check_grasp_on_cloth(scale_action(recover_action), np.array(cloth_intermediate_sim[1]))

    cloth_final_sim, execute_recover_action = env.recover_step(recover_action)
    observation, uncover_reward, recover_reward, done, info = env.get_info()

    if not recover:
        recover_action = []

    filename = f"c_{target}_{seed}_{int(time.time()*1000)}"
    with open(osp.join(pkl_loc, filename +".pkl"),"wb") as f:
        pickle.dump({
            "recovering":recover,
            "observation":observation,
            "info":info,
            "uncover_action":uncover_action,
            "recover_action":recover_action}, f)
    output = [i, filename, pid]
    del env

    return output

def counter_callback(output):
    global counter
    counter += 1
    print(f"{counter} - Trial Completed: {output[0]}, Worker: {os.getpid()}, Filename: {output[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data collection for gnn training')
    parser.add_argument('--env', default='RobeReversible-v1')
    parser.add_argument('--num_seeds', type=int, default=100)
    parser.add_argument('--rollouts', type=int, default=30000)
    parser.add_argument('--target_limb_list', type=str, default='2, 4, 5, 8, 10, 11, 12, 13, 14, 15')
    args = parser.parse_args()

    target_limb_list = [int(item) for item in args.target_limb_list.split(',')]
    assert args.rollouts % args.num_seeds == 0
    assert (args.rollouts // args.num_seeds) % len(target_limb_list) == 0

    #seed_list = pull_random_seeds.random_seeds(args.num_seeds, target_limb_list)

    current_dir = os.getcwd()
    recover = True

    recover_string = 'Uncover_Data'
    if recover:
        recover_string = 'Recover_Data'

    variation_type = f'TL_{args.target_limb_list}_{recover_string}_{args.num_seeds}_seeds_{args.rollouts}_states4' # for uncovering states are the random actions, for recovering states are = num_seeds
    pkl_loc = os.path.join(current_dir, "DATASETS",recover_string, variation_type, 'raw')
    pathlib.Path(pkl_loc).mkdir(parents=True, exist_ok=True)
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    counter = 0
    num_processes = 100
    trials = args.rollouts
    counter = 0
    result_objs = []

    dataset_path = '/home/kpputhuveetil/git/robe/robust-body-exposure/DATASETS/Recover_Data/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Recover_Data_100_seeds_30000_states3/raw'
    filenames_recover = list(Path(dataset_path).glob('*.pkl'))

    processed = {}
    for filename in filenames_recover:
        target = int(filename.name.split('_')[1])
        seed = int(filename.name.split('_')[2])

        if (target, seed) not in processed:
            processed[(target, seed)] = 0
        processed[(target, seed)] += 1

    filenames_iterated = []
    for filename in filenames * 30:
        target = int(filename.name.split('_')[0][2:])
        seed = int(filename.name.split('_')[2])
        if (target, seed) in processed and processed[(target, seed)] > 0:
            processed[(target, seed)] -= 1
            continue
        filenames_iterated.append(filename)
    trials = len(filenames_iterated)
    print(trials)

    filename_index = 0
    for j in range(math.ceil(trials/num_processes)):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(min(num_processes, trials - filename_index)):
                filename = filenames_iterated[filename_index]
                filename_index += 1
                with open(filename, 'rb') as f:
                    raw_data = pickle.load(f)
                    seed = int(filename.name.split('_')[2])
                result = pool.apply_async(gnn_data_collect, args = (args.env, i, filename.name, seed, raw_data), callback=counter_callback)
                result_objs.append(result)
            results = [result.get() for result in result_objs]