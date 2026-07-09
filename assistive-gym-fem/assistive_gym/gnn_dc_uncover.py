import sys, argparse, multiprocessing, time, os, math
import numpy as np
import pickle, pathlib
import os.path as osp
import random
import glob
from pathlib import Path
import gym
from gym.utils import seeding
from learn import make_env
from assistive_gym.envs.bu_gnn_util import scale_action, check_grasp_on_cloth

def sample_action(env):
    return env.action_space.sample()

def gnn_data_collect(env_name, i, target, post_release_steps=50, quiet_settle=False,
                     quiet_max_steps=20, quiet_speed_threshold=0.15):
    coop = 'Human' in env_name
    on_grasp = False
    uncover_action = []
    recover_action = []
    filename = []
    recover = False

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

    on_grasp = False

    done = False
    observation = env.reset()
    pid = os.getpid()

    # set up actions
    uncover_action = []
    recover_action = [0, 0, 0, 0]

    cloth_initial = env.get_cloth_state()

    while not on_grasp:
        uncover_action = sample_action(env)
        _, on_grasp = check_grasp_on_cloth(scale_action(uncover_action), np.array(cloth_initial))

    if hasattr(env, 'set_release_sim_steps'):
        env.set_release_sim_steps(post_release_steps=int(post_release_steps))
    if hasattr(env, 'set_release_quiet_settle'):
        env.set_release_quiet_settle(
            enabled=bool(quiet_settle),
            speed_threshold=float(quiet_speed_threshold),
            max_steps=int(quiet_max_steps),
        )

    cloth_initial_sim, cloth_intermediate_sim, execute_uncover_action = env.uncover_step(uncover_action)

    if not execute_uncover_action:
        return [-1, filename, target]

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
            "recover_action":recover_action,
            "data_collection_info": {
                "post_release_steps": int(post_release_steps),
                "quiet_settle": bool(quiet_settle),
                "quiet_max_steps": int(quiet_max_steps),
                "quiet_speed_threshold": float(quiet_speed_threshold),
            },
        }, f)

    output = [i, target, pid]

    del env

    return output

def counter_callback(output):
    global counter
    counter += 1
    print(f"{counter} - Trial Completed: {output[0]}, Worker: {output[2]}, Target: {output[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data collection for gnn training')
    parser.add_argument('--env', default='RobeReversible-v1')
    parser.add_argument('--num_seeds', type=int, default=100)
    parser.add_argument('--rollouts', type=int, default=10000)
    parser.add_argument('--target_limb_list', type=str, default='2, 4, 5, 8, 10, 11, 12, 13, 14, 15')
    parser.add_argument('--post-release-steps', type=int, default=50,
                        help='Post-release settle steps after uncover release.')
    parser.add_argument('--quiet-settle', action='store_true',
                        help='After min post-release steps, keep settling until cloth is quiet.')
    parser.add_argument('--quiet-max-steps', type=int, default=20,
                        help='Maximum post-release settle steps when --quiet-settle is enabled.')
    parser.add_argument('--quiet-speed-threshold', type=float, default=0.15,
                        help='P95 cloth vertex speed threshold for quiet settle.')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='Parallel PyBullet workers for data collection.')
    parser.add_argument('--output-dataset-dir', type=str, default=None,
                        help='Override dataset output directory (parent of raw/).')
    args = parser.parse_args()

    target_limb_list = [int(item.strip()) for item in args.target_limb_list.split(',') if item.strip()]
    if args.rollouts <= 0:
        raise ValueError("--rollouts must be > 0")
    if args.num_processes <= 0:
        raise ValueError("--num-processes must be > 0")

    repeats = math.ceil(args.rollouts / len(target_limb_list))
    target_list = (target_limb_list * repeats)[:args.rollouts]

    current_dir = os.getcwd()
    recover = False

    recover_string = 'Uncover_Data'
    if recover:
        recover_string = 'Recover_Data'

    variation_type = f'TL_{args.target_limb_list}_{recover_string}_{args.rollouts}_states_New_Grasp'
    if args.output_dataset_dir:
        output_dir = str(Path(args.output_dataset_dir).expanduser().resolve())
    else:
        output_dir = str(Path(current_dir) / 'DATASETS' / recover_string / variation_type)
    pkl_loc = osp.join(output_dir, 'raw')
    pathlib.Path(pkl_loc).mkdir(parents=True, exist_ok=True)

    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    counter = 0
    num_processes = min(args.num_processes, args.rollouts)
    trials = args.rollouts
    print(
        f'[Release Steps] post_release_steps={args.post_release_steps} '
        f'quiet_settle={args.quiet_settle} quiet_max_steps={args.quiet_max_steps} '
        f'num_processes={num_processes} rollouts={trials}'
    )
    print(f'[Output] raw dir: {pkl_loc}')

    target_iterated = iter(target_list)
    trials_started = 0

    for j in range(math.ceil(trials/num_processes)):
        batch_size = min(num_processes, trials - trials_started)
        with multiprocessing.Pool(processes=batch_size) as pool:
            result_objs = []
            for i in range(batch_size):
                target = next(target_iterated)
                result = pool.apply_async(
                    gnn_data_collect,
                    args=(
                        args.env,
                        trials_started + i,
                        target,
                        args.post_release_steps,
                        args.quiet_settle,
                        args.quiet_max_steps,
                        args.quiet_speed_threshold,
                    ),
                    callback=counter_callback,
                )
                result_objs.append(result)
            results = [result.get() for result in result_objs]
        trials_started += batch_size
