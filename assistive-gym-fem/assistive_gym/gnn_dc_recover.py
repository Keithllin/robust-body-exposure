import sys, argparse, multiprocessing, time, os, math
import numpy as np
import pickle, pathlib
import os.path as osp
import random
import glob
import pybullet as p
# import pull_random_seeds
from pathlib import Path

from assistive_gym.envs.bu_gnn_util import scale_action, check_grasp_on_cloth, get_body_points_from_obs, get_covered_status
from assistive_gym.envs.field_guided_policy import compute_field_guided_action

# Base directory of the repo
# REPO_ROOT = Path(__file__).resolve().parents[2]

# uncover_model_path = str(REPO_ROOT / 'trained_models/FINAL_MODELS/Recover/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Uncover_10000_states_New_Grasp_16000_epochs=250_batch=50_workers=4_1705905825')
uncover_model_path = '/mnt/data/MudkipUsersSu2025/kpputhuveetil/git/robe/robust-body-exposure_unstable/trained_models/FINAL_MODELS/Recover/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Uncover_10000_states_New_Grasp_16000_epochs=250_batch=50_workers=4_1705905825'

eval_dir_name = 'cma_evaluations'
threshold = 0.745

search_dir = osp.join(uncover_model_path, eval_dir_name)

eval_conditions = ['TL_All_Train_1k_states_RandomSearch_Opt_17074201118766']


def compute_fscore_uncover_local(initial_covered_status, final_covered_status):
    targ_uncov = 0
    nontarg_uncov = 0
    targ_cov = 0
    total_nontarg = 0

    for i in range(len(final_covered_status)):
        bod_point_type = final_covered_status[i][0]
        is_covered = final_covered_status[i][1]
        is_initially_covered = initial_covered_status[i][1]
        if bod_point_type == 1:
            if is_covered:
                targ_cov += 1
            else:
                targ_uncov += 1
        elif bod_point_type == 0 and is_initially_covered:
            total_nontarg += 1
            if not is_covered:
                nontarg_uncov += 1

    total_targ = targ_cov + targ_uncov
    if (total_targ + total_nontarg) == 0:
        return 0.0

    weight = total_targ / (total_targ + total_nontarg)
    penalties = []
    for i in range(1, nontarg_uncov + 1):
        penalty = i * weight
        penalties.append(penalty if penalty <= 1 else 1)

    tp = targ_uncov
    fp = np.sum(penalties)
    fn = targ_cov
    denom = tp + 0.5 * (fp + fn)
    if denom == 0:
        return 0.0
    return tp / denom


def compute_uncover_f1_from_raw(raw_data):
    info = raw_data.get('sim_info', {}).get('info', raw_data.get('info', {}))
    cloth_initial = np.array(info['cloth_initial'][1])
    cloth_final = np.array(info['cloth_final'][1])

    target = int(raw_data.get('target_limb_code', info['target_limb_code']))
    human_pose = np.reshape(raw_data['human_pose'], (-1, 2))
    body_info = info.get('human_body_info', None)
    all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target, body_info=body_info)

    initial_status = get_covered_status(all_body_points, np.delete(cloth_initial, 2, axis=1))
    final_status = get_covered_status(all_body_points, np.delete(cloth_final, 2, axis=1))
    return compute_fscore_uncover_local(initial_status, final_status)


def build_recover_reward_field(cloth_intermediate_3d, all_body_points, sigma_target=0.08, sigma_nontarget=0.06):
    """
    Build a recover-phase reward field for field-guided action selection.
    Negative = bad areas to fix (used by compute_field_guided_action thresholding).
    """
    cloth = np.asarray(cloth_intermediate_3d)
    cloth_xy = cloth[:, :2]

    covered_status = np.array(get_covered_status(all_body_points, cloth_xy))
    body_xy = np.asarray(all_body_points)[:, :2]

    target_uncovered = body_xy[(covered_status[:, 0] == 1) & (covered_status[:, 1] == 0)]
    nontarget_covered = body_xy[(covered_status[:, 0] == 0) & (covered_status[:, 1] == 1)]

    reward_field = np.zeros(len(cloth), dtype=np.float32)

    if len(target_uncovered) > 0:
        d_t = np.linalg.norm(cloth_xy[:, None, :] - target_uncovered[None, :, :], axis=2)
        min_dt = np.min(d_t, axis=1)
        target_term = np.exp(-(min_dt ** 2) / (2 * sigma_target ** 2))
        reward_field -= target_term.astype(np.float32)

    if len(nontarget_covered) > 0:
        d_nt = np.linalg.norm(cloth_xy[:, None, :] - nontarget_covered[None, :, :], axis=2)
        min_dnt = np.min(d_nt, axis=1)
        nontarget_term = 0.6 * np.exp(-(min_dnt ** 2) / (2 * sigma_nontarget ** 2))
        reward_field += nontarget_term.astype(np.float32)

    return reward_field


def sample_field_guided_recover_action(env, cloth_intermediate_sim, all_body_points, debug_field=False, debug_log=False):
    cloth_positions = np.asarray(cloth_intermediate_sim[1])
    reward_field = build_recover_reward_field(cloth_positions, all_body_points)
    body_pts = np.asarray(all_body_points)
    target_pts = body_pts[body_pts[:, 2] == 1]
    target_center = np.mean(target_pts[:, :3], axis=0) if len(target_pts) > 0 else np.mean(body_pts[:, :3], axis=0)

    if debug_log:
        neg_count = int(np.sum(reward_field < 0))
        print(
            f"[Field] reward[min/mean/max]={reward_field.min():.4f}/{reward_field.mean():.4f}/{reward_field.max():.4f}, "
            f"neg_vertices={neg_count}/{len(reward_field)}, target_points={len(target_pts)}"
        )

    if debug_field:
        try:
            p.removeAllUserDebugItems(physicsClientId=getattr(env, 'id', -1))
        except Exception:
            pass

    field_result = compute_field_guided_action(
        cloth_positions=cloth_positions,
        reward_field=reward_field,
        target_center=target_center,
        task_type='cover',
        threshold=0.0,
        step_size=0.16,
        debug_mode=debug_field,
        p_id=getattr(env, 'id', None),
    )
    if field_result is None:
        return None

    pick_pos, place_pos, _ = field_result
    action_world = np.array([pick_pos[0], pick_pos[1], place_pos[0], place_pos[1]], dtype=np.float32)
    action_policy = action_world / np.array([0.44, 1.05, 0.44, 1.05], dtype=np.float32)
    action_policy = np.clip(action_policy, -1.0, 1.0)

    if debug_log:
        print(
            f"[Field] pick_xy=({pick_pos[0]:.3f},{pick_pos[1]:.3f}) place_xy=({place_pos[0]:.3f},{place_pos[1]:.3f}) "
            f"action_policy={action_policy.round(3).tolist()}"
        )
    return action_policy

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

def gnn_data_collect(env_name, i, filename, seed, raw_data, action_mode='random', debug_field=False, render=False, debug_log=False):
    import gym
    from gym.utils import seeding
    from learn import make_env

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
        if debug_log:
            print(f"[Rollout {i}] source={filename}, seed={seed}, target_limb_code={target}, action_mode={action_mode}")
    else:
        seed = seeding.create_seed()

    # create environment
    env = make_env(env_name, coop=coop, seed=seed)
    try:
        if render:
            env.render()
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

        # Build target-aware body points from CURRENT env state (not source file)
        if isinstance(observation, (list, tuple)):
            human_pose_now = np.reshape(observation[0], (-1, 2))
        else:
            human_pose_now = np.reshape(observation, (-1, 2))
        all_body_points_now = np.array(
            get_body_points_from_obs(
                human_pose_now,
                target_limb_code=target,
                body_info=env.get_human_body_info(),
            )
        )
        if debug_log:
            n_t = int(np.sum(all_body_points_now[:, 2] == 1))
            n_nt = int(np.sum(all_body_points_now[:, 2] == 0))
            n_h = int(np.sum(all_body_points_now[:, 2] == -1))
            print(f"[Rollout {i}] body_points target/non-target/head = {n_t}/{n_nt}/{n_h}")

        # set up actions
        if not recover:
            uncover_action = sample_action(env)

        cloth_initial_sim, cloth_intermediate_sim, execute_uncover_action = env.uncover_step(uncover_action)

        if not execute_uncover_action:
            return [i, filename, pid]

        if recover:
            if action_mode == 'field_guided':
                recover_action = sample_field_guided_recover_action(
                    env,
                    cloth_intermediate_sim,
                    all_body_points_now,
                    debug_field=debug_field,
                    debug_log=debug_log,
                )
                if recover_action is not None:
                    _, on_grasp = check_grasp_on_cloth(scale_action(recover_action), np.array(cloth_intermediate_sim[1]))
                    if debug_log:
                        print(f"[Rollout {i}] field_guided grasp_on_cloth={on_grasp}")

            if recover_action is None or not on_grasp:
                while not on_grasp:
                    recover_action = sample_action(env)
                    _, on_grasp = check_grasp_on_cloth(scale_action(recover_action), np.array(cloth_intermediate_sim[1]))
                if debug_log and action_mode == 'field_guided':
                    print(f"[Rollout {i}] fallback=random (field action invalid/off-cloth)")

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
        return output
    finally:
        try:
            env.disconnect()
        except Exception:
            pass

def counter_callback(output):
    global counter
    counter += 1
    print(f"{counter} - Trial Completed: {output[0]}, Worker: {os.getpid()}, Filename: {output[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data collection for gnn training')
    parser.add_argument('--env', default='RobeReversible-v1')
    parser.add_argument('--num_seeds', type=int, default=100)
    parser.add_argument('--rollouts', type=int, default=10000)
    parser.add_argument('--target_limb_list', type=str, default='2, 4, 5, 8, 10, 11, 12, 13, 14, 15')
    parser.add_argument('--action_mode', type=str, default='random', choices=['random', 'field_guided'])
    parser.add_argument('--uncover_f1_threshold', type=float, default=0.745)
    parser.add_argument('--debug_field', action='store_true')
    parser.add_argument('--debug_log', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--uncover_model_path', type=str, default=uncover_model_path)
    parser.add_argument('--eval_condition', type=str, default=eval_conditions[0])
    parser.add_argument('--fallback_when_empty', type=str, default='topk', choices=['error', 'topk', 'all'])
    parser.add_argument('--fallback_topk', type=int, default=200)
    args = parser.parse_args()

    target_limb_list = [int(item) for item in args.target_limb_list.split(',')]
    if args.rollouts <= 0:
        raise ValueError("--rollouts must be > 0")
    if args.num_seeds <= 0:
        raise ValueError("--num_seeds must be > 0")

    #seed_list = pull_random_seeds.random_seeds(args.num_seeds, target_limb_list)

    current_dir = os.getcwd()
    recover = True

    recover_string = 'Uncover_Data'
    if recover:
        recover_string = 'Recover_Data'

    variation_type = f'TL_All_{recover_string}_{args.num_seeds}_seeds_{args.rollouts}_fix_nullgrasp' # for uncovering states are the random actions, for recovering states are = num_seeds
    pkl_loc = os.path.join(current_dir, "DATASETS",recover_string, variation_type, 'raw')
    pathlib.Path(pkl_loc).mkdir(parents=True, exist_ok=True)
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    counter = 0
    trials = args.rollouts
    use_visual_debug = args.debug_field or args.render
    num_processes = 1 if use_visual_debug else min(100, trials)
    if use_visual_debug:
        print('[Visual Debug] Enabling GUI render + single-process mode for visible PyBullet debug items.')
    counter = 0

    # Resolve source uncover-eval files
    data_path = osp.join(args.uncover_model_path, eval_dir_name, args.eval_condition, 'raw')
    source_files = list(Path(data_path).glob('*.pkl'))

    if len(source_files) == 0:
        robe_root = Path(__file__).resolve().parents[3]
        discovered = list(robe_root.glob(f"**/cma_evaluations/{args.eval_condition}/raw/*.pkl"))
        if len(discovered) > 0:
            source_files = discovered
            print(f"[DataPath] Default path empty. Auto-discovered {len(source_files)} files for eval_condition={args.eval_condition}.")
        else:
            raise RuntimeError(
                f"No source .pkl files found. Checked: {data_path} and auto-discovery under {robe_root} for eval_condition={args.eval_condition}."
            )

    eligible_records = []
    all_scored_records = []
    failed_records = 0
    for f in source_files:
        with open(f, 'rb') as rf:
            raw = pickle.load(rf)
        try:
            uncover_f1 = compute_uncover_f1_from_raw(raw)
            all_scored_records.append((f, raw, uncover_f1))
            if uncover_f1 >= args.uncover_f1_threshold:
                eligible_records.append((f, raw, uncover_f1))
        except Exception:
            failed_records += 1
            continue

    if len(all_scored_records) > 0:
        f1_vals = np.array([r[2] for r in all_scored_records])
        print(
            f"[F1 Filter] total={len(source_files)}, scored={len(all_scored_records)}, failed={failed_records}, "
            f"threshold={args.uncover_f1_threshold}, pass={len(eligible_records)}, "
            f"f1[min/mean/max]={f1_vals.min():.3f}/{f1_vals.mean():.3f}/{f1_vals.max():.3f}"
        )

    if len(eligible_records) == 0:
        if len(all_scored_records) == 0:
            raise RuntimeError("No source samples could be scored for uncover F1 (all failed).")

        if args.fallback_when_empty == 'error':
            raise RuntimeError(f"No source uncover states passed F1 threshold {args.uncover_f1_threshold}.")
        elif args.fallback_when_empty == 'all':
            print("[F1 Filter] No records passed threshold. Falling back to ALL scored records.")
            eligible_records = all_scored_records
        else:
            k = max(1, min(args.fallback_topk, len(all_scored_records)))
            eligible_records = sorted(all_scored_records, key=lambda x: x[2], reverse=True)[:k]
            print(f"[F1 Filter] No records passed threshold. Falling back to TOP-{k} scored records.")

    repeats = math.ceil(trials / len(eligible_records))
    eligible_records = (eligible_records * repeats)[:trials]
    filenames_iterated = iter(eligible_records)
    # dataset_path = '/home/kpputhuveetil/git/robe/robust-body-exposure/DATASETS/Recover_Data/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Recover_Data_100_seeds_30000_states3/raw'
    # filenames_recover = list(Path(dataset_path).glob('*.pkl'))

    # processed = {}
    # for filename in filenames_recover:
    #     target = int(filename.name.split('_')[1])
    #     seed = int(filename.name.split('_')[2])

    #     if (target, seed) not in processed:
    #         processed[(target, seed)] = 0
    #     processed[(target, seed)] += 1

    # filenames_iterated = []
    # for filename in filenames * 10:
    #     target = int(filename.name.split('_')[0][2:])
    #     seed = int(filename.name.split('_')[2])
    #     if (target, seed) in processed and processed[(target, seed)] > 0:
    #         processed[(target, seed)] -= 1
    #         continue
    #     filenames_iterated.append(filename)
    # trials = len(filenames_iterated)

    # filename_index = 0
    # for j in range(math.ceil(trials/num_processes)):
    #     with multiprocessing.Pool(processes=num_processes) as pool:
    #         for i in range(num_processes): #(min(num_processes, trials - filename_index)):
    #             filename = next(filenames_iterated) #filenames_iterated[filename_index]
    #             # filename_index += 1
    #             with open(filename, 'rb') as f:
    #                 raw_data = pickle.load(f)
    #                 seed = int(filename.name.split('_')[2])
    #             result = pool.apply_async(gnn_data_collect, args = (args.env, i, filename.name, seed, raw_data), callback=counter_callback)
    #             result_objs.append(result)
    #         results = [result.get() for result in result_objs]

    if use_visual_debug:
        for i in range(trials):
            filename, raw_data, uncover_f1 = next(filenames_iterated)
            seed = int(filename.name.split('_')[2])
            output = gnn_data_collect(
                args.env,
                i,
                filename.name,
                seed,
                raw_data,
                args.action_mode,
                args.debug_field,
                use_visual_debug,
                args.debug_log,
            )
            counter_callback(output)
    else:
        for j in range(math.ceil(trials/num_processes)):
            batch_size = min(num_processes, trials - j * num_processes)
            with multiprocessing.Pool(processes=batch_size) as pool:
                result_objs = []
                for i in range(batch_size):
                    filename, raw_data, uncover_f1 = next(filenames_iterated)
                    seed = int(filename.name.split('_')[2])
                    result = pool.apply_async(
                        gnn_data_collect,
                        args=(args.env, i, filename.name, seed, raw_data, args.action_mode, args.debug_field, use_visual_debug, args.debug_log),
                        callback=counter_callback,
                    )
                    result_objs.append(result)
                results = [result.get() for result in result_objs]