import sys, argparse, multiprocessing, time, os, math, json
import numpy as np
import pickle, pathlib
import os.path as osp
import random
import glob
import pybullet as p
from collections import Counter, defaultdict
# import pull_random_seeds
from pathlib import Path

from assistive_gym.envs.bu_gnn_util import scale_action, check_grasp_on_cloth, get_body_points_from_obs, get_covered_status
from assistive_gym.envs.field_guided_policy import compute_field_guided_action

# Base directory of the repository.  Callers can replace this model path
# when evaluating a particular checkpoint.
REPO_ROOT = Path(__file__).resolve().parents[2]
uncover_model_path = str(
    REPO_ROOT
    / 'trained_models/FINAL_MODELS/Recover/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Uncover_10000_states_New_Grasp_16000_epochs=250_batch=50_workers=4_1705905825'
)

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


def parse_seed_from_source_filename(filename):
    name = Path(filename).name
    parts = name.split('_')
    if len(parts) >= 3 and parts[0].startswith('tl') and parts[2].isdigit():
        return int(parts[2])
    resolved = Path(filename).resolve().name
    if resolved != name:
        return parse_seed_from_source_filename(resolved)
    raise ValueError(f"Cannot parse seed from source filename: {name}")


def load_source_manifest(manifest_path, pool_root):
    records = []
    pool_root = Path(pool_root).expanduser().resolve()
    with open(manifest_path, 'r') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rel_path = row.get('path')
            filename = row.get('filename')
            if rel_path:
                src_path = (pool_root / rel_path).resolve()
            else:
                src_path = (pool_root / 'raw' / filename).resolve()
            with open(src_path, 'rb') as rf:
                raw = pickle.load(rf)
            records.append((
                src_path,
                raw,
                float(row.get('f1', compute_uncover_f1_from_raw(raw))),
                row.get('quality_band', 'unknown'),
            ))
    return records


def set_collection_rng(collection_seed, env=None):
    """Seed only recover-action RNG. Do not call env.seed() here; sim uses source seed."""
    seed = int(collection_seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    if env is not None:
        try:
            env.action_space.seed(seed)
        except Exception:
            pass


def count_saved_for_run(raw_dir, run_id):
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return 0
    prefix = f"{run_id}_"
    return sum(1 for path in raw_dir.glob('*.pkl') if path.name.startswith(prefix))


def load_existing_collection_seeds(raw_dir, run_id):
    seeds = set()
    raw_dir = Path(raw_dir)
    prefix = f"{run_id}_"
    for path in raw_dir.glob('*.pkl'):
        if not path.name.startswith(prefix):
            continue
        parts = path.stem.split('_')
        if parts and parts[-1].isdigit():
            seeds.add(int(parts[-1]))
    return seeds


def cycle_records(records):
    while True:
        for record in records:
            yield record


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


def policy_action_from_world(action_world):
    scale = np.array([0.44, 1.05, 0.44, 1.05], dtype=np.float32)
    return np.clip(np.asarray(action_world, dtype=np.float32) / scale, -1.0, 1.0)


def compute_target_uncovered_points(all_body_points, cloth_intermediate_3d):
    cloth_xy = np.asarray(cloth_intermediate_3d, dtype=np.float32)[:, :2]
    covered_status = np.asarray(get_covered_status(all_body_points, cloth_xy))
    body_points = np.asarray(all_body_points, dtype=np.float32)
    return body_points[(covered_status[:, 0] == 1) & (covered_status[:, 1] == 0)]


def build_overlap_candidates(
    cloth_initial_3d,
    cloth_intermediate_3d,
    all_body_points,
    grid_size=0.05,
    min_count=3,
    min_z_range=0.025,
    min_top_sep=0.015,
    relevance_sigma=0.16,
):
    cloth_initial = np.asarray(cloth_initial_3d, dtype=np.float32)
    cloth = np.asarray(cloth_intermediate_3d, dtype=np.float32)
    if cloth_initial.shape != cloth.shape:
        raise RuntimeError(f"Initial/intermediate cloth shape mismatch: {cloth_initial.shape} != {cloth.shape}")

    xy = cloth[:, :2]
    xy_min = np.min(xy, axis=0)
    cell_idx = np.floor((xy - xy_min[None, :]) / float(grid_size)).astype(np.int64)
    groups = defaultdict(list)
    for idx, cell in enumerate(cell_idx):
        groups[(int(cell[0]), int(cell[1]))].append(idx)

    target_uncovered = compute_target_uncovered_points(all_body_points, cloth)
    candidates = []
    for cell, indices in groups.items():
        if len(indices) < int(min_count):
            continue

        local = cloth[indices]
        local_z = local[:, 2]
        local_z_range = float(np.max(local_z) - np.min(local_z))
        if local_z_range < float(min_z_range):
            continue

        top_local_rank = int(np.argmax(local_z))
        pick_vertex_idx = int(indices[top_local_rank])
        top_layer_separation = float(local_z[top_local_rank] - np.median(local_z))
        if top_layer_separation < float(min_top_sep):
            continue

        pick_xy = cloth[pick_vertex_idx, :2]
        if len(target_uncovered) > 0:
            dists = np.linalg.norm(target_uncovered[:, :2] - pick_xy[None, :], axis=1)
            min_dist = float(np.min(dists))
            recover_relevance = float(np.exp(-(min_dist ** 2) / (2.0 * float(relevance_sigma) ** 2)))
        else:
            min_dist = float('nan')
            recover_relevance = 1.0

        overlap_score = float(
            local_z_range
            * math.log1p(len(indices))
            * max(top_layer_separation, 0.0)
            * max(recover_relevance, 1e-6)
        )
        if overlap_score <= 0:
            continue

        candidates.append({
            'cell': cell,
            'indices': [int(v) for v in indices],
            'pick_vertex_idx': pick_vertex_idx,
            'local_point_count': int(len(indices)),
            'local_z_range': local_z_range,
            'top_layer_separation': top_layer_separation,
            'recover_relevance': recover_relevance,
            'nearest_target_uncovered_dist': min_dist,
            'overlap_score': overlap_score,
        })

    return candidates


def sample_overlap_guided_recover_action(
    cloth_initial_sim,
    cloth_intermediate_sim,
    all_body_points,
    grid_size=0.05,
    min_count=3,
    min_z_range=0.025,
    min_top_sep=0.015,
    initial_place_prob=0.7,
    debug_log=False,
):
    cloth_initial = np.asarray(cloth_initial_sim[1], dtype=np.float32)
    cloth_intermediate = np.asarray(cloth_intermediate_sim[1], dtype=np.float32)
    candidates = build_overlap_candidates(
        cloth_initial,
        cloth_intermediate,
        all_body_points,
        grid_size=grid_size,
        min_count=min_count,
        min_z_range=min_z_range,
        min_top_sep=min_top_sep,
    )
    if len(candidates) == 0:
        return None, {'skip_reason': 'no_overlap_candidate', 'num_overlap_candidates': 0}

    scores = np.asarray([candidate['overlap_score'] for candidate in candidates], dtype=np.float64)
    probs = scores / np.sum(scores)

    # Try several high-scoring stochastic candidates before giving up on grasp validity.
    for _ in range(min(10, len(candidates))):
        candidate = candidates[int(np.random.choice(len(candidates), p=probs))]
        pick_vertex_idx = int(candidate['pick_vertex_idx'])
        pick_xy = cloth_intermediate[pick_vertex_idx, :2] + np.random.normal(0.0, 0.005, size=2)

        if random.random() < float(initial_place_prob):
            place_xy = cloth_initial[pick_vertex_idx, :2].copy()
            place_policy_used = 'initial_vertex'
            if np.linalg.norm(place_xy - pick_xy) < 0.04:
                place_policy_used = 'random_radial_short_initial'
        else:
            place_policy_used = 'random_radial'

        if place_policy_used != 'initial_vertex':
            theta = random.uniform(0.0, 2.0 * math.pi)
            length = random.uniform(0.08, 0.30)
            place_xy = pick_xy + length * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)

        action_world = np.array([pick_xy[0], pick_xy[1], place_xy[0], place_xy[1]], dtype=np.float32)
        action_policy = policy_action_from_world(action_world)
        _, on_grasp = check_grasp_on_cloth(scale_action(action_policy), cloth_intermediate)
        if not on_grasp:
            continue

        debug = {
            'action_mode': 'overlap_guided',
            'pick_vertex_idx': pick_vertex_idx,
            'overlap_cell': [int(candidate['cell'][0]), int(candidate['cell'][1])],
            'local_point_count': int(candidate['local_point_count']),
            'local_z_range': float(candidate['local_z_range']),
            'top_layer_separation': float(candidate['top_layer_separation']),
            'overlap_score': float(candidate['overlap_score']),
            'recover_relevance': float(candidate['recover_relevance']),
            'nearest_target_uncovered_dist': float(candidate['nearest_target_uncovered_dist']),
            'place_policy_used': place_policy_used,
            'num_overlap_candidates': int(len(candidates)),
            'action_world': action_world.tolist(),
            'action_policy': action_policy.tolist(),
        }
        if debug_log:
            print(
                "[Overlap] pick_idx={pick_vertex_idx} cell={overlap_cell} "
                "count={local_point_count} z_range={local_z_range:.4f} "
                "top_sep={top_layer_separation:.4f} relevance={recover_relevance:.4f} "
                "place={place_policy_used}".format(**debug)
            )
        return action_policy, debug

    return None, {'skip_reason': 'overlap_pick_off_cloth', 'num_overlap_candidates': int(len(candidates))}


def save_overlap_debug_image(image_path, cloth_initial, cloth_intermediate, cloth_final, data_info):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[OverlapImage] Failed to import matplotlib: {exc}")
        return

    cloth_initial = np.asarray(cloth_initial, dtype=np.float32)
    cloth_intermediate = np.asarray(cloth_intermediate, dtype=np.float32)
    cloth_final = np.asarray(cloth_final, dtype=np.float32)
    action_world = np.asarray(data_info.get('action_world', [np.nan] * 4), dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    states = [
        ('initial', cloth_initial),
        ('intermediate', cloth_intermediate),
        ('final', cloth_final),
    ]
    for ax, (title, cloth) in zip(axes, states):
        ax.scatter(cloth[:, 0], cloth[:, 1], c=cloth[:, 2], s=4, cmap='viridis', alpha=0.75)
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)
    axes[1].scatter([action_world[0]], [action_world[1]], c='red', s=60, marker='x', label='pick')
    axes[1].scatter([action_world[2]], [action_world[3]], c='orange', s=50, marker='o', label='place')
    axes[1].arrow(
        action_world[0],
        action_world[1],
        action_world[2] - action_world[0],
        action_world[3] - action_world[1],
        color='red',
        width=0.003,
        length_includes_head=True,
    )
    axes[1].legend(loc='best')
    fig.suptitle(
        "overlap_guided idx={idx} count={count} z_range={z:.3f} top_sep={sep:.3f} relevance={rel:.3f}".format(
            idx=data_info.get('pick_vertex_idx'),
            count=data_info.get('local_point_count'),
            z=float(data_info.get('local_z_range', float('nan'))),
            sep=float(data_info.get('top_layer_separation', float('nan'))),
            rel=float(data_info.get('recover_relevance', float('nan'))),
        )
    )
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(image_path, dpi=140)
    plt.close(fig)

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

def gnn_data_collect(
    env_name,
    i,
    filename,
    seed,
    raw_data,
    action_mode='random',
    debug_field=False,
    render=False,
    debug_log=False,
    overlap_grid_size=0.05,
    overlap_min_count=3,
    overlap_min_z_range=0.025,
    overlap_min_top_sep=0.015,
    save_overlap_images=False,
    overlap_image_dir=None,
    uncover_post_release_steps=50,
    recover_post_release_steps=50,
    collection_seed=None,
    run_id='recover',
    source_f1=None,
    quality_band=None,
):
    import gym
    from gym.utils import seeding
    from learn import make_env

    coop = 'Human' in env_name
    seed_path = ''
    target = 0
    on_grasp = False
    uncover_action = []
    recover_action = []
    source_filename = filename
    recover = True
    data_collection_info = {
        'action_mode': action_mode,
        'source_uncover_pkl': source_filename,
        'uncover_post_release_steps': int(uncover_post_release_steps),
        'recover_post_release_steps': int(recover_post_release_steps),
        'run_id': str(run_id),
    }
    if source_f1 is not None:
        data_collection_info['source_uncover_f1'] = float(source_f1)
    if quality_band is not None:
        data_collection_info['source_quality_band'] = str(quality_band)
    if collection_seed is not None:
        data_collection_info['collection_seed'] = int(collection_seed)

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
            print(f"[Rollout {i}] source={source_filename}, seed={seed}, target_limb_code={target}, action_mode={action_mode}")
    else:
        seed = seeding.create_seed()

    # create environment (make_env seeds env with source sim seed)
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

        if hasattr(env, 'set_release_sim_steps'):
            env.set_release_sim_steps(post_release_steps=int(uncover_post_release_steps))
        cloth_initial_sim, cloth_intermediate_sim, execute_uncover_action = env.uncover_step(uncover_action)

        if not execute_uncover_action:
            return {'status': 'skipped', 'reason': 'uncover_not_executed', 'i': i, 'filename': source_filename, 'pid': pid}

        if recover:
            if collection_seed is not None:
                set_collection_rng(collection_seed, env)

            if action_mode == 'overlap_guided':
                recover_action, overlap_info = sample_overlap_guided_recover_action(
                    cloth_initial_sim,
                    cloth_intermediate_sim,
                    all_body_points_now,
                    grid_size=overlap_grid_size,
                    min_count=overlap_min_count,
                    min_z_range=overlap_min_z_range,
                    min_top_sep=overlap_min_top_sep,
                    debug_log=debug_log,
                )
                data_collection_info.update(overlap_info)
                if recover_action is None:
                    return {
                        'status': 'skipped',
                        'reason': overlap_info.get('skip_reason', 'overlap_action_not_found'),
                        'i': i,
                        'filename': source_filename,
                        'pid': pid,
                        'target_limb_code': int(target),
                        'data_collection_info': data_collection_info,
                    }
                _, on_grasp = check_grasp_on_cloth(scale_action(recover_action), np.array(cloth_intermediate_sim[1]))
                if debug_log:
                    print(f"[Rollout {i}] overlap_guided grasp_on_cloth={on_grasp}")

            elif action_mode == 'field_guided':
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
                if action_mode == 'overlap_guided':
                    return {
                        'status': 'skipped',
                        'reason': 'overlap_action_off_cloth',
                        'i': i,
                        'filename': source_filename,
                        'pid': pid,
                        'target_limb_code': int(target),
                        'data_collection_info': data_collection_info,
                    }
                while not on_grasp:
                    recover_action = sample_action(env)
                    _, on_grasp = check_grasp_on_cloth(scale_action(recover_action), np.array(cloth_intermediate_sim[1]))
                if debug_log and action_mode == 'field_guided':
                    print(f"[Rollout {i}] fallback=random (field action invalid/off-cloth)")

        if hasattr(env, 'set_release_sim_steps'):
            env.set_release_sim_steps(post_release_steps=int(recover_post_release_steps))
        cloth_final_sim, execute_recover_action = env.recover_step(recover_action)
        observation, uncover_reward, recover_reward, done, info = env.get_info()
        data_collection_info['execute_recover_action'] = bool(execute_recover_action)
        data_collection_info['target_limb_code'] = int(target)
        data_collection_info['seed'] = int(seed)
        try:
            anchor_idx = [int(v) for v in list(getattr(env, 'anchor_idx', []) or [])]
        except Exception:
            anchor_idx = list(info.get('anchor_idx', []) or [])
        data_collection_info['anchor_idx'] = anchor_idx
        data_collection_info['anchor_count'] = int(len(anchor_idx))
        if isinstance(info, dict) and 'anchor_idx' not in info:
            info['anchor_idx'] = anchor_idx
            info['anchor_count'] = int(len(anchor_idx))

        if not recover:
            recover_action = []

        if collection_seed is not None:
            filename = f"{run_id}_c_{target}_{seed}_{int(collection_seed)}"
        else:
            filename = f"c_{target}_{seed}_{int(time.time()*1000)}"
        if save_overlap_images and action_mode == 'overlap_guided' and overlap_image_dir is not None:
            image_path = osp.join(overlap_image_dir, filename + '.png')
            save_overlap_debug_image(
                image_path,
                cloth_initial_sim[1],
                cloth_intermediate_sim[1],
                cloth_final_sim[1],
                data_collection_info,
            )
            data_collection_info['debug_image'] = image_path

        with open(osp.join(pkl_loc, filename +".pkl"),"wb") as f:
            pickle.dump({
                "recovering":recover,
                "observation":observation,
                "info":info,
                "uncover_action":uncover_action,
                "recover_action":recover_action,
                "data_collection_info":data_collection_info}, f)
        output = {
            'status': 'saved',
            'i': i,
            'filename': filename,
            'pid': pid,
            'target_limb_code': int(target),
            'data_collection_info': data_collection_info,
        }
        return output
    finally:
        try:
            env.disconnect()
        except Exception:
            pass

def counter_callback(output):
    global counter, saved_counter, collection_results
    counter += 1
    if isinstance(output, dict):
        collection_results.append(output)
        if output.get('status') == 'saved':
            saved_counter += 1
        print(
            f"{counter} - Trial {output.get('status', 'unknown')}: {output.get('i')}, "
            f"Saved: {saved_counter}, Worker: {output.get('pid')}, "
            f"Filename: {output.get('filename')}, Reason: {output.get('reason', '')}"
        )
    else:
        saved_counter += 1
        collection_results.append({'status': 'saved', 'i': output[0], 'filename': output[1], 'pid': output[2]})
        print(f"{counter} - Trial Completed: {output[0]}, Saved: {saved_counter}, Worker: {os.getpid()}, Filename: {output[1]}")


def summarize_collection_outputs(outputs):
    saved = [out for out in outputs if out.get('status') == 'saved']
    skipped = [out for out in outputs if out.get('status') != 'saved']
    infos = [out.get('data_collection_info', {}) for out in saved]

    def numeric_values(field):
        vals = []
        for info in infos:
            if field not in info:
                continue
            try:
                value = float(info[field])
            except Exception:
                continue
            if np.isfinite(value):
                vals.append(value)
        return vals

    def mean_field(field):
        vals = numeric_values(field)
        return float(np.mean(vals)) if vals else float('nan')

    def median_field(field):
        vals = numeric_values(field)
        return float(np.median(vals)) if vals else float('nan')

    summary = {
        'attempted_count': int(len(outputs)),
        'saved_count': int(len(saved)),
        'skipped_count': int(len(skipped)),
        'skip_reasons': dict(Counter(out.get('reason', 'unknown') for out in skipped)),
        'saved_by_target_limb': dict(sorted(Counter(out.get('target_limb_code') for out in saved).items())),
        'saved_by_quality_band': dict(sorted(Counter(info.get('source_quality_band', 'unknown') for info in infos).items())),
        'collection_seed_count': len(set(info.get('collection_seed') for info in infos if info.get('collection_seed') is not None)),
        'local_z_range_mean': mean_field('local_z_range'),
        'local_z_range_median': median_field('local_z_range'),
        'top_layer_separation_mean': mean_field('top_layer_separation'),
        'top_layer_separation_median': median_field('top_layer_separation'),
        'local_point_count_mean': mean_field('local_point_count'),
        'overlap_score_mean': mean_field('overlap_score'),
        'recover_relevance_mean': mean_field('recover_relevance'),
        'execute_recover_rate': float(np.mean([bool(info.get('execute_recover_action')) for info in infos])) if infos else float('nan'),
        'place_policy_counts': dict(Counter(info.get('place_policy_used', 'unknown') for info in infos)),
    }
    return summary


def write_collection_summary(output_dir, summary, args):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        'summary': summary,
        'args': vars(args),
    }
    with open(osp.join(output_dir, 'collection_summary.json'), 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')

    lines = [
        '# Recover Data Collection Summary',
        '',
        f"- action_mode: `{args.action_mode}`",
        f"- uncover_post_release_steps: {getattr(args, 'uncover_post_release_steps', 50)}",
        f"- recover_post_release_steps: {getattr(args, 'recover_post_release_steps', 50)}",
        f"- attempted_count: {summary['attempted_count']}",
        f"- saved_count: {summary['saved_count']}",
        f"- skipped_count: {summary['skipped_count']}",
        f"- skip_reasons: `{summary['skip_reasons']}`",
        f"- saved_by_target_limb: `{summary['saved_by_target_limb']}`",
        f"- saved_by_quality_band: `{summary.get('saved_by_quality_band', {})}`",
        f"- collection_seed_count: {summary.get('collection_seed_count', 0)}",
        f"- local_z_range mean/median: {summary['local_z_range_mean']:.4f} / {summary['local_z_range_median']:.4f}",
        f"- top_layer_separation mean/median: {summary['top_layer_separation_mean']:.4f} / {summary['top_layer_separation_median']:.4f}",
        f"- local_point_count_mean: {summary['local_point_count_mean']:.2f}",
        f"- execute_recover_rate: {summary['execute_recover_rate']:.3f}",
        f"- place_policy_counts: `{summary['place_policy_counts']}`",
        '',
    ]
    with open(osp.join(output_dir, 'collection_summary.md'), 'w') as handle:
        handle.write('\n'.join(lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data collection for gnn training')
    parser.add_argument('--env', default='RobeReversible-v1')
    parser.add_argument('--num_seeds', type=int, default=100)
    parser.add_argument('--rollouts', type=int, default=10000)
    parser.add_argument('--target_limb_list', type=str, default='2, 4, 5, 8, 10, 11, 12, 13, 14, 15')
    parser.add_argument('--action_mode', type=str, default='random', choices=['random', 'field_guided', 'overlap_guided'])
    parser.add_argument('--uncover_f1_threshold', type=float, default=0.745)
    parser.add_argument('--debug_field', action='store_true')
    parser.add_argument('--debug_log', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--overlap_grid_size', type=float, default=0.05)
    parser.add_argument('--overlap_min_count', type=int, default=3)
    parser.add_argument('--overlap_min_z_range', type=float, default=0.025)
    parser.add_argument('--overlap_min_top_sep', type=float, default=0.015)
    parser.add_argument('--overlap_max_attempts', type=int, default=None,
                        help='Max source states to try for overlap_guided. Defaults to 5 * rollouts.')
    parser.add_argument('--save_overlap_images', action='store_true',
                        help='Save annotated overlap pick/place images for visual validation.')
    parser.add_argument('--overlap_image_limit', type=int, default=100,
                        help='Maximum overlap debug images to save.')
    parser.add_argument('--uncover_model_path', type=str, default=uncover_model_path)
    parser.add_argument('--eval_condition', type=str, default=eval_conditions[0])
    parser.add_argument('--balance-target-limbs', action='store_true')
    parser.add_argument('--fallback_when_empty', type=str, default='topk', choices=['error', 'topk', 'all'])
    parser.add_argument('--fallback_topk', type=int, default=200)
    parser.add_argument('--num-processes', type=int, default=4,
                        help='Parallel PyBullet workers for data collection. Ignored when --render or --debug_field is set.')
    parser.add_argument('--output-rollouts', type=int, default=None,
                        help='Rollouts value used in output dataset directory name. Defaults to --rollouts.')
    parser.add_argument('--uncover-post-release-steps', type=int, default=50,
                        help='Post-release settle steps after uncover release.')
    parser.add_argument('--recover-post-release-steps', type=int, default=50,
                        help='Post-release settle steps after recover release.')
    parser.add_argument('--output-dataset-dir', type=str, default=None,
                        help='Override dataset output directory (parent of raw/). Defaults to auto-generated DATASETS/Recover_Data/<variation_type>.')
    parser.add_argument('--run-id', type=str, default='recover',
                        help='Run identifier embedded in output PKL names and metadata.')
    parser.add_argument('--collection-seed-base', type=int, default=None,
                        help='Base seed for recover action RNG. Each rollout uses base + rollout_index.')
    parser.add_argument('--max-attempts', type=int, default=None,
                        help='Maximum source replay attempts. Defaults to max(rollouts, 3 * rollouts) for random mode.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume an interrupted run by counting existing PKLs for --run-id.')
    parser.add_argument('--source-manifest', type=str, default=None,
                        help='Optional source_manifest.jsonl from build_recover_source_pool.py.')
    args = parser.parse_args()

    target_limb_list = [int(item) for item in args.target_limb_list.split(',')]
    if args.rollouts <= 0:
        raise ValueError("--rollouts must be > 0")
    if args.num_seeds <= 0:
        raise ValueError("--num_seeds must be > 0")
    if args.num_processes <= 0:
        raise ValueError("--num-processes must be > 0")

    #seed_list = pull_random_seeds.random_seeds(args.num_seeds, target_limb_list)

    current_dir = os.getcwd()
    recover = True

    recover_string = 'Uncover_Data'
    if recover:
        recover_string = 'Recover_Data'

    action_tag = '' if args.action_mode == 'random' else f'_{args.action_mode}'
    rollouts_for_path = int(args.output_rollouts) if args.output_rollouts is not None else int(args.rollouts)
    variation_type = f'TL_All_{recover_string}_{args.num_seeds}_seeds_{rollouts_for_path}_fix_nullgrasp{action_tag}' # for uncovering states are the random actions, for recovering states are = num_seeds
    if args.output_dataset_dir:
        output_dir = str(Path(args.output_dataset_dir).expanduser().resolve())
    else:
        output_dir = str(Path(current_dir) / 'DATASETS' / recover_string / variation_type)
    pkl_loc = osp.join(output_dir, 'raw')
    pathlib.Path(pkl_loc).mkdir(parents=True, exist_ok=True)
    overlap_image_dir = osp.join(output_dir, 'overlap_debug_images')
    if args.save_overlap_images:
        pathlib.Path(overlap_image_dir).mkdir(parents=True, exist_ok=True)
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    counter = 0
    saved_counter = 0
    collection_results = []
    requested_rollouts = args.rollouts
    if args.max_attempts is not None:
        attempts_limit = int(args.max_attempts)
    elif args.action_mode == 'overlap_guided':
        attempts_limit = int(args.overlap_max_attempts) if args.overlap_max_attempts is not None else int(args.rollouts * 5)
    else:
        attempts_limit = max(int(args.rollouts), int(args.rollouts * 3))
    if args.action_mode == 'overlap_guided' and attempts_limit < requested_rollouts:
        raise ValueError("--overlap_max_attempts/--max-attempts must be >= --rollouts")
    if args.collection_seed_base is None:
        args.collection_seed_base = 1000000 if args.run_id == 'mudkip' else 2000000
    use_visual_debug = args.debug_field or args.render
    num_processes = 1 if use_visual_debug else min(args.num_processes, attempts_limit)
    if args.resume:
        saved_counter = count_saved_for_run(pkl_loc, args.run_id)
        print(f'[Resume] Found {saved_counter} existing saved PKLs for run_id={args.run_id}.')
    existing_collection_seeds = load_existing_collection_seeds(pkl_loc, args.run_id) if args.resume else set()
    if use_visual_debug:
        print('[Visual Debug] Enabling GUI render + single-process mode for visible PyBullet debug items.')
    else:
        print(f'[Parallel] Using {num_processes} worker process(es) (requested --num-processes={args.num_processes}).')
    print(
        f'[Release Steps] uncover_post_release_steps={args.uncover_post_release_steps} '
        f'recover_post_release_steps={args.recover_post_release_steps}'
    )
    print(f'[Output] raw dir: {pkl_loc}')
    print(f'[Run] run_id={args.run_id} collection_seed_base={args.collection_seed_base} max_attempts={attempts_limit}')
    counter = 0

    manifest_path = None
    pool_root = Path(args.uncover_model_path).expanduser().resolve()
    if args.source_manifest:
        manifest_path = Path(args.source_manifest).expanduser().resolve()
        if manifest_path.parent.name == args.eval_condition:
            pool_root = manifest_path.parents[2]
        else:
            pool_root = Path(args.uncover_model_path).expanduser().resolve()
        manifest_records = load_source_manifest(manifest_path, pool_root)
        all_scored_records = [(path, raw, f1) for path, raw, f1, _band in manifest_records]
        eligible_records = [(path, raw, f1, band) for path, raw, f1, band in manifest_records]
        failed_records = 0
        print(f"[Source Manifest] loaded={len(eligible_records)} from {manifest_path}")
    else:
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
            real_path = f.resolve()
            with open(real_path, 'rb') as rf:
                raw = pickle.load(rf)
            try:
                uncover_f1 = compute_uncover_f1_from_raw(raw)
                all_scored_records.append((real_path, raw, uncover_f1))
                if uncover_f1 >= args.uncover_f1_threshold:
                    eligible_records.append((real_path, raw, uncover_f1, 'high'))
            except Exception:
                failed_records += 1
                continue

    if len(all_scored_records) > 0 and not args.source_manifest:
        f1_vals = np.array([r[2] for r in all_scored_records])
        scored_counts = Counter(int(r[1]['target_limb_code']) for r in all_scored_records)
        pass_counts = Counter(int(r[1]['target_limb_code']) for r in eligible_records)
        print(
            f"[F1 Filter] total={len(all_scored_records)}, scored={len(all_scored_records)}, failed={failed_records}, "
            f"threshold={args.uncover_f1_threshold}, pass={len(eligible_records)}, "
            f"f1[min/mean/max]={f1_vals.min():.3f}/{f1_vals.mean():.3f}/{f1_vals.max():.3f}"
        )
        print(f"[F1 Filter] scored_by_target={dict(sorted(scored_counts.items()))}")
        print(f"[F1 Filter] pass_by_target={dict(sorted(pass_counts.items()))}")
    elif args.source_manifest:
        band_counts = Counter(r[3] for r in eligible_records)
        by_target = Counter(int(r[1]['target_limb_code']) for r in eligible_records)
        print(f"[Source Manifest] quality_band_counts={dict(sorted(band_counts.items()))}")
        print(f"[Source Manifest] by_target={dict(sorted(by_target.items()))}")

    if len(eligible_records) == 0:
        if len(all_scored_records) == 0:
            raise RuntimeError("No source samples could be scored for uncover F1 (all failed).")

        if args.fallback_when_empty == 'error':
            raise RuntimeError(f"No source uncover states passed F1 threshold {args.uncover_f1_threshold}.")
        elif args.fallback_when_empty == 'all':
            print("[F1 Filter] No records passed threshold. Falling back to ALL scored records.")
            eligible_records = [(path, raw, f1, 'fallback') for path, raw, f1 in all_scored_records]
        else:
            k = max(1, min(args.fallback_topk, len(all_scored_records)))
            eligible_records = [
                (path, raw, f1, 'fallback')
                for path, raw, f1 in sorted(all_scored_records, key=lambda x: x[2], reverse=True)[:k]
            ]
            print(f"[F1 Filter] No records passed threshold. Falling back to TOP-{k} scored records.")

    eligible_records = [
        record for record in eligible_records
        if int(record[1]['target_limb_code']) in target_limb_list
    ]
    if len(eligible_records) == 0:
        raise RuntimeError(f"No eligible records remain after target_limb_list filter: {target_limb_list}.")

    records_by_target = defaultdict(list)
    for record in eligible_records:
        records_by_target[int(record[1]['target_limb_code'])].append(record)

    missing_targets = [tl for tl in target_limb_list if len(records_by_target[tl]) == 0]
    if missing_targets:
        raise RuntimeError(f"No eligible records for target limbs: {missing_targets}")

    target_quota = requested_rollouts // len(target_limb_list)
    saved_by_target = Counter()
    if args.resume and saved_counter > 0:
        for path in Path(pkl_loc).glob(f"{args.run_id}_*.pkl"):
            try:
                with open(path, 'rb') as handle:
                    payload = pickle.load(handle)
                tl = int(payload.get('data_collection_info', {}).get('target_limb_code', payload.get('info', {}).get('target_limb_code', -1)))
                if tl in target_limb_list:
                    saved_by_target[tl] += 1
            except Exception:
                continue

    target_iterators = {tl: cycle_records(records_by_target[tl]) for tl in target_limb_list}
    flat_iterator = cycle_records(eligible_records)

    def quotas_met():
        if not args.balance_target_limbs:
            return saved_counter >= requested_rollouts
        return all(saved_by_target[tl] >= target_quota for tl in target_limb_list)

    def pick_next_record():
        if args.balance_target_limbs:
            candidates = [tl for tl in target_limb_list if saved_by_target[tl] < target_quota]
            if not candidates:
                return None
            target = min(candidates, key=lambda tl: saved_by_target[tl])
            return next(target_iterators[target])
        return next(flat_iterator)

    rollout_state = {'index': 0}

    def next_collection_seed():
        while True:
            candidate = int(args.collection_seed_base) + rollout_state['index']
            rollout_state['index'] += 1
            if candidate not in existing_collection_seeds:
                return candidate

    if args.balance_target_limbs:
        print(
            f"[Target Balance] requested_saved={requested_rollouts}, quota_per_limb={target_quota}, "
            f"source_pool_by_target={dict(sorted(Counter(int(r[1]['target_limb_code']) for r in eligible_records).items()))}, "
            f"resume_saved_by_target={dict(sorted(saved_by_target.items()))}"
        )
    else:
        print(f"[Target Balance] disabled, source_pool_size={len(eligible_records)}")

    def run_one_trial(trial_index):
        if quotas_met() or saved_counter >= requested_rollouts:
            return None
        filename, raw_data, uncover_f1, quality_band = pick_next_record()
        seed = parse_seed_from_source_filename(filename)
        collection_seed = next_collection_seed()
        return gnn_data_collect(
            args.env,
            trial_index,
            filename.name,
            seed,
            raw_data,
            args.action_mode,
            args.debug_field,
            use_visual_debug,
            args.debug_log,
            args.overlap_grid_size,
            args.overlap_min_count,
            args.overlap_min_z_range,
            args.overlap_min_top_sep,
            args.save_overlap_images and saved_counter < args.overlap_image_limit,
            overlap_image_dir,
            args.uncover_post_release_steps,
            args.recover_post_release_steps,
            collection_seed=collection_seed,
            run_id=args.run_id,
            source_f1=uncover_f1,
            quality_band=quality_band,
        )

    def handle_output(output):
        if isinstance(output, dict) and output.get('status') == 'saved':
            tl = output.get('target_limb_code')
            if tl is not None:
                saved_by_target[int(tl)] += 1
        counter_callback(output)

    if use_visual_debug:
        attempts_started = 0
        while attempts_started < attempts_limit and saved_counter < requested_rollouts and not quotas_met():
            output = run_one_trial(attempts_started)
            if output is None:
                break
            handle_output(output)
            attempts_started += 1
    else:
        attempts_started = 0
        while attempts_started < attempts_limit and saved_counter < requested_rollouts and not quotas_met():
            batch_size = min(num_processes, attempts_limit - attempts_started)
            with multiprocessing.Pool(processes=batch_size) as pool:
                result_objs = []
                for i in range(batch_size):
                    if saved_counter >= requested_rollouts or quotas_met():
                        break
                    filename, raw_data, uncover_f1, quality_band = pick_next_record()
                    seed = parse_seed_from_source_filename(filename)
                    collection_seed = next_collection_seed()
                    save_image = args.save_overlap_images and (saved_counter + len(result_objs)) < args.overlap_image_limit
                    result = pool.apply_async(
                        gnn_data_collect,
                        args=(
                            args.env,
                            attempts_started + i,
                            filename.name,
                            seed,
                            raw_data,
                            args.action_mode,
                            args.debug_field,
                            use_visual_debug,
                            args.debug_log,
                            args.overlap_grid_size,
                            args.overlap_min_count,
                            args.overlap_min_z_range,
                            args.overlap_min_top_sep,
                            save_image,
                            overlap_image_dir,
                            args.uncover_post_release_steps,
                            args.recover_post_release_steps,
                            collection_seed,
                            args.run_id,
                            uncover_f1,
                            quality_band,
                        ),
                        callback=handle_output,
                    )
                    result_objs.append(result)
                results = [result.get() for result in result_objs]
            attempts_started += len(result_objs)

    if saved_counter < requested_rollouts:
        print(
            f"[Warning] Saved {saved_counter}/{requested_rollouts} after {attempts_started} attempts. "
            f"Consider increasing --max-attempts or rerunning with --resume."
        )

    summary = summarize_collection_outputs(collection_results)
    write_collection_summary(output_dir, summary, args)
    print(f"[Summary] wrote collection summary to {output_dir}")
    print(f"[Summary] saved={summary['saved_count']} attempted={summary['attempted_count']} skipped={summary['skipped_count']}")
    if args.action_mode == 'overlap_guided' and summary['saved_count'] < requested_rollouts:
        raise RuntimeError(
            f"Only saved {summary['saved_count']} overlap_guided samples after {summary['attempted_count']} attempts; "
            f"requested {requested_rollouts}. Consider lowering overlap thresholds or increasing --overlap_max_attempts."
        )
