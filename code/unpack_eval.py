#%%
import argparse
import math
import os.path as osp
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'assistive-gym-fem'))

from assistive_gym.envs.bu_gnn_util import *  # noqa: F401,F403
from cma_gnn_util import *  # noqa: F401,F403


MODEL_DIR = Path('/mnt/data/MudkipUsersSu2025/kpputhuveetil/git/robe/robust-body-exposure_unstable/trained_models/FINAL_MODELS')
TARGET_LIMBS = [2, 4, 5, 8, 10, 11, 12, 13, 14, 15]


def resolve_model_path(arg_model):
    model_path = Path(arg_model)
    if model_path.exists():
        return model_path.resolve()
    return (MODEL_DIR / arg_model).resolve()


def resolve_raw_dir(model_path, eval_dir_name, eval_condition='', raw_dir=''):
    if raw_dir:
        resolved = Path(raw_dir).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f'raw dir does not exist: {resolved}')
        return resolved

    eval_root = model_path / eval_dir_name
    if not eval_root.exists():
        raise FileNotFoundError(f'eval dir does not exist: {eval_root}')

    if eval_condition:
        resolved = eval_root / eval_condition / 'raw'
        if not resolved.exists():
            raise FileNotFoundError(f'raw dir does not exist: {resolved}')
        return resolved

    candidates = [path / 'raw' for path in eval_root.iterdir() if path.is_dir() and (path / 'raw').exists()]
    if not candidates:
        raise FileNotFoundError(f'no raw directories found under: {eval_root}')
    candidates.sort(key=lambda path: path.parent.stat().st_mtime, reverse=True)
    return candidates[0]


def get_sim_reward(raw_data):
    sim_info = raw_data['sim_info']
    if raw_data['recovering']:
        return float(sim_info['recover_reward'])
    if 'uncover_reward' in sim_info:
        return float(sim_info['uncover_reward'])
    return float(sim_info['uncover reward'])


def get_optimizer_reward(raw_data, reward_key):
    cma_info = raw_data['cma_info']
    optimization_mode = cma_info.get('optimization_mode', '')
    recover = raw_data['recovering']

    if reward_key == 'joint':
        return float(cma_info.get('pred_joint_reward', cma_info.get('best_reward')))
    if reward_key == 'recover':
        return float(cma_info.get('pred_recover_reward', cma_info.get('best_reward')))
    if reward_key == 'uncover':
        return float(cma_info.get('pred_uncover_reward', cma_info.get('best_reward')))

    # auto
    if optimization_mode == 'sequential':
        if recover and 'pred_recover_reward' in cma_info:
            return float(cma_info['pred_recover_reward'])
        if not recover and 'pred_uncover_reward' in cma_info:
            return float(cma_info['pred_uncover_reward'])
        if 'pred_joint_reward' in cma_info:
            return float(cma_info['pred_joint_reward'])
    return float(cma_info.get('best_reward', cma_info.get('pred_joint_reward', cma_info.get('pred_recover_reward', cma_info.get('pred_uncover_reward', np.nan)))))


def get_body_info(raw_data):
    return raw_data['sim_info']['info'].get('human_body_info', raw_data['info'].get('human_body_info'))


def safe_mean(values):
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def safe_std(values):
    if len(values) == 0:
        return np.nan
    return float(np.std(values))


def safe_median(values):
    if len(values) == 0:
        return np.nan
    return float(np.median(values))


def format_failed(count, total):
    if total <= 0:
        return '0/0 (0.0%)'
    return f'{count}/{total} ({100.0 * count / total:.1f}%)'


def emit(lines, text=''):
    print(text)
    lines.append(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg_model', type=str, default='/mnt/data/MudkipUsersSu2025/kpputhuveetil/git/robe/robust-body-exposure_unstable/trained_models/FINAL_MODELS/Recover/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Recover_Data_100_seeds_30000_states_30000_epochs=250_batch=50_workers=4_1705986655')
    parser.add_argument('--eval-dir-name', type=str, default='joint_evaluations')
    parser.add_argument('--eval-condition', type=str, default='')
    parser.add_argument('--raw-dir', type=str, default='')
    parser.add_argument('--reward-key', type=str, default='auto', choices=['auto', 'joint', 'recover', 'uncover'])
    parser.add_argument('--tlc', type=int, default=-1)
    parser.add_argument('--save-md', action='store_true')
    parser.add_argument('--md-path', type=str, default='')
    args = parser.parse_args()

    model_path = resolve_model_path(args.arg_model)
    raw_dir = resolve_raw_dir(model_path, args.eval_dir_name, args.eval_condition, args.raw_dir)
    report_lines = []
    emit(report_lines, f'Model path: {model_path}')
    emit(report_lines, f'Raw dir: {raw_dir}')

    filenames = sorted(raw_dir.glob('*.pkl'))
    if len(filenames) == 0:
        raise FileNotFoundError(f'no pickle files found in {raw_dir}')

    num_targets = 16
    targ_data_reward = [[[] for _ in range(num_targets)], [[] for _ in range(num_targets)]]
    targ_data_task_fscore = [[[] for _ in range(num_targets)], [[] for _ in range(num_targets)]]
    targ_data_gap = [[] for _ in range(num_targets)]
    targ_pred_uncover_f1_saved = [[] for _ in range(num_targets)]
    targ_sim_uncover_f1 = [[] for _ in range(num_targets)]
    targ_pred_recover_f1 = [[] for _ in range(num_targets)]
    targ_sim_recover_f1 = [[] for _ in range(num_targets)]
    failed_grasps = np.zeros(num_targets)
    pred_joint_rewards = []
    pred_uncover_fscores = []
    count_ng = 0

    for filename in filenames:
        with open(filename, 'rb') as handle:
            raw_data = pickle.load(handle)

        target_limb_code = int(filename.name.split('_')[0][2:])
        if args.tlc >= 0 and target_limb_code != args.tlc:
            continue

        plt.close()
        recover = raw_data['recovering']
        optimizer_reward = get_optimizer_reward(raw_data, args.reward_key)
        sim_reward = get_sim_reward(raw_data)
        targ_data_reward[0][target_limb_code].append(optimizer_reward)
        targ_data_reward[1][target_limb_code].append(sim_reward)
        targ_data_gap[target_limb_code].append(float(optimizer_reward - sim_reward))

        cloth_initial = raw_data['sim_info']['info']['cloth_initial'][1]
        cloth_intermediate = raw_data['sim_info']['info'].get('cloth_intermediate', [None, None])[1] if recover else None
        cloth_final = raw_data['sim_info']['info']['cloth_final'][1]
        pred = raw_data['cma_info']['best_pred']
        human_pose = raw_data['human_pose']
        body_info = get_body_info(raw_data)
        count_ng += 1

        all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)
        initial_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_initial), 2, axis=1))
        if recover:
            intermediate_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_intermediate), 2, axis=1))

        pred_covered_status = get_covered_status(all_body_points, pred)
        sim_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_final), 2, axis=1))

        if recover:
            sim_uncover_f1 = compute_fscore_uncover(initial_covered_status, intermediate_covered_status)
            pred_uncover_f1 = float(raw_data['cma_info'].get('pred_uncover_f1', np.nan))
            sim_recover_f1 = compute_fscore_recover(initial_covered_status, intermediate_covered_status, sim_covered_status, False)
            pred_recover_f1 = compute_fscore_recover(initial_covered_status, intermediate_covered_status, pred_covered_status, False)
            sim_fscore = sim_recover_f1
            pred_fscore = pred_recover_f1
        else:
            sim_uncover_f1 = compute_fscore_uncover(initial_covered_status, sim_covered_status)
            pred_uncover_f1 = float(raw_data['cma_info'].get('pred_uncover_f1', compute_fscore_uncover(initial_covered_status, pred_covered_status)))
            sim_recover_f1 = np.nan
            pred_recover_f1 = np.nan
            sim_fscore = sim_uncover_f1
            pred_fscore = pred_uncover_f1

        info = raw_data['sim_info']['info']
        if recover:
            if not info.get('grasp_on_cloth_recover', True):
                failed_grasps[target_limb_code] += 1
                targ_data_task_fscore[0][target_limb_code].append(pred_fscore)
        else:
            if not info.get('grasp_on_cloth_uncover', True):
                failed_grasps[target_limb_code] += 1
                targ_data_task_fscore[0][target_limb_code].append(pred_fscore)

        if not np.isnan(sim_fscore):
            targ_data_task_fscore[1][target_limb_code].append(sim_fscore)
        if not np.isnan(pred_fscore) and not math.isnan(pred_fscore):
            targ_data_task_fscore[0][target_limb_code].append(pred_fscore)

        targ_data_task_fscore[0][target_limb_code] = [x for x in targ_data_task_fscore[0][target_limb_code] if str(x) != 'nan']
        targ_data_task_fscore[1][target_limb_code] = [x for x in targ_data_task_fscore[1][target_limb_code] if str(x) != 'nan']

        if not np.isnan(sim_uncover_f1) and not math.isnan(sim_uncover_f1):
            targ_sim_uncover_f1[target_limb_code].append(sim_uncover_f1)
        if not np.isnan(pred_uncover_f1) and not math.isnan(pred_uncover_f1):
            targ_pred_uncover_f1_saved[target_limb_code].append(pred_uncover_f1)
        if not np.isnan(pred_recover_f1) and not math.isnan(pred_recover_f1):
            targ_pred_recover_f1[target_limb_code].append(pred_recover_f1)
        if not np.isnan(sim_recover_f1) and not math.isnan(sim_recover_f1):
            targ_sim_recover_f1[target_limb_code].append(sim_recover_f1)

        if 'pred_joint_reward' in raw_data['cma_info']:
            pred_joint_rewards.append(float(raw_data['cma_info']['pred_joint_reward']))
        if 'pred_uncover_f1' in raw_data['cma_info']:
            pred_uncover_f1_value = float(raw_data['cma_info']['pred_uncover_f1'])
            pred_uncover_fscores.append(pred_uncover_f1_value)

    reward_means = [[], []]
    reward_stds = [[], []]
    reward_medians = [[], []]
    task_fscore_means = [[], []]
    task_fscore_stds = [[], []]
    pred_uncover_f1_saved_means = []
    pred_uncover_f1_saved_stds = []
    sim_uncover_f1_means = []
    sim_uncover_f1_stds = []
    pred_recover_f1_means = []
    pred_recover_f1_stds = []
    sim_recover_f1_means = []
    sim_recover_f1_stds = []
    gap_means = []
    gap_medians = []
    samples = []

    for target in range(num_targets):
        if target in TARGET_LIMBS:
            samples.append(len(targ_data_reward[0][target]))
            reward_means[0].append(safe_mean(targ_data_reward[0][target]))
            reward_stds[0].append(safe_std(targ_data_reward[0][target]))
            reward_medians[0].append(safe_median(targ_data_reward[0][target]))
            reward_means[1].append(safe_mean(targ_data_reward[1][target]))
            reward_stds[1].append(safe_std(targ_data_reward[1][target]))
            reward_medians[1].append(safe_median(targ_data_reward[1][target]))
            task_fscore_means[0].append(safe_mean(targ_data_task_fscore[0][target]))
            task_fscore_stds[0].append(safe_std(targ_data_task_fscore[0][target]))
            task_fscore_means[1].append(safe_mean(targ_data_task_fscore[1][target]))
            task_fscore_stds[1].append(safe_std(targ_data_task_fscore[1][target]))
            pred_uncover_f1_saved_means.append(safe_mean(targ_pred_uncover_f1_saved[target]))
            pred_uncover_f1_saved_stds.append(safe_std(targ_pred_uncover_f1_saved[target]))
            sim_uncover_f1_means.append(safe_mean(targ_sim_uncover_f1[target]))
            sim_uncover_f1_stds.append(safe_std(targ_sim_uncover_f1[target]))
            pred_recover_f1_means.append(safe_mean(targ_pred_recover_f1[target]))
            pred_recover_f1_stds.append(safe_std(targ_pred_recover_f1[target]))
            sim_recover_f1_means.append(safe_mean(targ_sim_recover_f1[target]))
            sim_recover_f1_stds.append(safe_std(targ_sim_recover_f1[target]))
            gap_means.append(safe_mean(targ_data_gap[target]))
            gap_medians.append(safe_median(targ_data_gap[target]))
        else:
            samples.append(0)
            reward_means[0].append(np.nan)
            reward_stds[0].append(np.nan)
            reward_medians[0].append(np.nan)
            reward_means[1].append(np.nan)
            reward_stds[1].append(np.nan)
            reward_medians[1].append(np.nan)
            task_fscore_means[0].append(np.nan)
            task_fscore_stds[0].append(np.nan)
            task_fscore_means[1].append(np.nan)
            task_fscore_stds[1].append(np.nan)
            pred_uncover_f1_saved_means.append(np.nan)
            pred_uncover_f1_saved_stds.append(np.nan)
            sim_uncover_f1_means.append(np.nan)
            sim_uncover_f1_stds.append(np.nan)
            pred_recover_f1_means.append(np.nan)
            pred_recover_f1_stds.append(np.nan)
            sim_recover_f1_means.append(np.nan)
            sim_recover_f1_stds.append(np.nan)
            gap_means.append(np.nan)
            gap_medians.append(np.nan)

    prop_failed = np.divide(failed_grasps, np.maximum(1, np.array(samples)))

    target_names_full = [
        '', '', 'R Arm',
        '', 'R L. Leg', 'R Leg',
        '', '', 'L Arm',
        '', 'L L. Leg', 'L Leg',
        'B L. Legs', 'Upper Body',
        'Lower Body', 'Whole Body',
    ]

    active_targets = [args.tlc] if args.tlc in TARGET_LIMBS else TARGET_LIMBS
    uncover_f1_rows = [[
        target,
        target_names_full[target],
        samples[target],
        pred_uncover_f1_saved_means[target],
        pred_uncover_f1_saved_stds[target],
        sim_uncover_f1_means[target],
        sim_uncover_f1_stds[target],
        task_fscore_means[0][target],
        task_fscore_stds[0][target],
        task_fscore_means[1][target],
        task_fscore_stds[1][target],
        format_failed(int(failed_grasps[target]), samples[target]),
    ] for target in active_targets]
    recover_f1_rows = [[
        target,
        target_names_full[target],
        samples[target],
        pred_recover_f1_means[target],
        pred_recover_f1_stds[target],
        sim_recover_f1_means[target],
        sim_recover_f1_stds[target],
    ] for target in active_targets]
    reward_rows = [[
        target,
        target_names_full[target],
        samples[target],
        reward_means[0][target],
        reward_means[1][target],
        reward_medians[1][target],
        gap_means[target],
        gap_medians[target],
    ] for target in active_targets]

    emit(report_lines, '')
    emit(report_lines, 'Per-target Uncover F1 summary')
    emit(report_lines, tabulate(
        uncover_f1_rows,
        headers=['TL#', 'Target', 'n', 'pred_uncover_f1_saved', 'pred_uncover_f1_std', 'sim_uncover_f1', 'sim_uncover_f1_std', 'task_pred_f', 'task_pred_f_std', 'task_sim_f', 'task_sim_f_std', 'failed grasp'],
        floatfmt='.3f',
    ))

    emit(report_lines, '')
    emit(report_lines, 'Per-target Recover F1 summary')
    emit(report_lines, tabulate(
        recover_f1_rows,
        headers=['TL#', 'Target', 'n', 'pred_recover_f1', 'pred_recover_f1_std', 'sim_recover_f1', 'sim_recover_f1_std'],
        floatfmt='.3f',
    ))

    emit(report_lines, '')
    emit(report_lines, 'Per-target reward summary')
    emit(report_lines, tabulate(
        reward_rows,
        headers=['TL#', 'Target', 'n', 'pred_r', 'sim_r_mean', 'sim_r_med', 'gap_mean', 'gap_med'],
        floatfmt='.2f',
    ))

    all_pred_reward = sum(targ_data_reward[0], [])
    all_sim_reward = sum(targ_data_reward[1], [])
    all_pred_fscore = sum(targ_data_task_fscore[0], [])
    all_sim_fscore = sum(targ_data_task_fscore[1], [])
    all_pred_sim_gap = sum(targ_data_gap, [])
    all_sim_uncover_f1 = sum(targ_sim_uncover_f1, [])
    all_pred_recover_f1 = sum(targ_pred_recover_f1, [])
    all_sim_recover_f1 = sum(targ_sim_recover_f1, [])

    global_summary_lines = [
        f'Total analyzed rollouts: {len(all_pred_reward)}',
        f'optimizer reward mean: {np.mean(all_pred_reward)}',
        f'optimizer reward median: {np.median(all_pred_reward)}',
        f'sim reward mean: {np.mean(all_sim_reward)}',
        f'sim reward median: {np.median(all_sim_reward)}',
        f'pred-sim reward gap mean: {np.mean(all_pred_sim_gap)}',
        f'pred-sim reward gap median: {np.median(all_pred_sim_gap)}',
        f'pred fscore: {round(np.mean(all_pred_fscore), 2)}',
        f'sim fscore: {round(np.mean(all_sim_fscore), 2)}',
        f'sim fscore std: {round(np.std(all_sim_fscore), 2)}',
        f'sim uncover F1 mean: {round(float(np.mean(all_sim_uncover_f1)), 3) if len(all_sim_uncover_f1) > 0 else np.nan}',
        f'pred recover F1 mean: {round(float(np.mean(all_pred_recover_f1)), 3) if len(all_pred_recover_f1) > 0 else np.nan}',
        f'sim recover F1 mean: {round(float(np.mean(all_sim_recover_f1)), 3) if len(all_sim_recover_f1) > 0 else np.nan}',
        f'Overall % Null Grasp: {np.sum(failed_grasps)}/{count_ng} = {np.sum(failed_grasps) / max(1, count_ng) * 100}',
    ]
    emit(report_lines, '')
    for line in global_summary_lines:
        emit(report_lines, line)

    if len(pred_joint_rewards) > 0:
        line = f'pred joint reward mean: {round(float(np.mean(pred_joint_rewards)), 2)}'
        global_summary_lines.append(line)
        emit(report_lines, line)
    if len(pred_uncover_fscores) > 0:
        line = f'pred uncover F1 mean: {round(float(np.mean(pred_uncover_fscores)), 3)}'
        global_summary_lines.append(line)
        emit(report_lines, line)

    if args.save_md:
        if args.md_path:
            md_path = Path(args.md_path).resolve()
        else:
            suffix = f'_tl{args.tlc}' if args.tlc in TARGET_LIMBS else ''
            md_path = (raw_dir.parent / f'eval_summary_{args.reward_key}{suffix}.md').resolve()
        md_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(md_path, 'w', encoding='utf-8') as handle:
            handle.write(f'# Eval Summary\n\n')
            handle.write(f'- Generated: {timestamp}\n')
            handle.write(f'- Model path: `{model_path}`\n')
            handle.write(f'- Raw dir: `{raw_dir}`\n')
            handle.write(f'- Reward key: `{args.reward_key}`\n')
            if args.tlc in TARGET_LIMBS:
                handle.write(f'- Target limb code filter: `{args.tlc}`\n')
            handle.write('\n## Per-target Uncover F1 summary\n\n')
            handle.write(tabulate(
                uncover_f1_rows,
                headers=['TL#', 'Target', 'n', 'pred_uncover_f1_saved', 'pred_uncover_f1_std', 'sim_uncover_f1', 'sim_uncover_f1_std', 'task_pred_f', 'task_pred_f_std', 'task_sim_f', 'task_sim_f_std', 'failed grasp'],
                floatfmt='.3f',
                tablefmt='github',
            ))
            handle.write('\n\n## Per-target Recover F1 summary\n\n')
            handle.write(tabulate(
                recover_f1_rows,
                headers=['TL#', 'Target', 'n', 'pred_recover_f1', 'pred_recover_f1_std', 'sim_recover_f1', 'sim_recover_f1_std'],
                floatfmt='.3f',
                tablefmt='github',
            ))
            handle.write('\n\n## Per-target reward summary\n\n')
            handle.write(tabulate(
                reward_rows,
                headers=['TL#', 'Target', 'n', 'pred_r', 'sim_r_mean', 'sim_r_med', 'gap_mean', 'gap_med'],
                floatfmt='.2f',
                tablefmt='github',
            ))
            handle.write('\n\n## Global summary\n\n')
            for line in global_summary_lines:
                handle.write(f'- {line}\n')
        emit(report_lines, f'Markdown saved to: {md_path}')


if __name__ == '__main__':
    main()
