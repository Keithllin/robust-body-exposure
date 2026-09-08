import argparse
import csv
import json
import math
import pickle
import sys
import numpy as np
from pathlib import Path
from statistics import mean, pstdev

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str((THIS_DIR / '../assistive-gym-fem').resolve()))

from assistive_gym.envs.bu_gnn_util import get_body_points_from_obs, get_covered_status
from cma_gnn_util import compute_fscore_recover, compute_fscore_uncover

TARGET_LIMBS = [2, 4, 5, 8, 10, 11, 12, 13, 14, 15]
TARGET_NAMES = {
    2: 'R Arm',
    4: 'R L. Leg',
    5: 'R Leg',
    8: 'L Arm',
    10: 'L L. Leg',
    11: 'L Leg',
    12: 'B L. Legs',
    13: 'Upper Body',
    14: 'Lower Body',
    15: 'Whole Body',
}


def safe_mean(x):
    return mean(x) if x else float('nan')


def safe_std(x):
    return pstdev(x) if len(x) > 1 else 0.0 if len(x) == 1 else float('nan')


def to_float(v, default=float('nan')):
    try:
        return float(v)
    except Exception:
        return default


def collect_metrics(raw_dir):
    raw_path = Path(raw_dir)
    files = sorted(raw_path.glob('*.pkl'))

    pred_rewards = []
    sim_rewards = []
    opt_times = []
    null_grasp_flags = []
    sim_f1 = []
    pred_f1 = []
    per_limb = {
        tl: {
            'pred_rewards': [],
            'sim_rewards': [],
            'pred_f1': [],
            'sim_f1': [],
            'failed_grasp': 0,
            'samples': 0,
        } for tl in TARGET_LIMBS
    }

    for f in files:
        try:
            d = pickle.load(open(f, 'rb'))
        except Exception:
            continue

        cma_info = d.get('cma_info', {})
        sim_info = d.get('sim_info', {})
        info = sim_info.get('info', d.get('info', {}))
        target_limb_code = d.get('target_limb_code', None)

        pred = to_float(cma_info.get('best_reward'))
        sim = to_float(sim_info.get('recover_reward'))

        # backward compatibility: 'best_time' vs 'optimizer_time'
        opt_t = cma_info.get('optimizer_time', cma_info.get('best_time', float('nan')))
        opt_t = to_float(opt_t)

        goc = info.get('grasp_on_cloth_recover', None)
        if goc is None:
            goc = info.get('grasp_on_cloth_uncover', None)

        if not math.isnan(pred):
            pred_rewards.append(pred)
            if target_limb_code in per_limb:
                per_limb[target_limb_code]['pred_rewards'].append(pred)
        if not math.isnan(sim):
            sim_rewards.append(sim)
            if target_limb_code in per_limb:
                per_limb[target_limb_code]['sim_rewards'].append(sim)
        if not math.isnan(opt_t):
            opt_times.append(opt_t)
        if isinstance(goc, bool):
            null_grasp_flags.append(0 if goc else 1)
            if target_limb_code in per_limb:
                per_limb[target_limb_code]['samples'] += 1
                if not goc:
                    per_limb[target_limb_code]['failed_grasp'] += 1

        try:
            info = sim_info.get('info', d.get('info', {}))
            cloth_initial = info.get('cloth_initial', [None, None])[1]
            cloth_intermediate = info.get('cloth_intermediate', [None, None])[1]
            cloth_final = info.get('cloth_final', [None, None])[1]
            pred_cloth = cma_info.get('best_pred', None)

            human_pose = d.get('human_pose', None)
            target_limb_code = d.get('target_limb_code', None)
            body_info = info.get('human_body_info', d.get('info', {}).get('human_body_info', None))

            if cloth_initial is not None and cloth_final is not None and pred_cloth is not None and human_pose is not None and target_limb_code is not None:
                all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)
                initial_status = get_covered_status(all_body_points, np.delete(np.array(cloth_initial), 2, axis=1))
                sim_status = get_covered_status(all_body_points, np.delete(np.array(cloth_final), 2, axis=1))
                pred_status = get_covered_status(all_body_points, np.array(pred_cloth))

                recovering = bool(d.get('recovering', True))
                if recovering and cloth_intermediate is not None and len(cloth_intermediate) > 0:
                    interm_status = get_covered_status(all_body_points, np.delete(np.array(cloth_intermediate), 2, axis=1))
                    sim_f = compute_fscore_recover(initial_status, interm_status, sim_status, False)
                    pred_f = compute_fscore_recover(initial_status, interm_status, pred_status, False)
                else:
                    sim_f = compute_fscore_uncover(initial_status, sim_status)
                    pred_f = compute_fscore_uncover(initial_status, pred_status)

                if not (isinstance(sim_f, float) and math.isnan(sim_f)):
                    sim_f1.append(float(sim_f))
                    if target_limb_code in per_limb:
                        per_limb[target_limb_code]['sim_f1'].append(float(sim_f))
                if not (isinstance(pred_f, float) and math.isnan(pred_f)):
                    pred_f1.append(float(pred_f))
                    if target_limb_code in per_limb:
                        per_limb[target_limb_code]['pred_f1'].append(float(pred_f))
        except Exception:
            pass

    n = len(files)
    valid_pairs = min(len(pred_rewards), len(sim_rewards))
    gap = [abs(pred_rewards[i] - sim_rewards[i]) for i in range(valid_pairs)]

    per_limb_rows = []
    for tl in TARGET_LIMBS:
        d_tl = per_limb[tl]
        n_samples = d_tl['samples']
        null_rate = (d_tl['failed_grasp'] / n_samples) if n_samples > 0 else float('nan')
        per_limb_rows.append({
            'target_limb_code': tl,
            'target_name': TARGET_NAMES.get(tl, str(tl)),
            'num_samples': n_samples,
            'pred_recover_reward_mean': safe_mean(d_tl['pred_rewards']),
            'sim_recover_reward_mean': safe_mean(d_tl['sim_rewards']),
            'pred_f1_mean': safe_mean(d_tl['pred_f1']),
            'sim_f1_mean': safe_mean(d_tl['sim_f1']),
            'null_grasp_rate': null_rate,
        })

    return {
        'num_files': n,
        'pred_recover_reward_mean': safe_mean(pred_rewards),
        'pred_recover_reward_std': safe_std(pred_rewards),
        'sim_recover_reward_mean': safe_mean(sim_rewards),
        'sim_recover_reward_std': safe_std(sim_rewards),
        'opt_sim_gap_mean': safe_mean(gap),
        'opt_sim_gap_std': safe_std(gap),
        'sim_f1_mean': safe_mean(sim_f1),
        'sim_f1_std': safe_std(sim_f1),
        'pred_f1_mean': safe_mean(pred_f1),
        'pred_f1_std': safe_std(pred_f1),
        'null_grasp_rate': safe_mean(null_grasp_flags),
        'optimizer_time_mean_sec': safe_mean(opt_times),
        'optimizer_time_std_sec': safe_std(opt_times),
        'per_limb_rows': per_limb_rows,
    }


def format_num(v):
    if isinstance(v, float):
        if math.isnan(v):
            return 'NaN'
        return f'{v:.4f}'
    return str(v)


def write_csv(rows, out_csv):
    cols = [
        'id', 'search', 'search_method', 'warm_start', 'budget', 'raw_dir',
        'num_files',
        'pred_recover_reward_mean', 'pred_recover_reward_std',
        'sim_recover_reward_mean', 'sim_recover_reward_std',
        'opt_sim_gap_mean', 'opt_sim_gap_std',
        'pred_f1_mean', 'pred_f1_std',
        'sim_f1_mean', 'sim_f1_std',
        'null_grasp_rate',
        'optimizer_time_mean_sec', 'optimizer_time_std_sec',
    ]
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in cols})


def write_md(rows, out_md):
    header = (
        '| id | search | method | warm | budget | n | sim_mean | sim_std | pred_mean | gap_mean | sim_f1 | pred_f1 | null_grasp | opt_time_mean(s) |\n'
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['search']} | {r.get('search_method', '')} | {r['warm_start']} | {r['budget']} | {r['num_files']} | "
            f"{format_num(r['sim_recover_reward_mean'])} | {format_num(r['sim_recover_reward_std'])} | "
            f"{format_num(r['pred_recover_reward_mean'])} | {format_num(r['opt_sim_gap_mean'])} | "
            f"{format_num(r['sim_f1_mean'])} | {format_num(r['pred_f1_mean'])} | "
            f"{format_num(r['null_grasp_rate'])} | {format_num(r['optimizer_time_mean_sec'])} |\n"
        )

    with open(out_md, 'w') as f:
        f.write(''.join(lines))


def write_per_limb_csv(rows, out_csv):
    cols = [
        'run_id', 'search', 'search_method', 'warm_start', 'budget',
        'target_limb_code', 'target_name', 'num_samples',
        'pred_recover_reward_mean', 'sim_recover_reward_mean',
        'pred_f1_mean', 'sim_f1_mean', 'null_grasp_rate',
    ]
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in cols})


def write_per_limb_md(rows, out_md):
    header = (
    '| run_id | target_limb | target_name | n | sim_reward | pred_reward | sim_f1 | pred_f1 | null_grasp |\n'
    '|---|---:|---|---:|---:|---:|---:|---:|---:|\n'
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['run_id']} | {r['target_limb_code']} | {r['target_name']} | {r['num_samples']} | "
            f"{format_num(r['sim_recover_reward_mean'])} | {format_num(r['pred_recover_reward_mean'])} | "
            f"{format_num(r['sim_f1_mean'])} | {format_num(r['pred_f1_mean'])} | {format_num(r['null_grasp_rate'])} |\n"
        )
    with open(out_md, 'w') as f:
        f.write(''.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', required=True, help='Path to matrix_runs.json')
    parser.add_argument('--out-csv', default='summary.csv')
    parser.add_argument('--out-md', default='summary.md')
    parser.add_argument('--out-per-limb-csv', default='summary_per_limb.csv')
    parser.add_argument('--out-per-limb-md', default='summary_per_limb.md')
    args = parser.parse_args()

    cfg = json.load(open(args.matrix, 'r'))
    runs = cfg.get('runs', [])

    rows = []
    per_limb_rows_all = []
    for run in runs:
        rid = run.get('id', 'unknown')
        raw_dir = run.get('raw_dir', '')
        m = collect_metrics(raw_dir)
        per_limb_rows = m.pop('per_limb_rows', [])
        row = {
            'id': rid,
            'search': run.get('search', ''),
            'search_method': run.get('search_method', ''),
            'warm_start': run.get('warm_start', ''),
            'budget': run.get('budget', ''),
            'raw_dir': raw_dir,
            **m,
        }
        rows.append(row)

        for tl_row in per_limb_rows:
            per_limb_rows_all.append({
                'run_id': rid,
                'search': run.get('search', ''),
                'search_method': run.get('search_method', ''),
                'warm_start': run.get('warm_start', ''),
                'budget': run.get('budget', ''),
                **tl_row,
            })

    # sort for readability
    rows = sorted(rows, key=lambda x: x['id'])

    write_csv(rows, args.out_csv)
    write_md(rows, args.out_md)
    write_per_limb_csv(per_limb_rows_all, args.out_per_limb_csv)
    write_per_limb_md(per_limb_rows_all, args.out_per_limb_md)

    print(f'Wrote CSV: {args.out_csv}')
    print(f'Wrote MD:  {args.out_md}')
    print(f'Wrote per-limb CSV: {args.out_per_limb_csv}')
    print(f'Wrote per-limb MD:  {args.out_per_limb_md}')


if __name__ == '__main__':
    main()
