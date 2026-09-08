import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'assistive-gym-fem'))

from assistive_gym.envs.bu_gnn_util import get_body_points_from_obs, get_covered_status  # noqa: E402
from cma_gnn_util import compute_fscore_uncover  # noqa: E402


TARGET_LIMBS = [2, 4, 5, 8, 10, 11, 12, 13, 14, 15]


def compute_sim_uncover_f1(raw_data):
    info = raw_data.get('sim_info', {}).get('info', raw_data.get('info', {}))
    cloth_initial = np.asarray(info['cloth_initial'][1])
    cloth_final = np.asarray(info['cloth_final'][1])
    target_limb_code = int(raw_data.get('target_limb_code', info['target_limb_code']))
    human_pose = np.reshape(raw_data['human_pose'], (-1, 2))
    body_info = info.get('human_body_info', raw_data.get('info', {}).get('human_body_info'))
    all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)
    initial_status = get_covered_status(all_body_points, np.delete(cloth_initial, 2, axis=1))
    final_status = get_covered_status(all_body_points, np.delete(cloth_final, 2, axis=1))
    return float(compute_fscore_uncover(initial_status, final_status))


def load_records(raw_dir):
    records = []
    for path in sorted(Path(raw_dir).glob('*.pkl')):
        with open(path, 'rb') as handle:
            raw_data = pickle.load(handle)
        target_limb_code = int(raw_data['target_limb_code'])
        seed = int(path.name.split('_')[2])
        sim_uncover_f1 = compute_sim_uncover_f1(raw_data)
        records.append({
            'source_uncover_pkl': str(path.resolve()),
            'target_limb_code': target_limb_code,
            'seed': seed,
            'sim_uncover_f1': sim_uncover_f1,
            'uncover_action': np.asarray(raw_data['uncover_action'], dtype=np.float32).tolist(),
        })
    return records


def main():
    parser = argparse.ArgumentParser(description='Build a fixed decoupled recover eval set manifest.')
    parser.add_argument('--raw-dir', type=str, required=True, help='Source uncover raw directory.')
    parser.add_argument('--output', type=str, required=True, help='Output JSON manifest path.')
    parser.add_argument('--per-limb', type=int, default=40, help='Number of eval cases per target limb.')
    parser.add_argument('--min-uncover-f1', type=float, default=1e-8, help='Exclude samples with sim_uncover_f1 <= this threshold.')
    parser.add_argument('--selection', type=str, default='random', choices=['random', 'top_f1'], help='How to choose the fixed subset per limb.')
    parser.add_argument('--seed', type=int, default=0, help='Deterministic RNG seed for random selection.')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_records = load_records(args.raw_dir)

    by_limb = {tl: [] for tl in TARGET_LIMBS}
    for record in all_records:
        if record['target_limb_code'] in by_limb and record['sim_uncover_f1'] > float(args.min_uncover_f1):
            by_limb[record['target_limb_code']].append(record)

    manifest_records = []
    summary = {}
    for tl in TARGET_LIMBS:
        limb_records = by_limb[tl]
        if len(limb_records) < args.per_limb:
            raise RuntimeError(
                f'Not enough eligible uncover samples for TL {tl}: '
                f'need {args.per_limb}, found {len(limb_records)} after filtering sim_uncover_f1 > {args.min_uncover_f1}'
            )
        if args.selection == 'top_f1':
            chosen = sorted(limb_records, key=lambda row: row['sim_uncover_f1'], reverse=True)[:args.per_limb]
        else:
            chosen = list(limb_records)
            rng.shuffle(chosen)
            chosen = chosen[:args.per_limb]
            chosen.sort(key=lambda row: (row['seed'], Path(row['source_uncover_pkl']).name))

        for idx, record in enumerate(chosen):
            manifest_records.append({
                'eval_id': f'tl{tl}_{idx:03d}',
                'target_limb_code': tl,
                'seed': int(record['seed']),
                'sim_uncover_f1': float(record['sim_uncover_f1']),
                'source_uncover_pkl': record['source_uncover_pkl'],
                'uncover_action': record['uncover_action'],
            })

        f1_values = [row['sim_uncover_f1'] for row in chosen]
        summary[str(tl)] = {
            'count': len(chosen),
            'sim_uncover_f1_mean': float(np.mean(f1_values)),
            'sim_uncover_f1_min': float(np.min(f1_values)),
            'sim_uncover_f1_max': float(np.max(f1_values)),
        }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'description': 'Fixed decoupled recover evaluation set',
        'source_raw_dir': str(Path(args.raw_dir).expanduser().resolve()),
        'per_limb': int(args.per_limb),
        'min_uncover_f1_exclusive': float(args.min_uncover_f1),
        'selection': args.selection,
        'selection_seed': int(args.seed),
        'target_limbs': TARGET_LIMBS,
        'summary': summary,
        'records': manifest_records,
    }
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)

    print(f'Wrote eval set: {output_path}')
    print(f'Total records: {len(manifest_records)}')
    for tl in TARGET_LIMBS:
        info = summary[str(tl)]
        print(
            f"  TL {tl}: n={info['count']} "
            f"f1[min/mean/max]={info['sim_uncover_f1_min']:.3f}/{info['sim_uncover_f1_mean']:.3f}/{info['sim_uncover_f1_max']:.3f}"
        )


if __name__ == '__main__':
    main()
