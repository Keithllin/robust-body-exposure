import json
import os
from pathlib import Path

from bm_dataset import BMDataset
from gnn_manager import GNN_Manager
import torch
import numpy as np
import argparse

GRAPH_CONFIGS = {
        '2D': {'use_3D': False, 'rot_draping': True},
        '3D': {'use_3D': True, 'rot_draping': False},
}


def parse_voxel_size(value):
        if value is None:
                return np.nan
        text = str(value).strip().lower()
        if text in {'nan', 'none', 'no', 'false'}:
                return np.nan
        return float(value)


parser = argparse.ArgumentParser()
parser.add_argument('--num_seeds', type=int, default=100)
parser.add_argument(
        '--dataset-dir',
        default='./DATASETS/Recover_Data/TL_All_Recover_Data_100_seeds_500000_fix_nullgrasp')
parser.add_argument('--dataset-sizes', default='250000,500000')
parser.add_argument('--description', default='Recovering_500k_fix_nullgrasp_Dataset')
parser.add_argument('--model-prefix', default='TL_All_Recover_Data_500k_Dynamic_Model')
parser.add_argument('--graph-config', choices=sorted(GRAPH_CONFIGS.keys()), default='2D')
parser.add_argument('--voxel-size', default='nan',
        help='Voxel size for BMDataset preprocessing. Use nan to disable subsampling.')
parser.add_argument('--edge-threshold', type=float, default=0.04)
parser.add_argument('--edge-mode', choices=['radius', 'mesh'], default='radius',
        help='Graph edge construction: radius keeps existing nearest-neighbor edges; mesh uses GT blanket mesh connectivity.')
parser.add_argument(
        '--use-oracle-anchor',
        action='store_true',
        help=(
            'Append a binary oracle grasp-attachment mask to each node feature. '
            'Uses stored info/anchor_idx when present; otherwise recomputes the '
            'simulator singulate/nearest attachment from cloth+action.'
        ),
)
parser.add_argument(
        '--no-singulate-layers',
        action='store_true',
        help='When recomputing oracle anchors, use nearest-4 instead of singulate_layers.',
)
parser.add_argument(
        '--action-mode',
        choices=['broadcast', 'oracle_triangle'],
        default='broadcast',
        help=(
            'Node action injection: broadcast tiles pick/place onto all nodes; '
            'oracle_triangle gates action to the simulator attached triangle only '
            '(implies --use-oracle-anchor and voxel_size=nan).'
        ),
)
parser.add_argument('--recover', action='store_true',
        help='Train a recover dynamics model (uses cloth_intermediate + recover_action).')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=35)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--process-workers', type=int, default=None,
        help='Workers for BMDataset preprocessing. Defaults to BMDataset capped CPU worker count (max 4).')
parser.add_argument('--learning-rate', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=1001)
parser.add_argument('--proc-layers', type=int, default=4)
parser.add_argument('--manifest-path', default='',
        help='Optional paired pilot manifest for deterministic train/held-out splits.')
parser.add_argument('--model-path', default='trained_models/FINAL_MODELS/Uncover_Lowering_AB',
        help='Model output root for pilot training.')
parser.add_argument('--lowering-arm', default='',
        help='Optional LowerOn/LowerOff tag appended to model description.')
parser.add_argument(
        '--layer-bit',
        choices=['none', 'oracle', 'shuffled'],
        default='none',
        help=(
            'Broadcast (layer_valid, is_topmost) onto nodes: '
            'none=baseline (node_dim=6), oracle=true labels, '
            'shuffled=valid bit kept + topmost shuffled among valid.'
        ),
)
parser.add_argument('--layer-labels-path', default='',
        help='JSONL from label_binary_grasp_layer.py (required unless --layer-bit none).')
parser.add_argument('--layer-shuffle-map', default='',
        help='JSON map sample_id->shuffled is_topmost for valid samples (--layer-bit shuffled).')
args = parser.parse_args()

dataset_dir = args.dataset_dir
recover = bool(args.recover)
graph_config = GRAPH_CONFIGS[args.graph_config]
voxel_size = parse_voxel_size(args.voxel_size)
process_workers = args.process_workers
use_oracle_anchor = bool(args.use_oracle_anchor)
singulate_layers = not bool(args.no_singulate_layers)
action_mode = str(args.action_mode).strip().lower()
layer_bit_mode = str(args.layer_bit).strip().lower()
if action_mode == 'oracle_triangle':
        use_oracle_anchor = True
if use_oracle_anchor and not np.isnan(voxel_size):
        raise ValueError('--use-oracle-anchor / --action-mode oracle_triangle requires --voxel-size nan')
if action_mode == 'oracle_triangle' and args.edge_mode != 'mesh':
        print('WARNING: action_mode=oracle_triangle is intended with --edge-mode mesh')
if layer_bit_mode != 'none' and not args.layer_labels_path:
        raise ValueError('--layer-bit %s requires --layer-labels-path' % layer_bit_mode)
if layer_bit_mode == 'shuffled' and not args.layer_shuffle_map:
        raise ValueError('--layer-bit shuffled requires --layer-shuffle-map')

manifest = None
train_ids = heldout_ids = None
if args.manifest_path:
        manifest = json.load(open(Path(args.manifest_path).expanduser().resolve()))
        train_ids = manifest['train_ids']
        heldout_ids = manifest['heldout_ids']

common_dataset_kwargs = dict(
        recover=recover,
        voxel_size=voxel_size,
        edge_threshold=args.edge_threshold,
        action_to_all=(action_mode == 'broadcast'),
        action_mode=action_mode,
        use_displacement=True,
        use_3D=graph_config['use_3D'],
        rot_draping=graph_config['rot_draping'],
        process_workers=process_workers,
        edge_mode=args.edge_mode,
        use_oracle_anchor=use_oracle_anchor,
        singulate_layers=singulate_layers,
        layer_bit_mode=layer_bit_mode,
        layer_labels_path=args.layer_labels_path or None,
        layer_shuffle_map_path=args.layer_shuffle_map or None,
)

if manifest is not None:
        train_dataset = BMDataset(
                root=dataset_dir,
                description=args.description,
                sample_ids=train_ids,
                manifest_path=args.manifest_path,
                **common_dataset_kwargs)
        heldout_dataset = BMDataset(
                root=dataset_dir,
                description=args.description,
                sample_ids=heldout_ids,
                manifest_path=args.manifest_path,
                **common_dataset_kwargs)
        pilot_runs = [(train_dataset, heldout_dataset, len(train_ids))]
else:
        not_subsampled = BMDataset(
                root=dataset_dir,
                description=args.description,
                **common_dataset_kwargs)
        dataset_sizes = [int(item.strip()) for item in args.dataset_sizes.split(',') if item.strip()]
        pilot_runs = [
                (not_subsampled[:size], not_subsampled[size:size + max(1, int(0.1 * size))], size)
                for size in dataset_sizes
        ]

model_path = args.model_path

for initial_dataset, heldout_dataset, train_count in pilot_runs:
        print(f'Training on {train_count} samples; held-out {len(heldout_dataset)}')
        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gnn_manager = GNN_Manager(device)
        gnn_manager.initial_dataset = initial_dataset
        gnn_manager.TRAIN_DATASET = initial_dataset
        gnn_manager.TEST_DATASET = heldout_dataset
        gnn_manager.dataset_dir = [initial_dataset.root]

        arm_tag = f'_{args.lowering_arm}' if args.lowering_arm else ''
        oracle_tag = '_OracleAnchor' if use_oracle_anchor else ''
        action_tag = '_MaskedAction' if action_mode == 'oracle_triangle' else ''
        layer_tag = ''
        if layer_bit_mode == 'oracle':
                layer_tag = '_TopmostLayerOracle'
        elif layer_bit_mode == 'shuffled':
                layer_tag = '_TopmostLayerShuffled'
        if manifest is not None and 'target_limbs' in manifest and not args.model_prefix:
                tl_tag = ','.join(str(x) for x in manifest.get('target_limbs', []))
                model_description = f'TL_{tl_tag}__Uncover_Lowering_Pilot{arm_tag}{oracle_tag}{action_tag}{layer_tag}_{train_count}'
        else:
                model_description = f'{args.model_prefix}{oracle_tag}{action_tag}{layer_tag}_{train_count}{arm_tag}'

        save_dir = os.path.abspath(os.path.join(os.getcwd(), model_path))
        epochs = args.epochs
        proc_layers = args.proc_layers
        learning_rate = args.learning_rate
        seed = args.seed
        batch_size = args.batch_size
        num_workers = args.num_workers
        use_displacement = True

        gnn_manager.initialize_new_model(
                save_dir, None,
                proc_layers, 100, epochs, learning_rate, seed, batch_size, num_workers,
                model_description, use_displacement)
        gnn_manager.set_dataloaders()
        gnn_manager.train(epochs, eval_every_epoch=True, heldout_tag='heldout')
