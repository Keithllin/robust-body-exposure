#%%
import argparse
import os.path as osp
import pickle
import sys
from pathlib import Path

import numpy as np

from cma_gnn_util import *  # noqa: F401,F403

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / 'code') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'code'))
sys.path.insert(0, str(REPO_ROOT / 'assistive-gym-fem'))

from assistive_gym.envs.bu_gnn_util import *  # noqa: F401,F403


MODEL_DIR = REPO_ROOT / 'trained_models/FINAL_MODELS/Recover'


GRAPH_NODE_CONFIGS = {
    '2D': {
        'voxel_size': float('nan'),
        'rot_drape': True,
        'filter_drape': False,
        'use_3D': False,
    },
    '2D_voxelxyz05': {
        'voxel_size': 0.05,
        'rot_drape': True,
        'filter_drape': False,
        'use_3D': False,
    },
}


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


def get_body_info(raw_data):
    return raw_data['sim_info']['info'].get('human_body_info', raw_data['info'].get('human_body_info'))


def _state_from_entry(entry, name):
    if isinstance(entry, (tuple, list)) and len(entry) >= 2:
        entry = entry[1]
    state = np.asarray(entry, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] not in (2, 3) or len(state) == 0:
        raise ValueError(f'{name} must be a non-empty N x 2/N x 3 state, got {state.shape}')
    return state


def _rotation_matrix(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.asarray([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
    ], dtype=np.float32)


def _apply_draping_rotation(state, enabled):
    state = np.asarray(state, dtype=np.float32).copy()
    if not enabled:
        return state

    right_rotation = _rotation_matrix([0.0, 1.0, 0.0], -np.pi / 2.0)
    left_rotation = _rotation_matrix([0.0, 1.0, 0.0], np.pi / 2.0)
    right_center = np.asarray([0.44, 0.0, 0.58], dtype=np.float32)
    left_center = np.asarray([-0.44, 0.0, 0.58], dtype=np.float32)

    movable = state[:, 2] < 0.575
    right_mask = movable & (state[:, 0] > 0.0)
    left_mask = movable & (state[:, 0] < 0.0)
    state[right_mask] = (
        (state[right_mask] - right_center) @ right_rotation.T + right_center
    )
    state[left_mask] = (
        (state[left_mask] - left_center) @ left_rotation.T + left_center
    )
    return state


def _reconstruct_graph_nodes(state, graph_config):
    """Reproduce Runtime_Graph preprocessing for a legacy result PKL."""
    state = _state_from_entry(state, 'Recover graph input state')
    if state.shape[1] != 3:
        raise ValueError(
            'Cannot reconstruct graph nodes from a non-3D preprocessed state: '
            f'{state.shape}'
        )

    state = _apply_draping_rotation(state, graph_config['rot_drape'])
    if graph_config['filter_drape']:
        state = state[state[:, 2] > 0.58]
    voxel_size = graph_config['voxel_size']
    if np.isfinite(voxel_size):
        from voxel_ops import xyz_voxel_representatives
        state, _ = xyz_voxel_representatives(state, voxel_size)
    if not graph_config['use_3D']:
        state = np.delete(state, 2, axis=1)
    return np.asarray(state, dtype=np.float32)


def get_intermediate_for_image(raw_data, source, graph_config=None):
    """Return the state drawn in the intermediate subplot.

    Evaluation metrics remain based on the simulator GT intermediate.  The
    optional prediction graph is only a visualization replacement, and is
    loaded from the exact state saved by run_robe_sim_new_opt.py.
    """
    gt_state = _state_from_entry(
        raw_data.get('info', {}).get('cloth_intermediate'),
        'GT intermediate',
    )
    cma_info = raw_data.get('cma_info', {})
    # Prefer the exact post-draping/post-voxelization node state.  The older
    # recover_graph_input_state is the full-resolution state before
    # Runtime_Graph preprocessing and must not be used for voxel-model
    # intermediate plots when the node state is available.
    prediction_state = None
    prediction_label = 'Recover graph nodes'
    exact_node_state = False
    if isinstance(cma_info, dict):
        prediction_state = cma_info.get('recover_graph_node_state')
        if prediction_state is not None:
            exact_node_state = True
            voxel_size = cma_info.get('recover_graph_voxel_size')
            state_source = cma_info.get('recover_state_source', 'unknown')
            source_label = {
                'gt_intermediate': 'GT',
                'uncover_prediction': 'Uncover prediction',
            }.get(str(state_source), str(state_source))
            if voxel_size is not None:
                prediction_label = (
                    f'{source_label} → Recover voxel graph '
                    f'(vs={float(voxel_size):g})'
                )
            else:
                prediction_label = f'{source_label} → Recover graph nodes'
        else:
            # Kept only as a compatibility check for old runs.  Do not
            # silently plot this as a voxel graph: it predates graph
            # preprocessing and is not the state consumed by a voxel model.
            prediction_state = cma_info.get('recover_graph_input_state')
    # ``predicted_graph`` is retained as a backwards-compatible alias for
    # existing prediction-state commands.  The stored state is actually the
    # post-preprocessing graph-node array and can also represent GT voxel
    # input, so ``graph_nodes`` is the precise name for new commands.
    if source == 'predicted_graph':
        source = 'graph_nodes'
    if source == 'auto':
        source = 'graph_nodes' if prediction_state is not None else 'gt'
    if source == 'gt':
        return gt_state, 'GT intermediate'
    if source not in ('graph_nodes',):
        raise ValueError(f'Unsupported intermediate source: {source}')

    if not exact_node_state and prediction_state is not None and graph_config is not None:
        prediction_state = _reconstruct_graph_nodes(prediction_state, graph_config)
        voxel_size = graph_config['voxel_size']
        state_source = cma_info.get('recover_state_source', 'unknown') if isinstance(cma_info, dict) else 'unknown'
        source_label = {
            'gt_intermediate': 'GT',
            'uncover_prediction': 'Uncover prediction',
        }.get(str(state_source), str(state_source))
        if np.isfinite(voxel_size):
            prediction_label = (
                f'{source_label} → Recover voxel graph '
                f'(vs={float(voxel_size):g}; reconstructed)'
            )
        else:
            prediction_label = f'{source_label} → Recover graph nodes (reconstructed)'
        exact_node_state = True

    if prediction_state is None:
        state_source = cma_info.get('recover_state_source') if isinstance(cma_info, dict) else None
        if graph_config is not None and state_source != 'uncover_prediction':
            prediction_state = _reconstruct_graph_nodes(gt_state, graph_config)
            voxel_size = graph_config['voxel_size']
            prediction_label = (
                f'GT → Recover voxel graph (vs={float(voxel_size):g}; reconstructed)'
                if np.isfinite(voxel_size)
                else 'GT → Recover graph nodes (reconstructed)'
            )
            exact_node_state = True
        else:
            raise ValueError(
                'graph_nodes requested, but this raw result does not contain '
                'a Recover graph state that can be reconstructed.'
            )

    if not exact_node_state:
        raise ValueError(
            'This raw result only contains the pre-voxel Recover input state, '
            'not the post-voxel graph nodes. Pass --graph-config '
            '2D_voxelxyz05 to reconstruct it or rerun the evaluation.'
        )
    return _state_from_entry(prediction_state, 'Recover graph nodes'), prediction_label


def add_intermediate_source_annotation(fig, label):
    if label == 'GT intermediate':
        return fig
    annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    annotations.append(dict(
        text=f'Intermediate shown: {label}',
        x=1.0,
        y=1.02,
        xref='paper',
        yref='paper',
        xanchor='right',
        yanchor='bottom',
        showarrow=False,
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='rgba(0,0,0,0.25)',
        borderwidth=1,
        font=dict(size=13, color='rgba(0,0,0,1)'),
    ))
    fig.update_layout(annotations=annotations)
    return fig


def add_uncover_entropy_annotation(fig, raw_data):
    cma_info = raw_data.get('cma_info', {})
    xy_entropy = cma_info.get('best_pred_xy_entropy')
    entropy_weight = cma_info.get('entropy_weight')
    overlap_penalty = cma_info.get('best_pred_overlap_penalty')
    regularized_cost = cma_info.get('best_pred_regularized_cost')
    reward_raw = cma_info.get('best_pred_uncover_reward_raw')

    fields = []
    if xy_entropy is not None:
        fields.append(f"Pred XY Entropy: {float(xy_entropy):.3f}")
    if entropy_weight is not None:
        fields.append(f"Entropy Weight: {float(entropy_weight):.1f}")
    if overlap_penalty is not None:
        fields.append(f"Overlap Penalty: {float(overlap_penalty):.3f}")
    if reward_raw is not None:
        fields.append(f"Pred Reward Raw: {float(reward_raw):.2f}")
    if regularized_cost is not None:
        fields.append(f"Regularized Cost: {float(regularized_cost):.2f}")
    if not fields:
        return fig

    annotation = dict(
        text="<br>".join(fields),
        x=1.0,
        y=1.0,
        xref='paper',
        yref='paper',
        xanchor='right',
        yanchor='top',
        showarrow=False,
        align='right',
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='rgba(0,0,0,0.25)',
        borderwidth=1,
        font=dict(size=14, color='rgba(0,0,0,1)'),
    )

    annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    annotations.append(annotation)
    fig.update_layout(annotations=annotations)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arg_model',
        type=str,
        required=True,
        help='Model directory or path relative to trained_models/FINAL_MODELS/Recover.',
    )
    parser.add_argument('--eval-dir-name', type=str, default='')
    parser.add_argument('--eval-condition', type=str, default='')
    parser.add_argument('--raw-dir', type=str, default='')
    parser.add_argument('--max-images', type=int, default=0, help='0 means all images')
    parser.add_argument('--tlc', type=int, default=-1)
    parser.add_argument('--output-format', type=str, default='auto', choices=['auto', 'png', 'html'])
    parser.add_argument('--image-style', type=str, default='compare', choices=['compare', 'sim-only'])
    parser.add_argument('--intermediate-source', type=str, default='gt',
                        choices=['gt', 'graph_nodes', 'predicted_graph', 'auto'],
                        help='State shown in Recover intermediate subplot; graph_nodes is the exact post-voxel Recover input. Metrics still use GT intermediate.')
    parser.add_argument('--graph-config', type=str, default=None,
                        choices=list(GRAPH_NODE_CONFIGS),
                        help='Optional graph config for reconstructing graph nodes in legacy PKLs that lack recover_graph_node_state.')
    args = parser.parse_args()

    model_path = resolve_model_path(args.arg_model)
    raw_dir = resolve_raw_dir(model_path, args.eval_dir_name, args.eval_condition, args.raw_dir)
    eval_name = raw_dir.parent.name
    image_dir = model_path / args.eval_dir_name / '_images' / eval_name
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f'Model path: {model_path}')
    print(f'Raw dir: {raw_dir}')
    print(f'Image dir: {image_dir}')

    filenames = sorted(raw_dir.glob('*.pkl'))
    if args.max_images > 0:
        filenames = filenames[:args.max_images]

    for path in filenames:
        with open(path, 'rb') as handle:
            raw_data = pickle.load(handle)

        target_limb_code = raw_data['target_limb_code']
        if args.tlc >= 0 and target_limb_code != args.tlc:
            continue

        pred = raw_data['cma_info']['best_pred']
        recover = raw_data['recovering']
        if len(raw_data['sim_info']['info']) <= 2:
            continue

        human_pose = raw_data['human_pose']
        body_info = get_body_info(raw_data)
        all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)

        cloth_initial = np.array(raw_data['info']['cloth_initial'][1])
        cloth_final = np.array(raw_data['info']['cloth_final'][1])
        initial_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_initial), 2, axis=1))
        sim_covered_status = get_covered_status(all_body_points, np.delete(np.array(cloth_final), 2, axis=1))
        pred_covered_status = get_covered_status(all_body_points, pred)

        uncover_action_raw = raw_data['uncover_action']
        uncover_action = scale_action(uncover_action_raw) if len(uncover_action_raw) == 4 else np.array([])
        seed = path.name.split('_')[2]

        if recover:
            recover_action_raw = raw_data['recover_action']
            if len(recover_action_raw) != 4:
                print(f'Skip {path.name}: recover action is not 4D')
                continue
            recover_action = scale_action(recover_action_raw)
            cloth_intermediate_gt = np.array(raw_data['info']['cloth_intermediate'][1])
            cloth_intermediate, intermediate_label = get_intermediate_for_image(
                raw_data,
                args.intermediate_source,
                GRAPH_NODE_CONFIGS.get(args.graph_config),
            )
            if args.image_style == 'sim-only':
                fig = generate_figure_data_collection(
                    target_limb_code,
                    uncover_action,
                    recover_action,
                    body_info,
                    all_body_points,
                    cloth_initial,
                    cloth_intermediate,
                    cloth_final,
                )
            else:
                intermediate_covered_status = get_covered_status(all_body_points, np.delete(cloth_intermediate_gt, 2, axis=1))

                sim_fscore, sim_info_fscore = compute_fscore_recover(initial_covered_status, intermediate_covered_status, sim_covered_status, True)
                pred_fscore, pred_info_fscore = compute_fscore_recover(initial_covered_status, intermediate_covered_status, pred_covered_status, True)

                sim_reward, sim_reward_info = get_recovering_reward(
                    recover_action,
                    all_body_points,
                    np.delete(np.array(cloth_initial), 2, axis=1),
                    np.delete(cloth_intermediate_gt, 2, axis=1),
                    np.delete(np.array(cloth_final), 2, axis=1),
                    True,
                )
                pred_reward, pred_reward_info = get_recovering_reward(
                    recover_action,
                    all_body_points,
                    np.delete(np.array(cloth_initial), 2, axis=1),
                    np.delete(cloth_intermediate_gt, 2, axis=1),
                    pred,
                    True,
                )

                fig = generate_figure_recover(
                    sim_info_fscore,
                    pred_info_fscore,
                    sim_reward,
                    sim_reward_info,
                    pred_reward,
                    pred_reward_info,
                    target_limb_code,
                    uncover_action,
                    recover_action,
                    body_info,
                    all_body_points,
                    cloth_initial,
                    final_cloths=[cloth_final, pred],
                    cloth_intermediate=cloth_intermediate,
                    initial_covered_status=[initial_covered_status, intermediate_covered_status],
                    covered_statuses=[sim_covered_status, pred_covered_status],
                    fscores=[sim_fscore, pred_fscore],
                    plot_initial=True,
                    compare_subplots=False,
                )
                fig = add_intermediate_source_annotation(fig, intermediate_label)
        else:
            sim_fscore = compute_fscore_uncover(initial_covered_status, sim_covered_status)
            pred_fscore = compute_fscore_uncover(initial_covered_status, pred_covered_status)
            sim_reward = get_uncovering_reward(
                uncover_action,
                all_body_points,
                np.delete(np.array(cloth_initial), 2, axis=1),
                np.delete(np.array(cloth_final), 2, axis=1),
            )
            pred_reward = get_uncovering_reward(
                uncover_action,
                all_body_points,
                np.delete(np.array(cloth_initial), 2, axis=1),
                pred,
            )
            fig = generate_figure_uncover(
                sim_reward,
                pred_reward,
                target_limb_code,
                uncover_action,
                body_info,
                all_body_points,
                cloth_initial,
                final_cloths=[cloth_final, pred],
                initial_covered_status=[initial_covered_status],
                covered_statuses=[sim_covered_status, pred_covered_status],
                fscores=[sim_fscore, pred_fscore],
                plot_initial=True,
                compare_subplots=False,
            )
            fig = add_uncover_entropy_annotation(fig, raw_data)

        stem = f'{target_limb_code}_{seed}'
        png_path = osp.join(image_dir, f'{stem}.png')
        html_path = osp.join(image_dir, f'{stem}.html')

        if args.output_format == 'html':
            fig.write_html(html_path)
            print(f'Wrote {html_path}')
            continue

        if args.output_format == 'png':
            fig.write_image(png_path)
            print(f'Wrote {png_path}')
            continue

        try:
            fig.write_image(png_path)
            print(f'Wrote {png_path}')
        except Exception as exc:
            fig.write_html(html_path)
            print(f'PNG export failed ({type(exc).__name__}); wrote fallback HTML: {html_path}')


if __name__ == '__main__':
    main()
