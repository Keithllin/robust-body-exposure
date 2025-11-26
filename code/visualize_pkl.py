#!/usr/bin/env python3
"""Utility to render a single CMA evaluation pickle as a figure."""
import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Ensure Assistive Gym modules are importable
sys.path.insert(0, '/home/keithlin/RCHI/robust-body-exposure-recovering/assistive-gym-fem')

from assistive_gym.envs.bu_gnn_util import (  # noqa: E402
    get_body_points_from_obs,
    get_covered_status,
    get_recovering_reward,
    get_uncovering_reward,
    scale_action,
)
from cma_gnn_util import (  # noqa: E402
    compute_fscore_recover,
    compute_fscore_uncover,
    generate_figure_recover,
    generate_figure_uncover,
)


def _to_xy(points: np.ndarray) -> np.ndarray:
    """Drop the z column if present so reward utilities get XY cloth coordinates."""
    arr = np.asarray(points)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array for cloth points, got shape {arr.shape}")
    if arr.shape[1] == 3:
        return np.delete(arr, 2, axis=1)
    return arr


def _load_raw_pkl(path: Path) -> dict:
    with path.open('rb') as handle:
        return pickle.load(handle)


def _ensure_key(data: dict, key: str):
    if key not in data:
        raise KeyError(f"Missing key '{key}' in pickle data")
    return data[key]


def _get_uncover_action(raw: dict) -> np.ndarray:
    action = raw.get('uncover_action')
    if action is None:
        action = raw.get('action')
    if action is None:
        raise KeyError("Neither 'uncover_action' nor fallback 'action' present in pickle")
    return scale_action(action)


def _get_recover_action(raw: dict) -> np.ndarray:
    action = raw.get('recover_action')
    if action is None:
        raise KeyError("Recover pickle missing 'recover_action'")
    return scale_action(action)


def _covered_status(body_pts: np.ndarray, cloth_pts: np.ndarray) -> np.ndarray:
    return get_covered_status(body_pts, _to_xy(cloth_pts))


def _build_output_paths(outdir: Path, stem: str, image_format: str, save_html: bool) -> Tuple[Path, Optional[Path]]:
    image_path = outdir / f"{stem}.{image_format}"
    html_path = outdir / f"{stem}.html" if save_html else None
    return image_path, html_path


def _write_figure(fig, image_path: Path, html_path: Optional[Path]) -> List[Path]:
    saved: List[Path] = []
    try:
        fig.write_image(str(image_path))
        saved.append(image_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Failed to write image via kaleido ({exc}); falling back to HTML only.")
        if html_path is None:
            html_path = image_path.with_suffix('.html')
    if html_path is not None:
        fig.write_html(str(html_path))
        saved.append(html_path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a single CMA evaluation pickle")
    parser.add_argument('--pkl', required=True, type=Path, help='Path to the raw evaluation .pkl file')
    parser.add_argument('--outdir', default=Path('./tmp_images'), type=Path, help='Directory to save figures')
    parser.add_argument('--image-format', default='png', choices=['png', 'jpg', 'jpeg', 'webp', 'svg'], help='Image format for static export')
    parser.add_argument('--tag', default=None, help='Optional suffix for the output filename')
    parser.add_argument('--compare-subplots', action='store_true', help='Show CMA vs sim results in separate subplots')
    parser.add_argument('--plot-initial', action='store_true', help='Overlay initial cloth state for context')
    parser.add_argument('--html', action='store_true', help='Always emit an interactive HTML file alongside the static image')
    args = parser.parse_args()

    raw = _load_raw_pkl(args.pkl)
    recover_mode = bool(raw.get('recovering', False))

    sim_info = _ensure_key(raw, 'sim_info')
    sim_info_body = _ensure_key(sim_info, 'info')
    body_info = _ensure_key(sim_info_body, 'human_body_info')

    target_limb_code = _ensure_key(raw, 'target_limb_code')
    human_pose = _ensure_key(raw, 'human_pose')
    all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)

    info_dict = _ensure_key(raw, 'info')
    cloth_initial = np.asarray(_ensure_key(info_dict, 'cloth_initial')[1])
    cloth_final = np.asarray(_ensure_key(info_dict, 'cloth_final')[1])
    cloth_intermediate = None
    if recover_mode:
        cloth_intermediate = np.asarray(_ensure_key(info_dict, 'cloth_intermediate')[1])

    pred = np.asarray(_ensure_key(_ensure_key(raw, 'cma_info'), 'best_pred'))

    initial_covered_status = _covered_status(all_body_points, cloth_initial)
    sim_covered_status = _covered_status(all_body_points, cloth_final)
    cma_covered_status = get_covered_status(all_body_points, pred)

    uncover_action = _get_uncover_action(raw)
    recover_action = _get_recover_action(raw) if recover_mode else None

    sim_reward = raw['sim_info'].get('recover_reward') if recover_mode else raw['sim_info'].get('reward')
    cma_reward = raw['cma_info'].get('best_reward')
    sim_reward_info = None
    cma_reward_info = None

    if recover_mode:
        intermediate_covered_status = _covered_status(all_body_points, cloth_intermediate)
        sim_fscore, sim_info_fscore = compute_fscore_recover(initial_covered_status, intermediate_covered_status, sim_covered_status, True)
        cma_fscore, cma_info_fscore = compute_fscore_recover(initial_covered_status, intermediate_covered_status, cma_covered_status, True)

        sim_reward, sim_reward_info = get_recovering_reward(
            recover_action,
            all_body_points,
            _to_xy(cloth_initial),
            _to_xy(cloth_intermediate),
            _to_xy(cloth_final),
            True,
        )
        cma_reward, cma_reward_info = get_recovering_reward(
            recover_action,
            all_body_points,
            _to_xy(cloth_initial),
            _to_xy(cloth_intermediate),
            pred,
            True,
        )

        fig = generate_figure_recover(
            sim_info_fscore,
            cma_info_fscore,
            sim_reward,
            sim_reward_info,
            cma_reward,
            cma_reward_info,
            target_limb_code,
            uncover_action,
            recover_action,
            body_info,
            all_body_points,
            cloth_initial,
            final_cloths=[cloth_final, pred],
            cloth_intermediate=cloth_intermediate,
            initial_covered_status=[initial_covered_status, intermediate_covered_status],
            covered_statuses=[sim_covered_status, cma_covered_status],
            fscores=[sim_fscore, cma_fscore],
            plot_initial=args.plot_initial,
            compare_subplots=args.compare_subplots,
        )
    else:
        sim_fscore = compute_fscore_uncover(initial_covered_status, sim_covered_status)
        cma_fscore = compute_fscore_uncover(initial_covered_status, cma_covered_status)

        sim_reward = get_uncovering_reward(
            uncover_action,
            all_body_points,
            _to_xy(cloth_initial),
            _to_xy(cloth_final),
        )
        cma_reward = get_uncovering_reward(
            uncover_action,
            all_body_points,
            _to_xy(cloth_initial),
            pred,
        )

        fig = generate_figure_uncover(
            sim_reward,
            cma_reward,
            target_limb_code,
            uncover_action,
            body_info,
            all_body_points,
            cloth_initial,
            final_cloths=[cloth_final, pred],
            initial_covered_status=[initial_covered_status],
            covered_statuses=[sim_covered_status, cma_covered_status],
            fscores=[sim_fscore, cma_fscore],
            plot_initial=args.plot_initial,
            compare_subplots=args.compare_subplots,
        )

    stem = args.tag or args.pkl.stem
    args.outdir.mkdir(parents=True, exist_ok=True)
    image_path, html_path = _build_output_paths(args.outdir, stem, args.image_format, args.html)
    saved_paths = _write_figure(fig, image_path, html_path)

    print("Saved:")
    for path in saved_paths:
        print(f"  - {path}")


if __name__ == '__main__':
    main()
