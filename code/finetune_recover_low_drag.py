#!/usr/bin/env python3
"""Fine-tune a recover GNN checkpoint on accepted low-drag data.

The fine-tuning set is split by source state (not individual actions) to avoid
putting different actions from the same uncover state in both train and
held-out sets.  A small replay subset from the broad recover dataset is
included in training and held-out evaluation to reduce catastrophic forgetting.

This entry point intentionally does not modify ``train_gnns.py``.  The latter
always initializes a fresh model; this script loads an existing checkpoint and
continues training it at a lower learning rate.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from bm_dataset import BMDataset
from gnn_manager import GNN_Manager


STATE_RE = re.compile(r"_s(\d+)(?:_|$)")
TL_RE = re.compile(r"(?:^|_)c_(\d+)(?:_|$)")
TARGET_TLS = (4, 5, 10, 11, 12, 14)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _dataset_root(path: Path) -> Path:
    """Accept either a dataset root or its raw/ directory."""
    path = path.expanduser().resolve()
    if path.name == "raw" and path.is_dir():
        return path.parent
    if (path / "raw").is_dir():
        return path
    raise FileNotFoundError("Expected dataset root containing raw/: %s" % path)


def _list_pkls(raw_dir: Path) -> List[Path]:
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError("Missing raw directory: %s" % raw_dir)
    files = sorted(raw_dir.glob("*.pkl"))
    if not files:
        raise RuntimeError("No PKLs found in %s" % raw_dir)
    return files


def _parse_voxel_size(value: str) -> float:
    text = str(value).strip().lower()
    if text in {"nan", "none", "off", "false"}:
        return float("nan")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError("--voxel-size must be positive or 'nan'")
    return parsed


def _metadata_from_name(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Read target TL and source state index from low-drag filenames."""
    name = path.name
    tl_match = TL_RE.search(name)
    state_match = STATE_RE.search(name)
    tl = int(tl_match.group(1)) if tl_match else None
    state = int(state_match.group(1)) if state_match else None
    return tl, state


def _split_lowdrag_by_state(
    files: Sequence[Path],
    holdout_fraction: float,
    seed: int,
) -> Tuple[List[Path], List[Path], Dict[str, int]]:
    """Stratified state-group split, keeping all actions from a state together."""
    if not 0.0 < float(holdout_fraction) < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")

    grouped = defaultdict(list)
    unknown_counter = 0
    for path in files:
        tl, state = _metadata_from_name(path)
        if state is None:
            # Do not silently group all malformed names together.
            state = "unknown_%06d" % unknown_counter
            unknown_counter += 1
        grouped[(tl, state)].append(path)

    rng = np.random.default_rng(int(seed))
    train, heldout = [], []
    train_groups = Counter()
    heldout_groups = Counter()
    by_tl = defaultdict(list)
    for (tl, state), paths in grouped.items():
        by_tl[tl].append(((tl, state), paths))

    for tl, tl_groups in sorted(by_tl.items(), key=lambda item: str(item[0])):
        order = np.arange(len(tl_groups))
        rng.shuffle(order)
        n_holdout = max(1, int(round(len(tl_groups) * float(holdout_fraction))))
        holdout_indices = set(order[:n_holdout].tolist())
        for idx, (key, paths) in enumerate(tl_groups):
            if idx in holdout_indices:
                heldout.extend(paths)
                heldout_groups[str(tl)] += 1
            else:
                train.extend(paths)
                train_groups[str(tl)] += 1

    train.sort()
    heldout.sort()
    stats = {
        "train_files": len(train),
        "heldout_files": len(heldout),
        "train_groups": int(sum(train_groups.values())),
        "heldout_groups": int(sum(heldout_groups.values())),
        "train_groups_by_tl": dict(train_groups),
        "heldout_groups_by_tl": dict(heldout_groups),
    }
    return train, heldout, stats


def _sample_without_replacement(
    files: Sequence[Path],
    count: int,
    rng: np.random.Generator,
) -> List[Path]:
    files = list(files)
    if count <= 0:
        return []
    if count >= len(files):
        return sorted(files)
    indices = rng.choice(len(files), size=int(count), replace=False)
    return sorted(files[int(i)] for i in indices)


def _make_dataset(
    root: Path,
    files: Sequence[Path],
    description: str,
    process_workers: int,
    edge_threshold: float,
    edge_mode: str,
    voxel_size: float,
    staging_root: Path,
):
    if not files:
        raise RuntimeError("Cannot build an empty BMDataset: %s" % description)
    if str(edge_mode).lower() not in {"radius", "mesh"}:
        raise ValueError("edge_mode must be 'radius' or 'mesh'")

    # The installed BMDataset scans root/raw/*.pkl and does not accept an
    # explicit raw_paths list. Stage symlinks for each split so the
    # state-group split and replay selection remain exact without copying
    # the large pickle files.
    split_root = staging_root / description
    raw_dir = split_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for source in files:
        link = raw_dir / source.name
        if link.is_symlink():
            if link.resolve() == source.resolve():
                continue
            raise FileExistsError("Staging link points to another file: %s" % link)
        if link.exists():
            raise FileExistsError("Staging path already exists: %s" % link)
        os.symlink(str(source), str(link))

    # Each split has a distinct root, so BMDataset's processed cache cannot
    # collide between train and held-out subsets.
    return BMDataset(
        root=str(split_root),
        description="processed",
        recover=True,
        voxel_size=float(voxel_size),
        edge_threshold=float(edge_threshold),
        action_to_all=True,
        use_displacement=True,
        use_3D=False,
        rot_draping=True,
        process_workers=int(process_workers),
        edge_mode=str(edge_mode).lower(),
    )


def _write_split_manifest(
    output_dir: Path,
    train_lowdrag: Sequence[Path],
    heldout_lowdrag: Sequence[Path],
    train_replay: Sequence[Path],
    heldout_replay: Sequence[Path],
):
    manifest = {
        "train_lowdrag": [str(p) for p in train_lowdrag],
        "heldout_lowdrag": [str(p) for p in heldout_lowdrag],
        "train_baseline_replay": [str(p) for p in train_replay],
        "heldout_baseline_replay": [str(p) for p in heldout_replay],
    }
    (output_dir / "finetune_split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def _copy_and_update_config(
    base_model: Path,
    output_dir: Path,
    train_count: int,
    heldout_count: int,
    learning_rate: float,
    epochs: int,
    edge_mode: str,
    edge_threshold: float,
    voxel_size: float,
):
    src = base_model / "config.ini"
    dst = output_dir / "config.ini"
    if not src.exists():
        return
    config = configparser.ConfigParser()
    config.read(str(src))
    if "Dataset" not in config:
        config["Dataset"] = {}
    if "Model" not in config:
        config["Model"] = {}
    config["Dataset"]["num_train_data"] = str(int(train_count))
    config["Dataset"]["num_test_data"] = str(int(heldout_count))
    config["Dataset"]["finetune_base_model"] = str(base_model)
    config["Dataset"]["edge_mode"] = str(edge_mode)
    config["Dataset"]["edge_threshold"] = str(float(edge_threshold))
    config["Dataset"]["voxel_size"] = (
        "nan" if np.isnan(voxel_size) else str(float(voxel_size))
    )
    config["Dataset"]["subsample"] = str(not np.isnan(voxel_size))
    config["Model"]["learning_rate"] = str(float(learning_rate))
    config["Model"]["epochs"] = str(int(epochs))
    with open(dst, "w") as handle:
        config.write(handle)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune a recover GNN checkpoint on accepted low-drag data."
    )
    parser.add_argument("--base-model", required=True, help="Existing GNN model directory.")
    parser.add_argument(
        "--lowdrag-root",
        required=True,
        help="accepted_low_drag_unfolding dataset root or its raw/ directory.",
    )
    parser.add_argument(
        "--baseline-root",
        required=True,
        help="Broad 70k recover dataset root or its raw/ directory.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--base-epoch",
        type=int,
        default=-1,
        help="Numeric base checkpoint epoch; -1 loads the latest model_N.pth.",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument(
        "--baseline-replay-ratio",
        type=float,
        default=0.25,
        help="Fraction of fine-tune train samples drawn from baseline replay.",
    )
    parser.add_argument(
        "--baseline-replay-count",
        type=int,
        default=0,
        help="Override replay count; 0 derives it from --baseline-replay-ratio.",
    )
    parser.add_argument(
        "--baseline-heldout-count",
        type=int,
        default=1000,
        help="Broad-data samples included in held-out evaluation.",
    )
    parser.add_argument("--state-split-seed", type=int, default=1001)
    parser.add_argument("--baseline-sample-seed", type=int, default=2001)
    parser.add_argument("--process-workers", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--edge-threshold", type=float, default=0.04)
    parser.add_argument(
        "--voxel-size",
        default="nan",
        help="Voxel cell size in metres; use 'nan' to disable subsampling.",
    )
    parser.add_argument(
        "--edge-mode",
        choices=["radius", "mesh"],
        default="radius",
        help="Graph edge construction used for all fine-tuning datasets.",
    )
    parser.add_argument("--no-baseline-replay", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    base_model = Path(args.base_model).expanduser().resolve()
    lowdrag_root = _dataset_root(Path(args.lowdrag_root))
    baseline_root = _dataset_root(Path(args.baseline_root))
    output_dir = Path(args.output_dir).expanduser().resolve()
    voxel_size = _parse_voxel_size(args.voxel_size)

    if str(args.edge_mode).lower() not in {"radius", "mesh"}:
        raise ValueError("edge_mode must be 'radius' or 'mesh'")

    if not base_model.is_dir():
        raise FileNotFoundError("Missing base model directory: %s" % base_model)
    if output_dir.exists() and any(output_dir.iterdir()):
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_files = (
            list(checkpoint_dir.glob("*.pth"))
            if checkpoint_dir.is_dir()
            else []
        )
        staging_dir = output_dir / "dataset_staging"
        if checkpoint_files or not staging_dir.is_dir():
            raise FileExistsError(
                "Fine-tune output is not empty and is not a safe staging-only "
                "resume directory; choose a new --output-dir: %s" % output_dir
            )
        print("[Resume] Reusing existing staging-only output directory: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lowdrag_files = _list_pkls(lowdrag_root / "raw")
    baseline_files = _list_pkls(baseline_root / "raw")
    lowdrag_train, lowdrag_heldout, split_stats = _split_lowdrag_by_state(
        lowdrag_files,
        holdout_fraction=float(args.holdout_fraction),
        seed=int(args.state_split_seed),
    )
    if not lowdrag_train or not lowdrag_heldout:
        raise RuntimeError("Low-drag state split produced an empty partition.")

    rng = np.random.default_rng(int(args.baseline_sample_seed))
    heldout_replay = _sample_without_replacement(
        baseline_files,
        int(args.baseline_heldout_count),
        rng,
    )
    heldout_set = set(heldout_replay)
    replay_candidates = [p for p in baseline_files if p not in heldout_set]

    if args.no_baseline_replay:
        train_replay = []
    elif int(args.baseline_replay_count) > 0:
        train_replay_count = int(args.baseline_replay_count)
        train_replay = _sample_without_replacement(
            replay_candidates,
            train_replay_count,
            rng,
        )
    else:
        ratio = float(args.baseline_replay_ratio)
        if not 0.0 <= ratio < 1.0:
            raise ValueError("--baseline-replay-ratio must be in [0, 1)")
        train_replay_count = int(
            round(len(lowdrag_train) * ratio / max(1e-12, 1.0 - ratio))
        )
        train_replay = _sample_without_replacement(
            replay_candidates,
            train_replay_count,
            rng,
        )

    _write_split_manifest(
        output_dir,
        lowdrag_train,
        lowdrag_heldout,
        train_replay,
        heldout_replay,
    )

    desc = output_dir.name.replace("-", "_")
    staging_root = output_dir / "dataset_staging"
    lowdrag_train_ds = _make_dataset(
        lowdrag_root,
        lowdrag_train,
        desc + "_lowdrag_train",
        args.process_workers,
        args.edge_threshold,
        args.edge_mode,
        voxel_size,
        staging_root,
    )
    lowdrag_heldout_ds = _make_dataset(
        lowdrag_root,
        lowdrag_heldout,
        desc + "_lowdrag_heldout",
        args.process_workers,
        args.edge_threshold,
        args.edge_mode,
        voxel_size,
        staging_root,
    )
    train_parts = [lowdrag_train_ds]
    heldout_parts = [lowdrag_heldout_ds]

    if train_replay:
        train_parts.append(
            _make_dataset(
                baseline_root,
                train_replay,
                desc + "_baseline_replay_train",
                args.process_workers,
                args.edge_threshold,
                args.edge_mode,
                voxel_size,
                staging_root,
            )
        )
    if heldout_replay:
        heldout_parts.append(
            _make_dataset(
                baseline_root,
                heldout_replay,
                desc + "_baseline_replay_heldout",
                args.process_workers,
                args.edge_threshold,
                args.edge_mode,
                voxel_size,
                staging_root,
            )
        )

    train_dataset = ConcatDataset(train_parts)
    heldout_dataset = ConcatDataset(heldout_parts)

    print(
        "[FineTune] lowdrag train=%d heldout=%d; baseline replay train=%d heldout=%d; "
        "total train=%d heldout=%d"
        % (
            len(lowdrag_train_ds),
            len(lowdrag_heldout_ds),
            len(train_replay),
            len(heldout_replay),
            len(train_dataset),
            len(heldout_dataset),
        )
    )
    print("[FineTune] split_stats=%s" % json.dumps(split_stats, sort_keys=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager = GNN_Manager(device)
    base_epoch = None if int(args.base_epoch) < 0 else int(args.base_epoch)
    manager.load_model_from_checkpoint(
        str(base_model),
        model_checkpoint_number=base_epoch,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        new_save_dir=str(output_dir),
    )

    # load_model_from_checkpoint restores the base optimizer state. Fine-tuning
    # uses a deliberately smaller LR while keeping the loaded model weights.
    manager.use_displacement = True
    manager.batch_size = int(args.batch_size)
    manager.num_workers = int(args.num_workers)
    manager.args.learning_rate = float(args.learning_rate)
    for group in manager.optimizer.param_groups:
        group["lr"] = float(args.learning_rate)
    manager.initial_dataset = lowdrag_train_ds
    manager.TRAIN_DATASET = train_dataset
    manager.TEST_DATASET = heldout_dataset
    manager.dataset_dir = [str(lowdrag_root), str(baseline_root)]
    manager.model_checkpoint_number = 0
    manager.set_dataloaders()
    manager.train(int(args.epochs), eval_every_epoch=True, heldout_tag="lowdrag_heldout")
    manager.writer.close()

    summary = {
        "base_model": str(base_model),
        "output_dir": str(output_dir),
        "lowdrag_root": str(lowdrag_root),
        "baseline_root": str(baseline_root),
        "base_epoch": base_epoch,
        "train_count": len(train_dataset),
        "heldout_count": len(heldout_dataset),
        "lowdrag_train_count": len(lowdrag_train),
        "lowdrag_heldout_count": len(lowdrag_heldout),
        "baseline_replay_train_count": len(train_replay),
        "baseline_replay_heldout_count": len(heldout_replay),
        "baseline_replay_ratio": (
            float(len(train_replay)) / float(len(train_dataset))
            if len(train_dataset)
            else 0.0
        ),
        "holdout_fraction": float(args.holdout_fraction),
        "learning_rate": float(args.learning_rate),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "process_workers": int(args.process_workers),
        "num_workers": int(args.num_workers),
        "graph_config": "2D",
        "edge_mode": args.edge_mode,
        "edge_threshold": float(args.edge_threshold),
        "voxel_size": "nan" if np.isnan(voxel_size) else float(voxel_size),
        "action_mode": "broadcast",
        "split_stats": split_stats,
    }
    (output_dir / "finetune_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"
    )
    _copy_and_update_config(
        base_model,
        output_dir,
        len(train_dataset),
        len(heldout_dataset),
        float(args.learning_rate),
        int(args.epochs),
        args.edge_mode,
        float(args.edge_threshold),
        voxel_size,
    )
    print("Fine-tune complete: %s" % output_dir)


if __name__ == "__main__":
    main()
