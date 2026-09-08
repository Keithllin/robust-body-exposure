"""Summarize pred / sensor / snap Recover plans from one pose directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from trial_layout import (
    recover_pred_dir,
    recover_sensor_dir,
    recover_snap_dir,
)


SOURCE_DIRS = {
    "pred": recover_pred_dir,
    "sensor": recover_sensor_dir,
    "snap": recover_snap_dir,
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _motion_stats(eval_path: Path) -> dict[str, float]:
    if not eval_path.is_file():
        return {}
    try:
        import pickle

        with eval_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return {}
    intermediate = np.asarray(payload.get("cloth_intermediate"), dtype=np.float64)
    final = np.asarray(payload.get("cloth_final"), dtype=np.float64)
    if (
        intermediate.ndim != 2
        or final.ndim != 2
        or len(intermediate) == 0
        or len(intermediate) != len(final)
    ):
        return {}
    delta = np.linalg.norm(final[:, :2] - intermediate[:, :2], axis=1)
    return {
        "median_m": float(np.median(delta)),
        "mean_m": float(np.mean(delta)),
        "p95_m": float(np.percentile(delta, 95)),
    }


def collect_source_row(output_dir: Path) -> dict[str, Any]:
    meta = _read_json(output_dir / "recover_runtime_metadata.json")
    stats = meta.get("graph_stats") or {}
    motion = _motion_stats(output_dir / "cma_eval_data.pkl")
    return {
        "output_dir": str(output_dir),
        "available": (output_dir / "recover_runtime_metadata.json").is_file(),
        "intermediate_source": meta.get("intermediate_source"),
        "num_nodes": stats.get("num_nodes"),
        "occupied_columns": stats.get("occupied_columns"),
        "pts_per_column": stats.get("pts_per_column"),
        "reward": meta.get("reward"),
        "scaled_action": meta.get("scaled_action"),
        "predicted_motion": motion,
        "sensor_snap": meta.get("sensor_snap"),
        "graph_stats": stats,
    }


def collect_pose_comparison(pose_dir: Path) -> dict[str, Any]:
    pose_dir = Path(pose_dir)
    sources = {
        name: collect_source_row(resolver(pose_dir))
        for name, resolver in SOURCE_DIRS.items()
    }
    actions = {
        name: np.asarray(row["scaled_action"], dtype=np.float64).reshape(4)
        if row.get("scaled_action") is not None
        else None
        for name, row in sources.items()
    }
    disagreement: dict[str, Any] = {}
    pairs = (("pred", "sensor"), ("pred", "snap"), ("sensor", "snap"))
    for left, right in pairs:
        a = actions[left]
        b = actions[right]
        if a is None or b is None:
            continue
        grasp = float(np.linalg.norm(a[:2] - b[:2]))
        release = float(np.linalg.norm(a[2:] - b[2:]))
        disagreement[f"{left}_vs_{right}"] = {
            "grasp_l2_m": grasp,
            "release_l2_m": release,
            "mean_endpoint_l2_m": float(0.5 * (grasp + release)),
        }
    return {
        "pose_dir": str(pose_dir),
        "sources": sources,
        "action_disagreement": disagreement,
    }


def write_pose_comparison(pose_dir: Path) -> dict[str, Any]:
    payload = collect_pose_comparison(pose_dir)
    out = Path(pose_dir) / "source_comparison.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def write_comparison_montage(pose_dir: Path, output: Path | None = None) -> Path | None:
    """Side-by-side recover_prediction.png for pred / sensor / snap."""

    pose_dir = Path(pose_dir)
    paths = [
        (name, resolver(pose_dir) / "recover_prediction.png")
        for name, resolver in SOURCE_DIRS.items()
    ]
    existing = [(name, path) for name, path in paths if path.is_file()]
    if len(existing) < 2:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    images = []
    for name, path in existing:
        image = Image.open(path).convert("RGB")
        images.append((name, image))
    height = min(image.height for _, image in images)
    label_h = 36
    resized = []
    for name, image in images:
        if image.height != height:
            width = int(round(image.width * height / image.height))
            image = image.resize((width, height))
        resized.append((name, image))
    montage = Image.new(
        "RGB",
        (sum(image.width for _, image in resized), height + label_h),
        color=(255, 255, 255),
    )
    try:
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(montage)
        font = ImageFont.load_default()
    except Exception:
        draw = None
        font = None
    x = 0
    for name, image in resized:
        montage.paste(image, (x, label_h))
        if draw is not None:
            draw.text((x + 8, 8), f"recover_{name}", fill=(0, 0, 0), font=font)
        x += image.width
    output = (
        Path(output)
        if output is not None
        else pose_dir / "recover_source_comparison.png"
    )
    montage.save(output)
    return output
