import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cma_gnn_util import compute_fscore_uncover, get_body_point_colors_uncovering, target_names

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "assistive-gym-fem"))

from assistive_gym.envs.bu_gnn_util import (  # noqa: E402
    get_body_points_from_obs,
    get_covered_status,
    get_uncovering_reward,
    scale_action,
)


def load_pkl(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def parse_seed(path, data):
    if "seed" in data:
        return int(data["seed"])
    parts = Path(path).name.split("_")
    if len(parts) >= 3:
        return int(parts[2])
    raise ValueError(f"Cannot parse seed from {path}")


def index_raw_dir(raw_dir):
    indexed = {}
    for path in sorted(Path(raw_dir).expanduser().resolve().glob("*.pkl")):
        data = load_pkl(path)
        tl = int(data["target_limb_code"])
        seed = parse_seed(path, data)
        indexed[(tl, seed)] = path
    return indexed


def fmt(value, digits=3):
    if value is None:
        return "nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def normalized_xy_occupancy_entropy(cloth_final_2d, cloth_initial_2d, grid_size=0.05):
    cloth_final_2d = np.asarray(cloth_final_2d, dtype=np.float32)
    cloth_initial_2d = np.asarray(cloth_initial_2d, dtype=np.float32)
    xy_stack = np.vstack([cloth_initial_2d[:, :2], cloth_final_2d[:, :2]])
    xy_min = np.min(xy_stack, axis=0)
    xy_max = np.max(xy_stack, axis=0)
    spans = np.maximum(xy_max - xy_min, float(grid_size))
    bins = np.maximum(np.ceil(spans / float(grid_size)).astype(int), 1)
    total_cells = int(bins[0] * bins[1])
    if total_cells <= 1:
        return 0.0

    coords = np.floor((cloth_final_2d[:, :2] - xy_min) / float(grid_size)).astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, bins[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, bins[1] - 1)
    flat = coords[:, 0] * bins[1] + coords[:, 1]
    counts = np.bincount(flat, minlength=total_cells).astype(np.float64)
    probs = counts[counts > 0] / float(np.sum(counts))
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / math.log(float(total_cells))


def target_name(tl):
    try:
        if isinstance(target_names, dict):
            return target_names.get(tl, "")
        return target_names[int(tl)]
    except (IndexError, KeyError, TypeError, ValueError):
        return ""


def axis_ref(row, col, ncols=3):
    axis_num = (row - 1) * ncols + col
    return "x" if axis_num == 1 else f"x{axis_num}", "y" if axis_num == 1 else f"y{axis_num}"


def add_action_arrow(fig, action, row, col):
    xref, yref = axis_ref(row, col)
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=[action[0]],
            y=[action[1]],
            marker=dict(color="rgba(0,0,0,1)", size=10),
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.add_annotation(
        x=action[2],
        y=action[3],
        ax=action[0],
        ay=action[1],
        xref=xref,
        yref=yref,
        axref=xref,
        ayref=yref,
        text="",
        showarrow=True,
        arrowhead=3,
        arrowwidth=3,
        arrowcolor="rgb(0,0,0)",
    )


def build_run_payload(path):
    data = load_pkl(path)
    target_limb_code = int(data["target_limb_code"])
    human_pose = data["human_pose"]
    body_info = data["sim_info"]["info"].get("human_body_info", data["info"].get("human_body_info"))
    all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code, body_info=body_info)

    cloth_initial = np.asarray(data["info"]["cloth_initial"][1], dtype=np.float32)
    cloth_final = np.asarray(data["info"]["cloth_final"][1], dtype=np.float32)
    pred = np.asarray(data["cma_info"]["best_pred"], dtype=np.float32)
    uncover_action_policy = np.asarray(data["uncover_action"], dtype=np.float32)
    uncover_action = scale_action(uncover_action_policy) if uncover_action_policy.shape[0] == 4 else np.array([])

    cloth_initial_2d = np.delete(cloth_initial, 2, axis=1)
    cloth_final_2d = np.delete(cloth_final, 2, axis=1)
    initial_status = get_covered_status(all_body_points, cloth_initial_2d)
    sim_status = get_covered_status(all_body_points, cloth_final_2d)
    pred_status = get_covered_status(all_body_points, pred)
    sim_f1 = compute_fscore_uncover(initial_status, sim_status)
    pred_f1 = compute_fscore_uncover(initial_status, pred_status)
    sim_reward = get_uncovering_reward(uncover_action, all_body_points, cloth_initial_2d, cloth_final_2d)
    pred_reward = get_uncovering_reward(uncover_action, all_body_points, cloth_initial_2d, pred)

    cma_info = data.get("cma_info", {})
    best_pred_xy_entropy = cma_info.get("best_pred_xy_entropy")
    if best_pred_xy_entropy is None:
        best_pred_xy_entropy = normalized_xy_occupancy_entropy(pred, cloth_initial_2d)
    return {
        "path": str(Path(path).resolve()),
        "data": data,
        "target_limb_code": target_limb_code,
        "seed": parse_seed(path, data),
        "body_info": body_info,
        "all_body_points": np.asarray(all_body_points, dtype=np.float32),
        "cloth_initial": cloth_initial,
        "cloth_final": cloth_final,
        "pred": pred,
        "uncover_action": np.asarray(uncover_action, dtype=np.float32),
        "initial_status": initial_status,
        "sim_status": sim_status,
        "pred_status": pred_status,
        "sim_f1": float(sim_f1),
        "pred_f1": float(pred_f1),
        "sim_reward": float(sim_reward[0] if isinstance(sim_reward, tuple) else sim_reward),
        "pred_reward": float(pred_reward[0] if isinstance(pred_reward, tuple) else pred_reward),
        "entropy_weight": cma_info.get("entropy_weight"),
        "best_pred_xy_entropy": best_pred_xy_entropy,
        "best_pred_overlap_penalty": cma_info.get("best_pred_overlap_penalty"),
        "best_pred_uncover_reward_raw": cma_info.get("best_pred_uncover_reward_raw"),
        "best_pred_regularized_cost": cma_info.get("best_pred_regularized_cost"),
    }


def add_uncover_row(fig, payload, label, row):
    all_body_points = payload["all_body_points"]
    cloth_initial = payload["cloth_initial"]
    sim_colors = get_body_point_colors_uncovering(payload["initial_status"], payload["sim_status"])
    pred_colors = get_body_point_colors_uncovering(payload["initial_status"], payload["pred_status"])

    for col in range(1, 4):
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=cloth_initial[:, 0],
                y=cloth_initial[:, 1],
                marker=dict(color="rgba(99, 190, 242, 0.12)", size=7),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=all_body_points[:, 0],
            y=all_body_points[:, 1],
            marker=dict(color="rgba(255, 186, 71, 1)", size=7),
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=all_body_points[:, 0],
            y=all_body_points[:, 1],
            marker=dict(color=sim_colors, size=7),
            showlegend=False,
        ),
        row=row,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=all_body_points[:, 0],
            y=all_body_points[:, 1],
            marker=dict(color=pred_colors, size=7),
            showlegend=False,
        ),
        row=row,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=payload["cloth_final"][:, 0],
            y=payload["cloth_final"][:, 1],
            marker=dict(color="rgba(38, 60, 201, 0.5)", size=7),
            showlegend=False,
        ),
        row=row,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=payload["pred"][:, 0],
            y=payload["pred"][:, 1],
            marker=dict(color="rgba(38, 60, 201, 0.5)", size=7),
            showlegend=False,
        ),
        row=row,
        col=3,
    )

    if payload["uncover_action"].shape[0] == 4:
        add_action_arrow(fig, payload["uncover_action"], row, 2)
        add_action_arrow(fig, payload["uncover_action"], row, 3)

    for col in range(1, 4):
        fig.update_xaxes(autorange="reversed", visible=False, row=row, col=col)
        fig.update_yaxes(autorange="reversed", visible=False, row=row, col=col)

    text = (
        f"{label}<br>"
        f"entropy={fmt(payload['best_pred_xy_entropy'])}, "
        f"w={fmt(payload['entropy_weight'], 1)}<br>"
        f"sim_f1={fmt(payload['sim_f1'])}, pred_f1={fmt(payload['pred_f1'])}<br>"
        f"sim_reward={fmt(payload['sim_reward'], 1)}, pred_reward={fmt(payload['pred_reward'], 1)}"
    )
    fig.add_annotation(
        text=text,
        x=0.0,
        y=1.0 if row == 1 else 0.47,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.86)",
        bordercolor="rgba(0,0,0,0.25)",
        borderwidth=1,
        font=dict(size=13, color="rgba(0,0,0,1)"),
    )


def build_pair_figure(payload_a, payload_b, label_a, label_b):
    tl = payload_a["target_limb_code"]
    seed = payload_a["seed"]
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            f"{label_a}: Initial",
            f"{label_a}: Sim",
            f"{label_a}: Pred",
            f"{label_b}: Initial",
            f"{label_b}: Sim",
            f"{label_b}: Pred",
        ],
        vertical_spacing=0.06,
        horizontal_spacing=0.02,
    )
    add_uncover_row(fig, payload_a, label_a, 1)
    add_uncover_row(fig, payload_b, label_b, 2)
    fig.update_layout(
        width=1500,
        height=980,
        plot_bgcolor="rgba(255,255,255,1)",
        paper_bgcolor="rgba(255,255,255,1)",
        title=dict(
            text=f"TL {tl} {target_name(tl)} | seed {seed}",
            x=0.5,
            y=0.985,
            xanchor="center",
            yanchor="top",
        ),
        margin=dict(l=20, r=20, t=90, b=20),
    )
    return fig


def write_manifest(records, path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"records": records}, handle, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-a", required=True)
    parser.add_argument("--raw-b", required=True)
    parser.add_argument("--label-a", default="w0")
    parser.add_argument("--label-b", default="w200")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--target-limb-filter", default="12,14,15")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    allowed = {int(part.strip()) for part in args.target_limb_filter.split(",") if part.strip()}
    index_a = index_raw_dir(args.raw_a)
    index_b = index_raw_dir(args.raw_b)
    keys = sorted(set(index_a) & set(index_b))
    if allowed:
        keys = [key for key in keys if key[0] in allowed]
    if args.max_images > 0:
        keys = keys[: args.max_images]
    if not keys:
        raise RuntimeError("No paired PKLs found for the requested filters.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    csv_rows = []
    for idx, key in enumerate(keys):
        payload_a = build_run_payload(index_a[key])
        payload_b = build_run_payload(index_b[key])
        tl, seed = key
        fig = build_pair_figure(payload_a, payload_b, args.label_a, args.label_b)
        image_path = output_dir / f"tl{tl}_{seed}_{args.label_a}_vs_{args.label_b}.png"
        fig.write_image(str(image_path))

        record = {
            "eval_id": f"tl{tl}_{seed}",
            "seed": seed,
            "target_limb_code": tl,
            "source_uncover_pkl": payload_b["path"],
            "compare_image": str(image_path),
            f"{args.label_a}_source_uncover_pkl": payload_a["path"],
            f"{args.label_b}_source_uncover_pkl": payload_b["path"],
            f"{args.label_a}_pred_xy_entropy": payload_a["best_pred_xy_entropy"],
            f"{args.label_b}_pred_xy_entropy": payload_b["best_pred_xy_entropy"],
            f"{args.label_a}_sim_uncover_f1": payload_a["sim_f1"],
            f"{args.label_b}_sim_uncover_f1": payload_b["sim_f1"],
            f"{args.label_a}_pred_uncover_f1": payload_a["pred_f1"],
            f"{args.label_b}_pred_uncover_f1": payload_b["pred_f1"],
        }
        manifest_records.append(record)
        csv_rows.append(record)
        print(f"[{idx + 1}/{len(keys)}] wrote {image_path}")

    if args.manifest_out:
        manifest_path = write_manifest(manifest_records, args.manifest_out)
        print(f"Wrote manifest: {manifest_path}")

    if args.csv_out:
        csv_path = Path(args.csv_out).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({field for row in csv_rows for field in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
