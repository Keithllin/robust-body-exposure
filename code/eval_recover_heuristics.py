import argparse
import json
import math
import multiprocessing as mp
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go

from replay_recover_interactive import (
    bool_arg,
    compute_bottom_corner_recover_action,
    compute_line_stacking_recover_action,
    invert_action_policy,
    parse_seed_from_filename,
    parse_tl_from_filename,
)

from assistive_gym.learn import make_env
from assistive_gym.envs.bu_gnn_util import (
    get_body_points_from_obs,
    get_covered_status,
    get_recovering_reward,
    get_uncovering_reward,
    scale_action,
)
from cma_gnn_util import compute_fscore_recover, compute_fscore_uncover, generate_figure_data_collection, target_names


def load_pkl(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_eval_set(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "records" in payload:
        return payload.get("records", [])
    if isinstance(payload, list):
        return payload
    raise RuntimeError(f"Unsupported eval set format: {path}")


def apply_release_settings(env, args):
    env.set_release_sim_steps(post_release_steps=args.post_release_steps)
    if hasattr(env, "set_release_quiet_settle"):
        env.set_release_quiet_settle(
            enabled=getattr(args, "quiet_settle", False),
            speed_threshold=getattr(args, "quiet_speed_threshold", 0.15),
            max_steps=getattr(args, "quiet_max_steps", 20),
        )
    if hasattr(env, "set_uncover_release_gravity_boost"):
        env.set_uncover_release_gravity_boost(
            enabled=getattr(args, "uncover_release_gravity_boost", False),
            gravity_z=getattr(args, "uncover_release_gravity_z", -39.24),
        )


def args_to_worker_dict(args):
    return {
        "mode": args.mode,
        "env_name": args.env_name,
        "blanket_pose_var": args.blanket_pose_var,
        "high_pose_var": args.high_pose_var,
        "body_shape_var": args.body_shape_var,
        "singulate_layers": args.singulate_layers,
        "save_images": args.save_images,
        "output_format": args.output_format,
        "image_dir": str(args.image_dir) if hasattr(args, "image_dir") and args.image_dir is not None else "",
        "post_release_steps": args.post_release_steps,
        "quiet_settle": getattr(args, "quiet_settle", False),
        "quiet_max_steps": getattr(args, "quiet_max_steps", 20),
        "quiet_speed_threshold": getattr(args, "quiet_speed_threshold", 0.15),
        "uncover_release_gravity_boost": getattr(args, "uncover_release_gravity_boost", False),
        "uncover_release_gravity_z": getattr(args, "uncover_release_gravity_z", -39.24),
    }


def worker_dict_to_args(args_dict):
    ns = SimpleNamespace(**args_dict)
    ns.image_dir = Path(ns.image_dir).expanduser().resolve() if getattr(ns, "image_dir", "") else None
    return ns


def summarize(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    arr = np.asarray(finite, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def format_float(value):
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.3f}"


def serialize_debug(value):
    if isinstance(value, dict):
        return {str(k): serialize_debug(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_debug(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_target_limb_filter(value):
    if value is None or str(value).strip() == "":
        return None
    return {int(part.strip()) for part in str(value).split(",") if part.strip()}


def to_2d(points_3d):
    return np.delete(np.asarray(points_3d, dtype=np.float32), 2, axis=1)


def compute_rollout_metrics(env, uncover_action_policy, recover_action_policy):
    human_pose = np.reshape(env.human_pose, (-1, 2))
    all_body_points = get_body_points_from_obs(
        human_pose,
        target_limb_code=env.target_limb_code,
        body_info=env.get_human_body_info(),
    )

    cloth_initial = np.asarray(env.cloth_initial[1], dtype=np.float32)
    cloth_intermediate = np.asarray(env.cloth_intermediate[1], dtype=np.float32)
    cloth_final = np.asarray(env.cloth_final[1], dtype=np.float32)

    cloth_initial_2d = to_2d(cloth_initial)
    cloth_intermediate_2d = to_2d(cloth_intermediate)
    cloth_final_2d = to_2d(cloth_final)

    initial_status = get_covered_status(all_body_points, cloth_initial_2d)
    intermediate_status = get_covered_status(all_body_points, cloth_intermediate_2d)
    final_status = get_covered_status(all_body_points, cloth_final_2d)

    uncover_reward, _ = get_uncovering_reward(
        np.asarray(uncover_action_policy, dtype=np.float32),
        all_body_points,
        cloth_initial_2d,
        cloth_intermediate_2d,
    )
    recover_reward, _ = get_recovering_reward(
        np.asarray(recover_action_policy, dtype=np.float32),
        all_body_points,
        cloth_initial_2d,
        cloth_intermediate_2d,
        cloth_final_2d,
    )

    uncover_f1 = compute_fscore_uncover(initial_status, intermediate_status)
    recover_f1 = compute_fscore_recover(initial_status, intermediate_status, final_status)

    return {
        "sim_uncover_reward": float(uncover_reward),
        "sim_recover_reward": float(recover_reward),
        "sim_uncover_f1": float(uncover_f1),
        "sim_recover_f1": float(recover_f1),
        "execute_uncover_action": bool(env.execute_uncover_action),
        "execute_recover_action": bool(env.execute_recover_action),
    }


def choose_recover_action(mode, env, uncover_action_policy, raw):
    if mode in {"line", "field"}:
        action_policy, action_world, pick_pos, place_pos, force_direction, debug = compute_line_stacking_recover_action(
            env,
            uncover_action_policy,
        )
        return {
            "recover_action_policy": np.asarray(action_policy, dtype=np.float32),
            "recover_action_world": np.asarray(action_world, dtype=np.float32),
            "pick_pos": np.asarray(pick_pos, dtype=np.float32),
            "place_pos": np.asarray(place_pos, dtype=np.float32),
            "force_direction": np.asarray(force_direction, dtype=np.float32),
            "debug": debug,
        }

    if mode == "bottom-corner":
        action_policy, action_world, pick_pos, place_pos, force_direction, debug = compute_bottom_corner_recover_action(env)
        return {
            "recover_action_policy": np.asarray(action_policy, dtype=np.float32),
            "recover_action_world": np.asarray(action_world, dtype=np.float32),
            "pick_pos": np.asarray(pick_pos, dtype=np.float32),
            "place_pos": np.asarray(place_pos, dtype=np.float32),
            "force_direction": np.asarray(force_direction, dtype=np.float32),
            "debug": debug,
        }

    if mode == "inverse-uncover":
        action_policy = invert_action_policy(uncover_action_policy)
        return {
            "recover_action_policy": np.asarray(action_policy, dtype=np.float32),
            "recover_action_world": np.asarray(scale_action(action_policy), dtype=np.float32),
        }

    if mode == "saved":
        recover_action = np.asarray(raw.get("recover_action", []), dtype=np.float32)
        if recover_action.shape[0] != 4:
            raise RuntimeError("saved mode requires recover_action in the pkl")
        return {
            "recover_action_policy": recover_action,
            "recover_action_world": np.asarray(scale_action(recover_action), dtype=np.float32),
        }

    raise ValueError(f"Unsupported mode: {mode}")


def evaluate_single_pkl(path, args, eval_record=None):
    raw = load_pkl(path)
    cma_info = raw.get("cma_info", {})
    seed = int((eval_record or {}).get("seed", raw.get("seed", parse_seed_from_filename(path))))
    target_limb_code = int((eval_record or {}).get("target_limb_code", raw.get("target_limb_code", parse_tl_from_filename(path))))
    uncover_action = np.asarray((eval_record or {}).get("uncover_action", raw.get("uncover_action", [])), dtype=np.float32)
    if uncover_action.shape[0] != 4:
        raise RuntimeError(f"Missing uncover_action in {path}")

    env = make_env(args.env_name, coop=False, seed=seed)
    env.set_env_variations(
        collect_data=False,
        blanket_pose_var=args.blanket_pose_var,
        high_pose_var=args.high_pose_var,
        body_shape_var=args.body_shape_var,
    )
    env.set_singulate(args.singulate_layers)
    env.set_target_limb_code(target_limb_code)
    env.set_recover(True)
    env.set_seed_val(seed)

    try:
        env.reset()
        apply_release_settings(env, args)

        env.uncover_step(uncover_action)
        choice = choose_recover_action(args.mode, env, uncover_action, raw)
        env.recover_step(choice["recover_action_policy"])
        metrics = compute_rollout_metrics(env, uncover_action, choice["recover_action_policy"])
        result = {
            "pkl": str(path),
            "seed": seed,
            "target_limb_code": target_limb_code,
            "mode": args.mode,
            "eval_id": (eval_record or {}).get("eval_id", cma_info.get("eval_id")),
            "post_release_steps": int(args.post_release_steps),
            "uncover_release_gravity_boost": bool(getattr(args, "uncover_release_gravity_boost", False)),
            "uncover_release_gravity_z": float(getattr(args, "uncover_release_gravity_z", -39.24)),
            "entropy_weight": cma_info.get("entropy_weight"),
            "entropy_grid_size": cma_info.get("entropy_grid_size"),
            "best_pred_xy_entropy": cma_info.get("best_pred_xy_entropy"),
            "best_pred_overlap_penalty": cma_info.get("best_pred_overlap_penalty"),
            "best_pred_uncover_reward_raw": cma_info.get("best_pred_uncover_reward_raw"),
            "best_pred_regularized_cost": cma_info.get("best_pred_regularized_cost"),
            "uncover_action_policy": uncover_action.tolist(),
            "recover_action_policy": choice["recover_action_policy"].tolist(),
            "recover_action_world": choice["recover_action_world"].tolist(),
        }
        if "pick_pos" in choice:
            result["pick_pos"] = choice["pick_pos"].tolist()
            result["place_pos"] = choice["place_pos"].tolist()
            result["force_direction"] = choice["force_direction"].tolist()
            result["heuristic_debug"] = serialize_debug(choice["debug"])
            if args.mode == "line":
                result["line_debug"] = result["heuristic_debug"]
            elif args.mode == "bottom-corner":
                result["bottom_corner_debug"] = result["heuristic_debug"]
        result["_figure_payload"] = {
            "body_info": env.get_human_body_info(),
            "all_body_points": np.asarray(
                get_body_points_from_obs(
                    np.reshape(env.human_pose, (-1, 2)),
                    target_limb_code=env.target_limb_code,
                    body_info=env.get_human_body_info(),
                ),
                dtype=np.float32,
            ),
            "cloth_initial": np.asarray(env.cloth_initial[1], dtype=np.float32),
            "cloth_intermediate": np.asarray(env.cloth_intermediate[1], dtype=np.float32),
            "cloth_final": np.asarray(env.cloth_final[1], dtype=np.float32),
            "uncover_action_world": np.asarray(scale_action(uncover_action), dtype=np.float32),
        }
        result.update(metrics)
        return result
    finally:
        try:
            env.disconnect()
        except Exception:
            try:
                env.close()
            except Exception:
                pass


def process_eval_job(job):
    idx, total, path_str, args_dict, eval_record = job
    args = worker_dict_to_args(args_dict)
    path = Path(path_str).expanduser().resolve()
    result = evaluate_single_pkl(path, args, eval_record=eval_record)
    if args.save_images:
        payload = result["_figure_payload"]
        recover_action = np.asarray(result["recover_action_world"], dtype=np.float32)
        fig = generate_figure_data_collection(
            result["target_limb_code"],
            payload["uncover_action_world"],
            recover_action,
            payload["body_info"],
            payload["all_body_points"],
            payload["cloth_initial"],
            payload["cloth_intermediate"],
            payload["cloth_final"],
            metrics={
                "uncover_reward": result["sim_uncover_reward"],
                "uncover_f1": result["sim_uncover_f1"],
                "recover_reward": result["sim_recover_reward"],
                "recover_f1": result["sim_recover_f1"],
            },
        )
        stem = f"tl{result['target_limb_code']}_{result['seed']}"
        written_kind, written_path = write_figure(
            fig,
            args.image_dir / f"{stem}.png",
            args.image_dir / f"{stem}.html",
            args.output_format,
        )
        result["image_path"] = str(written_path)
        result["image_format"] = written_kind
    result.pop("_figure_payload", None)
    log_line = (
        f"[{idx}/{total}] tl={result['target_limb_code']} seed={result['seed']} "
        f"recover_reward={result['sim_recover_reward']:.2f} recover_f1={result['sim_recover_f1']:.3f} "
        f"exec_recover={result['execute_recover_action']}"
    )
    return result, log_line


def build_markdown(results, mode, raw_dir):
    lines = [
        f"# Recover Heuristic Eval",
        "",
        f"- mode: `{mode}`",
        f"- raw_dir: `{raw_dir}`",
        f"- num_rollouts: `{len(results)}`",
        "",
    ]

    overall_fields = [
        "best_pred_xy_entropy",
        "sim_uncover_reward",
        "sim_recover_reward",
        "sim_uncover_f1",
        "sim_recover_f1",
    ]
    lines.append("## Overall")
    lines.append("")
    lines.append("| metric | mean | median | std |")
    lines.append("|---|---:|---:|---:|")
    for key in overall_fields:
        stats = summarize([row.get(key) for row in results])
        lines.append(
            f"| {key} | {format_float(stats['mean'])} | {format_float(stats['median'])} | {format_float(stats['std'])} |"
        )

    exec_recover = [1.0 if row.get("execute_recover_action") else 0.0 for row in results]
    exec_stats = summarize(exec_recover)
    lines.append(
        f"| execute_recover_rate | {format_float(exec_stats['mean'])} | {format_float(exec_stats['median'])} | {format_float(exec_stats['std'])} |"
    )
    lines.append("")

    per_limb = {}
    for row in results:
        per_limb.setdefault(int(row["target_limb_code"]), []).append(row)

    lines.append("## Per Limb Objective And Recoverability")
    lines.append("")
    lines.append("| tl | n | pred_xy_entropy_mean | pred_xy_entropy_median | sim_uncover_reward_mean | sim_uncover_reward_median | sim_uncover_f1_mean | sim_uncover_f1_median | sim_recover_reward_mean | sim_recover_reward_median | sim_recover_f1_mean | sim_recover_f1_median | exec_recover_rate |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tl in sorted(per_limb):
        rows = per_limb[tl]
        entropy_stats = summarize([r.get("best_pred_xy_entropy") for r in rows])
        uncover_reward_stats = summarize([r.get("sim_uncover_reward") for r in rows])
        uncover_f1_stats = summarize([r.get("sim_uncover_f1") for r in rows])
        recover_reward_stats = summarize([r.get("sim_recover_reward") for r in rows])
        recover_f1_stats = summarize([r.get("sim_recover_f1") for r in rows])
        exec_stats = summarize([1.0 if r.get("execute_recover_action") else 0.0 for r in rows])
        lines.append(
            f"| {tl} | {len(rows)} | {format_float(entropy_stats['mean'])} | {format_float(entropy_stats['median'])} | "
            f"{format_float(uncover_reward_stats['mean'])} | {format_float(uncover_reward_stats['median'])} | "
            f"{format_float(uncover_f1_stats['mean'])} | {format_float(uncover_f1_stats['median'])} | "
            f"{format_float(recover_reward_stats['mean'])} | {format_float(recover_reward_stats['median'])} | "
            f"{format_float(recover_f1_stats['mean'])} | {format_float(recover_f1_stats['median'])} | "
            f"{format_float(exec_stats['mean'])} |"
        )

    lines.append("")
    lines.append("## Per Limb Uncover")
    lines.append("")
    lines.append("| tl | n | uncover_reward_mean | uncover_reward_median | uncover_reward_min | uncover_reward_max | uncover_f1_mean | uncover_f1_median |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tl in sorted(per_limb):
        rows = per_limb[tl]
        uncover_reward_stats = summarize([r["sim_uncover_reward"] for r in rows])
        uncover_f1_stats = summarize([r["sim_uncover_f1"] for r in rows])
        lines.append(
            f"| {tl} | {len(rows)} | {format_float(uncover_reward_stats['mean'])} | {format_float(uncover_reward_stats['median'])} | "
            f"{format_float(uncover_reward_stats['min'])} | {format_float(uncover_reward_stats['max'])} | "
            f"{format_float(uncover_f1_stats['mean'])} | {format_float(uncover_f1_stats['median'])} |"
        )

    lines.append("")
    lines.append("## Per Limb Recover")
    lines.append("")
    lines.append("| tl | n | recover_reward_mean | recover_reward_median | recover_reward_min | recover_reward_max | recover_f1_mean | recover_f1_median | exec_recover_rate |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tl in sorted(per_limb):
        rows = per_limb[tl]
        reward_stats = summarize([r["sim_recover_reward"] for r in rows])
        f1_stats = summarize([r["sim_recover_f1"] for r in rows])
        exec_stats = summarize([1.0 if r.get("execute_recover_action") else 0.0 for r in rows])
        lines.append(
            f"| {tl} | {len(rows)} | {format_float(reward_stats['mean'])} | {format_float(reward_stats['median'])} | "
            f"{format_float(reward_stats['min'])} | {format_float(reward_stats['max'])} | "
            f"{format_float(f1_stats['mean'])} | {format_float(f1_stats['median'])} | {format_float(exec_stats['mean'])} |"
        )

    lines.append("")
    lines.append("## Per Rollout")
    lines.append("")
    lines.append("| idx | tl | seed | pred_xy_entropy | uncover_reward | uncover_f1 | recover_reward | recover_f1 | exec_recover | pkl | image |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for idx, row in enumerate(results, start=1):
        pkl_name = Path(row["pkl"]).name
        image_name = Path(row["image_path"]).name if row.get("image_path") else ""
        lines.append(
            f"| {idx} | {int(row['target_limb_code'])} | {int(row['seed'])} | "
            f"{format_float(row.get('best_pred_xy_entropy'))} | "
            f"{format_float(row.get('sim_uncover_reward'))} | {format_float(row.get('sim_uncover_f1'))} | "
            f"{format_float(row.get('sim_recover_reward'))} | {format_float(row.get('sim_recover_f1'))} | "
            f"{1 if row.get('execute_recover_action') else 0} | {pkl_name} | {image_name} |"
        )

    return "\n".join(lines) + "\n"


def write_figure(fig, png_path: Path, html_path: Path, output_format: str):
    if output_format == "html":
        fig.write_html(str(html_path))
        return "html", html_path

    if output_format == "png":
        fig.write_image(str(png_path))
        return "png", png_path

    try:
        fig.write_image(str(png_path))
        return "png", png_path
    except Exception:
        fig.write_html(str(html_path))
        return "html", html_path


def write_plot(fig, png_path: Path, html_path: Path, output_format: str):
    if output_format == "html":
        fig.write_html(str(html_path))
        return "html", html_path

    if output_format == "png":
        fig.write_image(str(png_path))
        return "png", png_path

    try:
        fig.write_image(str(png_path))
        return "png", png_path
    except Exception:
        fig.write_html(str(html_path))
        return "html", html_path


def rebuild_image_from_result(row, args):
    path = Path(row["pkl"]).expanduser().resolve()
    raw = load_pkl(path)
    seed = int(row.get("seed", raw.get("seed", parse_seed_from_filename(path))))
    target_limb_code = int(row.get("target_limb_code", raw.get("target_limb_code", parse_tl_from_filename(path))))
    uncover_action = np.asarray(row.get("uncover_action_policy", raw.get("uncover_action", [])), dtype=np.float32)
    recover_action = np.asarray(row.get("recover_action_policy", []), dtype=np.float32)
    if uncover_action.shape[0] != 4 or recover_action.shape[0] != 4:
        return None

    env = make_env(args.env_name, coop=False, seed=seed)
    env.set_env_variations(
        collect_data=False,
        blanket_pose_var=args.blanket_pose_var,
        high_pose_var=args.high_pose_var,
        body_shape_var=args.body_shape_var,
    )
    env.set_singulate(args.singulate_layers)
    env.set_target_limb_code(target_limb_code)
    env.set_recover(True)
    env.set_seed_val(seed)
    try:
        human_pose = env.reset()
        env.set_release_sim_steps(post_release_steps=3)#cut off post manipulation waiting time

        env.uncover_step(uncover_action)
        env.recover_step(recover_action)
        body_info = env.get_human_body_info()
        all_body_points = get_body_points_from_obs(
            np.reshape(human_pose, (-1, 2)),
            target_limb_code=target_limb_code,
            body_info=body_info,
        )
        fig = generate_figure_data_collection(
            target_limb_code,
            np.asarray(scale_action(uncover_action), dtype=np.float32),
            np.asarray(scale_action(recover_action), dtype=np.float32),
            body_info,
            np.asarray(all_body_points, dtype=np.float32),
            np.asarray(env.cloth_initial[1], dtype=np.float32),
            np.asarray(env.cloth_intermediate[1], dtype=np.float32),
            np.asarray(env.cloth_final[1], dtype=np.float32),
            metrics={
                "uncover_reward": float(row.get("sim_uncover_reward", 0.0)),
                "uncover_f1": float(row.get("sim_uncover_f1", 0.0)),
                "recover_reward": float(row.get("sim_recover_reward", 0.0)),
                "recover_f1": float(row.get("sim_recover_f1", 0.0)),
            },
        )
        return fig
    finally:
        try:
            env.disconnect()
        except Exception:
            try:
                env.close()
            except Exception:
                pass


def rebuild_image_job(job):
    row, args_dict = job
    args = worker_dict_to_args(args_dict)
    fig = rebuild_image_from_result(row, args)
    if fig is None:
        return row
    stem = f"tl{int(row['target_limb_code'])}_{int(row['seed'])}"
    written_kind, written_path = write_figure(
        fig,
        args.image_dir / f"{stem}.png",
        args.image_dir / f"{stem}.html",
        args.output_format,
    )
    updated_row = dict(row)
    updated_row["image_path"] = str(written_path)
    updated_row["image_format"] = written_kind
    return updated_row


def make_limb_scatter(results, metric_key, title, yaxis_title):
    grouped = {}
    for row in results:
        tl = int(row["target_limb_code"])
        grouped.setdefault(tl, []).append(row)

    fig = go.Figure()
    for tl in sorted(grouped):
        rows = grouped[tl]
        xs = []
        ys = []
        hover = []
        for idx, row in enumerate(rows):
            jitter = ((idx % 7) - 3) * 0.06
            xs.append(tl + jitter)
            ys.append(row.get(metric_key))
            hover.append(
                f"tl={tl}<br>"
                f"name={target_names[tl] if tl < len(target_names) else tl}<br>"
                f"seed={row['seed']}<br>"
                f"{metric_key}={format_float(row.get(metric_key))}<br>"
                f"pkl={Path(row['pkl']).name}"
            )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=target_names[tl] if tl < len(target_names) and target_names[tl] else f"TL {tl}",
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    tickvals = sorted(grouped)
    ticktext = [target_names[tl] if tl < len(target_names) and target_names[tl] else f"TL {tl}" for tl in tickvals]
    fig.update_layout(
        title=title,
        xaxis=dict(title="Target Limb", tickmode="array", tickvals=tickvals, ticktext=ticktext),
        yaxis=dict(title=yaxis_title),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig


def make_limb_scatter_clipped(results, metric_key, title, yaxis_title, y_min=-300.0, y_max=150.0):
    grouped = {}
    for row in results:
        tl = int(row["target_limb_code"])
        grouped.setdefault(tl, []).append(row)

    fig = go.Figure()
    for tl in sorted(grouped):
        rows = grouped[tl]
        xs = []
        ys = []
        hover = []
        symbols = []
        for idx, row in enumerate(rows):
            raw_y = row.get(metric_key)
            if raw_y is None or not math.isfinite(float(raw_y)):
                continue
            raw_y = float(raw_y)
            clipped_y = min(max(raw_y, y_min), y_max)
            xs.append(tl + ((idx % 7) - 3) * 0.06)
            ys.append(clipped_y)
            if raw_y < y_min:
                symbols.append("triangle-down")
            elif raw_y > y_max:
                symbols.append("triangle-up")
            else:
                symbols.append("circle")
            hover.append(
                f"tl={tl}<br>"
                f"name={target_names[tl] if tl < len(target_names) else tl}<br>"
                f"seed={row['seed']}<br>"
                f"{metric_key}={format_float(raw_y)}<br>"
                f"shown_y={format_float(clipped_y)}<br>"
                f"pkl={Path(row['pkl']).name}"
            )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=target_names[tl] if tl < len(target_names) and target_names[tl] else f"TL {tl}",
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(symbol=symbols),
            )
        )

    tickvals = sorted(grouped)
    ticktext = [target_names[tl] if tl < len(target_names) and target_names[tl] else f"TL {tl}" for tl in tickvals]
    fig.update_layout(
        title=title,
        xaxis=dict(title="Target Limb", tickmode="array", tickvals=tickvals, ticktext=ticktext),
        yaxis=dict(title=yaxis_title, range=[y_min, y_max]),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig


def make_reward_f1_scatter(results):
    grouped = {}
    for row in results:
        tl = int(row["target_limb_code"])
        grouped.setdefault(tl, []).append(row)

    fig = go.Figure()
    for tl in sorted(grouped):
        rows = grouped[tl]
        fig.add_trace(
            go.Scatter(
                x=[row.get("sim_recover_f1") for row in rows],
                y=[row.get("sim_recover_reward") for row in rows],
                mode="markers",
                name=target_names[tl] if tl < len(target_names) and target_names[tl] else f"TL {tl}",
                text=[
                    f"tl={tl}<br>"
                    f"name={target_names[tl] if tl < len(target_names) else tl}<br>"
                    f"seed={row['seed']}<br>"
                    f"recover_f1={format_float(row.get('sim_recover_f1'))}<br>"
                    f"recover_reward={format_float(row.get('sim_recover_reward'))}<br>"
                    f"pkl={Path(row['pkl']).name}"
                    for row in rows
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Recover Reward vs Recover F1",
        xaxis=dict(title="Recover F1"),
        yaxis=dict(title="Recover Reward"),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig

    if output_format == "png":
        fig.write_image(str(png_path))
        return "png", png_path

    try:
        fig.write_image(str(png_path))
        return "png", png_path
    except Exception:
        fig.write_html(str(html_path))
        return "html", html_path


def main():
    parser = argparse.ArgumentParser(description="Batch-evaluate recover heuristics on uncover raw PKLs.")
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Directory containing fixed Uncover evaluation PKLs.",
    )
    parser.add_argument("--results-json", default="")
    parser.add_argument("--eval-set", default="")
    parser.add_argument(
        "--mode",
        choices=["line", "field", "bottom-corner", "inverse-uncover", "saved"],
        default="line",
        help="'field' is a deprecated alias for the line-stacking heuristic.",
    )
    parser.add_argument("--env-name", default="RobeReversible-v1")
    parser.add_argument("--blanket-pose-var", type=bool_arg, default=False)
    parser.add_argument("--high-pose-var", type=bool_arg, default=False)
    parser.add_argument("--body-shape-var", type=bool_arg, default=False)
    parser.add_argument("--singulate-layers", type=bool_arg, default=True)
    parser.add_argument("--target-limb-filter", default="", help="Comma-separated target limb codes to evaluate, e.g. 12,14,15")
    parser.add_argument("--post-release-steps", type=int, default=3)
    parser.add_argument("--quiet-settle", type=bool_arg, default=False)
    parser.add_argument("--quiet-max-steps", type=int, default=20)
    parser.add_argument("--quiet-speed-threshold", type=float, default=0.15)
    parser.add_argument("--uncover-release-gravity-boost", type=bool_arg, default=False)
    parser.add_argument("--uncover-release-gravity-z", type=float, default=-39.24)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-images", type=bool_arg, default=False)
    parser.add_argument("--rebuild-images-from-results", type=bool_arg, default=False)
    parser.add_argument("--save-plots", type=bool_arg, default=False)
    parser.add_argument("--output-format", type=str, default="png", choices=["auto", "png", "html"])
    parser.add_argument("--num-processes", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "field":
        args.mode = "line"
    target_limb_filter = parse_target_limb_filter(args.target_limb_filter)

    if args.results_json:
        json_path = Path(args.results_json).expanduser().resolve()
        if not json_path.exists():
            raise FileNotFoundError(f"results json does not exist: {json_path}")
        with json_path.open("r") as handle:
            results = json.load(handle)
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else json_path.parent
        raw_dir = Path(args.raw_dir).expanduser().resolve() if args.raw_dir else output_dir.parent / "raw"
        if args.rebuild_images_from_results:
            image_dir = output_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            args.image_dir = image_dir
            worker_args = args_to_worker_dict(args)
            rebuilt_results = []
            if args.num_processes > 1:
                ctx = mp.get_context("spawn")
                jobs = [(row, worker_args) for row in results]
                with ctx.Pool(processes=args.num_processes) as pool:
                    for updated_row in pool.imap(rebuild_image_job, jobs):
                        rebuilt_results.append(updated_row)
            else:
                for row in results:
                    rebuilt_results.append(rebuild_image_job((row, worker_args)))
            results = rebuilt_results
            with json_path.open("w") as handle:
                json.dump(results, handle, indent=2)
        if target_limb_filter is not None:
            results = [row for row in results if int(row["target_limb_code"]) in target_limb_filter]
    else:
        eval_records = None
        if args.eval_set:
            eval_set_path = Path(args.eval_set).expanduser().resolve()
            if not eval_set_path.exists():
                raise FileNotFoundError(f"eval set does not exist: {eval_set_path}")
            eval_records = load_eval_set(eval_set_path)
            if target_limb_filter is not None:
                eval_records = [
                    row for row in eval_records
                    if int(row.get("target_limb_code", row.get("tl", -1))) in target_limb_filter
                ]
            if args.max_files is not None:
                eval_records = eval_records[: args.max_files]
            pkl_paths = [Path(row["source_uncover_pkl"]).expanduser().resolve() for row in eval_records]
            raw_dir = eval_set_path
            if len(pkl_paths) == 0:
                raise RuntimeError(f"No records found in eval set {eval_set_path}")
            output_dir = (
                Path(args.output_dir).expanduser().resolve()
                if args.output_dir
                else eval_set_path.parent / f"{eval_set_path.stem}_{args.mode}_heuristic_eval"
            )
        else:
            raw_dir = Path(args.raw_dir).expanduser().resolve()
            pkl_paths = sorted(raw_dir.glob("*.pkl"))
            if target_limb_filter is not None:
                pkl_paths = [path for path in pkl_paths if int(parse_tl_from_filename(path) or -1) in target_limb_filter]
            if args.max_files is not None:
                pkl_paths = pkl_paths[: args.max_files]
            if len(pkl_paths) == 0:
                raise RuntimeError(f"No PKLs found in {raw_dir}")
            output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else raw_dir.parent / f"{args.mode}_heuristic_eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = output_dir / "images"
        if args.save_images:
            image_dir.mkdir(parents=True, exist_ok=True)
        args.image_dir = image_dir

        results = []
        worker_args = args_to_worker_dict(args)
        jobs = []
        total = len(pkl_paths)
        for idx, path in enumerate(pkl_paths, start=1):
            eval_record = eval_records[idx - 1] if eval_records is not None else None
            jobs.append((idx, total, str(path), worker_args, eval_record))
        if args.num_processes > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=args.num_processes) as pool:
                for result, log_line in pool.imap(process_eval_job, jobs):
                    results.append(result)
                    print(log_line)
        else:
            for job in jobs:
                result, log_line = process_eval_job(job)
                results.append(result)
                print(log_line)

        json_path = output_dir / "results.json"
        with json_path.open("w") as handle:
            json.dump(results, handle, indent=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "summary.md"
    with md_path.open("w") as handle:
        handle.write(build_markdown(results, args.mode, raw_dir))

    if args.save_plots:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plots = [
            ("recover_reward_by_limb", make_limb_scatter(results, "sim_recover_reward", "Recover Reward by Limb", "Recover Reward")),
            (
                "recover_reward_by_limb_clipped",
                make_limb_scatter_clipped(
                    results,
                    "sim_recover_reward",
                    "Recover Reward by Limb (Clipped)",
                    "Recover Reward",
                ),
            ),
            ("recover_f1_by_limb", make_limb_scatter(results, "sim_recover_f1", "Recover F1 by Limb", "Recover F1")),
            ("recover_reward_vs_f1", make_reward_f1_scatter(results)),
        ]
        for stem, fig in plots:
            fmt, path = write_plot(
                fig,
                plot_dir / f"{stem}.png",
                plot_dir / f"{stem}.html",
                args.output_format,
            )
            print(f"Saved plot ({fmt}): {path}")

    print("=" * 80)
    print(f"Using JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
