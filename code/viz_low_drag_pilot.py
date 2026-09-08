#!/usr/bin/env python3
"""gen_images-style viz for low-drag unfold PKLs, split by bucket.

Default audit buckets (separate dirs, not mixed):
  1. accepted_low_drag_unfolding
  2. dragging_contaminated
  3. no_effective_unfolding
  4. boundary_unsafe_sibling  (off unless --include-boundary)

Each sample writes:
  * comparison PNG/HTML (Uncover | Recover via generate_figure_data_collection)
  * metrics.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "assistive-gym-fem"))

from assistive_gym.envs.bu_gnn_util import (  # noqa: E402
    get_covered_status,
    get_recovering_reward,
    get_uncovering_reward,
    scale_action,
)
from cma_gnn_util import (  # noqa: E402
    add_action_arrow,
    add_corner_annotation,
    compute_fscore_recover,
    compute_fscore_uncover,
    generate_figure_data_collection,
    write_gen_images_figure,
)
from low_drag_metrics import (  # noqa: E402
    BUCKET_ACCEPTED,
    BUCKET_BOUNDARY,
    BUCKET_DRAGGING,
    BUCKET_NO_UNFOLD,
    upper_body_points,
)

DEFAULT_AUDIT_BUCKETS = (
    BUCKET_ACCEPTED,
    BUCKET_DRAGGING,
    BUCKET_NO_UNFOLD,
)


def _as_points(arr_like):
    try:
        arr = np.asarray(arr_like, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        elif arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)
        else:
            return None
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 2:
        return None
    return arr


def _cloth_xyz(raw, key):
    if raw is None:
        return None
    val = raw.get(key)
    if val is None and isinstance(raw.get("info"), dict):
        val = raw["info"].get(key)
    if val is None:
        return None
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        pts = _as_points(val[1])
        if pts is not None:
            return pts
    return _as_points(val)


def _scale_action_4(raw_action):
    act = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if act.size != 4:
        return None
    return np.asarray(scale_action(act), dtype=np.float64)


def _build_metrics(cloth_i, cloth_m, cloth_f, uncover_action, recover_action, all_body_points, raw):
    try:
        cloth_i_2d = np.delete(np.asarray(cloth_i), 2, axis=1) if cloth_i.shape[1] > 2 else cloth_i
        cloth_m_2d = np.delete(np.asarray(cloth_m), 2, axis=1) if cloth_m.shape[1] > 2 else cloth_m
        cloth_f_2d = np.delete(np.asarray(cloth_f), 2, axis=1) if cloth_f.shape[1] > 2 else cloth_f
        initial_status = get_covered_status(all_body_points, cloth_i_2d)
        intermediate_status = get_covered_status(all_body_points, cloth_m_2d)
        final_status = get_covered_status(all_body_points, cloth_f_2d)
        uncover_reward, _ = get_uncovering_reward(
            uncover_action, all_body_points, cloth_i_2d, cloth_m_2d
        )
        recover_reward, _ = get_recovering_reward(
            recover_action, all_body_points, cloth_i_2d, cloth_m_2d, cloth_f_2d, True
        )
        return {
            "uncover_reward": float(uncover_reward),
            "uncover_f1": float(compute_fscore_uncover(initial_status, intermediate_status)),
            "recover_reward": float(recover_reward),
            "recover_f1": float(
                compute_fscore_recover(initial_status, intermediate_status, final_status, False)
            ),
        }
    except Exception:
        dci = raw.get("data_collection_info") or {}
        return {
            "uncover_reward": float(raw.get("uncover_reward") or 0.0),
            "uncover_f1": float(dci.get("live_uncover_f1") or 0.0),
            "recover_reward": float(raw.get("recover_reward") or 0.0),
            "recover_f1": 0.0,
        }


def _pose_from_obs(observation):
    if isinstance(observation, (list, tuple)):
        return np.reshape(observation[0], (-1, 2))
    return np.reshape(observation, (-1, 2))


def _mark_newly_exposed_upper(fig, pose, body_info, cloth_m, cloth_f, col=2):
    try:
        upper = upper_body_points(pose, body_info=body_info)
        if upper.size == 0:
            return fig
        s0 = get_covered_status(upper, cloth_m[:, :2])
        s1 = get_covered_status(upper, cloth_f[:, :2])
        pts = []
        for a, b, p in zip(s0, s1, upper):
            if a[0] == 1 and a[1] and (not b[1]):
                pts.append(p[:2])
        if not pts:
            return fig
        pts = np.asarray(pts, dtype=np.float64)
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=pts[:, 0],
                y=pts[:, 1],
                name="newly exposed upper",
                showlegend=True,
                marker=dict(color="rgb(220,20,60)", size=12, symbol="x", line=dict(width=2)),
            ),
            row=1,
            col=int(col),
        )
    except Exception:
        pass
    return fig


def _annotate(fig, dci):
    reason = dci.get("rejection_reason") or dci.get("accept_label") or "-"
    dE = dci.get("upper_exposure_delta_final")
    e0 = dci.get("upper_exposed_start")
    e_rel = dci.get("upper_exposed_at_release")
    e_f = dci.get("upper_exposed_final")
    dC = dci.get("lower_recovery_gain")
    text = (
        "bucket=%s | accepted=%s | role=%s<br>"
        "reason=%s | cutoff=%s α=%.2f<br>"
        "<b>ΔE_upper(final)=%s</b>  (start_uncovered=%s, at_release=%s, final_uncovered=%s)<br>"
        "ΔC_lower=%s (no-op gate) | Len i/a=%.3f/%.3f<br>"
        "τ_online=%s  τ_final=%s  η=%s"
        % (
            dci.get("bucket"),
            dci.get("accepted"),
            dci.get("boundary_role"),
            reason,
            dci.get("cutoff_triggered"),
            float(dci.get("cutoff_fraction") or 1.0),
            dE,
            e0,
            e_rel,
            e_f,
            dC,
            float(dci.get("intended_action_length") or 0.0),
            float(dci.get("actual_action_length") or 0.0),
            dci.get("tau_online"),
            dci.get("tau_final"),
            dci.get("eta_lower_gain"),
        )
    )
    fig = add_corner_annotation(fig, text, x=0.0, y=1.0, xanchor="left", yanchor="top", font_size=12)
    title = "Low-drag | tl=%s state=%s a=%s | %s | ΔE_upper=%s" % (
        dci.get("target_limb_code"),
        dci.get("state_idx"),
        dci.get("action_index"),
        "ACCEPTED" if dci.get("accepted") else "REJECTED",
        dE,
    )
    annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    annotations.append(
        go.layout.Annotation(
            dict(x=0.5, y=1.12, xref="paper", yref="paper", text=title, showarrow=False, font=dict(size=15))
        )
    )
    fig.update_layout(annotations=annotations, margin=dict(t=110))
    return fig


def render_one(path: Path, image_dir: Path, output_format: str):
    with path.open("rb") as handle:
        raw = pickle.load(handle)

    dci = raw.get("data_collection_info") or {}
    cloth_i = _cloth_xyz(raw, "cloth_initial")
    cloth_m = _cloth_xyz(raw, "cloth_intermediate")
    cloth_f = _cloth_xyz(raw, "cloth_final")
    if cloth_i is None or cloth_m is None or cloth_f is None:
        return {"status": "skip", "path": str(path), "reason": "missing_cloth"}

    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
    all_body_points = np.asarray(info.get("all_body_points", []), dtype=np.float64)
    if all_body_points.size == 0 or all_body_points.ndim != 2 or all_body_points.shape[1] < 2:
        return {"status": "skip", "path": str(path), "reason": "missing_all_body_points"}
    if not np.isfinite(all_body_points).all() or float(np.max(np.abs(all_body_points[:, :2]))) > 2.5:
        return {"status": "skip", "path": str(path), "reason": "bad_all_body_points"}

    uncover_action = _scale_action_4(raw.get("uncover_action"))
    recover_action = _scale_action_4(raw.get("recover_action"))  # actual stop-in-place
    if uncover_action is None or recover_action is None:
        return {"status": "skip", "path": str(path), "reason": "bad_action"}

    tl = int(dci.get("target_limb_code", info.get("target_limb_code", -1)))
    seed = int(dci.get("seed", -1))
    body_info = info.get("human_body_info")
    metrics = _build_metrics(
        cloth_i, cloth_m, cloth_f, uncover_action, recover_action, all_body_points, raw
    )

    try:
        fig = generate_figure_data_collection(
            tl,
            uncover_action,
            recover_action,
            body_info,
            all_body_points,
            cloth_i,
            cloth_m,
            cloth_f,
            metrics=metrics,
        )
        # Intended trajectory (orange) vs actual (default recover arrow).
        intended = dci.get("intended_release_xy") or dci.get("intended_release")
        grasp = dci.get("grasp")
        if intended is not None and grasp is not None:
            g = np.asarray(grasp, dtype=np.float64).reshape(-1)
            r = np.asarray(intended, dtype=np.float64).reshape(-1)
            if g.size >= 2 and r.size >= 2 and np.isfinite(g[:2]).all() and np.isfinite(r[:2]).all():
                intended_w = np.array([g[0], g[1], r[0], r[1]], dtype=np.float64)
                if not np.allclose(intended_w[2:4], recover_action[2:4], atol=1e-4):
                    fig = add_action_arrow(
                        fig, intended_w, col=2, color="rgb(243,156,18)", width=3,
                        pick_size=11, pick_name="intended", showlegend=True,
                    )
                    fig = add_action_arrow(
                        fig, intended_w, col=1, color="rgb(243,156,18)", width=2,
                        pick_size=9, pick_name="intended", showlegend=False,
                    )
        try:
            pose = _pose_from_obs(raw.get("observation"))
            fig = _mark_newly_exposed_upper(fig, pose, body_info, cloth_m, cloth_f, col=2)
        except Exception:
            pass
        fig = _annotate(fig, dci)
    except Exception as exc:
        return {
            "status": "skip",
            "path": str(path),
            "reason": "figure_error:%s:%s" % (type(exc).__name__, exc),
        }

    family = str(dci.get("action_family") or dci.get("family") or dci.get("slot") or "act")
    family_safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in family)
    state_idx = int(dci.get("state_idx", -1))
    a_idx = int(dci.get("action_index", -1))
    stem = "tl%s_s%03d_a%02d_%s_%s_%s" % (
        tl, state_idx, a_idx, seed, family_safe, int(time.time() * 1000) % 100000
    )
    png_path = image_dir / ("%s.png" % stem)
    html_path = image_dir / ("%s.html" % stem)
    if output_format == "html":
        fig.write_html(str(html_path))
        out_path, kind = str(html_path), "html"
    elif output_format == "png":
        fig.write_image(str(png_path))
        out_path, kind = str(png_path), "png"
    else:
        out_path, err = write_gen_images_figure(fig, png_path, html_path)
        kind = "png" if str(out_path).endswith(".png") else "html"

    meta = {
        "source_pkl": str(path),
        "image_path": out_path,
        "bucket": dci.get("bucket"),
        "accepted": dci.get("accepted"),
        "rejection_reason": dci.get("rejection_reason"),
        "boundary_role": dci.get("boundary_role"),
        "cutoff_triggered": dci.get("cutoff_triggered"),
        "cutoff_fraction": dci.get("cutoff_fraction"),
        "upper_exposure_delta_final": dci.get("upper_exposure_delta_final"),
        "lower_recovery_gain": dci.get("lower_recovery_gain"),
        "actual_action_length": dci.get("actual_action_length"),
        "intended_action_length": dci.get("intended_action_length"),
        "target_limb_code": tl,
        "state_idx": state_idx,
        "action_index": a_idx,
        "family": dci.get("action_family") or dci.get("family"),
        "metrics": metrics,
    }
    (image_dir / ("%s_metrics.json" % stem)).write_text(json.dumps(meta, indent=2) + "\n")
    return {"status": "ok", **meta, "image_format": kind}


def _discover_buckets(ds: Path, include_boundary: bool, buckets_arg):
    if buckets_arg:
        return [b for b in buckets_arg if (ds / b / "raw").is_dir()]
    found = []
    for b in DEFAULT_AUDIT_BUCKETS:
        if (ds / b / "raw").is_dir():
            found.append(b)
    # Legacy Stage-0 dirs
    if not found:
        for legacy, alias in (
            ("accepted_low_drag_unfold", BUCKET_ACCEPTED),
            ("boundary_and_rejected", "legacy_rejected"),
        ):
            if (ds / legacy / "raw").is_dir():
                found.append(legacy)
    if include_boundary and (ds / BUCKET_BOUNDARY / "raw").is_dir():
        found.append(BUCKET_BOUNDARY)
    return found


def main():
    parser = argparse.ArgumentParser(description="Bucketed gen_images viz for low-drag PKLs.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-per-bucket", type=int, default=0)
    parser.add_argument("--output-format", default="auto", choices=["auto", "png", "html"])
    parser.add_argument(
        "--buckets",
        nargs="*",
        default=None,
        help="Bucket names to viz (default: accepted / dragging / no_effective)",
    )
    parser.add_argument(
        "--include-boundary",
        action="store_true",
        help="Also viz boundary_unsafe_sibling (separate dir)",
    )
    args = parser.parse_args()

    ds = Path(args.dataset_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else ds / "viz"
    out_root.mkdir(parents=True, exist_ok=True)

    buckets = _discover_buckets(ds, args.include_boundary, args.buckets)
    if not buckets:
        raise SystemExit("No bucket raw/ dirs found under %s" % ds)

    report = {"viz_style": "gen_images_bucketed", "dataset": str(ds), "buckets": {}, "n_written": 0, "n_skipped": 0}

    for bucket in buckets:
        raw_dir = ds / bucket / "raw"
        image_dir = out_root / bucket
        image_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(raw_dir.glob("*.pkl"))
        if args.max_per_bucket > 0:
            paths = paths[: int(args.max_per_bucket)]
        written, skipped = [], []
        for path in paths:
            if args.max_images > 0 and report["n_written"] >= int(args.max_images):
                break
            try:
                result = render_one(path, image_dir, args.output_format)
            except Exception as exc:
                skipped.append({"path": str(path), "reason": "crash:%s" % type(exc).__name__})
                print("skip", path.name, type(exc).__name__)
                continue
            if result.get("status") != "ok":
                skipped.append(result)
                print("skip", path.name, result.get("reason"))
                continue
            written.append(result)
            report["n_written"] += 1
            print("wrote", result["image_path"])

        report["n_skipped"] += len(skipped)
        report["buckets"][bucket] = {
            "n_written": len(written),
            "n_skipped": len(skipped),
            "output_dir": str(image_dir),
            "written": written,
            "skipped": skipped,
        }
        md = ["# %s" % bucket, ""]
        for row in written:
            md.append(
                "- state=%s a=%s accepted=%s **ΔE_upper=%s** ΔC_lower=%s cut=%s → `%s`"
                % (
                    row.get("state_idx"),
                    row.get("action_index"),
                    row.get("accepted"),
                    row.get("upper_exposure_delta_final"),
                    row.get("lower_recovery_gain"),
                    row.get("cutoff_triggered"),
                    Path(row["image_path"]).name,
                )
            )
        (image_dir / "index.md").write_text("\n".join(md) + "\n")

    (out_root / "gen_images_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out_root / "gen_images_report.json")
    print("done buckets=%s written=%d skipped=%d" % (buckets, report["n_written"], report["n_skipped"]))


if __name__ == "__main__":
    main()
