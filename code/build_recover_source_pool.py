#!/usr/bin/env python3
"""Build a variance-aware recover source pool from CMA uncover-eval PKLs."""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "assistive-gym-fem"))
from assistive_gym.gnn_dc_recover import compute_uncover_f1_from_raw
from recover_source_pool_utils import dedupe_records, select_for_limb, summarize_selection


def parse_seed_from_name(name):
    parts = Path(name).name.split("_")
    if len(parts) >= 3 and parts[0].startswith("tl") and parts[2].isdigit():
        return int(parts[2])
    raise ValueError(f"Cannot parse seed from filename: {name}")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_record(src_path, dst_path, copy_file=False):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    if copy_file:
        shutil.copy2(str(src_path), str(dst_path))
    else:
        os.link(str(src_path), str(dst_path))


def build_parser():
    parser = argparse.ArgumentParser(description="Build variance-aware recover source pool.")
    parser.add_argument("--input-raw-dir", required=True, help="Combined CMA raw dir, e.g. 1640 PKLs.")
    parser.add_argument("--output-root", required=True, help="Output root; writes raw/ and manifest files.")
    parser.add_argument("--eval-condition", default="Combined_1k_plus_boost_varaware",
                        help="Eval condition folder name under cma_evaluations/.")
    parser.add_argument("--threshold", type=float, default=0.745)
    parser.add_argument("--near-floor", type=float, default=0.60,
                        help="Minimum F1 for near-threshold band.")
    parser.add_argument("--max-high-per-limb", type=int, default=100)
    parser.add_argument("--max-near-per-limb", type=int, default=25)
    parser.add_argument("--target-limbs", default="2,4,5,8,10,11,12,13,14,15")
    parser.add_argument("--copy", action="store_true", help="Copy PKLs instead of hardlinking.")
    return parser


def main():
    args = build_parser().parse_args()
    input_raw = Path(args.input_raw_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    eval_root = output_root / "cma_evaluations" / args.eval_condition
    output_raw = eval_root / "raw"
    output_raw.mkdir(parents=True, exist_ok=True)

    target_limbs = [int(x.strip()) for x in args.target_limbs.split(",") if x.strip()]
    source_files = sorted(input_raw.glob("*.pkl"))
    if not source_files:
        raise RuntimeError(f"No PKLs found in {input_raw}")

    scored_records = []
    failed = 0
    for path in source_files:
        real_path = path.resolve()
        try:
            with open(real_path, "rb") as handle:
                raw = pickle.load(handle)
            f1 = float(compute_uncover_f1_from_raw(raw))
            target_limb = int(raw["target_limb_code"])
            seed = parse_seed_from_name(real_path.name)
            scored_records.append({
                "filename": real_path.name,
                "source_path": str(real_path),
                "target_limb": target_limb,
                "seed": seed,
                "f1": f1,
            })
        except Exception:
            failed += 1

    deduped = dedupe_records(scored_records)
    by_limb = defaultdict(list)
    for record in deduped:
        if record["target_limb"] in target_limbs:
            by_limb[record["target_limb"]].append(record)

    selected_all = []
    limb_reports = {}
    for tl in target_limbs:
        records = by_limb.get(tl, [])
        report = select_for_limb(
            records,
            threshold=args.threshold,
            max_high=args.max_high_per_limb,
            max_near=args.max_near_per_limb,
            floor=args.near_floor,
        )
        limb_reports[tl] = {k: v for k, v in report.items() if k != "selected"}
        selected_all.extend(report["selected"])

    manifest_path = eval_root / "source_manifest.jsonl"
    manifest_sha_path = eval_root / "source_manifest.sha256"
    with open(manifest_path, "w") as handle:
        for record in sorted(selected_all, key=lambda r: (r["target_limb"], r["seed"])):
            src = Path(record["source_path"])
            dst = output_raw / record["filename"]
            materialize_record(src, dst, copy_file=args.copy)
            digest = sha256_file(dst)
            row = {
                "path": str(dst.relative_to(output_root)),
                "filename": record["filename"],
                "target_limb": record["target_limb"],
                "seed": record["seed"],
                "f1": round(record["f1"], 6),
                "quality_band": record["quality_band"],
                "source_sha256": digest,
                "original_path": record["source_path"],
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest_digest = sha256_file(manifest_path)
    manifest_sha_path.write_text(manifest_digest + "\n")

    band_counts = Counter(r["quality_band"] for r in selected_all)
    report = {
        "input_raw_dir": str(input_raw),
        "output_root": str(output_root),
        "eval_condition": args.eval_condition,
        "threshold": args.threshold,
        "near_floor": args.near_floor,
        "max_high_per_limb": args.max_high_per_limb,
        "max_near_per_limb": args.max_near_per_limb,
        "input_count": len(source_files),
        "scored_count": len(scored_records),
        "deduped_count": len(deduped),
        "failed_count": failed,
        "selected_count": len(selected_all),
        "quality_band_counts": dict(sorted(band_counts.items())),
        "selected_by_target": dict(sorted(Counter(r["target_limb"] for r in selected_all).items())),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "limb_reports": {str(k): v for k, v in sorted(limb_reports.items())},
    }
    with open(eval_root / "pool_report.json", "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    lines = [
        "# Recover Source Pool Report",
        "",
        f"- input_raw_dir: `{report['input_raw_dir']}`",
        f"- output_root: `{report['output_root']}`",
        f"- eval_condition: `{report['eval_condition']}`",
        f"- threshold: {args.threshold}",
        f"- selected_count: {report['selected_count']}",
        f"- quality_band_counts: `{report['quality_band_counts']}`",
        f"- manifest_sha256: `{manifest_digest}`",
        "",
        "## Per-limb stats",
        "",
        "| limb | scored | mean | std | lower | high | near | selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tl in target_limbs:
        row = limb_reports[tl]
        lines.append(
            f"| {tl} | {row['scored_count']} | {row['mean_f1']:.3f} | {row['std_f1']:.3f} | "
            f"{row['lower_bound']:.3f} | {row['high_count']} | {row['near_count']} | {row['selected_count']} |"
        )
    with open(eval_root / "pool_report.md", "w") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"Wrote curated pool: {output_raw} ({len(selected_all)} files)")
    print(f"manifest_sha256={manifest_digest}")
    print(f"quality_band_counts={report['quality_band_counts']}")
    print(f"selected_by_target={report['selected_by_target']}")


if __name__ == "__main__":
    main()
