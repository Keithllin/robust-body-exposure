"""Pure selection helpers for recover source pool building."""
from collections import Counter


def dedupe_records(scored_records):
    best = {}
    for record in scored_records:
        key = (record["target_limb"], record["seed"])
        if key not in best or record["f1"] > best[key]["f1"]:
            best[key] = record
    return list(best.values())


def lower_bound_for_limb(threshold, mean_f1, std_f1, floor):
    return max(floor, threshold - std_f1, mean_f1 - std_f1)


def select_for_limb(records, threshold, max_high, max_near, floor):
    import numpy as np

    f1_vals = np.array([r["f1"] for r in records], dtype=float)
    mean_f1 = float(np.mean(f1_vals)) if len(f1_vals) else float("nan")
    std_f1 = float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0
    lower = lower_bound_for_limb(threshold, mean_f1, std_f1, floor)

    high = sorted(
        [r for r in records if r["f1"] >= threshold],
        key=lambda r: (-r["f1"], r["seed"], r["filename"]),
    )[:max_high]
    high_keys = {(r["target_limb"], r["seed"]) for r in high}

    near = sorted(
        [
            r for r in records
            if (r["target_limb"], r["seed"]) not in high_keys
            and lower <= r["f1"] < threshold
        ],
        key=lambda r: (-r["f1"], r["seed"], r["filename"]),
    )[:max_near]

    selected = high + near
    for row in high:
        row["quality_band"] = "high"
    for row in near:
        row["quality_band"] = "near_threshold"

    return {
        "target_limb": records[0]["target_limb"] if records else None,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "lower_bound": lower,
        "scored_count": len(records),
        "high_count": len(high),
        "near_count": len(near),
        "selected_count": len(selected),
        "selected": selected,
    }


def summarize_selection(selected_all):
    return {
        "quality_band_counts": dict(Counter(r["quality_band"] for r in selected_all)),
        "selected_by_target": dict(Counter(r["target_limb"] for r in selected_all)),
    }
