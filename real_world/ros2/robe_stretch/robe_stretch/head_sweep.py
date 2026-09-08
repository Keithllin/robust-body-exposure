"""Head pan sweep: several D435i views, fuse in odom until ≥3 bed posts.

Per-view PnP may use 1 physical post (narrow FOV). Union across pans
must reach 3 posts. Freeze on that union; do not add extra head pans.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from stretch_localize import (
    LocalizationSample,
    fuse_localization_samples,
    fusion_acceptable,
    union_posts,
)

GrabFn = Callable[[], Optional[LocalizationSample]]
LookFn = Callable[[float], None]


def pan_angles_from_range(start: float, end: float, n_stops: int) -> list[float]:
    """Inclusive linspace. One stop looks only at ``end`` (toward the bed)."""

    n = max(1, int(n_stops))
    if n == 1:
        return [float(end)]
    return [float(v) for v in np.linspace(float(start), float(end), n)]


def resolve_pan_angles(
    *,
    pan_start: float,
    pan_end: float,
    n_stops: int,
    explicit: Sequence[float] | None = None,
) -> list[float]:
    if explicit is not None and len(explicit) >= 2:
        return [float(v) for v in explicit]
    return pan_angles_from_range(pan_start, pan_end, n_stops)


def best_of_frames(grab: GrabFn, n_frames: int) -> Optional[LocalizationSample]:
    best: Optional[LocalizationSample] = None
    for _ in range(max(1, int(n_frames))):
        sample = grab()
        if sample is None:
            continue
        if best is None or sample.reprojection_rms_px < best.reprojection_rms_px:
            best = sample
    return best


def sweep_head_views(
    *,
    look: LookFn,
    grab: GrabFn,
    pan_angles_rad: list[float],
    frames_per_view: int = 5,
    min_posts: int = 3,
    translation_std_max_m: float = 0.010,
    rotation_max_deg: float = 2.0,
    origin_spread_max_m: float = 0.03,
    pan_sign: float = 1.0,
    log: Callable[[str], None] | None = None,
) -> tuple[Optional[LocalizationSample], list[LocalizationSample], str]:
    """Look at the given pans only. Freeze when the union of posts is ≥3.

    Independent per-view rotations are not gated; extra search pans are not
    added. Returns (fused sample, per-view, reason).
    """

    views: list[LocalizationSample] = []
    emit = log or (lambda _msg: None)
    sign = 1.0 if float(pan_sign) >= 0 else -1.0
    for pan in pan_angles_rad:
        commanded = sign * float(pan)
        look(commanded)
        sample = best_of_frames(grab, frames_per_view)
        if sample is None:
            emit(f"pan={commanded:+.2f} rad: no usable PnP")
            continue
        views.append(sample)
        emit(
            f"pan={commanded:+.2f} rad: posts={sorted(sample.posts)} "
            f"ids={list(sample.marker_ids)} rms={sample.reprojection_rms_px:.2f}px"
        )
        ok, reason = fusion_acceptable(
            views,
            min_posts=min_posts,
            translation_std_max_m=translation_std_max_m,
            rotation_max_deg=rotation_max_deg,
            origin_spread_max_m=origin_spread_max_m,
            check_rotation=False,
        )
        if ok:
            fused = fuse_localization_samples(views)
            emit(
                f"multiview fuse posts={sorted(fused.posts)} "
                f"rms={fused.reprojection_rms_px:.2f}px"
            )
            emit(reason)
            return fused, views, reason
        emit(f"  not frozen yet: {reason}")
    if views:
        ok, reason = fusion_acceptable(
            views,
            min_posts=min_posts,
            translation_std_max_m=translation_std_max_m,
            rotation_max_deg=rotation_max_deg,
            origin_spread_max_m=origin_spread_max_m,
            check_rotation=False,
        )
        if ok:
            return fuse_localization_samples(views), views, reason
        return None, views, reason
    return (
        None,
        views,
        f"no usable views; union posts {sorted(union_posts(views))}",
    )
