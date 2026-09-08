"""Head-camera 5x5 board PnP. Broadcasts a frozen odom→layout TF during a pull.

Uses ``DICT_5X5_100`` and ``marker_layout.json``. Do not call Hello Robot
``detect_aruco_markers`` (that stack is ``DICT_6X6_250``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .path_setup import add_workstation_code

add_workstation_code()

from canonical_bed import CORNER_POST, TOP_MARKER_IDS, physical_posts  # noqa: E402
from marker_utils import (  # noqa: E402
    DEFAULT_LAYOUT_PATH,
    MarkerLayoutError,
    collect_correspondences,
    detect_markers,
    load_layout,
    solve_bed_to_camera,
)
from stretch_localize import (  # noqa: E402
    LocalizationSample,
    compose_t_odom_layout,
    fuse_odom_layout,
    marker_pixel_sizes,
    sample_acceptable,
    union_posts,
)

DICTIONARY = "DICT_5X5_100"
SIDE_MARKER_IDS = frozenset({10, 11, 12, 13})


def _markers_for_pnp(detections, sizes, min_pixel_size: float | None):
    """Prefer coplanar tags. Mixing a top tag with a side tag inflates RMS.

    View one saw 0+1+10; picking the larger side tag 10 with top tag 1
    gave ~20 px RMS. Two tops (0 and 1) stay on the mattress plane.
    """

    thresh = 0.0 if min_pixel_size is None else float(min_pixel_size)
    top: dict = {}
    side: dict = {}
    for key, corners in detections.items():
        mid = int(key)
        if float(sizes.get(mid, 0.0)) < thresh:
            continue
        if mid in TOP_MARKER_IDS:
            top[mid] = corners
        elif mid in SIDE_MARKER_IDS:
            side[mid] = corners
    if len(physical_posts(top)) >= 2:
        return top
    if len(physical_posts(side)) >= 2:
        return side
    best: dict[int, tuple[bool, float, object]] = {}
    for key, corners in detections.items():
        mid = int(key)
        post = CORNER_POST.get(mid)
        if post is None:
            continue
        size = float(sizes.get(mid, 0.0))
        if size < thresh:
            continue
        is_top = mid in TOP_MARKER_IDS
        prev = best.get(post)
        if prev is None:
            best[post] = (is_top, size, key)
            continue
        prev_top, prev_size, _ = prev
        if is_top and not prev_top:
            best[post] = (is_top, size, key)
        elif is_top == prev_top and size > prev_size:
            best[post] = (is_top, size, key)
    return {int(item[2]): detections[item[2]] for item in best.values()}


def pnp_layout_from_image(
    image_bgr,
    camera_matrix,
    distortion,
    *,
    layout_path=DEFAULT_LAYOUT_PATH,
    dictionary=DICTIONARY,
    min_posts=3,
    min_pixel_size: float | None = 20.0,
):
    """Return (pose_dict or None, used_ids, posts, pixel_sizes).

    ``pose_dict`` contains ``T_camera_bed`` (OpenCV optical → layout board).
    """

    layout, specs = load_layout(str(layout_path))
    detections = detect_markers(image_bgr, dictionary)
    sizes = marker_pixel_sizes(detections)
    detections = _markers_for_pnp(detections, sizes, min_pixel_size)
    object_points, image_points, used_ids = collect_correspondences(
        detections,
        specs,
        float(layout["marker_size_m"]),
    )
    posts = physical_posts(used_ids)
    if not sample_acceptable(
        used_ids,
        min_posts=min_posts,
        min_pixel_size=min_pixel_size,
        pixel_sizes=sizes,
    ):
        return None, used_ids, posts, sizes
    if len(object_points) < 4:
        return None, used_ids, posts, sizes
    try:
        pose = solve_bed_to_camera(
            object_points, image_points, camera_matrix, distortion
        )
    except MarkerLayoutError:
        return None, used_ids, posts, sizes
    pose["object_points"] = object_points
    pose["image_points"] = image_points
    return pose, used_ids, posts, sizes


def sample_from_image(
    image_bgr,
    camera_matrix,
    distortion,
    t_odom_camera: np.ndarray,
    *,
    layout_path=DEFAULT_LAYOUT_PATH,
    min_posts: int = 3,
    min_pixel_size: float | None = 20.0,
    reprojection_rms_max_px: float | None = 5.0,
) -> Optional[LocalizationSample]:
    if not layout_path:
        layout_path = DEFAULT_LAYOUT_PATH
    pose, used_ids, posts, sizes = pnp_layout_from_image(
        image_bgr,
        camera_matrix,
        distortion,
        layout_path=layout_path,
        min_posts=min_posts,
        min_pixel_size=min_pixel_size,
    )
    if pose is None:
        return None
    rms = float(pose["reprojection_rms_px"])
    if reprojection_rms_max_px is not None and rms > reprojection_rms_max_px:
        return None
    t_camera_layout = np.asarray(pose["T_camera_bed"], dtype=np.float64)
    t_odom_layout = compose_t_odom_layout(t_odom_camera, t_camera_layout)
    return LocalizationSample(
        t_odom_layout=t_odom_layout,
        marker_ids=tuple(int(v) for v in used_ids),
        reprojection_rms_px=rms,
        posts=frozenset(int(v) for v in posts),
        t_odom_camera=np.asarray(t_odom_camera, dtype=np.float64),
        object_points=np.asarray(pose["object_points"], dtype=np.float64),
        image_points=np.asarray(pose["image_points"], dtype=np.float64),
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        distortion=np.asarray(distortion, dtype=np.float64).reshape(-1),
    )


def fuse_samples(samples) -> np.ndarray:
    if union_posts(samples) and len(samples) > 1:
        return fuse_odom_layout(samples)
    if not samples:
        raise ValueError("no localization samples")
    return np.asarray(samples[-1].t_odom_layout, dtype=np.float64)
