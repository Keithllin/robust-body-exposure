"""Layout-board localization helpers for Stretch head D435i.

Uses the existing 8-tag ``marker_layout`` correspondences. Gate on unique
physical posts, not raw marker-ID count. Multi-view poses are fused in odom.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from canonical_bed import CORNER_POST, physical_posts
from marker_utils import invert_transform, make_transform


@dataclass(frozen=True)
class LocalizationSample:
    t_odom_layout: np.ndarray
    marker_ids: tuple[int, ...]
    reprojection_rms_px: float
    posts: frozenset[int]
    t_odom_camera: np.ndarray | None = None
    object_points: np.ndarray | None = None
    image_points: np.ndarray | None = None
    camera_matrix: np.ndarray | None = None
    distortion: np.ndarray | None = None


def _rotation_to_quat(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
    else:
        if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _quat_to_rotation(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def sample_acceptable(
    marker_ids: Sequence[int],
    *,
    min_posts: int = 3,
    min_pixel_size: float | None = None,
    pixel_sizes: dict[int, float] | None = None,
) -> bool:
    if len(physical_posts(marker_ids)) < min_posts:
        return False
    if min_pixel_size is None or not pixel_sizes:
        return True
    return all(
        float(pixel_sizes.get(int(marker_id), 0.0)) >= min_pixel_size
        for marker_id in marker_ids
        if int(marker_id) in CORNER_POST
    )


def _has_view_correspondences(sample: LocalizationSample) -> bool:
    return (
        sample.t_odom_camera is not None
        and sample.object_points is not None
        and sample.image_points is not None
        and sample.camera_matrix is not None
        and len(np.asarray(sample.object_points).reshape(-1, 3)) >= 4
    )


def _project_layout_in_view(
    t_odom_layout: np.ndarray,
    sample: LocalizationSample,
) -> np.ndarray:
    t_camera_odom = invert_transform(np.asarray(sample.t_odom_camera, dtype=np.float64))
    t_camera_layout = t_camera_odom @ np.asarray(t_odom_layout, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(t_camera_layout[:3, :3])
    projected, _ = cv2.projectPoints(
        np.asarray(sample.object_points, dtype=np.float64).reshape(-1, 3),
        rvec,
        t_camera_layout[:3, 3],
        np.asarray(sample.camera_matrix, dtype=np.float64),
        np.asarray(sample.distortion if sample.distortion is not None else np.zeros(5)),
    )
    return projected.reshape(-1, 2)


def _multiview_residuals(x: np.ndarray, samples: Sequence[LocalizationSample]) -> np.ndarray:
    rvec = np.asarray(x[:3], dtype=np.float64).reshape(3, 1)
    rotation, _ = cv2.Rodrigues(rvec)
    t_odom_layout = make_transform(rotation, np.asarray(x[3:6], dtype=np.float64))
    chunks = []
    for sample in samples:
        if not _has_view_correspondences(sample):
            continue
        proj = _project_layout_in_view(t_odom_layout, sample)
        observed = np.asarray(sample.image_points, dtype=np.float64).reshape(-1, 2)
        chunks.append((proj - observed).reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks)


def refine_t_odom_layout_multiview(
    samples: Sequence[LocalizationSample],
) -> tuple[np.ndarray, float]:
    """One T_odom_layout from all view correspondences, seeded by the richest view."""

    usable = [s for s in samples if _has_view_correspondences(s)]
    if not usable:
        raise ValueError("No view correspondences to fuse")
    anchor = max(usable, key=lambda sample: (len(sample.posts), -sample.reprojection_rms_px))
    t0 = np.asarray(anchor.t_odom_layout, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(t0[:3, :3])
    x = np.concatenate([rvec.reshape(3), t0[:3, 3]])
    residual = _multiview_residuals(x, usable)
    for _ in range(12):
        if residual.size == 0:
            break
        jac = np.zeros((residual.size, 6), dtype=np.float64)
        eps = 1e-6
        for i in range(6):
            stepped = x.copy()
            stepped[i] += eps
            jac[:, i] = (_multiview_residuals(stepped, usable) - residual) / eps
        delta, _, _, _ = np.linalg.lstsq(jac, -residual, rcond=None)
        x = x + delta
        residual = _multiview_residuals(x, usable)
        if float(np.linalg.norm(delta)) < 1e-8:
            break
    rvec = x[:3].reshape(3, 1)
    rotation, _ = cv2.Rodrigues(rvec)
    fused = make_transform(rotation, x[3:6])
    rms = float(np.sqrt(np.mean(residual**2))) if residual.size else float(anchor.reprojection_rms_px)
    return fused, rms


def fuse_odom_layout(samples: Sequence[LocalizationSample]) -> np.ndarray:
    """Average translations and unit-quaternion rotations in odom."""

    if not samples:
        raise ValueError("Need at least one localization sample")
    translations = np.array(
        [np.asarray(sample.t_odom_layout, dtype=np.float64)[:3, 3] for sample in samples]
    )
    quats = np.array(
        [
            _rotation_to_quat(np.asarray(sample.t_odom_layout)[:3, :3])
            for sample in samples
        ]
    )
    # Flip hemisphere so the mean is well defined.
    ref = quats[0]
    aligned = []
    for quat in quats:
        aligned.append(quat if float(np.dot(quat, ref)) >= 0 else -quat)
    mean_q = np.mean(aligned, axis=0)
    mean_q = mean_q / np.linalg.norm(mean_q)
    fused = np.eye(4, dtype=np.float64)
    fused[:3, :3] = _quat_to_rotation(mean_q)
    fused[:3, 3] = translations.mean(axis=0)
    return fused


def union_posts(samples: Sequence[LocalizationSample]) -> frozenset[int]:
    posts: set[int] = set()
    for sample in samples:
        posts |= set(sample.posts)
    return frozenset(posts)


def _geodesic_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = np.asarray(rotation_a, dtype=np.float64).T @ np.asarray(
        rotation_b, dtype=np.float64
    )
    cosine = float((np.trace(delta) - 1.0) / 2.0)
    cosine = min(1.0, max(-1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def pairwise_origin_spread_xy_m(samples: Sequence[LocalizationSample]) -> float:
    origins = [
        np.asarray(sample.t_odom_layout, dtype=np.float64)[:2, 3] for sample in samples
    ]
    if len(origins) < 2:
        return 0.0
    spread = 0.0
    for i, a in enumerate(origins):
        for b in origins[i + 1 :]:
            spread = max(spread, float(np.linalg.norm(a - b)))
    return spread


def pairwise_rotation_max_deg(samples: Sequence[LocalizationSample]) -> float:
    rotations = [
        np.asarray(sample.t_odom_layout, dtype=np.float64)[:3, :3] for sample in samples
    ]
    if len(rotations) < 2:
        return 0.0
    worst = 0.0
    for i, a in enumerate(rotations):
        for b in rotations[i + 1 :]:
            worst = max(worst, _geodesic_deg(a, b))
    return worst


def origin_translation_std_m(samples: Sequence[LocalizationSample]) -> float:
    origins = np.array(
        [np.asarray(s.t_odom_layout, dtype=np.float64)[:3, 3] for s in samples]
    )
    if len(origins) < 2:
        return 0.0
    return float(np.max(origins.std(axis=0)))


def fuse_localization_samples(
    samples: Sequence[LocalizationSample],
) -> LocalizationSample:
    """Fuse views in odom. Prefer multi-view reprojection over averaging PnPs."""

    if not samples:
        raise ValueError("Need at least one localization sample")
    if len(samples) == 1:
        return samples[0]
    ids = tuple(sorted({int(i) for sample in samples for i in sample.marker_ids}))
    posts = union_posts(samples)
    if all(_has_view_correspondences(s) for s in samples):
        fused, rms = refine_t_odom_layout_multiview(samples)
        return LocalizationSample(
            t_odom_layout=fused,
            marker_ids=ids,
            reprojection_rms_px=rms,
            posts=posts,
        )
    fused = fuse_odom_layout(samples)
    return LocalizationSample(
        t_odom_layout=fused,
        marker_ids=ids,
        reprojection_rms_px=max(float(s.reprojection_rms_px) for s in samples),
        posts=posts,
    )


def fusion_acceptable(
    samples: Sequence[LocalizationSample],
    *,
    min_posts: int = 3,
    translation_std_max_m: float = 0.010,
    rotation_max_deg: float = 2.0,
    origin_spread_max_m: float = 0.03,
    check_rotation: bool = True,
) -> tuple[bool, str]:
    """Whether these views can freeze T_odom_layout.

    A single view is enough only if it already contains ``min_posts``.
    Multiple views must cover ``min_posts`` in union. Independent per-view
    PnP agreement is only required when ``check_rotation`` is True.
    """

    if not samples:
        return False, "no views with a usable PnP"
    posts = union_posts(samples)
    if len(posts) < min_posts:
        return False, f"union posts {sorted(posts)} < {min_posts}"
    if len(samples) == 1:
        return True, "single view with ≥3 posts"
    spread = pairwise_origin_spread_xy_m(samples)
    std_m = origin_translation_std_m(samples)
    rot_deg = pairwise_rotation_max_deg(samples)
    if check_rotation:
        if spread > origin_spread_max_m:
            return (
                False,
                f"view origin XY spread {spread:.3f} m > {origin_spread_max_m:.3f} m",
            )
        if std_m > translation_std_max_m:
            return False, f"origin axis std {std_m:.3f} m > {translation_std_max_m:.3f} m"
        if rot_deg > rotation_max_deg:
            return False, f"view rotation {rot_deg:.2f} deg > {rotation_max_deg:.2f} deg"
    return True, (
        f"union posts={sorted(posts)} from {len(samples)} views"
        if not check_rotation
        else (
            f"fused {len(samples)} views posts={sorted(posts)} "
            f"spread={spread:.3f}m std={std_m:.3f}m rot={rot_deg:.2f}deg"
        )
    )


def marker_pixel_sizes(detections: Mapping[int, np.ndarray]) -> dict[int, float]:
    """Largest side length in pixels for each detected marker."""

    sizes: dict[int, float] = {}
    for marker_id, corners in detections.items():
        pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        closed = np.vstack([pts, pts[:1]])
        sides = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        sizes[int(marker_id)] = float(np.max(sides))
    return sizes


def compose_t_odom_layout(
    t_odom_camera: np.ndarray, t_camera_layout: np.ndarray
) -> np.ndarray:
    """``p_odom = T_odom_layout @ p_layout`` with layout = marker_layout board."""

    return np.asarray(t_odom_camera, dtype=np.float64).reshape(4, 4) @ np.asarray(
        t_camera_layout, dtype=np.float64
    ).reshape(4, 4)


def quat_xyzw_to_rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    return _quat_to_rotation(np.array([w, x, y, z], dtype=np.float64))


def transform_to_mat(
    translation_xyz: Sequence[float], quat_xyzw: Sequence[float]
) -> np.ndarray:
    rotation = quat_xyzw_to_rotation(*[float(v) for v in quat_xyzw])
    return make_transform(rotation, np.asarray(translation_xyz, dtype=np.float64))


def mat_to_quat_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    w, x, y, z = _rotation_to_quat(rotation)
    return float(x), float(y), float(z), float(w)


def yaw_align_error_rad(
    t_odom_layout: np.ndarray,
    t_odom_base: np.ndarray,
    layout_x_axis: np.ndarray | None = None,
) -> float:
    """Signed yaw (rad) to rotate the base so ``+X_base`` matches bed ``+X``.

    Unused by the executor (no auto ALIGNING). Kept for tests.
    """

    x_layout = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if layout_x_axis is None
        else np.asarray(layout_x_axis, dtype=np.float64).reshape(3)
    )
    return _planar_yaw_error_rad(
        np.asarray(t_odom_layout, dtype=np.float64)[:3, :3] @ x_layout,
        np.asarray(t_odom_base, dtype=np.float64)[:3, :3]
        @ np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )


def bedside_parking_error_rad(
    t_odom_layout: np.ndarray,
    t_odom_base: np.ndarray,
    *,
    layout_y_axis: np.ndarray | None = None,
) -> float:
    """Signed yaw: ``+X_base`` vs bed ``+Y`` (drive along the long side).

    Raw Stretch PnP is Z-down, same as ceiling overlay. After that, bed
    ``+Y`` may be antiparallel to ``+X_base`` (~180°) even when the base
    is correctly parallel and the arm points into the bed. Use
    ``bedside_parallel_error_rad`` plus ``bedside_arm_into_bed_error_rad``
    to accept either heading.
    """

    y_layout = (
        np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if layout_y_axis is None
        else np.asarray(layout_y_axis, dtype=np.float64).reshape(3)
    )
    return _planar_yaw_error_rad(
        np.asarray(t_odom_layout, dtype=np.float64)[:3, :3] @ y_layout,
        np.asarray(t_odom_base, dtype=np.float64)[:3, :3]
        @ np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )


def bedside_parallel_error_rad(
    t_odom_layout: np.ndarray,
    t_odom_base: np.ndarray,
    *,
    layout_y_axis: np.ndarray | None = None,
) -> float:
    """Abs yaw from the nearest parallel alignment (0° or 180°)."""

    err = bedside_parking_error_rad(
        t_odom_layout, t_odom_base, layout_y_axis=layout_y_axis
    )
    return float(min(abs(err), abs(abs(err) - np.pi)))


def bedside_arm_into_bed_error_rad(
    t_odom_layout: np.ndarray,
    t_odom_base: np.ndarray,
    *,
    layout_x_axis: np.ndarray | None = None,
) -> float:
    """Signed yaw: arm ``-Y_base`` vs bed ``+X`` (into the mattress)."""

    x_layout = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if layout_x_axis is None
        else np.asarray(layout_x_axis, dtype=np.float64).reshape(3)
    )
    return _planar_yaw_error_rad(
        np.asarray(t_odom_layout, dtype=np.float64)[:3, :3] @ x_layout,
        np.asarray(t_odom_base, dtype=np.float64)[:3, :3]
        @ np.array([0.0, -1.0, 0.0], dtype=np.float64),
    )


def _planar_yaw_error_rad(desired_odom: np.ndarray, actual_odom: np.ndarray) -> float:
    yaw_des = np.arctan2(float(desired_odom[1]), float(desired_odom[0]))
    yaw_act = np.arctan2(float(actual_odom[1]), float(actual_odom[0]))
    err = yaw_des - yaw_act
    return float(np.arctan2(np.sin(err), np.cos(err)))


def ensure_layout_z_up(t_odom_layout: np.ndarray) -> np.ndarray:
    """If PnP left layout +Z downward, rotate 180° about layout X (Y and Z flip).

    Do **not** use this for base/arm motion or VERIFY. Rx(π) reverses
    layout +Y, so ``translate_mobile_base`` drives toward the opposite
    end of the bed from the CMA / ceiling-overlay grasp. Motion uses
    ``t_odom_layout_for_motion``.
    """

    t_ol = np.asarray(t_odom_layout, dtype=np.float64).reshape(4, 4).copy()
    if float(t_ol[2, 2]) >= 0.0:
        return t_ol
    rx_pi = np.diag([1.0, -1.0, -1.0, 1.0])
    return t_ol @ rx_pi


def t_odom_layout_for_motion(t_odom_layout: np.ndarray) -> np.ndarray:
    """Stretch PnP as stored — same Z-down T as ceiling overlay / sim_origin."""

    return np.asarray(t_odom_layout, dtype=np.float64).reshape(4, 4).copy()


def layout_xy_at_height(
    p_layout: np.ndarray, z_layout: float
) -> np.ndarray:
    """Keep layout XY, replace Z. Hover is same XY at the gripper height.

    ``T_odom_layout`` couples Z into odom XY (third column). Sending the
    cloth-plane point as the EE target therefore misses by
    ``T[:2, 2] * (z_ee − z_cloth)`` — about 3 cm from a 40 cm standoff.
    """

    p = np.asarray(p_layout, dtype=np.float64).reshape(3).copy()
    p[2] = float(z_layout)
    return p


def _registration_dirs(start: Path) -> tuple[Path, ...]:
    """Trial dir first, then the subject/session dir (parent of ``pose_*``)."""

    start = Path(start).resolve()
    if start.name.startswith("pose_") or (start / "canonical_bed_frame.json").is_file():
        return (start, start.parent)
    return (start,)


def resolve_session_dir_once(explicit=None, *, force: bool = False):
    """Re-export: resolve ``sessions/current`` once per process."""

    from session_paths import resolve_session_dir_once as _resolve

    return _resolve(explicit, force=force)


def origin_paths(session_dir=None):
    """Re-export: stretch/ paths for a resolved session directory."""

    from session_paths import origin_paths as _origin_paths

    return _origin_paths(session_dir)


def resolve_stretch_origin_path(trial_dir: Path) -> Path:
    """Prefer the process-frozen session corrected origin, then legacy paths.

    Session directory is resolved once (see ``session_paths.resolve_session_dir_once``).
    """

    try:
        from session_paths import session_paths

        paths = session_paths()
        if paths.origin_corrected.is_file():
            return paths.origin_corrected
        if paths.origin_raw.is_file():
            return paths.origin_raw
        if paths.origin_json.is_file():
            return paths.origin_json
    except FileNotFoundError:
        pass
    trial = Path(trial_dir)
    dirs = _registration_dirs(trial)
    for directory in dirs:
        corrected = directory / "stretch_origin_corrected.json"
        if corrected.is_file():
            return corrected
    for directory in dirs:
        raw = directory / "stretch_origin.json"
        if raw.is_file():
            return raw
    return trial / "stretch_origin.json"


def translate_t_odom_layout(
    t_odom_layout: np.ndarray, delta_layout: np.ndarray
) -> np.ndarray:
    """Return ``T_odom_layout`` that shifts layout points by ``delta_layout``.

    ``p_odom = T @ p_layout``. Adding ``delta`` in layout (rotation unchanged)
    requires ``t' = t - R @ delta``. Choose ``delta = p_ceiling - p_tf`` so
    the same physical wrist maps to the ceiling measurement.
    """

    t_ol = np.asarray(t_odom_layout, dtype=np.float64).reshape(4, 4).copy()
    delta = np.asarray(delta_layout, dtype=np.float64).reshape(3)
    t_ol[:3, 3] = t_ol[:3, 3] - t_ol[:3, :3] @ delta
    return t_ol


def layout_canonical_disagreement_m(origin_layout: np.ndarray) -> float:
    """XY metres between layout-board origin (0,0) and trial centroid ``O``.

    This is the stored ``origin_layout`` of ``canonical_bed_frame.json``, not a
    live Stretch-vs-ceiling comparison. The executor maps pickle through
    ``(O, R)``; a ~6 cm Y offset on this bed is expected.
    """

    origin = np.asarray(origin_layout, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(origin[:2]))


def apply_base_yaw(t_odom_base: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Return a new ``T_odom_base`` after a +Z base rotation of ``yaw_rad``."""

    delta = np.eye(4, dtype=np.float64)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    delta[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(t_odom_base, dtype=np.float64).reshape(4, 4) @ delta


def se3_a_from_b(t_parent_a: np.ndarray, t_parent_b: np.ndarray) -> np.ndarray:
    """``T_a_b`` such that ``p_a = T_a_b @ p_b``."""

    return invert_transform(t_parent_a) @ np.asarray(t_parent_b, dtype=np.float64).reshape(
        4, 4
    )
