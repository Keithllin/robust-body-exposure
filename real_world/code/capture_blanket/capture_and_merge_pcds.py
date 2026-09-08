#!/usr/bin/env python3
"""Capture one or more ZED blanket PCDs and merge them in the bed frame.

The calibration file contains one fixed ``T_bed_camera`` per camera role.
ZED is opened with ``COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP`` so that this
transform can be applied directly to the XYZ measurement returned by the SDK.

Ceiling-only backward-compatible capture:

  conda run -n robe-zed python code/capture_blanket/capture_and_merge_pcds.py \
      --pose-dir /path/to/pose_dir --roles ceiling

Three-camera capture after calibration and side filter tuning:

  conda run -n robe-zed python code/capture_blanket/capture_and_merge_pcds.py \
      --pose-dir /path/to/pose_dir \
      --roles ceiling,side_left,side_right --require-calibration

With the default ``--merge-mode auto``, the multi-camera merge keeps the
ceiling cloud as the geometry and uses left/right only to drop ceiling
points whose XY neighbourhood was seen from the side at a disagreeing 3D
location.  That avoids punching holes where a side camera is occluded.
Use ``--merge-mode top_supported`` for the previous side-geometry merge,
and ``--merge-mode union`` for an unfiltered union.

All requested roles are opened once at the start of a capture, used for
filtering, then closed together — avoiding per-role USB reopen cycles.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from canonical_bed import (  # noqa: E402
    CanonicalBedError,
    canonical_bed_frame,
    layout_points_to_canonical,
    load_canonical_frame,
    resolve_canonical_path,
    sample_xyz_at_pixels,
    save_canonical_frame,
)
from marker_utils import (  # noqa: E402
    crop_pcd_to_bed_xy,
    detect_markers,
    transform_pcd,
    transform_points,
)
from pcd_merge import merge_clouds  # noqa: E402
from zed_util import (  # noqa: E402
    DEFAULT_FILTER_JSON,
    PCD_EXPOSURE,
    PCD_GAIN,
    capture_filtered_blanket,
    configure_exposure_gain,
    grab_frame,
    open_zed,
    resolution_from_name,
)


REAL_WORLD_DIR = CODE_DIR.parent
DEFAULT_EXTRINSICS = REAL_WORLD_DIR / "calibration" / "zed_extrinsics.json"
ROLE_NAMES = ("ceiling", "side_left", "side_right")
DEFAULT_FILTERS = {
    "ceiling": DEFAULT_FILTER_JSON,
    "side_left": REAL_WORLD_DIR / "calibration" / "zed_blanket_filter_side_left.json",
    "side_right": REAL_WORLD_DIR / "calibration" / "zed_blanket_filter_side_right.json",
}


def parse_roles(value: str) -> tuple[str, ...]:
    roles = tuple(item.strip() for item in value.split(",") if item.strip())
    if not roles:
        raise ValueError("At least one camera role is required")
    unknown = sorted(set(roles) - set(ROLE_NAMES))
    if unknown:
        raise ValueError(f"Unknown camera role(s): {unknown}")
    if len(set(roles)) != len(roles):
        raise ValueError("Camera roles must be unique")
    return roles


def load_extrinsics(path: Path) -> dict:
    if not path.exists():
        return {"cameras": {}}
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("cameras"), dict):
        raise ValueError(f"Invalid camera map in {path}")
    return payload


def explicit_serial(args, role: str) -> int:
    return int(
        {
            "ceiling": args.serial,
            "side_left": args.serial_side_left,
            "side_right": args.serial_side_right,
        }[role]
    )


def filter_path(args, role: str) -> Path:
    configured = {
        "ceiling": args.filter_json,
        "side_left": args.filter_json_side_left,
        "side_right": args.filter_json_side_right,
    }[role]
    if configured:
        return Path(configured)
    if getattr(args, "forbid_calibration_fallback", False):
        raise RuntimeError(
            f"REJECT: no session filter for {role}. "
            "Pass --filter-json / --filter-json-side-* from session/zed. "
            "Calibration defaults are not a silent fallback."
        )
    return Path(DEFAULT_FILTERS[role])


def get_camera_config(
    role: str,
    args,
    extrinsics: dict,
) -> tuple[int, dict | None]:
    entry = extrinsics.get("cameras", {}).get(role)
    serial = explicit_serial(args, role)
    if not serial and entry:
        serial = int(entry.get("serial", 0))
    if not serial and role != "ceiling":
        raise RuntimeError(
            f"No serial for {role}; pass --serial-{role.replace('_', '-')} "
            "or calibrate that role first."
        )
    return serial, entry


def role_capture_plan(
    role: str,
    args,
    extrinsics: dict,
) -> dict:
    """Resolve serial / filter / transform for one role without opening the camera."""

    serial, entry = get_camera_config(role, args, extrinsics)
    filter_json = filter_path(args, role)
    if not filter_json.exists():
        raise FileNotFoundError(
            f"Missing {role} filter: {filter_json}. "
            "Tune this camera independently before merging."
        )

    transform = None
    if entry is not None and entry.get("T_bed_camera") is not None:
        transform = np.asarray(entry["T_bed_camera"], dtype=np.float64).reshape(4, 4)
    if transform is None and (
        args.require_calibration or not getattr(args, "allow_image_frame", False)
    ):
        raise RuntimeError(
            f"No T_bed_camera for {role} in {args.extrinsics}; "
            "IMAGE-frame PCD cannot be written as bed-frame output. "
            "Calibrate the camera or pass --allow-image-frame (debug only)."
        )

    resolution_name = (
        entry.get("resolution", args.resolution) if entry else args.resolution
    )
    return {
        "role": role,
        "serial": int(serial),
        "entry": entry,
        "filter_json": filter_json,
        "transform": transform,
        "resolution": resolution_from_name(resolution_name),
        "resolution_name": resolution_name,
    }


def open_role_cameras(
    plans: list[dict],
    *,
    exposure: int | None,
    gain: int | None,
) -> dict[str, object]:
    """Open every requested ZED once and lock PCD exposure/gain.

    Cameras stay open for the whole capture session so USB renegotiation
    only happens at the start/end of ``capture_and_merge_pcds``, not once
    per role.
    """

    cameras: dict[str, object] = {}
    try:
        for plan in plans:
            role = plan["role"]
            print(
                f"Opening {role} ZED (serial={plan['serial'] or 'auto'}, "
                f"resolution={plan['resolution_name']})..."
            )
            print(f"Filter: {plan['filter_json']}")
            camera = open_zed(
                serial=plan["serial"],
                resolution=plan["resolution"],
            )
            cameras[role] = camera
            camera_settings = configure_exposure_gain(
                camera,
                exposure=exposure,
                gain=gain,
                lock=True,
            )
            plan["camera_settings"] = camera_settings
            print(
                f"Camera settings ({role}): PCD lock "
                f"exposure={camera_settings['exposure']} "
                f"gain={camera_settings['gain']} "
                "(filter/PCD pair; marker calib uses zed_extrinsics.json)"
            )
    except Exception:
        close_role_cameras(cameras)
        raise
    return cameras


def close_role_cameras(cameras: dict[str, object]) -> None:
    for role, camera in list(cameras.items()):
        try:
            camera.close()
        except Exception as exc:
            print(f"WARN: failed to close {role} ZED: {exc}")
        cameras.pop(role, None)


def capture_role_with_camera(
    plan: dict,
    camera,
    args,
    *,
    warm_frames: int,
    grab_attempts: int,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Grab + filter one already-open camera into the bed frame."""

    role = plan["role"]
    filter_json = plan["filter_json"]
    transform = plan["transform"]
    pcd_camera, bgr, mask, params, xyz, hsv_mask = capture_filtered_blanket(
        camera,
        filter_json=str(filter_json),
        warm_frames=warm_frames,
        grab_attempts=grab_attempts,
        mask_backend=args.mask_backend,
    )

    point_count = len(pcd_camera.points)
    if point_count == 0:
        raise RuntimeError(
            f"No points survived the {role} filter. "
            "Inspect the saved mask and re-tune this camera."
        )

    if transform is None:
        if not getattr(args, "allow_image_frame", False):
            raise RuntimeError(
                f"REJECT: {role} has no T_bed_camera. "
                "IMAGE-frame PCD cannot be written as bed-frame output."
            )
        print(
            f"WARN: {role} has no T_bed_camera; writing IMAGE-frame PCD "
            "(--allow-image-frame)."
        )
        pcd_bed = pcd_camera
    else:
        pcd_bed = transform_pcd(pcd_camera, transform)
        print(
            f"Layout-bed transform for {role}: {point_count} -> "
            f"{len(pcd_bed.points)} points (crop deferred until canonicalization)"
        )
    if len(pcd_bed.points) == 0:
        raise RuntimeError(
            f"No {role} points remain after the layout-bed transform; "
            "retune the camera filter."
        )

    info = {
        "role": role,
        "serial": int(plan["serial"]),
        "filter_json": str(filter_json.resolve()),
        "mask_backend": str(args.mask_backend),
        "hsv_px": int(np.count_nonzero(hsv_mask)),
        "mask_px": int(np.count_nonzero(mask)),
        "num_points_camera": point_count,
        "num_points_bed": len(pcd_bed.points),
        "T_bed_camera": transform.tolist() if transform is not None else None,
        "camera_settings": plan["camera_settings"],
        "bgr": bgr,
        "mask": mask,
        "hsv_mask": hsv_mask,
        "xyz": xyz if role == "ceiling" else None,
        "filter_params": params.__dict__,
    }
    return pcd_bed, info


def marker_calib_exposure_gain(entry) -> tuple[int | None, int | None]:
    settings = (entry or {}).get("camera_settings") or {}
    exposure = settings.get("exposure")
    gain = settings.get("gain")
    return (
        None if exposure is None else int(exposure),
        None if gain is None else int(gain),
    )


def grab_ceiling_aruco_rgbd(camera, *, exposure: int, gain: int, warm_frames: int = 8):
    """RGB-D for top-marker PnP using zed_extrinsics.json exposure/gain."""

    settings = configure_exposure_gain(
        camera, exposure=exposure, gain=gain, lock=True
    )
    print(
        "Ceiling ArUco grab: marker-calib lock "
        f"exposure={settings['exposure']} gain={settings['gain']} "
        "(PCD stays at the filter pair; this frame is detection only)"
    )
    bgr = xyz = None
    for _ in range(max(1, int(warm_frames))):
        bgr, _depth, xyz = grab_frame(camera)
    if bgr is None or xyz is None:
        raise RuntimeError("Ceiling ArUco grab failed after switching to calib exposure")
    return bgr, xyz, settings


def marker_centers_layout_bed(
    image_bgr: np.ndarray,
    xyz_camera: np.ndarray,
    t_bed_camera: np.ndarray,
    *,
    dictionary: str,
) -> dict[int, np.ndarray]:
    """Detect top markers and lift their pixel centers into layout-bed metres."""

    detections = detect_markers(image_bgr, dictionary)
    missing = [marker_id for marker_id in (0, 1, 2, 3) if marker_id not in detections]
    if missing:
        raise CanonicalBedError(
            "Ceiling image is missing top ArUco IDs "
            f"{missing}; saw {sorted(detections)}"
        )
    pixels = np.array(
        [np.asarray(detections[marker_id], dtype=np.float64).mean(axis=0) for marker_id in (0, 1, 2, 3)],
        dtype=np.float64,
    )
    camera_xyz = sample_xyz_at_pixels(xyz_camera, pixels, radius_px=3)
    if not np.isfinite(camera_xyz).all():
        raise CanonicalBedError(
            "Ceiling depth is missing at one or more top-marker centers; "
            f"finite={np.isfinite(camera_xyz).all(axis=1).tolist()}"
        )
    layout_xyz = transform_points(camera_xyz, t_bed_camera)
    return {marker_id: layout_xyz[index] for index, marker_id in enumerate((0, 1, 2, 3))}


def canonicalize_merged_cloud(
    merged: o3d.geometry.PointCloud,
    pose_dir: Path,
    *,
    ceiling_bgr,
    ceiling_xyz,
    ceiling_transform,
    dictionary: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    detect_top_markers: bool = False,
):
    """Reuse the trial canonical frame when present; detect 0–3 only if asked."""

    existing = resolve_canonical_path(pose_dir)
    if existing is not None:
        frame = load_canonical_frame(existing)
        source = str(existing)
        print(f"Reusing canonical bed frame: {existing}")
    else:
        from canonical_bed import write_canonical_from_sim_origin

        try:
            existing = write_canonical_from_sim_origin(pose_dir)
        except CanonicalBedError:
            existing = None
        if existing is not None:
            frame = load_canonical_frame(existing)
            source = str(existing)
            print(f"Canonical bed frame from uncovered sim_origin: {existing}")
        else:
            if not detect_top_markers:
                raise CanonicalBedError(
                    "No canonical_bed_frame.json or sim_origin_data.pkl. "
                    "Reuse the session pose pack, or pass --detect-top-markers "
                    "/ --require-aruco once for this pose."
                )
            if ceiling_bgr is None or ceiling_xyz is None or ceiling_transform is None:
                raise CanonicalBedError(
                    "Need a ceiling RGB-D frame and T_bed_camera to define "
                    "canonical_bed_frame.json"
                )
            centers = marker_centers_layout_bed(
                ceiling_bgr,
                ceiling_xyz,
                ceiling_transform,
                dictionary=dictionary,
            )
            frame = canonical_bed_frame(centers)
            source = "ceiling_top_markers"
            dest = pose_dir / "canonical_bed_frame.json"
            if pose_dir.name in {"initial", "intermediate", "final", "uncover"}:
                dest = pose_dir.parent / "canonical_bed_frame.json"
            save_canonical_frame(dest, frame)
            print(f"Wrote canonical bed frame: {dest}")

    points = np.asarray(merged.points, dtype=np.float64)
    converted = layout_points_to_canonical(points, frame)
    canonical = o3d.geometry.PointCloud(merged)
    canonical.points = o3d.utility.Vector3dVector(converted)
    if merged.has_colors():
        canonical.colors = merged.colors
    before = len(canonical.points)
    canonical = crop_pcd_to_bed_xy(
        canonical,
        x_limits=x_limits,
        y_limits=y_limits,
    )
    print(
        f"Canonical crop: {before} -> {len(canonical.points)} points "
        f"(x={x_limits}, y={y_limits})"
    )
    if len(canonical.points) == 0:
        raise RuntimeError(
            "No points remain after canonical bed crop; "
            "widen --bed-x/--bed-y limits."
        )
    meta = {
        "frame": "canonical_bed",
        "source": source,
        "origin_layout": frame.origin.tolist(),
        "detected_ids": list(frame.detected_ids),
        "num_points_before_crop": before,
        "num_points_after_crop": len(canonical.points),
        "bed_crop": {
            "x_limits": list(x_limits),
            "y_limits": list(y_limits),
        },
    }
    return canonical, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=str, default="TEST")
    parser.add_argument("--pose-dir", type=str, default="TEST")
    parser.add_argument("--manikin", type=int, default=0)
    parser.add_argument(
        "--roles",
        type=str,
        default="ceiling",
        help="Comma-separated subset of ceiling,side_left,side_right",
    )
    parser.add_argument(
        "--extrinsics",
        type=str,
        default=str(DEFAULT_EXTRINSICS),
        help="zed_extrinsics.json produced by calibrate_zed_markers.py",
    )
    parser.add_argument(
        "--filter-json",
        type=str,
        default=str(DEFAULT_FILTER_JSON),
        help="Ceiling filter JSON",
    )
    parser.add_argument("--filter-json-side-left", type=str, default=None)
    parser.add_argument("--filter-json-side-right", type=str, default=None)
    parser.add_argument(
        "--mask-backend",
        choices=("sam2", "hsv"),
        default="sam2",
        help="sam2: HSV box then SAM2 pixels on all roles (default). "
        "hsv: previous color+depth filter only.",
    )
    parser.add_argument(
        "--serial",
        type=int,
        default=0,
        help="Ceiling ZED serial; 0 = use calibrated serial or first device",
    )
    parser.add_argument("--serial-side-left", type=int, default=0)
    parser.add_argument("--serial-side-right", type=int, default=0)
    parser.add_argument(
        "--resolution",
        type=str,
        default="HD720",
        help="Fallback resolution when a calibration entry has none",
    )
    parser.add_argument("--warm-frames", type=int, default=30)
    parser.add_argument(
        "--exposure",
        type=int,
        default=None,
        help="PCD exposure 0-100; default 55 (not the marker-calib value)",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=None,
        help="PCD gain 0-100; default 40 (not the marker-calib value)",
    )
    parser.add_argument(
        "--lock-exposure-gain",
        action="store_true",
        help="Warm auto exposure, then freeze the selected/current values",
    )
    parser.add_argument("--grab-attempts", type=int, default=15)
    parser.add_argument("--bed-x-min", type=float, default=-0.55)
    parser.add_argument("--bed-x-max", type=float, default=0.55)
    parser.add_argument("--bed-y-min", type=float, default=-1.10)
    parser.add_argument("--bed-y-max", type=float, default=1.10)
    parser.add_argument(
        "--require-calibration",
        action="store_true",
        help="Reject any role without a fixed T_bed_camera",
    )
    parser.add_argument(
        "--allow-image-frame",
        action="store_true",
        help="Debug: allow missing T_bed_camera (IMAGE-frame PCD). Not production.",
    )
    parser.add_argument(
        "--forbid-calibration-fallback",
        action="store_true",
        help="Reject missing session filter JSON instead of calibration defaults.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Optional bed-frame voxel downsample size in meters",
    )
    parser.add_argument(
        "--merge-mode",
        choices=("auto", "union", "top_supported", "ceiling_primary"),
        default="auto",
        help="Merge policy; auto keeps ceiling geometry and uses sides as a "
        "noise veto (ceiling_primary)",
    )
    parser.add_argument(
        "--support-radius-3d",
        "--support-radius-xy",
        dest="support_radius",
        type=float,
        default=0.03,
        help="XYZ support radius in meters for pairwise top-supported merging",
    )
    parser.add_argument(
        "--no-fill-holes-from-support",
        dest="fill_holes_from_support",
        action="store_false",
        help="Do not restore ceiling coverage in XY columns no side camera reached",
    )
    parser.add_argument(
        "--fill-cell",
        type=float,
        default=0.05,
        help="XY cell size in meters for hole filling; match the model voxel size",
    )
    parser.add_argument(
        "--fill-dilation-cells",
        type=int,
        default=2,
        help="How many cells a filled column may sit from existing coverage",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an Open3D viewer after capture (interactive)",
    )
    parser.add_argument(
        "--no-canonicalize",
        action="store_true",
        help="Keep the layout-bed frame; skip 4-marker origin/axes",
    )
    parser.add_argument(
        "--detect-top-markers",
        action="store_true",
        help="Detect bed-corner markers 0-3 if no canonical frame exists. "
        "Default: reuse canonical_bed_frame.json / sim_origin.",
    )
    parser.add_argument(
        "--aruco-dictionary",
        type=str,
        default="DICT_5X5_100",
        help="OpenCV ArUco dictionary for ceiling top markers",
    )
    args = parser.parse_args()

    try:
        roles = parse_roles(args.roles)
    except ValueError as exc:
        parser.error(str(exc))
    if args.voxel_size < 0:
        parser.error("--voxel-size must be non-negative")
    if args.bed_x_min >= args.bed_x_max:
        parser.error("--bed-x-min must be smaller than --bed-x-max")
    if args.bed_y_min >= args.bed_y_max:
        parser.error("--bed-y-min must be smaller than --bed-y-max")
    if args.support_radius <= 0:
        parser.error("--support-radius-3d must be positive")
    if args.fill_cell <= 0:
        parser.error("--fill-cell must be positive")
    if args.fill_dilation_cells < 1:
        parser.error("--fill-dilation-cells must be >= 1")
    if len(roles) > 1:
        args.require_calibration = True

    pose_dir = Path(args.pose_dir)
    pose_dir.mkdir(parents=True, exist_ok=True)
    extrinsics_path = Path(args.extrinsics)
    extrinsics = load_extrinsics(extrinsics_path)

    metadata = {
        "frame": "bed",
        "coordinate_system": extrinsics.get(
            "coordinate_system", "RIGHT_HANDED_Y_UP"
        ),
        "extrinsics": str(extrinsics_path.resolve()),
        "roles": {},
        "camera_session": "keep_open_all_roles",
        "mask_backend": args.mask_backend,
    }
    pcds = {}
    ceiling_bgr = None
    ceiling_xyz = None
    ceiling_transform = None
    plans = [role_capture_plan(role, args, extrinsics) for role in roles]
    exposure = PCD_EXPOSURE if args.exposure is None else args.exposure
    gain = PCD_GAIN if args.gain is None else args.gain
    # Keep every selected ZED open for the whole multi-role capture so USB
    # open/close only happens once per capture_and_merge invocation.
    cameras = open_role_cameras(plans, exposure=exposure, gain=gain)
    try:
        for plan in plans:
            role = plan["role"]
            print(f"Capturing {role} while other roles stay open...")
            pcd_bed, info = capture_role_with_camera(
                plan,
                cameras[role],
                args,
                warm_frames=5,
                grab_attempts=args.grab_attempts,
            )
            if args.voxel_size > 0:
                pcd_bed = pcd_bed.voxel_down_sample(args.voxel_size)
                info["num_points_bed_after_voxel"] = len(pcd_bed.points)
            pcds[role] = pcd_bed

            pcd_path = pose_dir / f"pcd_filtered_{role}.pcd"
            rgb_path = pose_dir / f"covered_rgb_{role}.png"
            mask_path = pose_dir / f"blanket_mask_{role}.png"
            if not o3d.io.write_point_cloud(str(pcd_path), pcd_bed):
                raise RuntimeError(f"Failed to write {pcd_path}")
            bgr = info.pop("bgr")
            xyz = info.pop("xyz", None)
            mask = info.pop("mask")
            hsv_mask = info.pop("hsv_mask")
            cv2.imwrite(str(rgb_path), bgr)
            cv2.imwrite(str(mask_path), mask)
            if args.mask_backend == "sam2":
                cv2.imwrite(str(pose_dir / f"blanket_mask_{role}_hsv.png"), hsv_mask)
            if role == "ceiling":
                ceiling_bgr = bgr
                ceiling_xyz = xyz
                ceiling_transform = plan["transform"]
                o3d.io.write_point_cloud(
                    str(pose_dir / "pcd_filtered_top.pcd"), pcd_bed
                )

            metadata["roles"][role] = info
            print(f"Saved {pcd_path} ({len(pcd_bed.points)} bed-frame points)")

        need_aruco = (
            args.detect_top_markers
            and not args.no_canonicalize
            and resolve_canonical_path(pose_dir) is None
            and "ceiling" in cameras
        )
        if need_aruco:
            ceiling_plan = next(p for p in plans if p["role"] == "ceiling")
            calib_exposure, calib_gain = marker_calib_exposure_gain(
                ceiling_plan.get("entry")
            )
            if calib_exposure is None or calib_gain is None:
                print(
                    "WARN: zed_extrinsics.json has no ceiling camera_settings; "
                    "ArUco will use the PCD exposure frame"
                )
            else:
                ceiling_bgr, ceiling_xyz, aruco_settings = grab_ceiling_aruco_rgbd(
                    cameras["ceiling"],
                    exposure=calib_exposure,
                    gain=calib_gain,
                )
                aruco_rgb = pose_dir / "covered_rgb_ceiling_aruco.png"
                cv2.imwrite(str(aruco_rgb), ceiling_bgr)
                metadata["ceiling_aruco_camera_settings"] = aruco_settings
                print(f"Saved {aruco_rgb} (marker-calib exposure)")
    finally:
        print(f"Closing {len(cameras)} ZED camera(s)...")
        close_role_cameras(cameras)

    merged, merge_info = merge_clouds(
        pcds,
        roles,
        mode=args.merge_mode,
        radius_m=args.support_radius,
        fill_holes_from_support=args.fill_holes_from_support,
        fill_cell_m=args.fill_cell,
        fill_dilation_cells=args.fill_dilation_cells,
    )
    metadata["merge"] = merge_info
    hole_fill = merge_info.get("hole_fill")
    if hole_fill:
        print(
            f"Hole fill: +{hole_fill['num_filled_points']} support points "
            f"covering {hole_fill['num_filled_columns']} columns "
            f"({hole_fill['num_columns_before']} -> "
            f"{hole_fill['num_columns_after']} occupied)"
        )
    if args.no_canonicalize:
        before = len(merged.points)
        merged = crop_pcd_to_bed_xy(
            merged,
            x_limits=(args.bed_x_min, args.bed_x_max),
            y_limits=(args.bed_y_min, args.bed_y_max),
        )
        metadata["canonical"] = {
            "skipped": True,
            "num_points_before_crop": before,
            "num_points_after_crop": len(merged.points),
        }
        metadata["frame"] = "layout_bed"
        print("Skipping 4-marker canonicalization (--no-canonicalize)")
    else:
        merged, canonical_meta = canonicalize_merged_cloud(
            merged,
            pose_dir,
            ceiling_bgr=ceiling_bgr,
            ceiling_xyz=ceiling_xyz,
            ceiling_transform=ceiling_transform,
            dictionary=args.aruco_dictionary,
            detect_top_markers=args.detect_top_markers,
            x_limits=(args.bed_x_min, args.bed_x_max),
            y_limits=(args.bed_y_min, args.bed_y_max),
        )
        metadata["canonical"] = canonical_meta
        metadata["frame"] = "canonical_bed"
    merged_path = pose_dir / "blanket_pcd.pcd"
    if not o3d.io.write_point_cloud(str(merged_path), merged):
        raise RuntimeError(f"Failed to write {merged_path}")
    metadata_path = pose_dir / "pcd_capture_metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    try:
        from artifact_contract import build_capture_contract, write_capture_contract
        from sam_blanket import resolve_sam2_weights, resolve_sam_device

        sam_w = resolve_sam2_weights() if args.mask_backend == "sam2" else None
        sam_d = resolve_sam_device() if args.mask_backend == "sam2" else None
        contract = build_capture_contract(
            pose_dir,
            metadata=metadata,
            sam_weights=sam_w,
            sam_device=sam_d,
        )
        contract_path = write_capture_contract(pose_dir, contract)
        print(f"Saved capture contract: {contract_path}")
    except Exception as exc:  # noqa: BLE001
        if not getattr(args, "allow_image_frame", False):
            raise
        print(f"WARN: capture contract not written: {exc}")

    print(f"Saved merged bed-frame PCD: {merged_path}")
    print(f"Saved capture metadata: {metadata_path}")
    for role, pcd in pcds.items():
        if len(pcd.points) < 500:
            print(f"WARN: {role} has only {len(pcd.points)} points")

    if args.show:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        o3d.visualization.draw_geometries(
            [merged, axis], window_name="blanket_pcd (bed frame)"
        )


if __name__ == "__main__":
    main()
