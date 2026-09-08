#!/usr/bin/env python3
"""Calibrate one ZED to the fixed bed frame using ID-aware ArUco markers.

Examples:

  # First validate that marker_layout.json has been filled:
  conda run -n robe-zed python calibration/calibrate_zed_markers.py \
      --role ceiling --check-layout

  # Ceiling sees IDs 0,1,2,3:
  conda run -n robe-zed python calibration/calibrate_zed_markers.py \
      --role ceiling --serial <SERIAL> --sim-origin-dir <POSE_DIR>

  # A side camera sees the fixed side markers 10,11,12,13:
  conda run -n robe-zed python calibration/calibrate_zed_markers.py \
      --role side_left --serial <SERIAL>

The layout file contains the measured 3D center and in-plane axes for every
marker.  Do not infer marker meaning from OpenCV's detection order. The
saved production transform uses ZED RIGHT_HANDED_Y_UP coordinates; the raw
OpenCV image-frame pose is retained for diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from conda_python import drop_foreign_site_packages  # noqa: E402

drop_foreign_site_packages()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from marker_utils import (  # noqa: E402
    DEFAULT_LAYOUT_PATH,
    ROLE_MARKERS,
    MarkerLayoutError,
    build_sim_origin_data,
    bed_from_image_camera_to_bed_from_zed_camera,
    collect_correspondences,
    detect_markers,
    draw_detections,
    draw_reprojection_overlay,
    invert_transform,
    load_layout,
    project_marker_corners,
    solve_bed_to_camera,
)
from zed_util import (  # noqa: E402
    configure_exposure_gain,
    set_auto_exposure_gain,
    get_left_intrinsics,
    grab_rgb_bgr,
    open_zed,
    resolution_from_name,
    warm_up,
)


DEFAULT_EXTRINSICS_PATH = Path(__file__).resolve().parent / "zed_extrinsics.json"


def parse_ids(value: str | None, role: str) -> tuple[int, ...]:
    if value:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(ROLE_MARKERS[role])


def load_extrinsics(path: Path) -> dict:
    if not path.exists():
        return {
            "version": 1,
            "frame": "bed",
            "coordinate_system": "IMAGE",
            "cameras": {},
        }
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_extrinsics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def put_pose_overlay(
    image: np.ndarray,
    pose: dict,
    expected_ids: tuple[int, ...],
) -> np.ndarray:
    output = image.copy()
    text = (
        f"IDs={','.join(map(str, expected_ids))} "
        f"RMS={pose['reprojection_rms_px']:.2f}px "
        f"max={pose['reprojection_max_px']:.2f}px"
    )
    cv2.putText(
        output,
        text,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        text,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return output


def marker_reprojection_errors(
    detections: dict,
    specs: dict,
    marker_size_m: float,
    pose: dict,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_ids: tuple[int, ...],
) -> dict:
    projected = project_marker_corners(
        specs, marker_size_m, pose, camera_matrix, distortion, marker_ids
    )
    errors = {}
    for marker_id in marker_ids:
        residual_vectors = projected[marker_id] - np.asarray(
            detections[marker_id], dtype=np.float64
        )
        residual = np.linalg.norm(residual_vectors, axis=1)
        mean_vector = residual_vectors.mean(axis=0)
        errors[str(marker_id)] = {
            "rms_px": float(np.sqrt(np.mean(residual**2))),
            "max_px": float(np.max(residual)),
            "mean_dx_px": float(mean_vector[0]),
            "mean_dy_px": float(mean_vector[1]),
        }
    return errors


def build_residual_overlay(
    image: np.ndarray,
    detections: dict,
    specs: dict,
    marker_size_m: float,
    pose: dict,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_ids: tuple[int, ...],
) -> np.ndarray:
    projected = project_marker_corners(
        specs, marker_size_m, pose, camera_matrix, distortion, marker_ids
    )
    overlay = draw_reprojection_overlay(
        image, detections, projected, marker_ids=marker_ids, arrow_scale=2.0
    )
    return put_pose_overlay(overlay, pose, marker_ids)


def run_role_calibration(
    camera,
    *,
    role: str,
    output_path: Path,
    layout_path: Path,
    expected_ids: tuple[int, ...],
    min_markers: int,
    frames: int,
    exposure: int | None,
    gain: int | None,
    lock_exposure_gain: bool = True,
    preview: bool = False,
    warmup: int = 30,
    max_reprojection_rms: float = 10.0,
    close_camera: bool = False,
    resolution: str = "HD720",
    save_extrinsics_file: bool = True,
) -> dict:
    """PnP one already-open ZED into ``output_path``. Does not open USB."""

    layout, specs = load_layout(str(layout_path), required_ids=expected_ids)
    marker_size_m = float(layout["marker_size_m"])
    output_path = Path(output_path)
    existing = load_extrinsics(output_path)
    camera_info = camera.get_camera_information()
    actual_serial = int(camera_info.serial_number)
    camera_matrix, distortion = get_left_intrinsics(camera)
    best = None
    best_image = None
    best_raw_image = None
    seen_ids = set()
    preview_name = f"ArUco calibration: {role}"
    if preview:
        cv2.namedWindow(preview_name, cv2.WINDOW_NORMAL)

    try:
        if exposure is not None or gain is not None:
            camera_settings = configure_exposure_gain(
                camera, exposure=exposure, gain=gain, lock=True
            )
            print(
                "Camera settings: reused recorded marker-calib "
                f"exposure={camera_settings['exposure']} "
                f"gain={camera_settings['gain']}",
                flush=True,
            )
        else:
            if lock_exposure_gain:
                set_auto_exposure_gain(camera, True)
            warm_up(camera, warmup)
            camera_settings = configure_exposure_gain(
                camera, exposure=None, gain=None, lock=lock_exposure_gain
            )
            print(
                "Camera settings: "
                f"locked={camera_settings['locked']} "
                f"auto={camera_settings['auto_exposure']} "
                f"exposure={camera_settings['exposure']} "
                f"gain={camera_settings['gain']}",
                flush=True,
            )
        if camera_settings["locked"]:
            warm_up(camera, 5)
        print(
            f"Sampling {frames} frames for {role} "
            f"(serial={actual_serial}, need>={min_markers} of {expected_ids})...",
            flush=True,
        )
        for frame_idx in range(frames):
            if frame_idx == 0 or frame_idx + 1 == frames or (frame_idx + 1) % 50 == 0:
                print(f"  {role} frame {frame_idx + 1}/{frames}", flush=True)
            image = grab_rgb_bgr(camera)
            if image is None:
                continue
            if preview and frame_idx == 0:
                h, w = image.shape[:2]
                cv2.resizeWindow(preview_name, w, h)
            detections = detect_markers(image, layout["dictionary"])
            seen_ids.update(detections)
            visible = set(detections).intersection(expected_ids)
            if len(visible) < min_markers:
                if preview:
                    diagnostic = draw_detections(image, detections, expected_ids)
                    cv2.imshow(preview_name, diagnostic)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                continue
            object_points, image_points, used_ids = collect_correspondences(
                detections, specs, marker_size_m, required_ids=expected_ids
            )
            try:
                pose = solve_bed_to_camera(
                    object_points, image_points, camera_matrix, distortion
                )
            except (MarkerLayoutError, cv2.error):
                continue
            if pose["reprojection_rms_px"] > max_reprojection_rms:
                continue
            t_bed_cam = np.asarray(pose["T_bed_camera"], dtype=np.float64)
            cam_bed = t_bed_cam[:3, 3]
            plane_point = object_points.mean(axis=0)
            centered = object_points - plane_point
            normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
            if set(used_ids) <= {10, 13}:
                if normal[0] > 0:
                    normal = -normal
                if float(normal @ (cam_bed - plane_point)) <= 0.0:
                    continue
            elif set(used_ids) <= {11, 12}:
                if normal[0] < 0:
                    normal = -normal
                if float(normal @ (cam_bed - plane_point)) <= 0.0:
                    continue
            if best is None or pose["reprojection_rms_px"] < best["reprojection_rms_px"]:
                best = {
                    **pose,
                    "frame_index": frame_idx,
                    "visible_ids": list(used_ids),
                    "detected_corners_px": {
                        str(marker_id): detections[marker_id].tolist()
                        for marker_id in used_ids
                    },
                    "marker_reprojection": marker_reprojection_errors(
                        detections,
                        specs,
                        marker_size_m,
                        pose,
                        camera_matrix,
                        distortion,
                        used_ids,
                    ),
                }
                best_image = build_residual_overlay(
                    image,
                    detections,
                    specs,
                    marker_size_m,
                    pose,
                    camera_matrix,
                    distortion,
                    used_ids,
                )
                best_raw_image = image.copy()
            if preview:
                diagnostic = build_residual_overlay(
                    image,
                    detections,
                    specs,
                    marker_size_m,
                    pose,
                    camera_matrix,
                    distortion,
                    used_ids,
                )
                cv2.imshow(preview_name, diagnostic)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        if preview:
            cv2.destroyWindow(preview_name)
        if close_camera:
            camera.close()

    if best is None or best_image is None:
        raise RuntimeError(
            f"Could not solve {role}: saw IDs {sorted(seen_ids)}, "
            f"needed at least {min_markers} of {expected_ids}"
        )

    entry = None
    if save_extrinsics_file:
        payload = existing
        payload["marker_layout"] = str(Path(layout_path).resolve())
        payload["marker_size_m"] = marker_size_m
        payload["dictionary"] = layout["dictionary"]
        payload["coordinate_system"] = "RIGHT_HANDED_Y_UP"
        t_bed_camera_image = np.asarray(best["T_bed_camera"], dtype=np.float64)
        t_bed_camera_zed = bed_from_image_camera_to_bed_from_zed_camera(
            t_bed_camera_image
        )
        t_camera_bed_zed = invert_transform(t_bed_camera_zed)
        entry = {
            "serial": actual_serial,
            "role": role,
            "resolution": str(resolution).upper(),
            "camera_settings": camera_settings,
            "T_camera_bed": t_camera_bed_zed.tolist(),
            "T_bed_camera": t_bed_camera_zed.tolist(),
            "T_camera_bed_image": best["T_camera_bed"],
            "T_bed_camera_image": best["T_bed_camera"],
            "visible_ids": best["visible_ids"],
            "detected_corners_px": best["detected_corners_px"],
            "marker_reprojection": best["marker_reprojection"],
            "reprojection_rms_px": best["reprojection_rms_px"],
            "reprojection_max_px": best["reprojection_max_px"],
            "pnp_method": best.get("pnp_method"),
            "num_points": best["num_points"],
            "camera_matrix": camera_matrix.tolist(),
            "distortion": distortion.tolist(),
            "image_size_wh": [
                int(best_raw_image.shape[1]),
                int(best_raw_image.shape[0]),
            ],
            "frame_index": best["frame_index"],
            "captured_at_unix": time.time(),
            "pnp_ids": list(expected_ids),
        }
        payload.setdefault("cameras", {})[role] = entry
        save_extrinsics(output_path, payload)
        diagnostic_path = output_path.parent / f"aruco_{role}_diagnostic.png"
        residual_path = output_path.parent / f"aruco_{role}_residuals.png"
        cv2.imwrite(str(diagnostic_path), best_image)
        cv2.imwrite(str(residual_path), best_image)
        print(
            f"Saved {role} serial={actual_serial}; "
            f"IDs={best['visible_ids']}; "
            f"RMS={best['reprojection_rms_px']:.3f}px -> {output_path}",
            flush=True,
        )
    return {
        "best": best,
        "best_image": best_image,
        "best_raw_image": best_raw_image,
        "camera_settings": camera_settings,
        "actual_serial": actual_serial,
        "camera_matrix": camera_matrix,
        "distortion": distortion,
        "entry": entry,
        "layout": layout,
        "specs": specs,
        "marker_size_m": marker_size_m,
        "seen_ids": seen_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=tuple(ROLE_MARKERS), required=True)
    parser.add_argument("--serial", type=int, default=0)
    parser.add_argument(
        "--resolution",
        type=str,
        default="HD720",
        help="ZED image resolution; use the same mode as ZED SDK preview",
    )
    parser.add_argument("--layout", type=str, default=str(DEFAULT_LAYOUT_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_EXTRINSICS_PATH))
    parser.add_argument(
        "--sim-origin-dir",
        type=str,
        default=None,
        help="For ceiling: also write sim_origin_data.pkl / uncovered_rgb.png here",
    )
    parser.add_argument(
        "--no-save-extrinsics",
        action="store_true",
        help="Solve ArUco / sim_origin without rewriting zed_extrinsics.json "
        "(use during run_trial pose capture)",
    )
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument(
        "--exposure",
        type=int,
        default=None,
        help="Manual ZED exposure 0-100; disables auto exposure. "
        "Default: locked value already stored for this role in --output",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=None,
        help="Manual ZED gain 0-100; disables auto gain. "
        "Default: locked value already stored for this role in --output",
    )
    parser.add_argument(
        "--lock-exposure-gain",
        action="store_true",
        help="Warm auto exposure, then freeze the selected/current values",
    )
    parser.add_argument(
        "--min-markers",
        type=int,
        default=None,
        help="Minimum visible IDs (default: 4 ceiling, 3 side; 2 is "
        "allowed for an explicit planar marker pair)",
    )
    parser.add_argument(
        "--max-reprojection-rms",
        type=float,
        default=10.0,
        help="Reject frames above this reprojection RMS in pixels",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated override, e.g. 10,11,12,13",
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--check-layout",
        action="store_true",
        help="Validate layout and exit without opening a camera",
    )
    args = parser.parse_args()

    expected_ids = parse_ids(args.ids, args.role)
    min_markers = args.min_markers
    if min_markers is None:
        min_markers = 4 if args.role == "ceiling" else 3
    if min_markers < 2:
        raise SystemExit("--min-markers must be at least 2")
    if min_markers == 2 and not args.ids:
        raise SystemExit(
            "--min-markers 2 requires an explicit --ids marker pair"
        )

    try:
        layout, specs = load_layout(args.layout, required_ids=expected_ids)
    except MarkerLayoutError as exc:
        raise SystemExit(f"Marker layout is not ready: {exc}") from exc

    marker_size_m = float(layout["marker_size_m"])
    if args.check_layout:
        print(
            f"Layout OK: {args.layout}; IDs={expected_ids}; "
            f"marker_size={marker_size_m:.3f}m"
        )
        return

    output_path = Path(args.output)
    existing = load_extrinsics(output_path)
    role_entry = (existing.get("cameras") or {}).get(args.role) or {}
    stored_settings = role_entry.get("camera_settings") or {}
    serial = int(args.serial) or int(role_entry.get("serial") or 0)
    exposure = (
        args.exposure if args.exposure is not None else stored_settings.get("exposure")
    )
    gain = args.gain if args.gain is not None else stored_settings.get("gain")
    lock_exposure_gain = bool(
        args.lock_exposure_gain
        or args.exposure is not None
        or args.gain is not None
        or stored_settings.get("locked")
    )
    resolution = role_entry.get("resolution", args.resolution)

    camera = open_zed(
        serial=serial,
        resolution=resolution_from_name(resolution),
    )
    camera_info = camera.get_camera_information()
    actual_serial = int(camera_info.serial_number)
    camera_matrix, distortion = get_left_intrinsics(camera)
    best = None
    best_image = None
    best_raw_image = None
    seen_ids = set()
    preview_name = f"ArUco calibration: {args.role}"
    if args.preview:
        cv2.namedWindow(preview_name, cv2.WINDOW_NORMAL)

    try:
        if exposure is not None or gain is not None:
            camera_settings = configure_exposure_gain(
                camera,
                exposure=exposure,
                gain=gain,
                lock=True,
            )
            print(
                "Camera settings: reused recorded marker-calib "
                f"exposure={camera_settings['exposure']} "
                f"gain={camera_settings['gain']}"
            )
        else:
            # No stored pair: auto-settle, then optionally freeze.
            if lock_exposure_gain:
                set_auto_exposure_gain(camera, True)
            warm_up(camera, args.warmup)
            camera_settings = configure_exposure_gain(
                camera,
                exposure=None,
                gain=None,
                lock=lock_exposure_gain,
            )
            print(
                "Camera settings: "
                f"locked={camera_settings['locked']} "
                f"auto={camera_settings['auto_exposure']} "
                f"exposure={camera_settings['exposure']} "
                f"gain={camera_settings['gain']}"
            )
        if camera_settings["locked"]:
            warm_up(camera, 5)
        print(
            f"Sampling {args.frames} frames for {args.role} "
            f"(serial={actual_serial}, need>={min_markers} of {expected_ids})..."
        )
        for frame_idx in range(args.frames):
            image = grab_rgb_bgr(camera)
            if image is None:
                continue
            if args.preview and frame_idx == 0:
                # Match SDK-style large preview. SDK SIDE_BY_SIDE is 2x wider
                # because it shows LEFT|RIGHT; this window is LEFT only.
                h, w = image.shape[:2]
                cv2.resizeWindow(preview_name, w, h)
                print(
                    f"Preview: LEFT eye only {w}x{h}. "
                    f"ZED SDK SIDE_BY_SIDE is typically {2 * w}x{h} (LEFT|RIGHT)."
                )
            detections = detect_markers(image, layout["dictionary"])
            seen_ids.update(detections)
            visible = set(detections).intersection(expected_ids)
            if len(visible) < min_markers:
                if args.preview:
                    diagnostic = draw_detections(image, detections, expected_ids)
                    lines = [
                        f"SN={actual_serial} LEFT {image.shape[1]}x{image.shape[0]}",
                        f"visible={sorted(visible)} need={min_markers} saw={sorted(seen_ids)}",
                        "SDK SBS looks wider because it shows LEFT|RIGHT",
                    ]
                    y = 28
                    for line in lines:
                        cv2.putText(
                            diagnostic,
                            line,
                            (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 0),
                            3,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            diagnostic,
                            line,
                            (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        y += 28
                    cv2.imshow(preview_name, diagnostic)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                continue

            object_points, image_points, used_ids = collect_correspondences(
                detections,
                specs,
                marker_size_m,
                required_ids=expected_ids,
            )
            try:
                pose = solve_bed_to_camera(
                    object_points, image_points, camera_matrix, distortion
                )
            except (MarkerLayoutError, cv2.error):
                continue

            if pose["reprojection_rms_px"] > args.max_reprojection_rms:
                continue

            # Planar side markers admit a same-RMS flip through the bed.
            # Reject poses whose camera origin is not on the outward side of
            # the marker plane (left face outward -X; right face outward +X).
            t_bed_cam = np.asarray(pose["T_bed_camera"], dtype=np.float64)
            cam_bed = t_bed_cam[:3, 3]
            plane_point = object_points.mean(axis=0)
            centered = object_points - plane_point
            normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
            # Outward for left IDs points -X; for right IDs +X.
            if set(used_ids) <= {10, 13}:
                if normal[0] > 0:
                    normal = -normal
                if float(normal @ (cam_bed - plane_point)) <= 0.0:
                    continue
            elif set(used_ids) <= {11, 12}:
                if normal[0] < 0:
                    normal = -normal
                if float(normal @ (cam_bed - plane_point)) <= 0.0:
                    continue

            if best is None or pose["reprojection_rms_px"] < best["reprojection_rms_px"]:
                best = {
                    **pose,
                    "frame_index": frame_idx,
                    "visible_ids": list(used_ids),
                    "detected_corners_px": {
                        str(marker_id): detections[marker_id].tolist()
                        for marker_id in used_ids
                    },
                    "marker_reprojection": marker_reprojection_errors(
                        detections,
                        specs,
                        marker_size_m,
                        pose,
                        camera_matrix,
                        distortion,
                        used_ids,
                    ),
                }
                best_image = build_residual_overlay(
                    image,
                    detections,
                    specs,
                    marker_size_m,
                    pose,
                    camera_matrix,
                    distortion,
                    used_ids,
                )
                best_raw_image = image.copy()

            if args.preview:
                diagnostic = build_residual_overlay(
                    image,
                    detections,
                    specs,
                    marker_size_m,
                    pose,
                    camera_matrix,
                    distortion,
                    used_ids,
                )
                cv2.imshow(preview_name, diagnostic)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        camera.close()
        if args.preview:
            cv2.destroyAllWindows()

    if best is None or best_image is None:
        raise SystemExit(
            f"Could not solve {args.role}: saw IDs {sorted(seen_ids)}, "
            f"needed at least {min_markers} of {expected_ids}"
        )

    if not args.no_save_extrinsics:
        payload = existing
        payload["marker_layout"] = str(Path(args.layout).resolve())
        payload["marker_size_m"] = marker_size_m
        payload["dictionary"] = layout["dictionary"]
        payload["coordinate_system"] = "RIGHT_HANDED_Y_UP"
        t_bed_camera_image = np.asarray(best["T_bed_camera"], dtype=np.float64)
        t_bed_camera_zed = bed_from_image_camera_to_bed_from_zed_camera(
            t_bed_camera_image
        )
        t_camera_bed_zed = invert_transform(t_bed_camera_zed)
        payload.setdefault("cameras", {})[args.role] = {
            "serial": actual_serial,
            "role": args.role,
            "resolution": str(resolution).upper(),
            "camera_settings": camera_settings,
            # T_bed_camera is the production transform for ZED XYZ in
            # RIGHT_HANDED_Y_UP coordinates.
            "T_camera_bed": t_camera_bed_zed.tolist(),
            "T_bed_camera": t_bed_camera_zed.tolist(),
            # Keep the raw OpenCV pose for reprojection/diagnostics.
            "T_camera_bed_image": best["T_camera_bed"],
            "T_bed_camera_image": best["T_bed_camera"],
            "visible_ids": best["visible_ids"],
            "detected_corners_px": best["detected_corners_px"],
            "marker_reprojection": best["marker_reprojection"],
            "reprojection_rms_px": best["reprojection_rms_px"],
            "reprojection_max_px": best["reprojection_max_px"],
            "pnp_method": best.get("pnp_method"),
            "num_points": best["num_points"],
            "camera_matrix": camera_matrix.tolist(),
            "distortion": distortion.tolist(),
            "image_size_wh": [
                int(best_raw_image.shape[1]),
                int(best_raw_image.shape[0]),
            ],
            "frame_index": best["frame_index"],
            "captured_at_unix": time.time(),
        }
        save_extrinsics(output_path, payload)
        diagnostic_path = output_path.parent / f"aruco_{args.role}_diagnostic.png"
        residual_path = output_path.parent / f"aruco_{args.role}_residuals.png"
        cv2.imwrite(str(diagnostic_path), best_image)
        cv2.imwrite(str(residual_path), best_image)
        print(
            f"Saved {args.role} serial={actual_serial}; "
            f"IDs={best['visible_ids']}; "
            f"RMS={best['reprojection_rms_px']:.3f}px -> {output_path}"
        )
        print(f"Diagnostic image: {diagnostic_path}")
        print(f"Residual overlay: {residual_path}")
    else:
        print(
            f"Solved {args.role} serial={actual_serial}; "
            f"IDs={best['visible_ids']}; "
            f"RMS={best['reprojection_rms_px']:.3f}px "
            f"(skipped rewriting {output_path})"
        )

    for marker_id, stats in best["marker_reprojection"].items():
        print(
            f"  ID {marker_id}: RMS={stats['rms_px']:.2f}px "
            f"mean_d=({stats['mean_dx_px']:+.1f}, {stats['mean_dy_px']:+.1f})px"
        )

    if args.role == "ceiling" and args.sim_origin_dir:
        # Re-detect the selected diagnostic frame so the compatibility pickle
        # retains the actual marker centers used by the saved pose.
        detections = detect_markers(best_raw_image, layout["dictionary"])
        sim_data = build_sim_origin_data(
            best_raw_image,
            camera_matrix,
            distortion,
            detections,
            specs,
            best,
            marker_size_m,
        )
        sim_dir = Path(args.sim_origin_dir)
        sim_dir.mkdir(parents=True, exist_ok=True)
        with (sim_dir / "sim_origin_data.pkl").open("wb") as handle:
            pickle.dump(sim_data, handle)
        rgb_path = sim_dir / "uncovered_rgb.png"
        viz_path = sim_dir / "uncovered_aruco_viz.png"
        if not cv2.imwrite(str(rgb_path), best_raw_image):
            raise SystemExit(f"Failed to write {rgb_path}")
        if not cv2.imwrite(str(viz_path), best_image):
            raise SystemExit(f"Failed to write {viz_path}")
        print(f"Saved {sim_dir / 'sim_origin_data.pkl'}")
        print(f"Saved {rgb_path}")
        print(f"Saved {viz_path}")
        from canonical_bed import write_canonical_from_sim_origin

        wrote = write_canonical_from_sim_origin(sim_dir)
        print(f"Saved {wrote}")


if __name__ == "__main__":
    main()
