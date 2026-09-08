#!/usr/bin/env python3
"""Robot↔planning registration: ceiling 136 vs Stretch wrist TF.

Session-only. Does not read trial sim_origin_data.pkl or canonical_bed_frame.
Does not re-PnP Stretch, does not change CMA / tool frame.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
CALIB_DIR = Path(__file__).resolve().parents[1] / "calibration"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from conda_python import env_without_foreign_site_packages, zed_python  # noqa: E402
from marker_utils import invert_transform  # noqa: E402
from overlay_action_on_ceiling import bed_xyz_to_pixel  # noqa: E402
from overlay_stretch_tf_ceiling import (  # noqa: E402
    TfNode,
    _T_WRIST_TO_BIG,
    _detect_136,
    _draw,
    _pose_marker_camera,
)
from session_paths import (  # noqa: E402
    ACCEPT_XY_M,
    DEFAULT_LAYOUT,
    atomic_write_json,
    freeze_registration,
    print_session_resolved,
    resolve_session_dir_once,
    session_paths,
    sha256_file,
)
from stretch_localize import translate_t_odom_layout  # noqa: E402


def _grab_ceiling(session_dir: Path) -> None:
    zed = zed_python()
    cmd = [zed, str(CALIB_DIR / "grab_ceiling_now.py")]
    env = env_without_foreign_site_packages(str(Path(zed).resolve().parents[1]))
    env["ROBE_SESSION_DIR"] = str(session_dir)
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=False, env=env)
    if completed.returncode != 0:
        raise SystemExit(f"ceiling grab failed (exit {completed.returncode})")


def _lookup_tf():
    import rclpy  # noqa: WPS433

    rclpy.init()
    node = TfNode()
    end = time.monotonic() + 4.0
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    t_odom_wrist = node.lookup("odom", "link_aruco_top_wrist")
    t_odom_wrist_big = t_odom_wrist @ _T_WRIST_TO_BIG
    node.destroy_node()
    rclpy.shutdown()
    return t_odom_wrist_big


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument(
        "--reuse-image",
        action="store_true",
        help="Use existing session/stretch/ceiling_now_ee.png",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Ceiling RGB to use instead of grabbing (trial initial capture).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Also write the registration payload here (trial wrist_xy_match.json).",
    )
    parser.add_argument(
        "--on-missing-136",
        choices=("fail", "keep"),
        default=None,
        help="fail: abort if ArUco 136 is not in the ceiling image (session SOP). "
        "keep: leave stretch_origin_corrected.json unchanged. "
        "Default is keep when --image is set, otherwise fail.",
    )
    args = parser.parse_args()
    if args.on_missing_136 is None:
        args.on_missing_136 = "keep" if args.image is not None else "fail"
    session_dir = resolve_session_dir_once(args.session_dir)
    paths = session_paths(session_dir)
    print_session_resolved(session_dir, paths.origin_raw)
    if not paths.zed_extrinsics.is_file():
        raise SystemExit(f"missing session ZED extrinsics: {paths.zed_extrinsics}")
    if not paths.origin_raw.is_file() and not paths.origin_json.is_file():
        raise SystemExit(f"missing Stretch origin raw under {paths.stretch_dir}")
    raw_path = paths.origin_raw if paths.origin_raw.is_file() else paths.origin_json

    if args.image is not None:
        image_path = Path(args.image).expanduser().resolve()
        if not image_path.is_file():
            raise SystemExit(f"missing --image {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"could not read --image {image_path}")
        paths.ceiling_now.parent.mkdir(parents=True, exist_ok=True)
        if image_path != paths.ceiling_now.resolve():
            cv2.imwrite(str(paths.ceiling_now), image)
    else:
        if not args.reuse_image or not paths.ceiling_now.is_file():
            _grab_ceiling(session_dir)
        image = cv2.imread(str(paths.ceiling_now), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"could not read {paths.ceiling_now}")

    ext = json.loads(paths.zed_extrinsics.read_text())
    ceil = (ext.get("cameras") or {}).get("ceiling") or {}
    k_zed = np.asarray(ceil["camera_matrix"], dtype=np.float64)
    dist_zed = np.asarray(ceil["distortion"], dtype=np.float64)
    t_bed_cam = np.asarray(ceil["T_bed_camera_image"], dtype=np.float64)
    h, w = image.shape[:2]
    calib_wh = ceil.get("image_size_wh")
    if calib_wh is not None and (int(calib_wh[0]), int(calib_wh[1])) != (w, h):
        sx = w / float(calib_wh[0])
        sy = h / float(calib_wh[1])
        k_zed = k_zed.copy()
        k_zed[0, :] *= sx
        k_zed[1, :] *= sy

    try:
        _det_uv, corners = _detect_136(image)
    except RuntimeError as exc:
        if "136 not detected" not in str(exc) or args.on_missing_136 != "keep":
            raise
        kept = (
            paths.origin_corrected
            if paths.origin_corrected.is_file()
            else raw_path
        )
        print(
            "WARN: ArUco 136 not in ceiling view; "
            f"keeping last alignment {kept}"
        )
        if args.output_json is not None:
            atomic_write_json(
                Path(args.output_json).expanduser().resolve(),
                {
                    "accepted": True,
                    "fallback": True,
                    "fallback_reason": "aruco_136_not_detected",
                    "kept_origin": str(kept),
                    "image": str(paths.ceiling_now),
                    "rotation_corrected": False,
                    "z_corrected": False,
                },
            )
        return 0
    t_cam_marker = _pose_marker_camera(corners, 0.1, k_zed, dist_zed)
    p_ceil = (t_bed_cam @ t_cam_marker)[:3, 3].copy()

    origin = json.loads(raw_path.read_text())
    t_ol = np.asarray(origin["T_odom_layout"], dtype=np.float64).reshape(4, 4)
    t_odom_wrist_big = _lookup_tf()
    p_tf = (invert_transform(t_ol) @ np.append(t_odom_wrist_big[:3, 3], 1.0))[:3]

    before = p_tf - p_ceil
    before_xy = float(np.linalg.norm(before[:2]))
    delta = np.zeros(3, dtype=np.float64)
    delta[:2] = p_ceil[:2] - p_tf[:2]
    t_new = translate_t_odom_layout(t_ol, delta)
    if float(np.linalg.norm(t_new[:3, :3] - t_ol[:3, :3])) > 1e-12:
        raise SystemExit("REJECT: rotation would change")
    p_after = (invert_transform(t_new) @ (t_ol @ np.append(p_tf, 1.0)))[:3]
    after = p_after - p_ceil
    after_xy = float(np.linalg.norm(after[:2]))
    accept = ACCEPT_XY_M
    if paths.session_json.is_file():
        accept = float(
            json.loads(paths.session_json.read_text()).get("accept_xy_m") or ACCEPT_XY_M
        )

    overlay = {
        "image": str(paths.ceiling_now),
        "origin": str(raw_path),
        "projection": "session ceiling T_bed_camera_image (no trial sim_origin)",
        "points_layout_Zdown_m": {
            "ceil_det_136": p_ceil.round(4).tolist(),
            "tf_wrist_big": p_tf.round(4).tolist(),
        },
        "deltas_layout_Zdown_m": {
            "tf_wrist_big_minus_ceil136_3D": {
                "dx": round(float(before[0]), 4),
                "dy": round(float(before[1]), 4),
                "dz": round(float(before[2]), 4),
                "norm_xy": round(before_xy, 4),
            }
        },
    }
    atomic_write_json(paths.overlay_json, overlay)

    canvas = image.copy()
    for key, xyz, color, label in (
        ("ceil", p_ceil, (0, 255, 255), "ceil136"),
        ("tf", p_tf, (0, 140, 255), "wrist_big"),
        ("after", p_after, (0, 255, 0), "after"),
    ):
        uv = bed_xyz_to_pixel(
            xyz.reshape(1, 3),
            bed_from_camera=t_bed_cam,
            camera_matrix=k_zed,
            distortion=dist_zed,
        )[0]
        _draw(canvas, uv, color, label)
    cv2.imwrite(str(paths.overlay_corrected_png), canvas)

    registration = {
        "reference": "aruco_136",
        "correction_xyz_m": [float(delta[0]), float(delta[1]), 0.0],
        "before_xy_error_m": before_xy,
        "after_xy_error_m": after_xy,
        "rotation_corrected": False,
        "z_corrected": False,
        "accepted": after_xy <= accept,
        "stretch_origin_raw_sha256": sha256_file(raw_path),
        "zed_extrinsics_sha256": sha256_file(paths.zed_extrinsics),
        "marker_layout_sha256": sha256_file(DEFAULT_LAYOUT),
    }
    print(json.dumps(registration, indent=2))
    if after_xy > accept:
        print("REJECT")
        print(f"XY residual {after_xy:.4f} m > accept {accept:.4f} m")
        atomic_write_json(paths.registration_json, registration)
        if args.output_json is not None:
            out = Path(args.output_json).expanduser().resolve()
            atomic_write_json(out, dict(registration, accepted=False))
        return 2

    payload = dict(origin)
    payload["T_odom_layout"] = t_new.tolist()
    payload["layout_origin_odom_m"] = t_new[:3, 3].tolist()
    payload["xy_registration"] = {
        "method": "ceiling_136_vs_tf_wrist_big_translation_xy_only",
        "source_origin": str(raw_path),
        "delta_layout_m": delta.tolist(),
        "before_tf_minus_ceil_m": before.round(4).tolist(),
        "predicted_after_tf_minus_ceil_m": after.round(6).tolist(),
        "rotation_unchanged": True,
        "z_corrected": False,
    }
    atomic_write_json(paths.origin_corrected, payload)
    freeze_registration(
        session_dir,
        xy_registration_m=delta[:2].tolist(),
        residual_xy_m=after_xy,
        registration=registration,
    )
    if args.output_json is not None:
        out = Path(args.output_json).expanduser().resolve()
        payload = dict(registration)
        payload["image"] = str(paths.ceiling_now)
        payload["overlay_png"] = str(paths.overlay_corrected_png)
        atomic_write_json(out, payload)
        print(f"wrote {out}")
    print("SESSION REGISTRATION FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
