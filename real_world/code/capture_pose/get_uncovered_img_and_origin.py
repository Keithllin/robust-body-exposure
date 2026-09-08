#!/usr/bin/env python3
"""Capture uncovered RGB + sim_origin for run_trial pose capture.

This is a thin wrapper around the validated
``calibration/calibrate_zed_markers.py`` path (multi-frame ArUco solve,
extrinsics exposure/gain/serial).  Trial capture uses ``--no-save-extrinsics``
so it does not rewrite ``zed_extrinsics.json``.

Run with the ``robe-zed`` environment (usually via capture_human_pose.py):

  conda run -n robe-zed python code/capture_pose/get_uncovered_img_and_origin.py \
      --save-dir /path/to/pose_dir
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

CODE_DIR = Path(__file__).resolve().parents[1]
REAL_WORLD_DIR = CODE_DIR.parent
CALIBRATE_SCRIPT = REAL_WORLD_DIR / "calibration" / "calibrate_zed_markers.py"
DEFAULT_EXTRINSICS = REAL_WORLD_DIR / "calibration" / "zed_extrinsics.json"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from conda_python import drop_foreign_site_packages  # noqa: E402

drop_foreign_site_packages()

import cv2  # noqa: E402

from zed_util import (  # noqa: E402
    configure_exposure_gain,
    grab_rgb_bgr,
    open_ceiling_zed,
    warm_up,
)


def _ceiling_from_extrinsics(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    cameras = payload.get("cameras") or {}
    entry = cameras.get("ceiling") or {}
    return entry if isinstance(entry, dict) else {}


def _capture_rgb_only(
    save_dir: Path,
    *,
    serial: int,
    exposure,
    gain,
    warmup: int,
) -> None:
    """Fallback when ArUco is skipped: one uncovered RGB with calib settings."""

    cam = open_ceiling_zed(serial=serial)
    try:
        warm_up(cam, warmup)
        if exposure is not None or gain is not None:
            settings = configure_exposure_gain(
                cam, exposure=exposure, gain=gain, lock=True
            )
            print(
                f"Camera settings: locked exposure={settings['exposure']} "
                f"gain={settings['gain']}"
            )
            warm_up(cam, 5)
        bgr = None
        for _ in range(20):
            bgr = grab_rgb_bgr(cam)
            if bgr is not None:
                break
        if bgr is None:
            raise RuntimeError("Failed to grab uncovered RGB from ZED")
    finally:
        cam.close()

    out_path = save_dir / "uncovered_rgb.png"
    if not cv2.imwrite(str(out_path), bgr):
        raise RuntimeError(f"Failed to write {out_path}")
    print(f"saved: {out_path}")
    print("Skipping ArUco / sim_origin (--skip-aruco).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=str, default="TEST")
    parser.add_argument(
        "--serial",
        type=int,
        default=0,
        help="Ceiling ZED serial; 0 uses zed_extrinsics.json",
    )
    parser.add_argument(
        "--extrinsics",
        type=str,
        default=str(DEFAULT_EXTRINSICS),
        help="Passed through to calibrate_zed_markers --output",
    )
    parser.add_argument("--exposure", type=int, default=None)
    parser.add_argument("--gain", type=int, default=None)
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Frames for calibrate_zed_markers (default 100)",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument(
        "--max-rms-px",
        type=float,
        default=10.0,
        help="Passed as --max-reprojection-rms",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the same live ArUco preview as calibrate_zed_markers",
    )
    parser.add_argument(
        "--skip-aruco",
        action="store_true",
        help="Only save uncovered_rgb.png; skip sim_origin_data.pkl",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    extrinsics = Path(args.extrinsics)
    ceiling = _ceiling_from_extrinsics(extrinsics)
    serial = int(args.serial) or int(ceiling.get("serial") or 0)
    stored = ceiling.get("camera_settings") or {}
    exposure = (
        args.exposure if args.exposure is not None else stored.get("exposure")
    )
    gain = args.gain if args.gain is not None else stored.get("gain")

    if args.skip_aruco:
        _capture_rgb_only(
            save_dir,
            serial=serial,
            exposure=exposure,
            gain=gain,
            warmup=args.warmup,
        )
        return

    if not CALIBRATE_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing validated calibrator: {CALIBRATE_SCRIPT}")

    command = [
        sys.executable,
        str(CALIBRATE_SCRIPT),
        "--role",
        "ceiling",
        "--serial",
        str(serial),
        "--output",
        str(extrinsics),
        "--sim-origin-dir",
        str(save_dir),
        "--no-save-extrinsics",
        "--frames",
        str(args.frames),
        "--warmup",
        str(args.warmup),
        "--max-reprojection-rms",
        str(args.max_rms_px),
        "--lock-exposure-gain",
    ]
    if exposure is not None:
        command.extend(["--exposure", str(int(exposure))])
    if gain is not None:
        command.extend(["--gain", str(int(gain))])
    if args.preview:
        command.append("--preview")

    print("Using validated calibrator for uncovered RGB + sim_origin:")
    print(">>", " ".join(command))
    subprocess.check_call(command)

    required = (
        save_dir / "sim_origin_data.pkl",
        save_dir / "uncovered_rgb.png",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "calibrate_zed_markers finished but missing: " + ", ".join(missing)
        )


if __name__ == "__main__":
    main()
