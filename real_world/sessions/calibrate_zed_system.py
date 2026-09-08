#!/usr/bin/env python3
"""Three-ZED layout calibration: PnP, PCD config, freeze.

Plane-residual fusion lives in run_trial --validate-fusion
(validate_zed_fusion.py), not here. Does not touch Stretch, ArUco 136,
or trial canonical_bed_frame.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
CALIB_DIR = Path(__file__).resolve().parents[1] / "calibration"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(CALIB_DIR))

from conda_python import drop_foreign_site_packages, zed_python  # noqa: E402

drop_foreign_site_packages()

from session_paths import (  # noqa: E402
    ACCEPT_XY_M,
    DEFAULT_LAYOUT,
    atomic_write_json,
    freeze_zed,
    print_session_resolved,
    resolve_session_dir_once,
    session_paths,
    sha256_file,
    write_pcd_config,
)

ROLES = ("ceiling", "side_left", "side_right")
# Marker PnP is per-camera (proven CLI). Filter/PCD stays 55/40 for all roles.
PNP_PRESETS = {
    "ceiling": {
        "serial": 0,
        "exposure": 40,
        "gain": 25,
        "ids": (0, 1, 2, 3),
        "min_markers": 4,
        "frames": 100,
    },
    "side_left": {
        "serial": 0,
        "exposure": 30,
        "gain": 15,
        "ids": (0, 3),
        "min_markers": 2,
        "frames": 100,
    },
    "side_right": {
        "serial": 0,
        "exposure": 35,
        "gain": 20,
        "ids": (1, 2),
        "min_markers": 2,
        "frames": 100,
    },
}
PCD_EXPOSURE = 55
PCD_GAIN = 40


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def _readline_tty(prompt: str) -> str | None:
    """Read a confirm line. ``conda run`` has no stdin unless --no-capture-output."""

    if sys.stdin.isatty():
        try:
            return input(prompt)
        except EOFError:
            return None
    try:
        with open("/dev/tty", encoding="utf-8") as tty_in, open(
            "/dev/tty", "w", encoding="utf-8"
        ) as tty_out:
            tty_out.write(prompt)
            tty_out.flush()
            return tty_in.readline()
    except OSError:
        return None


def _confirm_freeze(*, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    reply = _readline_tty("Freeze ZED system? [y/N] ")
    if reply is None:
        _log("No TTY (conda run ate stdin). Not frozen. Re-run with:")
        _log("  conda run -n robe-zed --no-capture-output python -u \\")
        _log("    real_world/sessions/calibrate_zed_system.py")
        _log("or pass --yes after all three roles are DONE.")
        return False
    return reply.strip().lower() in ("y", "yes")


def _load_cameras(dest: Path) -> dict:
    if not dest.is_file():
        return {}
    try:
        return json.loads(dest.read_text()).get("cameras") or {}
    except json.JSONDecodeError:
        return {}


def _role_complete(dest: Path, role: str) -> dict | None:
    preset = PNP_PRESETS[role]
    entry = _load_cameras(dest).get(role) or {}
    if not entry.get("T_bed_camera"):
        return None
    settings = entry.get("camera_settings") or {}
    if int(settings.get("exposure") or -1) != int(preset["exposure"]):
        return None
    if int(settings.get("gain") or -1) != int(preset["gain"]):
        return None
    if int(entry.get("serial") or 0) != int(preset["serial"]):
        return None
    return entry


def _print_board(dest: Path) -> None:
    _log("PROGRESS")
    for role in ROLES:
        preset = PNP_PRESETS[role]
        entry = _role_complete(dest, role)
        if entry is None:
            _log(
                f"  {role:<12} PENDING  "
                f"PnP exp={preset['exposure']} gain={preset['gain']} "
                f"ids={list(preset['ids'])} frames={preset['frames']}"
            )
            continue
        _log(
            f"  {role:<12} DONE     serial={entry.get('serial')} "
            f"IDs={entry.get('visible_ids')} RMS={entry.get('reprojection_rms_px')} "
            f"PnP {preset['exposure']}/{preset['gain']}"
        )


def _close_cameras(cameras: dict) -> None:
    for role, camera in list(cameras.items()):
        try:
            camera.close()
        except Exception as exc:  # noqa: BLE001
            _log(f"WARN close {role}: {exc}")
        cameras.pop(role, None)


def _open_all_cameras() -> dict:
    """Keep every ZED open for the whole module, like run_trial capture."""

    from zed_util import configure_exposure_gain, open_zed, resolution_from_name

    cameras: dict = {}
    for role in ROLES:
        preset = PNP_PRESETS[role]
        _log(
            f"Opening {role} serial={preset['serial']} "
            f"PnP exposure={preset['exposure']} gain={preset['gain']}"
        )
        try:
            camera = open_zed(
                serial=int(preset["serial"]),
                resolution=resolution_from_name("HD720"),
            )
            configure_exposure_gain(
                camera,
                exposure=int(preset["exposure"]),
                gain=int(preset["gain"]),
                lock=True,
            )
            cameras[role] = camera
            _log(f"  {role} open")
        except Exception as exc:  # noqa: BLE001
            _log(f"  FAIL open {role}: {exc}")
    return cameras


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--yes", action="store_true", help="Freeze without prompt")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--serial-ceiling", type=int, default=None)
    parser.add_argument("--serial-side-left", type=int, default=None)
    parser.add_argument("--serial-side-right", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redo roles that are already in the session",
    )
    parser.add_argument(
        "--pcd-exposure",
        type=int,
        default=PCD_EXPOSURE,
        help="Unified filter/PCD exposure (default 55). Marker PnP uses per-role presets.",
    )
    parser.add_argument(
        "--pcd-gain",
        type=int,
        default=PCD_GAIN,
        help="Unified filter/PCD gain (default 40). Marker PnP uses per-role presets.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show ArUco preview while sampling each pending role",
    )
    args = parser.parse_args()
    if args.serial_ceiling is not None:
        PNP_PRESETS["ceiling"]["serial"] = int(args.serial_ceiling)
    if args.serial_side_left is not None:
        PNP_PRESETS["side_left"]["serial"] = int(args.serial_side_left)
    if args.serial_side_right is not None:
        PNP_PRESETS["side_right"]["serial"] = int(args.serial_side_right)
    session_dir = resolve_session_dir_once(args.session_dir)
    paths = session_paths(session_dir)
    paths.zed_dir.mkdir(parents=True, exist_ok=True)
    print_session_resolved(session_dir, paths.zed_extrinsics)

    if args.tune:
        _log("Run tune_zed_blanket_filter.py per role, then re-run this module.")
        _log(f"  {zed_python()} {CALIB_DIR.parent / 'tune_zed_blanket_filter.py'}")
        return 0

    pcd_exposure = int(args.pcd_exposure)
    pcd_gain = int(args.pcd_gain)
    _log("PNP PRESETS (per camera)  |  PCD/filter unified "
         f"exposure={pcd_exposure} gain={pcd_gain}")
    for role, preset in PNP_PRESETS.items():
        _log(
            f"  {role}: serial={preset['serial']} "
            f"exp={preset['exposure']} gain={preset['gain']} "
            f"ids={list(preset['ids'])} min={preset['min_markers']} "
            f"frames={preset['frames']}"
        )
    _print_board(paths.zed_extrinsics)
    pending = [
        role
        for role in ROLES
        if args.force or _role_complete(paths.zed_extrinsics, role) is None
    ]
    cameras = {}
    try:
        _log("Opening all 3 ZEDs and keeping them open (same as run_trial capture)")
        cameras = _open_all_cameras()
        sys.path.insert(0, str(CALIB_DIR))
        from calibrate_zed_markers import run_role_calibration  # noqa: WPS433

        for role in pending:
            preset = PNP_PRESETS[role]
            if role not in cameras:
                _log(f"FAIL {role}: camera not open")
                continue
            _log(f"--- PnP {role} on already-open camera ---")
            try:
                run_role_calibration(
                    cameras[role],
                    role=role,
                    output_path=paths.zed_extrinsics,
                    layout_path=DEFAULT_LAYOUT,
                    expected_ids=tuple(preset["ids"]),
                    min_markers=int(preset["min_markers"]),
                    frames=int(preset["frames"]),
                    exposure=int(preset["exposure"]),
                    gain=int(preset["gain"]),
                    lock_exposure_gain=True,
                    preview=bool(args.preview),
                    close_camera=False,
                    save_extrinsics_file=True,
                )
            except Exception as exc:  # noqa: BLE001
                _log(f"FAIL {role}: {exc}; session file kept")
            _print_board(paths.zed_extrinsics)
        for role in ROLES:
            if role not in pending:
                _log(f"skip {role} (already in session, camera stayed open)")
    finally:
        _close_cameras(cameras)

    if paths.zed_extrinsics.is_file():
        payload = json.loads(paths.zed_extrinsics.read_text())
        payload["marker_layout_sha256"] = sha256_file(DEFAULT_LAYOUT)
        payload["session_id"] = paths.session_id
        atomic_write_json(paths.zed_extrinsics, payload)
    still = [role for role in ROLES if _role_complete(paths.zed_extrinsics, role) is None]
    if still:
        _log("PARTIAL: kept completed roles in session. Re-run to resume:")
        _log("  conda run -n robe-zed --no-capture-output python -u \\")
        _log("    real_world/sessions/calibrate_zed_system.py")
        _print_board(paths.zed_extrinsics)
        return 2

    write_pcd_config(
        session_dir,
        pcd_exposure=pcd_exposure,
        pcd_gain=pcd_gain,
        support_radius_3d=0.03,
        merge_mode="auto",
    )
    _log("pcd_config written (filter names relative to session/zed/)")
    _log("Fusion plane residual is run_trial --validate-fusion, not this module.")
    _log(f"accept_xy_m (registration, later) = {ACCEPT_XY_M}")
    if not _confirm_freeze(assume_yes=bool(args.yes)):
        _log("not frozen (extrinsics still in session)")
        return 1
    freeze_zed(session_dir)
    _log("ZED SYSTEM FROZEN")
    _log("registration_status = needs_alignment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
