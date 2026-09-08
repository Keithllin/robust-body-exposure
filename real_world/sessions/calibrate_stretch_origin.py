#!/usr/bin/env python3
"""Stretch origin only: D435i bed-post PnP → T_odom_layout.

Does not look at ceiling 136, does not apply XY correction.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from session_paths import (  # noqa: E402
    print_session_resolved,
    receive_stretch_raw,
    resolve_session_dir_once,
    session_paths,
)


def _on_stretch() -> bool:
    host = os.uname().nodename.lower()
    return bool(os.environ.get("HELLO_FLEET_ID")) or host.startswith("stretch")


def _stretch_ssh() -> str:
    host = os.environ.get("ROBE_STRETCH_SSH", "").strip()
    if not host:
        raise RuntimeError(
            "Set ROBE_STRETCH_SSH (user@host) before pulling the Stretch origin."
        )
    return host


def _pull_raw(paths) -> None:
    """Stretch writes /tmp/robe only. RCHI installs session raw (never corrected)."""

    host = os.environ.get("ROBE_STRETCH_SCP", _stretch_ssh()).strip()
    paths.stretch_dir.mkdir(parents=True, exist_ok=True)
    tmp = paths.stretch_dir / "stretch_origin.json.tmp"
    scp = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
    origin = subprocess.run(
        scp + [f"{host}:/tmp/robe/stretch_origin.json", str(tmp)],
        check=False,
    )
    if origin.returncode != 0:
        raise RuntimeError(
            f"scp {host}:/tmp/robe/stretch_origin.json failed. "
            "File still on Stretch? Re-run sample, or: "
            "bash real_world/sessions/pull_stretch_origin.sh"
        )
    for name in ("stretch_origin_tags.png", "head_sweep_range.json"):
        extra = subprocess.run(
            scp + [f"{host}:/tmp/robe/{name}", str(paths.stretch_dir / name)],
            check=False,
        )
        if extra.returncode != 0:
            print(f"WARN scp {name} failed (optional)")
    dest = receive_stretch_raw(paths.stretch_dir)
    print(f"installed raw {dest}")
    print("stretch_origin_status = raw_ready (corrected untouched)")


def _sample_cmd() -> str:
    return (
        "set -eo pipefail; set +u; "
        "source /opt/ros/humble/setup.bash; "
        "source \"$HOME/ament_ws/install/setup.bash\"; "
        "if [ -f \"$HOME/robe_ros2/dds_env.sh\" ]; then source \"$HOME/robe_ros2/dds_env.sh\"; fi; "
        "export ROBE_CODE_DIR=\"$HOME/robe/real_world/code\"; "
        "export PYTHONPATH=\"$ROBE_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}\"; "
        "ros2 run robe_stretch sample_origin --ros-args -p output_dir:=/tmp/robe"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, default=None)
    args = parser.parse_args()
    session_dir = resolve_session_dir_once(args.session_dir)
    paths = session_paths(session_dir)
    print_session_resolved(session_dir, paths.origin_raw)
    print("layout frame: marker_layout.json")
    print("this module writes Stretch origin (T_odom_layout) only")
    if _on_stretch():
        completed = subprocess.run(["bash", "-lc", _sample_cmd()], check=False)
        return int(completed.returncode)
    host = _stretch_ssh()
    print(f"SSH {host} ros2 run robe_stretch sample_origin")
    completed = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            host,
            f"bash -lc {_sample_cmd()!r}",
        ],
        check=False,
    )
    if completed.returncode != 0:
        print("ERROR sample_origin failed.")
        print("If /tmp/robe/stretch_origin.json exists on Stretch, pull only:")
        print("  bash real_world/sessions/pull_stretch_origin.sh")
        return int(completed.returncode or 1)
    try:
        _pull_raw(paths)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR pull raw to session: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
