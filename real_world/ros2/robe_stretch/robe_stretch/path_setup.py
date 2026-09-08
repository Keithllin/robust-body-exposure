"""Add the workstation ``real_world/code`` tree to sys.path from a ROS2 node."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def add_workstation_code() -> Path:
    here = Path(__file__).resolve()
    env = os.environ.get("ROBE_CODE_DIR", "").strip()
    candidates = [
        Path(env) if env else None,
        here.parents[4] / "code" if len(here.parents) >= 4 else None,
        here.parents[3] / "code" if len(here.parents) >= 3 else None,
        here.parents[2] / "code" if len(here.parents) >= 2 else None,
    ]
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry:
            candidates.append(Path(entry))
    for path in candidates:
        if path is None:
            continue
        if (path / "canonical_bed.py").is_file():
            resolved = path.resolve()
            if str(resolved) not in sys.path:
                sys.path.insert(0, str(resolved))
            return resolved
    raise FileNotFoundError(
        "real_world/code with canonical_bed.py not found; "
        "rsync real_world/code onto the robot or set ROBE_CODE_DIR / PYTHONPATH"
    )


def on_stretch_robot() -> bool:
    """True on the Hello Robot PC. D435i stays here; RCHI must not subscribe."""
    host = os.uname().nodename.lower()
    return bool(os.environ.get("HELLO_FLEET_ID")) or host.startswith("stretch")


def require_stretch_for_d435i() -> None:
    if on_stretch_robot():
        return
    print(
        "D435i stays on Stretch; RCHI does not subscribe to camera topics.\n"
        "On the robot desktop (AnyDesk, DISPLAY=:0):\n"
        "  source /opt/ros/humble/setup.bash\n"
        "  source ~/ament_ws/install/setup.bash\n"
        "  ros2 run robe_stretch preview_origin --ros-args "
        "-p output_dir:=/tmp/robe -p localization.n_stops:=2\n"
        "Then copy /tmp/robe/stretch_origin.json to sessions/current/stretch "
        "on RCHI (never overwrite stretch_origin_corrected.json)."
    )
    raise SystemExit(2)
