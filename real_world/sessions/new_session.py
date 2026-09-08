#!/usr/bin/env python3
"""Create a session directory and point ``current`` at it. No geometry."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from session_paths import (  # noqa: E402
    CURRENT_LINK,
    SESSIONS_ROOT,
    SessionPaths,
    default_session_contract,
    save_session_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Suffix after YYYYMMDD (default: smoke)",
    )
    parser.add_argument("--session-id", type=str, default="")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing session.json",
    )
    args = parser.parse_args()
    if args.session_id:
        session_id = args.session_id
    else:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        suffix = args.name.strip() or "smoke"
        session_id = f"{day}_{suffix}"
    session_dir = SESSIONS_ROOT / session_id
    paths = SessionPaths(session_dir)
    paths.zed_dir.mkdir(parents=True, exist_ok=True)
    paths.stretch_dir.mkdir(parents=True, exist_ok=True)
    paths.poses_dir.mkdir(parents=True, exist_ok=True)
    if paths.session_json.is_file() and not args.force:
        print(f"keep existing {paths.session_json}")
    else:
        save_session_json(session_dir, default_session_contract(session_id))
    if CURRENT_LINK.is_symlink() or CURRENT_LINK.exists():
        CURRENT_LINK.unlink()
    CURRENT_LINK.symlink_to(session_id, target_is_directory=True)
    print(f"created {session_dir}")
    print(f"current -> {session_id}")
    print("statuses missing (run the three calibration modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
