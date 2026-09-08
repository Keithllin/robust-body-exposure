#!/usr/bin/env python3
"""Print session READY FOR TRIAL. Gate only — not a calibration module."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from session_paths import (  # noqa: E402
    evaluate_session,
    resolve_session_dir_once,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, default=None)
    args = parser.parse_args()
    session_dir = resolve_session_dir_once(args.session_dir)
    report = evaluate_session(session_dir)
    print(report.text())
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
