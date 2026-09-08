#!/usr/bin/env python3
"""Install a just-scp'd stretch_origin.json.tmp as session raw. Never touches corrected."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from session_paths import receive_stretch_raw  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stretch-dir", type=Path, required=True)
    args = parser.parse_args()
    dest = receive_stretch_raw(args.stretch_dir.resolve())
    print(f"installed raw {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
