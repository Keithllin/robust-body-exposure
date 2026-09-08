#!/usr/bin/env bash
# RCHI: pull Stretch /tmp/robe raw origin into sessions/current/stretch.
# Atomic install; never overwrites stretch_origin_corrected.json.
set -euo pipefail
HOST="${ROBE_STRETCH_SCP:?Set ROBE_STRETCH_SCP to user@host:/remote/session/stretch}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRETCH_DIR="${ROBE_SESSION_STRETCH:-${HERE}/current/stretch}"
mkdir -p "${STRETCH_DIR}"
scp -o BatchMode=yes "${HOST}:/tmp/robe/stretch_origin.json" \
  "${STRETCH_DIR}/stretch_origin.json.tmp"
scp -o BatchMode=yes "${HOST}:/tmp/robe/stretch_origin_tags.png" \
  "${STRETCH_DIR}/stretch_origin_tags.png" || true
scp -o BatchMode=yes "${HOST}:/tmp/robe/head_sweep_range.json" \
  "${STRETCH_DIR}/head_sweep_range.json" || true
python3 "${HERE}/../code/receive_stretch_raw.py" --stretch-dir "${STRETCH_DIR}"
