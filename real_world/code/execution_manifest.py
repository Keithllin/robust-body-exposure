"""Trial execution log for attributing pred-real gaps."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


def _sha256(path: Optional[Path]) -> Optional[str]:
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_execution_manifest(
    dest: Path,
    *,
    trial_id: str,
    planned_action_bed: list[float],
    mirror_x_applied: bool,
    reachability: Mapping[str, Any],
    canonical_frame_path: Optional[Path] = None,
    marker_layout_path: Optional[Path] = None,
    zed_extrinsics_path: Optional[Path] = None,
    executor_yaml_path: Optional[Path] = None,
    session_id: Optional[str] = None,
    session_json_path: Optional[Path] = None,
    stretch_origin_corrected_path: Optional[Path] = None,
    pcd_config_path: Optional[Path] = None,
    registration_path: Optional[Path] = None,
    git_commit: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial_id": trial_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned_action_bed": [float(v) for v in planned_action_bed],
        "frame_id": "bed",
        "mirror_x_applied": bool(mirror_x_applied),
        "session_id": session_id,
        "reachability": dict(reachability),
        "hashes": {
            "canonical_bed_frame": _sha256(canonical_frame_path),
            "marker_layout": _sha256(marker_layout_path),
            "zed_extrinsics": _sha256(zed_extrinsics_path),
            "executor_yaml": _sha256(executor_yaml_path),
            "session_json": _sha256(session_json_path),
            "stretch_origin_corrected": _sha256(stretch_origin_corrected_path),
            "pcd_config": _sha256(pcd_config_path),
            "registration": _sha256(registration_path),
        },
        "git_commit": git_commit,
    }
    if extra:
        payload.update(dict(extra))
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    return dest
