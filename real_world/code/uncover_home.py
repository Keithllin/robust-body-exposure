"""Uncover start parking: record once, return after Recover.

Session ``stretch/uncover_home.json`` is the first Uncover bedside pose of
the day. A later trial's Recover snapshot must not overwrite it.
Return is ``translate_mobile_base`` only — no ``rotate_mobile_base``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

UNCOVER_HOME_NAME = "uncover_home.json"
YAW_LIMIT_RAD = 0.15
LATERAL_WARN_M = 0.05
DONE_XY_M = 0.04


def _as_t(matrix) -> np.ndarray:
    t = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return t


def yaw_about_z(t_odom_base: np.ndarray) -> float:
    x = _as_t(t_odom_base)[:2, 0]
    return float(np.arctan2(x[1], x[0]))


def along_bed_return(
    t_now: np.ndarray, t_home: np.ndarray
) -> dict[str, float]:
    """Incremental +X_base needed to match home XY. Lateral is leftover."""

    now = _as_t(t_now)
    home = _as_t(t_home)
    x_now = now[:2, 0]
    n = float(np.linalg.norm(x_now))
    if n < 1e-9:
        raise ValueError("T_odom_base +X is zero")
    x_now = x_now / n
    delta = home[:2, 3] - now[:2, 3]
    d_along = float(delta @ x_now)
    lateral = delta - d_along * x_now
    yaw = yaw_about_z(home) - yaw_about_z(now)
    yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
    return {
        "translate_mobile_base": d_along,
        "lateral_m": float(np.linalg.norm(lateral)),
        "yaw_err_rad": yaw,
        "xy_err_m": float(np.linalg.norm(delta)),
    }


def payload_from_snapshot(snapshot: dict, *, source: str | Path) -> dict:
    t_ob = snapshot.get("T_odom_base")
    if t_ob is None:
        raise KeyError("snapshot missing T_odom_base")
    return {
        "role": "uncover_start_parking",
        "T_odom_base": _as_t(t_ob).tolist(),
        "source": str(source),
        "note": "translate_mobile_base only after Recover; do not overwrite "
        "from a Recover / mid-pull snapshot.",
    }


def write_uncover_home(path: Path, payload: dict, *, overwrite: bool) -> bool:
    path = Path(path)
    if path.is_file() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return True


def load_uncover_home(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if "T_odom_base" not in payload:
        raise KeyError(f"{path} missing T_odom_base")
    return payload


def resolve_uncover_home(
    pose_dir: Path, session_dir: Path | None
) -> Path | None:
    """Prefer session home (first Uncover of the day), else this trial."""

    from session_paths import session_paths

    if session_dir is not None:
        session = session_paths(session_dir).uncover_home_json
        if session.is_file():
            return session
    trial = Path(pose_dir) / UNCOVER_HOME_NAME
    if trial.is_file():
        return trial
    return None


def record_uncover_home(
    snapshot_path: Path,
    pose_dir: Path,
    session_dir: Path | None,
) -> dict[str, str]:
    """Write trial home always; write session home only if missing."""

    snap = json.loads(Path(snapshot_path).read_text())
    payload = payload_from_snapshot(snap, source=snapshot_path)
    trial = Path(pose_dir) / UNCOVER_HOME_NAME
    write_uncover_home(trial, payload, overwrite=True)
    wrote = {"trial": str(trial)}
    if session_dir is not None:
        from session_paths import session_paths

        session = session_paths(session_dir).uncover_home_json
        if write_uncover_home(session, payload, overwrite=False):
            wrote["session"] = str(session)
        else:
            wrote["session"] = f"{session} (kept)"
    return wrote
