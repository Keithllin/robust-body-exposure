"""Session registration directory, freeze contract, and one-shot resolve.

``sessions/current`` is resolved to an absolute path once per process.
Never re-read the symlink after that. Raw sync must not overwrite
``stretch_origin_corrected.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

CODE_DIR = Path(__file__).resolve().parent
REAL_WORLD_DIR = CODE_DIR.parent
SESSIONS_ROOT = REAL_WORLD_DIR / "sessions"
CURRENT_LINK = SESSIONS_ROOT / "current"
DEFAULT_LAYOUT = REAL_WORLD_DIR / "calibration" / "marker_layout.json"
DEFAULT_ZED_EXTRINSICS = REAL_WORLD_DIR / "calibration" / "zed_extrinsics.json"
CALIBRATION_DIR = REAL_WORLD_DIR / "calibration"
ACCEPT_XY_M = 0.02
REDO_IF = (
    "stretch manually moved",
    "stretch_driver odom reset",
    "bed or marker rig moved",
    "D435i mechanically moved",
    "ceiling ZED moved",
)

DEFAULT_FILTER_NAMES = {
    "ceiling": "zed_blanket_filter.json",
    "side_left": "zed_blanket_filter_side_left.json",
    "side_right": "zed_blanket_filter_side_right.json",
}
DEFAULT_PCD_CONFIG = {
    "pcd_exposure": 55,
    "pcd_gain": 40,
    "support_radius_3d": 0.03,
    "merge_mode": "auto",
    "bed_crop": None,
    "filter_json": dict(DEFAULT_FILTER_NAMES),
}

# Set after the first resolve_session_dir_once() in this process.
_RESOLVED_SESSION_DIR: Optional[Path] = None

STATUS_MISSING = "missing"
STATUS_NEEDS_CALIBRATION = "needs_calibration"
STATUS_RAW_READY = "raw_ready"
STATUS_NEEDS_ALIGNMENT = "needs_alignment"
STATUS_FROZEN = "frozen"
# Legacy alias kept so older session.json / tests still parse.
STATUS_UNALIGNED = "unaligned"


def sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class SessionPaths:
    session_dir: Path

    @property
    def session_id(self) -> str:
        return self.session_dir.name

    @property
    def session_json(self) -> Path:
        return self.session_dir / "session.json"

    @property
    def zed_dir(self) -> Path:
        return self.session_dir / "zed"

    @property
    def zed_extrinsics(self) -> Path:
        return self.zed_dir / "zed_extrinsics.json"

    @property
    def pcd_config(self) -> Path:
        return self.zed_dir / "pcd_config.json"

    @property
    def zed_preview_dir(self) -> Path:
        return self.zed_dir / "preview"

    @property
    def stretch_dir(self) -> Path:
        return self.session_dir / "stretch"

    @property
    def origin_json(self) -> Path:
        return self.stretch_dir / "stretch_origin.json"

    @property
    def origin_raw(self) -> Path:
        return self.stretch_dir / "stretch_origin_raw.json"

    @property
    def origin_corrected(self) -> Path:
        return self.stretch_dir / "stretch_origin_corrected.json"

    @property
    def origin_tags(self) -> Path:
        return self.stretch_dir / "stretch_origin_tags.png"

    @property
    def head_sweep(self) -> Path:
        return self.stretch_dir / "head_sweep_range.json"

    @property
    def ceiling_now(self) -> Path:
        return self.stretch_dir / "ceiling_now_ee.png"

    @property
    def overlay_json(self) -> Path:
        return self.stretch_dir / "ceiling_stretch_tf_overlay.json"

    @property
    def overlay_corrected_png(self) -> Path:
        return self.stretch_dir / "ceiling_stretch_tf_overlay_corrected.png"

    @property
    def registration_json(self) -> Path:
        return self.stretch_dir / "registration.json"

    @property
    def uncover_home_json(self) -> Path:
        """First Uncover bedside parking of the session. Return-to after Recover."""

        return self.stretch_dir / "uncover_home.json"

    @property
    def poses_dir(self) -> Path:
        return self.session_dir / "poses"

    def pose_pack_dir(self, pose_num: str | int) -> Path:
        return self.poses_dir / f"pose_{pose_num}"

    @property
    def active_trial_json(self) -> Path:
        return self.session_dir / "active_trial.json"


def resolve_session_dir_once(
    explicit: Optional[Path] = None,
    *,
    force: bool = False,
) -> Path:
    """Resolve session directory once per process. Do not follow ``current`` again."""

    global _RESOLVED_SESSION_DIR
    if _RESOLVED_SESSION_DIR is not None and not force:
        return _RESOLVED_SESSION_DIR
    env = os.environ.get("ROBE_SESSION_DIR", "").strip()
    if explicit is not None:
        raw = Path(explicit)
    elif env:
        raw = Path(env)
    elif CURRENT_LINK.is_symlink() or CURRENT_LINK.is_dir():
        raw = CURRENT_LINK
    else:
        raise FileNotFoundError(
            f"No session: set ROBE_SESSION_DIR or create {CURRENT_LINK}"
        )
    resolved = raw.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Session dir missing: {resolved}")
    _RESOLVED_SESSION_DIR = resolved
    return resolved


def session_paths(session_dir: Optional[Path] = None) -> SessionPaths:
    directory = session_dir or resolve_session_dir_once()
    return SessionPaths(Path(directory).resolve())


def origin_paths(session_dir: Optional[Path] = None) -> SessionPaths:
    """Stretch origin files under a (once-resolved) session directory."""

    return session_paths(session_dir)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON via tmp + validate + fsync + os.replace."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=2) + "\n"
    json.loads(text)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_session_json(session_dir: Path) -> dict[str, Any]:
    path = SessionPaths(session_dir).session_json
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_session_json(session_dir: Path, payload: Mapping[str, Any]) -> Path:
    path = SessionPaths(session_dir).session_json
    atomic_write_json(path, payload)
    return path


def default_session_contract(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "marker_layout_sha256": sha256_file(DEFAULT_LAYOUT),
        "zed_extrinsics_sha256": None,
        "pcd_config_sha256": None,
        "stretch_origin_raw_sha256": None,
        "stretch_origin_corrected_sha256": None,
        "corrected_from_raw_sha256": None,
        "xy_registration_m": None,
        "registration_residual_xy_m": None,
        "accept_xy_m": ACCEPT_XY_M,
        "zed_status": STATUS_MISSING,
        "stretch_origin_status": STATUS_MISSING,
        "registration_status": STATUS_MISSING,
        "stretch_registration_status": STATUS_MISSING,
        "zed_fusion": None,
        "redo_if": list(REDO_IF),
    }


def _update_contract(session_dir: Path, **fields: Any) -> dict[str, Any]:
    paths = SessionPaths(session_dir)
    if paths.session_json.is_file():
        payload = load_session_json(session_dir)
    else:
        payload = default_session_contract(paths.session_id)
    payload.update(fields)
    save_session_json(session_dir, payload)
    return payload


def mark_raw_arrived(session_dir: Path) -> dict[str, Any]:
    """New raw is on disk. Old corrected is no longer valid for execution."""

    paths = SessionPaths(session_dir)
    raw_sha = sha256_file(paths.origin_raw) or sha256_file(paths.origin_json)
    return _update_contract(
        session_dir,
        stretch_origin_raw_sha256=raw_sha,
        stretch_origin_status=STATUS_RAW_READY,
        registration_status=STATUS_NEEDS_ALIGNMENT,
        stretch_registration_status=STATUS_NEEDS_ALIGNMENT,
    )


def freeze_zed(
    session_dir: Path,
    *,
    fusion: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Freeze 3-ZED + pcd_config. Invalidates robot-planning registration."""

    paths = SessionPaths(session_dir)
    return _update_contract(
        session_dir,
        zed_status=STATUS_FROZEN,
        zed_extrinsics_sha256=sha256_file(paths.zed_extrinsics),
        pcd_config_sha256=sha256_file(paths.pcd_config),
        zed_fusion=None if fusion is None else dict(fusion),
        registration_status=STATUS_NEEDS_ALIGNMENT,
        stretch_registration_status=STATUS_NEEDS_ALIGNMENT,
    )


def freeze_registration(
    session_dir: Path,
    *,
    xy_registration_m: list[float],
    residual_xy_m: float,
    registration: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    paths = SessionPaths(session_dir)
    raw_sha = sha256_file(paths.origin_raw) or sha256_file(paths.origin_json)
    if registration is not None:
        atomic_write_json(paths.registration_json, registration)
    return _update_contract(
        session_dir,
        stretch_origin_raw_sha256=raw_sha,
        stretch_origin_corrected_sha256=sha256_file(paths.origin_corrected),
        corrected_from_raw_sha256=raw_sha,
        xy_registration_m=[float(v) for v in xy_registration_m],
        registration_residual_xy_m=float(residual_xy_m),
        zed_extrinsics_sha256=sha256_file(paths.zed_extrinsics),
        stretch_origin_status=STATUS_RAW_READY,
        registration_status=STATUS_FROZEN,
        stretch_registration_status=STATUS_FROZEN,
    )


class SessionRegistrationError(RuntimeError):
    """Stretch registration is not frozen or raw/corrected hashes drifted."""


def _registration_status(contract: Mapping[str, Any]) -> str:
    return str(
        contract.get("registration_status")
        or contract.get("stretch_registration_status")
        or ""
    )


def assert_registration_frozen(session_dir: Path) -> dict[str, Any]:
    """Reject execution unless status is frozen and dependency hashes match."""

    report = evaluate_session(session_dir)
    if not report.registration_ok:
        raise SessionRegistrationError(
            report.registration_error
            or f"session {SessionPaths(session_dir).session_id} registration not frozen"
        )
    return load_session_json(session_dir)


def validate_origin_payload(payload: Mapping[str, Any]) -> None:
    if not payload.get("ok"):
        raise ValueError(f"origin ok={payload.get('ok')}")
    if "T_odom_layout" not in payload:
        raise ValueError("origin missing T_odom_layout")


def receive_stretch_raw(stretch_dir: Path) -> Path:
    """Validate ``stretch_origin.json.tmp``, atomically install raw copies.

    Never touches ``stretch_origin_corrected.json``.
    """

    stretch_dir = Path(stretch_dir)
    tmp = stretch_dir / "stretch_origin.json.tmp"
    dest = stretch_dir / "stretch_origin.json"
    raw = stretch_dir / "stretch_origin_raw.json"
    if not tmp.is_file():
        raise FileNotFoundError(f"missing {tmp}")
    payload = json.loads(tmp.read_text(encoding="utf-8"))
    validate_origin_payload(payload)
    tmp.replace(dest)
    shutil.copy2(dest, raw)
    session_dir = stretch_dir.parent
    mark_raw_arrived(session_dir)
    return dest


def workstation_stretch_scp_url() -> str:
    return os.environ.get("ROBE_WORKSTATION_SCP", "").strip()


def sync_raw_to_workstation(local_out_dir: Path) -> None:
    """From Stretch: scp raw artifacts then run receive on RCHI.

    Syncs only stretch_origin.json (+ tags/sweep). Never corrected.
    """

    local_out_dir = Path(local_out_dir)
    origin = local_out_dir / "stretch_origin.json"
    if not origin.is_file():
        raise FileNotFoundError(origin)
    payload = json.loads(origin.read_text(encoding="utf-8"))
    validate_origin_payload(payload)
    url = workstation_stretch_scp_url()
    if ":" not in url:
        raise ValueError(f"ROBE_WORKSTATION_SCP must be user@host:path, got {url}")
    host, remote_dir = url.split(":", 1)
    remote_tmp = remote_dir.rstrip("/") + "/stretch_origin.json.tmp"
    files = [origin]
    for name in ("stretch_origin_tags.png", "head_sweep_range.json"):
        extra = local_out_dir / name
        if extra.is_file():
            files.append(extra)
    scp = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
    cmd_origin = scp + [str(origin), f"{host}:{remote_tmp}"]
    completed = subprocess.run(cmd_origin, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"scp raw origin failed: {completed.stderr.strip() or completed.stdout}"
        )
    extras = [p for p in files if p != origin]
    for extra in extras:
        extra_cmd = scp + [str(extra), f"{host}:{remote_dir.rstrip('/')}/{extra.name}"]
        extra_run = subprocess.run(extra_cmd, capture_output=True, text=True)
        if extra_run.returncode != 0:
            print(
                f"WARN scp {extra.name} failed: "
                f"{extra_run.stderr.strip() or extra_run.stdout}"
            )
    code_dir = Path(
        os.environ.get("ROBE_CODE_DIR", str(CODE_DIR))
    ).expanduser().resolve()
    receive_script = code_dir / "receive_stretch_raw.py"
    receive = (
        f"python3 {shlex.quote(str(receive_script))} "
        f"--stretch-dir {shlex.quote(remote_dir.rstrip('/'))}"
    )
    ssh = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        host,
        receive,
    ]
    finished = subprocess.run(ssh, capture_output=True, text=True)
    if finished.returncode != 0:
        raise RuntimeError(
            f"receive_stretch_raw failed: {finished.stderr.strip() or finished.stdout}"
        )
    print(finished.stdout.strip())


def infer_pose_dir(path: Path) -> Path:
    """Trial pose directory from a pickle, json, or subdirectory."""

    directory = Path(path).resolve()
    if directory.is_file():
        directory = directory.parent
    if directory.name in {"initial", "intermediate", "final", "uncover"}:
        return directory.parent
    return directory


def write_active_trial(session_dir: Path, pose_dir: Path) -> Path:
    """Point the executor at this trial. Written just before /bed_pull."""

    pose = infer_pose_dir(pose_dir)
    candidates = (
        pose / "canonical_bed_frame.json",
        pose.parent / "canonical_bed_frame.json",
    )
    canon = next((item for item in candidates if item.is_file()), None)
    if canon is None:
        raise FileNotFoundError(
            f"no canonical_bed_frame.json under {pose} "
            "(capture the initial blanket first)"
        )
    dest = session_paths(session_dir).active_trial_json
    atomic_write_json(
        dest,
        {
            "pose_dir": str(pose),
            "canonical_frame_path": str(canon.resolve()),
            "written_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return dest


def load_active_trial(session_dir: Path) -> Optional[dict[str, Any]]:
    dest = session_paths(session_dir).active_trial_json
    if not dest.is_file():
        return None
    try:
        payload = json.loads(dest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    canon = Path(str(payload.get("canonical_frame_path") or ""))
    pose = Path(str(payload.get("pose_dir") or ""))
    if not canon.is_file() or not pose.is_dir():
        return None
    return payload


def print_session_resolved(session_dir: Path, origin_path: Path) -> str:
    text = (
        f"SESSION RESOLVED:\n  {session_dir}\n"
        f"STRETCH ORIGIN:\n  {origin_path}"
    )
    print(text)
    return text


def load_pcd_config(session_dir: Path) -> dict[str, Any]:
    path = SessionPaths(session_dir).pcd_config
    if not path.is_file():
        return dict(DEFAULT_PCD_CONFIG)
    payload = json.loads(path.read_text(encoding="utf-8"))
    merged = dict(DEFAULT_PCD_CONFIG)
    merged.update(payload)
    filters = dict(DEFAULT_FILTER_NAMES)
    filters.update(payload.get("filter_json") or {})
    merged["filter_json"] = filters
    return merged


def resolve_pcd_filters(session_dir: Path) -> dict[str, Path]:
    """Resolve filter JSON names relative to ``session/zed/`` only."""

    paths = SessionPaths(session_dir)
    cfg = load_pcd_config(session_dir)
    resolved: dict[str, Path] = {}
    for role, name in (cfg.get("filter_json") or {}).items():
        candidate = Path(str(name))
        if candidate.is_absolute():
            raise ValueError(
                f"pcd_config filter_json.{role} must be a name relative to "
                f"{paths.zed_dir}, not {candidate}"
            )
        resolved[role] = paths.zed_dir / candidate.name
    return resolved


def snapshot_filter_jsons(session_dir: Path) -> dict[str, str]:
    """Copy calibration filter JSONs into session/zed/. Return name→sha."""

    paths = SessionPaths(session_dir)
    paths.zed_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for role, name in DEFAULT_FILTER_NAMES.items():
        src = CALIBRATION_DIR / name
        dest = paths.zed_dir / name
        if src.is_file():
            shutil.copy2(src, dest)
        digest = sha256_file(dest)
        if digest:
            hashes[role] = digest
    return hashes


def write_pcd_config(session_dir: Path, **overrides: Any) -> Path:
    paths = SessionPaths(session_dir)
    hashes = snapshot_filter_jsons(session_dir)
    payload = dict(DEFAULT_PCD_CONFIG)
    payload.update(overrides)
    payload["filter_json"] = dict(DEFAULT_FILTER_NAMES)
    payload["filter_sha256"] = hashes
    atomic_write_json(paths.pcd_config, payload)
    return paths.pcd_config


@dataclass
class SessionReadyReport:
    session_id: str
    session_dir: Path
    ready: bool
    zed_ok: bool
    stretch_ok: bool
    registration_ok: bool
    zed_status: str
    stretch_origin_status: str
    registration_status: str
    registration_error: Optional[str] = None
    lines: list[str] = field(default_factory=list)
    hashes: dict[str, Optional[str]] = field(default_factory=dict)

    def text(self) -> str:
        return "\n".join(self.lines)


def evaluate_session(session_dir: Path) -> SessionReadyReport:
    """Gate used by session_status and run_trial. Not a geometry module."""

    paths = SessionPaths(session_dir)
    if paths.session_json.is_file():
        contract = load_session_json(session_dir)
    else:
        contract = default_session_contract(paths.session_id)
    zed_status = str(contract.get("zed_status") or STATUS_MISSING)
    stretch_status = str(contract.get("stretch_origin_status") or STATUS_MISSING)
    reg_status = _registration_status(contract)
    accept = float(contract.get("accept_xy_m") or ACCEPT_XY_M)
    residual = contract.get("registration_residual_xy_m")
    layout_sha = sha256_file(DEFAULT_LAYOUT)
    zed_sha = sha256_file(paths.zed_extrinsics)
    pcd_sha = sha256_file(paths.pcd_config)
    raw_sha = sha256_file(paths.origin_raw) or sha256_file(paths.origin_json)
    corrected_sha = sha256_file(paths.origin_corrected)
    session_sha = sha256_file(paths.session_json)
    reg_file_sha = sha256_file(paths.registration_json)

    zed_ok = (
        zed_status == STATUS_FROZEN
        and paths.zed_extrinsics.is_file()
        and zed_sha is not None
        and zed_sha == contract.get("zed_extrinsics_sha256")
        and paths.pcd_config.is_file()
        and pcd_sha is not None
        and pcd_sha == contract.get("pcd_config_sha256")
    )
    stretch_ok = (
        stretch_status == STATUS_RAW_READY
        and raw_sha is not None
        and raw_sha == contract.get("stretch_origin_raw_sha256")
    )

    registration_error = None
    registration_ok = False
    if reg_status != STATUS_FROZEN:
        registration_error = (
            f"registration_status={reg_status!r} (need frozen). "
            "Run register_robot_to_planning.py"
        )
    elif not paths.origin_corrected.is_file():
        registration_error = f"missing {paths.origin_corrected}"
    elif not raw_sha or raw_sha != contract.get("corrected_from_raw_sha256"):
        registration_error = (
            f"raw hash {raw_sha} != corrected_from_raw "
            f"{contract.get('corrected_from_raw_sha256')}"
        )
    elif contract.get("stretch_origin_corrected_sha256") and (
        corrected_sha != contract.get("stretch_origin_corrected_sha256")
    ):
        registration_error = (
            f"corrected hash {corrected_sha} != contract "
            f"{contract.get('stretch_origin_corrected_sha256')}"
        )
    elif residual is not None and float(residual) > accept:
        registration_error = (
            f"XY residual {float(residual):.4f} m > accept {accept:.4f} m"
        )
    elif paths.registration_json.is_file():
        recorded = json.loads(paths.registration_json.read_text(encoding="utf-8"))
        expected = {
            "stretch_origin_raw_sha256": raw_sha,
            "zed_extrinsics_sha256": zed_sha,
            "marker_layout_sha256": layout_sha,
        }
        for key, current in expected.items():
            stored = recorded.get(key)
            if not stored or not current or stored != current:
                registration_error = (
                    f"{key} drifted (registration {stored} != current {current})"
                )
                break
        if registration_error is None:
            registration_ok = True
    else:
        registration_error = f"missing {paths.registration_json}"

    if registration_error and reg_status == STATUS_FROZEN:
        # Stale freeze: treat as needs_alignment for the gate.
        reg_status = STATUS_NEEDS_ALIGNMENT

    ready = bool(zed_ok and stretch_ok and registration_ok)
    fusion = contract.get("zed_fusion") or {}
    fusion_verdict = str(fusion.get("verdict") or "—")
    before = contract.get("xy_registration_m")
    before_xy = None
    if isinstance(before, (list, tuple)) and len(before) >= 2:
        before_xy = (float(before[0]) ** 2 + float(before[1]) ** 2) ** 0.5
    after_xy = None if residual is None else float(residual)
    recorded = {}
    if paths.registration_json.is_file():
        try:
            recorded = json.loads(paths.registration_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            recorded = {}
    if recorded.get("before_xy_error_m") is not None:
        before_xy = float(recorded["before_xy_error_m"])
    if recorded.get("after_xy_error_m") is not None:
        after_xy = float(recorded["after_xy_error_m"])

    def _cm(value: Optional[float]) -> str:
        if value is None:
            return "—"
        return f"{100.0 * float(value):.1f} cm"

    lines = [
        f"SESSION: {paths.session_id}",
        "",
        "ZED SYSTEM",
        f"  PnP                    {'PASS' if zed_ok else zed_status.upper()}",
        f"  Fusion                 {fusion_verdict}",
        f"  PCD config             {'FROZEN' if zed_ok else zed_status.upper()}",
        "",
        "STRETCH ORIGIN",
        f"  Raw origin             {'PASS' if stretch_ok else stretch_status.upper()}",
        f"  Physical posts         {'PASS' if stretch_ok else '—'}",
        "",
        "ROBOT-PLANNING REGISTRATION",
        f"  Reference              ArUco 136",
        f"  Before XY              {_cm(before_xy)}",
        f"  After XY               {_cm(after_xy)}",
        f"  Threshold              {_cm(accept)}",
        f"  Status                 {reg_status.upper()}",
        "",
        f"READY FOR TRIAL: {'YES' if ready else 'NO'}",
    ]
    if registration_error and not registration_ok:
        lines.append(f"  reason                 {registration_error}")
    return SessionReadyReport(
        session_id=paths.session_id,
        session_dir=paths.session_dir,
        ready=ready,
        zed_ok=zed_ok,
        stretch_ok=stretch_ok,
        registration_ok=registration_ok,
        zed_status=zed_status,
        stretch_origin_status=stretch_status,
        registration_status=reg_status,
        registration_error=registration_error,
        lines=lines,
        hashes={
            "session_sha256": session_sha,
            "zed_extrinsics_sha256": zed_sha,
            "pcd_config_sha256": pcd_sha,
            "stretch_origin_corrected_sha256": corrected_sha,
            "registration_sha256": reg_file_sha,
            "stretch_origin_raw_sha256": raw_sha,
            "marker_layout_sha256": layout_sha,
        },
    )


def session_manifest_fields(session_dir: Path) -> dict[str, Any]:
    report = evaluate_session(session_dir)
    return {
        "session_id": report.session_id,
        "session_sha256": report.hashes.get("session_sha256"),
        "zed_extrinsics_sha256": report.hashes.get("zed_extrinsics_sha256"),
        "pcd_config_sha256": report.hashes.get("pcd_config_sha256"),
        "stretch_origin_corrected_sha256": report.hashes.get(
            "stretch_origin_corrected_sha256"
        ),
        "registration_sha256": report.hashes.get("registration_sha256"),
    }
