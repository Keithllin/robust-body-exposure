"""Trial identity, named stages, and atomically updated progress.

``trial_identity.json`` is written once at start. ``trial_progress.json`` is
replaced via tmp+fsync so a crash mid-stage cannot leave a half-written
resume file. ``--resume`` reads this; ``--start-here > 0`` with a new
timestamp or a new random TL is rejected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from session_paths import atomic_write_json
from trial_config import (
    NAMED_STAGES,
    named_stage_from_start_here,
    start_here_map,
)

STAGE_PENDING = "pending"
STAGE_RUNNING = "running"
STAGE_SUCCEEDED = "succeeded"
STAGE_FAILED = "failed"
STAGE_SKIPPED = "skipped"

IDENTITY_NAME = "trial_identity.json"
PROGRESS_NAME = "trial_progress.json"


class ResumeError(RuntimeError):
    """Unsafe resume / start-here combination."""


@dataclass
class StageRecord:
    name: str
    status: str = STAGE_PENDING
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageRecord":
        return cls(
            name=str(payload["name"]),
            status=str(payload.get("status") or STAGE_PENDING),
            inputs=dict(payload.get("inputs") or {}),
            outputs=dict(payload.get("outputs") or {}),
            error=payload.get("error"),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stages_for_loop(loop: str) -> tuple[str, ...]:
    if loop == "open":
        return (
            "pose",
            "initial-capture",
            "intermediate-capture",
            "recover-plan",
            "recover-exec",
            "score",
        )
    if loop == "closed-compare":
        return NAMED_STAGES
    return NAMED_STAGES


def identity_path(pose_dir: Path) -> Path:
    return Path(pose_dir) / IDENTITY_NAME


def progress_path(pose_dir: Path) -> Path:
    return Path(pose_dir) / PROGRESS_NAME


def write_identity(pose_dir: Path, payload: Mapping[str, Any]) -> Path:
    dest = identity_path(pose_dir)
    body = dict(payload)
    body.setdefault("written_at", _now())
    atomic_write_json(dest, body)
    return dest


def load_identity(pose_dir: Path) -> dict[str, Any]:
    path = identity_path(pose_dir)
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text())


def empty_progress(loop: str, *, start_stage: str = "pose") -> dict[str, Any]:
    records = []
    reached = False
    for name in stages_for_loop(loop):
        if name == start_stage:
            reached = True
        status = STAGE_PENDING if reached else STAGE_SKIPPED
        records.append(StageRecord(name=name, status=status).to_dict())
    return {
        "loop": loop,
        "start_stage": start_stage,
        "current": start_stage,
        "stages": records,
        "updated_at": _now(),
    }


def write_progress(pose_dir: Path, payload: Mapping[str, Any]) -> Path:
    dest = progress_path(pose_dir)
    body = dict(payload)
    body["updated_at"] = _now()
    atomic_write_json(dest, body)
    return dest


def load_progress(pose_dir: Path) -> dict[str, Any]:
    path = progress_path(pose_dir)
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text())


def _records(progress: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(progress.get("stages") or [])


def update_stage(
    pose_dir: Path,
    name: str,
    *,
    status: str,
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    progress = load_progress(pose_dir)
    found = False
    for row in _records(progress):
        if row.get("name") != name:
            continue
        found = True
        row["status"] = status
        if inputs is not None:
            row["inputs"] = dict(inputs)
        if outputs is not None:
            row["outputs"] = dict(outputs)
        if status == STAGE_RUNNING:
            row["started_at"] = row.get("started_at") or _now()
            row["error"] = None
        if status in (STAGE_SUCCEEDED, STAGE_FAILED, STAGE_SKIPPED):
            row["finished_at"] = _now()
            row["error"] = error
        progress["current"] = name
        break
    if not found:
        raise KeyError(f"stage {name!r} is not in {progress_path(pose_dir)}")
    write_progress(pose_dir, progress)
    return progress


def next_pending_stage(progress: Mapping[str, Any]) -> str | None:
    for row in _records(progress):
        if row.get("status") in (STAGE_PENDING, STAGE_FAILED, STAGE_RUNNING):
            return str(row["name"])
    return None


def stage_succeeded(progress: Mapping[str, Any], name: str) -> bool:
    for row in _records(progress):
        if row.get("name") == name:
            return row.get("status") == STAGE_SUCCEEDED
    return False


def should_run_stage(progress: Mapping[str, Any], name: str) -> bool:
    """True when this named stage still needs to run (resume-safe)."""

    reached_pending = False
    for row in _records(progress):
        status = row.get("status")
        if row.get("name") == name:
            if status == STAGE_SUCCEEDED or status == STAGE_SKIPPED:
                return False
            return True
        if status in (STAGE_PENDING, STAGE_FAILED, STAGE_RUNNING):
            reached_pending = True
        if reached_pending:
            return False
    return False


def latest_trial_dir(subject_dir: Path) -> Path | None:
    subject = Path(subject_dir)
    if not subject.is_dir():
        return None
    candidates = [
        path
        for path in subject.iterdir()
        if path.is_dir() and progress_path(path).is_file()
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (progress_path(path).stat().st_mtime, path.name),
    )


def resolve_resume_dir(
    *,
    resume: str,
    subject_dir: Path,
    pose_dir: Path | None = None,
) -> Path:
    text = str(resume or "").strip()
    if not text:
        raise ResumeError("--resume is empty")
    if text == "latest":
        found = latest_trial_dir(subject_dir)
        if found is None:
            raise ResumeError(f"no trial_progress.json under {subject_dir}")
        return found
    raw = Path(text).expanduser()
    if raw.is_dir() and progress_path(raw).is_file():
        return raw.resolve()
    sibling = subject_dir / text
    if sibling.is_dir() and progress_path(sibling).is_file():
        return sibling.resolve()
    if pose_dir is not None and pose_dir.is_dir() and progress_path(pose_dir).is_file():
        return Path(pose_dir).resolve()
    raise ResumeError(f"cannot resume {resume!r}; no {PROGRESS_NAME}")


def assert_safe_start(
    *,
    start_here: int,
    start_stage: str,
    tl_code: str,
    sub_id: int | None,
    resume: str,
) -> None:
    """Forbid --start-here > 0 with a brand-new timestamp / random TL."""

    if resume:
        return
    named = str(start_stage or "").strip()
    if int(start_here) <= 0 and named in ("", "pose"):
        return
    if str(tl_code) == "random":
        raise ResumeError(
            "--start-here/--start-stage after pose cannot be paired with "
            "a new random TL. Pass --tl-code <n> or --resume <trial|latest>."
        )
    if sub_id is None:
        raise ResumeError(
            "--start-here/--start-stage after pose requires --sub-id "
            "(the existing trial timestamp). Otherwise this creates a new "
            "directory and skips earlier stages unsafely."
        )


def resolve_start_stage(*, loop: str, start_here: int, start_stage: str) -> str:
    named = str(start_stage or "").strip()
    if named:
        if named not in NAMED_STAGES:
            raise ResumeError(f"unknown --start-stage {named!r}")
        return named
    return named_stage_from_start_here(loop, int(start_here))


def stages_from(
    loop: str,
    start_stage: str,
    *,
    include: Iterable[str] | None = None,
) -> tuple[str, ...]:
    available = stages_for_loop(loop)
    if start_stage not in available:
        raise ResumeError(
            f"stage {start_stage!r} is not in loop={loop} ({available})"
        )
    index = available.index(start_stage)
    wanted = available[index:]
    if include is None:
        return wanted
    allow = set(include)
    return tuple(name for name in wanted if name in allow)


@dataclass
class TrialStage:
    """Uniform stage interface. Algorithms stay in the helper modules."""

    name: str
    prerequisites: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()
    validate: Callable[[Any], None] | None = None
    run: Callable[[Any], dict[str, Any] | None] | None = None

    def execute(self, ctx: Any) -> dict[str, Any]:
        if self.validate is not None:
            self.validate(ctx)
        payload = self.run(ctx) if self.run is not None else {}
        return dict(payload or {})
