"""Operator prompts and structured failure records for run_trial.

RETRY / QUIT / YES stay here so the orchestrator does not embed stdin
policy. Failures are written onto trial_progress.json with a fault code.
"""

from __future__ import annotations

import subprocess
import sys
import termios
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from trial_state import (
    STAGE_FAILED,
    STAGE_RUNNING,
    STAGE_SUCCEEDED,
    update_stage,
)


class OperatorQuit(SystemExit):
    """Operator typed QUIT. Progress already recorded."""


def drain_stdin() -> None:
    """Drop leftover keystrokes so a previous ENTER cannot skip the next gate."""

    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def wait_yes(message: str, *, prompt: str = "Type YES then ENTER when ready: ") -> None:
    print(message)
    drain_stdin()
    while True:
        reply = input(prompt).strip().upper()
        if reply == "YES":
            return
        print("Still waiting — type YES and press ENTER.")


def wait_retry_or_quit(message: str) -> str:
    drain_stdin()
    while True:
        reply = input(message).strip().upper()
        if reply in {"RETRY", "QUIT"}:
            return reply
        print("Still waiting — type RETRY or QUIT.")


def prompt_execution_quality() -> str:
    while True:
        raw = input("Execution quality [good / moderate / bad]: ").strip().lower()
        aliases = {
            "g": "good",
            "m": "moderate",
            "b": "bad",
            "good": "good",
            "moderate": "moderate",
            "bad": "bad",
        }
        if raw in aliases:
            return aliases[raw]
        print("Please enter good, moderate, or bad.")


def prompt_failure_type() -> str | None:
    raw = input(
        "Optional failure type (dragging / folding / occlusion / "
        "extent / other / blank): "
    ).strip().lower()
    return raw or None


def record_failure(
    pose_dir: Path,
    stage: str,
    *,
    fault_code: str,
    error: str,
    extras: Mapping[str, Any] | None = None,
) -> None:
    payload = {"fault_code": fault_code, **dict(extras or {})}
    update_stage(
        pose_dir,
        stage,
        status=STAGE_FAILED,
        outputs=payload,
        error=f"{fault_code}: {error}",
    )


def run_command(
    cmd: Sequence[str],
    *,
    pose_dir: Path | None = None,
    stage: str | None = None,
    context: str | None = None,
    allow_retry: bool = True,
    env: Mapping[str, str] | None = None,
    env_for: Callable[[str], dict[str, str] | None] | None = None,
) -> None:
    """Run a subprocess; on failure wait for RETRY / QUIT instead of exiting."""

    label = context or (Path(str(cmd[1])).name if len(cmd) > 1 else "command")
    while True:
        print(">>", " ".join(str(c) for c in cmd))
        run_env = env
        if run_env is None and env_for is not None:
            run_env = env_for(str(cmd[0]))
        result = subprocess.run(cmd, check=False, env=run_env)
        if result.returncode == 0:
            return
        print(f"ERROR: {label} failed with exit code {result.returncode}.")
        if pose_dir is not None and stage is not None:
            record_failure(
                pose_dir,
                stage,
                fault_code="COMMAND",
                error=f"{label} exit {result.returncode}",
                extras={"cmd": [str(c) for c in cmd]},
            )
        if not allow_retry:
            raise subprocess.CalledProcessError(result.returncode, cmd)
        reply = wait_retry_or_quit(
            "Type RETRY then ENTER to try again, or QUIT then ENTER to stop: "
        )
        if reply == "RETRY":
            if pose_dir is not None and stage is not None:
                update_stage(pose_dir, stage, status=STAGE_RUNNING)
            continue
        raise OperatorQuit(f"QUIT after {label} failed")


def mark_running(pose_dir: Path, stage: str, *, inputs: Mapping[str, Any] | None = None) -> None:
    update_stage(pose_dir, stage, status=STAGE_RUNNING, inputs=inputs)


def mark_succeeded(
    pose_dir: Path,
    stage: str,
    *,
    outputs: Mapping[str, Any] | None = None,
) -> None:
    update_stage(pose_dir, stage, status=STAGE_SUCCEEDED, outputs=outputs)
