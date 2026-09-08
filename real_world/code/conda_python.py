"""Resolve conda env python executables for dual-env trial orchestration."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _candidate(env_name: str) -> Path:
    # Prefer CONDA_EXE prefix, then ~/miniconda3, then ~/anaconda3
    roots = []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        roots.append(Path(conda_exe).resolve().parents[1])
    home = Path.home()
    roots.extend([home / "miniconda3", home / "anaconda3", home / "mambaforge"])
    for root in roots:
        py = root / "envs" / env_name / "bin" / "python"
        if py.is_file():
            return py
    return Path()


def python_for_env(env_name: str) -> str:
    """Return absolute python path for a conda env, or fall back to PATH/`python`."""
    explicit = os.environ.get(f"ROBE_{env_name.upper().replace('-', '_')}_PYTHON")
    if explicit and Path(explicit).is_file():
        return explicit
    cand = _candidate(env_name)
    if cand.is_file():
        return str(cand)
    which = shutil.which("python")
    if which:
        return which
    return "python"


def zed_python() -> str:
    return python_for_env("robe-zed")


def robe_python() -> str:
    return python_for_env("robe")


def _is_foreign_site_packages(entry: str, prefix: Path) -> bool:
    if not entry:
        return False
    try:
        resolved = Path(entry).resolve()
    except OSError:
        return False
    if "site-packages" not in resolved.parts:
        return False
    try:
        resolved.relative_to(prefix)
    except ValueError:
        return True
    return False


def drop_foreign_site_packages() -> None:
    """Keep this interpreter's site-packages; drop other conda envs.

    ``conda run -n robe-zed`` inherits PYTHONPATH. If Humble / robe-ros2
    was sourced in the same shell, NumPy 3.11 lands on a 3.10 process.
    """

    prefix = Path(sys.prefix).resolve()
    sys.path[:] = [
        entry
        for entry in sys.path
        if not _is_foreign_site_packages(entry, prefix)
    ]
    kept: list[str] = []
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry or _is_foreign_site_packages(entry, prefix):
            continue
        kept.append(entry)
    if kept:
        os.environ["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        os.environ.pop("PYTHONPATH", None)


def env_without_foreign_site_packages(prefix: str | None = None) -> dict[str, str]:
    """Environment for a subprocess of ``prefix`` (default: this interpreter)."""

    env = os.environ.copy()
    root = Path(prefix or sys.prefix).resolve()
    kept: list[str] = []
    for entry in env.get("PYTHONPATH", "").split(os.pathsep):
        if not entry or _is_foreign_site_packages(entry, root):
            continue
        kept.append(entry)
    if kept:
        env["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        env.pop("PYTHONPATH", None)
    home = env.get("PYTHONHOME", "").strip()
    if home:
        try:
            Path(home).resolve().relative_to(root)
        except ValueError:
            env.pop("PYTHONHOME", None)
    return env


def env_for_python(python_exe: str) -> dict[str, str]:
    """Env for a conda python so parent PYTHONPATH cannot leak site-packages."""

    prefix = Path(python_exe).resolve().parents[1]
    env = env_without_foreign_site_packages(str(prefix))
    env["CONDA_PREFIX"] = str(prefix)
    env["CONDA_DEFAULT_ENV"] = prefix.name
    return env
