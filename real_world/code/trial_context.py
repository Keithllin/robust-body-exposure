"""Shared trial runtime context passed to capture / plan / execute / score."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from trial_config import ResolvedProfile


@dataclass
class TrialContext:
    args: Any
    profile: ResolvedProfile
    pose_dir: Path
    subject_dir: Path
    study_root: Path
    target_limb_code: str
    sub_id: int
    robe_py: str
    zed_py: str
    code_dir: Path
    real_world_dir: Path
    session_dir: Path | None = None
    session_filters: dict = field(default_factory=dict)
    session_paths: Any = None
    pack_source: Path | None = None
    pack_copied: list[str] = field(default_factory=list)
    recover_aligned_at_capture: bool = False
    run: Callable[..., None] | None = None
    wait_yes: Callable[[str], None] | None = None
    env_for_python: Callable[[str], dict | None] | None = None

    @property
    def initial_dir(self) -> Path:
        from trial_layout import initial_dir

        return initial_dir(self.pose_dir)

    @property
    def uncover_dir(self) -> Path:
        from trial_layout import uncover_dir

        return uncover_dir(self.pose_dir)

    @property
    def intermediate_dir(self) -> Path:
        from trial_layout import intermediate_dir

        return intermediate_dir(self.pose_dir)

    @property
    def final_dir(self) -> Path:
        from trial_layout import final_dir

        return final_dir(self.pose_dir)

    @property
    def is_recover(self) -> bool:
        return self.args.approach in ("recover", "dyn")

    @property
    def closed_loop(self) -> bool:
        return self.profile.loop in ("closed", "closed-compare")

    @property
    def compare_loop(self) -> bool:
        return self.profile.loop == "closed-compare"

    @property
    def uncover_fevals(self) -> int:
        if self.args.uncover_max_fevals is None:
            return int(self.args.max_fevals)
        return int(self.args.uncover_max_fevals)
