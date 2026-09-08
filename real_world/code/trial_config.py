"""Resolved production profile and CLI identity for a real-world trial.

CLI stays compatible with ``run_trial.py``. Production defaults live here so
an operator can inspect the resolved profile without reading 2000 lines of
orchestration. Every CLI override is recorded as ``debug_overrides``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from stretch_cartesian import execution_wrist_as_dict
from stretch_limits import DEFAULT_CMA_BOUNDS, PLANNER_ARM_MAX_M

TL_LABELS = {
    0: "right hand",
    1: "right hand + forearm",
    2: "right arm",
    3: "right foot",
    4: "right foot + shin",
    5: "right leg",
    6: "left hand",
    7: "left hand + forearm",
    8: "left arm",
    9: "left foot",
    10: "left foot + shin",
    11: "left leg",
    12: "both lower legs",
    13: "upper body",
    14: "lower body",
    15: "whole body",
}
STUDY_TLS = (2, 4, 5, 12, 13, 14, 15)

MARKER_POLICIES = ("reuse_pack", "detect_if_missing", "force_redetect")
REGISTRATION_POLICIES = ("keep_session", "trial_check", "refreeze_session")
PROFILES = ("production", "debug")

NAMED_STAGES = (
    "pose",
    "initial-capture",
    "uncover-plan",
    "uncover-exec",
    "intermediate-capture",
    "recover-plan",
    "recover-exec",
    "final-capture",
    "score",
)

# Old --start-here integers. Recover plan/exec are split; 5 still means plan.
CLOSED_START_HERE = {
    0: "pose",
    1: "initial-capture",
    2: "uncover-plan",
    3: "uncover-exec",
    4: "intermediate-capture",
    5: "recover-plan",
    6: "final-capture",
    7: "score",
}
OPEN_START_HERE = {
    0: "pose",
    1: "initial-capture",
    2: "intermediate-capture",
    3: "recover-plan",
}
COMPARE_START_HERE = {
    0: "pose",
    1: "initial-capture",
    2: "uncover-plan",
    3: "uncover-exec",
    4: "intermediate-capture",
    5: "recover-plan",
    6: "recover-exec",
    7: "final-capture",
    8: "score",
}


@dataclass(frozen=True)
class ResolvedProfile:
    name: str
    loop: str
    send_bed_pull: bool
    tl_codes: tuple[int, ...]
    tl_code_mode: str
    remeasure_body: bool
    mask_backend: str
    detect_top_markers: bool
    trial_wrist_match: bool
    marker_policy: str
    registration_policy: str
    live_geometry_policy: str
    controller: str
    recipe: str
    cma_bounds: str
    arm_max_m: float
    camera_roles: str
    approach: str
    wrist: dict[str, float] = field(default_factory=execution_wrist_as_dict)
    debug_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def production_profile(**overrides: Any) -> ResolvedProfile:
    payload = {
        "name": "production",
        "loop": "closed",
        "send_bed_pull": True,
        "tl_codes": STUDY_TLS,
        "tl_code_mode": "random",
        "remeasure_body": True,
        "mask_backend": "sam2",
        "detect_top_markers": False,
        "trial_wrist_match": False,
        "marker_policy": "reuse_pack",
        "registration_policy": "keep_session",
        "live_geometry_policy": "reject",
        "controller": "streaming",
        "recipe": "full-pull",
        "cma_bounds": DEFAULT_CMA_BOUNDS,
        "arm_max_m": PLANNER_ARM_MAX_M,
        "camera_roles": "ceiling,side_left,side_right",
        "approach": "recover",
        "wrist": execution_wrist_as_dict(),
        "debug_overrides": {},
    }
    payload.update(overrides)
    return ResolvedProfile(**payload)


def debug_profile(**overrides: Any) -> ResolvedProfile:
    base = production_profile(
        name="debug",
        loop="open",
        send_bed_pull=False,
        live_geometry_policy="warn",
    )
    data = asdict(base)
    data.update(overrides)
    return ResolvedProfile(**data)


def marker_policy_from_flags(*, detect_top_markers: bool, recapture_pose: bool) -> str:
    if detect_top_markers or recapture_pose:
        return "force_redetect"
    return "reuse_pack"


def registration_policy_from_flags(*, trial_wrist_match: bool) -> str:
    if trial_wrist_match:
        return "trial_check"
    return "keep_session"


def start_here_map(loop: str) -> dict[int, str]:
    if loop == "closed-compare":
        return dict(COMPARE_START_HERE)
    if loop in ("closed",):
        return dict(CLOSED_START_HERE)
    return dict(OPEN_START_HERE)


def named_stage_from_start_here(loop: str, start_here: int) -> str:
    mapping = start_here_map(loop)
    if start_here not in mapping:
        raise ValueError(
            f"--start-here {start_here} is not valid for loop={loop}. "
            f"Use {sorted(mapping)} or --start-stage {', '.join(NAMED_STAGES)}"
        )
    return mapping[start_here]


def argv_has(argv: Sequence[str], *flags: str) -> bool:
    for arg in argv:
        if arg.split("=", 1)[0] in flags:
            return True
    return False


def add_trial_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the public run_trial CLI. Keep flag names stable."""

    parser.add_argument("--subject-id", type=str, default="TEST", required=True)
    parser.add_argument("--pose-num", type=str, default="TEST", required=True)
    parser.add_argument(
        "--tl-code",
        type=str,
        default="random",
        help="Target limb. Default: random among "
        + ",".join(str(t) for t in STUDY_TLS)
        + ".",
    )
    parser.add_argument("--start-here", type=int, default=0)
    parser.add_argument(
        "--start-stage",
        type=str,
        default="",
        choices=("",) + NAMED_STAGES,
        help="Named stage to start from. Preferred over --start-here.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume <trial-dir|latest> from trial_progress.json.",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="debug",
        help="debug keeps the legacy open-loop CLI. production: closed loop, "
        "SAM2, random study TL, reject geometry, send /bed_pull.",
    )
    parser.add_argument("--sub-id", type=int, default=None)
    parser.add_argument("--manikin", type=str, default="0")
    parser.add_argument(
        "--approach",
        type=str,
        choices=("recover", "dyn", "naive"),
        default="recover",
        help="recover is the new two-stage radius planner; dyn is an alias",
    )
    parser.add_argument(
        "--loop",
        choices=("open", "closed", "closed-compare"),
        default=None,
        help="Default comes from --profile (production=closed, debug=open).",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--uncover-model-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-number", type=int, default=None)
    parser.add_argument("--uncover-checkpoint-number", type=int, default=None)
    parser.add_argument("--residual-model-dir", type=Path, default=None)
    parser.add_argument("--residual-checkpoint-number", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-fevals", type=int, default=150)
    parser.add_argument("--uncover-max-fevals", type=int, default=None)
    parser.add_argument("--entropy-weight", type=float, default=200.0)
    parser.add_argument("--entropy-grid-size", type=float, default=0.05)
    parser.add_argument(
        "--execute-uncover",
        choices=("prompt", "skip"),
        default="prompt",
    )
    parser.add_argument("--max-points", type=int, default=1061)
    parser.add_argument("--ceiling-serial", type=int, default=0)
    parser.add_argument(
        "--camera-roles",
        type=str,
        default="ceiling,side_left,side_right",
    )
    parser.add_argument("--camera-extrinsics", type=str, default=None)
    parser.add_argument(
        "--mask-backend",
        choices=("sam2", "hsv"),
        default="sam2",
    )
    parser.add_argument("--require-camera-calibration", action="store_true")
    parser.add_argument(
        "--merge-mode",
        choices=("auto", "union", "top_supported", "ceiling_primary"),
        default="auto",
    )
    parser.add_argument(
        "--support-radius-3d",
        "--support-radius-xy",
        dest="support_radius",
        type=float,
        default=0.03,
    )
    parser.add_argument("--pcd-exposure", type=int, default=40)
    parser.add_argument("--pcd-gain", type=int, default=25)
    parser.add_argument("--require-aruco", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--execute-source",
        choices=("pred", "sensor"),
        default=None,
    )
    parser.add_argument("--a-is", choices=("pred", "sensor"), default=None)
    parser.add_argument("--trial-id", type=str, default=None)
    parser.add_argument("--skip-real-f1", action="store_true")
    parser.add_argument(
        "--ros2-execute",
        dest="ros2_execute",
        action="store_true",
        default=None,
        help="After overlay confirm, preflight and send /bed_pull.",
    )
    parser.add_argument(
        "--no-ros2-execute",
        dest="ros2_execute",
        action="store_false",
        help="Plan only; do not send /bed_pull.",
    )
    parser.add_argument(
        "--cma-bounds",
        choices=("symmetric", "asymmetric"),
        default=DEFAULT_CMA_BOUNDS,
    )
    parser.add_argument("--arm-max-m", type=float, default=PLANNER_ARM_MAX_M)
    parser.add_argument("--stretch-reach-snapshot", type=Path, default=None)
    parser.add_argument("--no-stretch-reach", action="store_true")
    parser.add_argument("--resnapshot-stretch", action="store_true")
    parser.add_argument("--skip-prepare-wrist-down", action="store_true")
    parser.add_argument("--skip-trial-wrist-match", action="store_true")
    parser.add_argument("--trial-wrist-match", action="store_true")
    parser.add_argument("--detect-top-markers", action="store_true")
    parser.add_argument(
        "--marker-policy",
        choices=MARKER_POLICIES,
        default=None,
        help="0–3 strategy. Default reuse_pack unless --detect-top-markers.",
    )
    parser.add_argument(
        "--registration-policy",
        choices=REGISTRATION_POLICIES,
        default=None,
        help="136 strategy. Default keep_session unless --trial-wrist-match.",
    )
    parser.add_argument("--reuse-subject-body", action="store_true")
    parser.add_argument("--skip-return-home", action="store_true")
    parser.add_argument("--stretch-ssh", type=str, default="")
    parser.add_argument("--coord-contract", action="store_true")
    parser.add_argument("--validate-fusion", action="store_true")
    parser.add_argument("--allow-unfrozen-session", action="store_true")
    parser.add_argument("--reuse-human-pose", type=Path, default=None)
    parser.add_argument("--no-reuse-human-pose", action="store_true")
    parser.add_argument("--recapture-pose", action="store_true")
    parser.add_argument(
        "--remeasure-body",
        dest="remeasure_body",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-remeasure-body",
        dest="remeasure_body",
        action="store_false",
    )
    parser.add_argument(
        "--live-geometry-policy",
        choices=("reject", "warn"),
        default=None,
    )
    parser.add_argument(
        "--recipe",
        choices=("verify-only", "contact-only", "grasp-only", "full-pull"),
        default="full-pull",
    )
    return parser


def resolve_profile(args: argparse.Namespace, argv: Sequence[str]) -> ResolvedProfile:
    base = production_profile() if args.profile == "production" else debug_profile()
    marker = args.marker_policy or marker_policy_from_flags(
        detect_top_markers=bool(args.detect_top_markers),
        recapture_pose=bool(args.recapture_pose),
    )
    registration = args.registration_policy or registration_policy_from_flags(
        trial_wrist_match=bool(args.trial_wrist_match),
    )
    tl_mode = "random" if str(args.tl_code) == "random" else "fixed"
    loop = args.loop if args.loop is not None else base.loop
    send_bed_pull = (
        base.send_bed_pull if args.ros2_execute is None else bool(args.ros2_execute)
    )
    live_policy = (
        base.live_geometry_policy
        if args.live_geometry_policy is None
        else args.live_geometry_policy
    )
    overrides: dict[str, Any] = {}
    tracked = {
        "loop": loop,
        "send_bed_pull": send_bed_pull,
        "tl_code_mode": tl_mode,
        "remeasure_body": bool(args.remeasure_body),
        "mask_backend": args.mask_backend,
        "detect_top_markers": bool(args.detect_top_markers),
        "trial_wrist_match": bool(args.trial_wrist_match),
        "marker_policy": marker,
        "registration_policy": registration,
        "live_geometry_policy": live_policy,
        "controller": "streaming",
        "recipe": args.recipe,
        "cma_bounds": args.cma_bounds,
        "arm_max_m": float(args.arm_max_m),
        "camera_roles": args.camera_roles,
        "approach": args.approach,
    }
    for key, value in tracked.items():
        if getattr(base, key) != value:
            overrides[key] = value
    data = asdict(base)
    data.update(tracked)
    data["debug_overrides"] = overrides
    data["name"] = args.profile
    return ResolvedProfile(**data)


def identity_fields(
    *,
    args: argparse.Namespace,
    profile: ResolvedProfile,
    target_limb_code: str,
    sub_id: int,
    pose_dir: Path,
    session_hashes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "trial_id": args.trial_id or pose_dir.name,
        "subject_id": args.subject_id,
        "pose_num": str(args.pose_num),
        "tl_code": int(target_limb_code),
        "tl_code_mode": profile.tl_code_mode,
        "sub_id": int(sub_id),
        "loop": profile.loop,
        "approach": profile.approach,
        "profile": profile.to_dict(),
        "mask_backend": profile.mask_backend,
        "camera_roles": profile.camera_roles,
        "marker_policy": profile.marker_policy,
        "registration_policy": profile.registration_policy,
        "remeasure_body": profile.remeasure_body,
        "send_bed_pull": profile.send_bed_pull,
        "wrist": profile.wrist,
        "session_hashes": dict(session_hashes or {}),
        "debug_overrides": dict(profile.debug_overrides),
    }
