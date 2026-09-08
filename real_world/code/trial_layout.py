"""Canonical on-disk layout for a closed-compare trial pose directory.

Root stays operator-facing (pose, overlays, manifest). Capture and planner
artifacts go in stage subdirectories. Readers accept the old flat layout so
existing trials keep working.

Same ``pose_num`` inside one session: copy skeleton, diameters, uncovered RGB,
sim_origin, and pose viz from ``session/poses/pose_<n>/``. Trials still keep
their own copies so STUDY_DATA is self-contained. Do not copy from another
session. Camera / robot / 136 stay under ``session/zed`` and ``session/stretch``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

POSE_PACK_FILES = (
    "human_pose.pkl",
    "sim_origin_data.pkl",
    "uncovered_rgb.png",
    "canonical_bed_frame.json",
)
POSE_PACK_BODY = "body_info.pkl"
POSE_PACK_VIZ = (
    "all_body_points_over_rgb.png",
    "uncovered_aruco_viz.png",
)
POSE_PACK_COPY_FILES = POSE_PACK_FILES + (POSE_PACK_BODY,) + POSE_PACK_VIZ


def initial_dir(pose_dir: Path) -> Path:
    return Path(pose_dir) / "initial"


def uncover_dir(pose_dir: Path) -> Path:
    return Path(pose_dir) / "uncover"


def intermediate_dir(pose_dir: Path) -> Path:
    return Path(pose_dir) / "intermediate"


def final_dir(pose_dir: Path) -> Path:
    return Path(pose_dir) / "final"


def recover_pred_dir(pose_dir: Path) -> Path:
    return Path(pose_dir) / "recover_pred"


def recover_pred_lowdrag_dir(pose_dir: Path) -> Path:
    """Pred-graph Recover planned with PredLowDrag voxel FT (ablation)."""

    return Path(pose_dir) / "recover_pred_lowdrag"


def recover_sensor_dir(pose_dir: Path) -> Path:
    return Path(pose_dir) / "recover_sensor"


def recover_snap_dir(pose_dir: Path) -> Path:
    return Path(pose_dir) / "recover_snap"


def recover_density_dir(pose_dir: Path, mode: str) -> Path:
    names = {
        "density-1": "recover_density1",
        "density-50": "recover_density50",
        "density-full": "recover_density_full",
    }
    if mode not in names:
        raise ValueError(f"Unknown density mode: {mode}")
    return Path(pose_dir) / names[mode]


def first_existing(candidates: list[Path], *, required: bool = True) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    if required:
        listed = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Missing required artifact. Looked in: {listed}")
    return candidates[0]


def resolve_initial_pcd(pose_dir: Path) -> Path:
    pose_dir = Path(pose_dir)
    return first_existing(
        [
            initial_dir(pose_dir) / "blanket_pcd.pcd",
            pose_dir / "blanket_pcd.pcd",
        ]
    )


def resolve_ceiling_rgb(pose_dir: Path, stage: str = "initial") -> Path:
    pose_dir = Path(pose_dir)
    if stage == "intermediate":
        return first_existing(
            [
                intermediate_dir(pose_dir) / "covered_rgb_ceiling.png",
                pose_dir / "covered_rgb_ceiling.png",
            ]
        )
    if stage == "final":
        return first_existing(
            [
                final_dir(pose_dir) / "covered_rgb_ceiling.png",
                pose_dir / "covered_rgb_ceiling.png",
            ]
        )
    return first_existing(
        [
            initial_dir(pose_dir) / "covered_rgb_ceiling.png",
            pose_dir / "covered_rgb_ceiling.png",
        ]
    )


def resolve_uncover_file(
    pose_dir: Path, name: str, *, required: bool = True
) -> Path:
    pose_dir = Path(pose_dir)
    return first_existing(
        [
            uncover_dir(pose_dir) / name,
            pose_dir / name,
        ],
        required=required,
    )


def resolve_canonical_frame(pose_dir: Path, *, required: bool = False) -> Path:
    pose_dir = Path(pose_dir)
    return first_existing(
        [
            pose_dir / "canonical_bed_frame.json",
            pose_dir.parent / "canonical_bed_frame.json",
            initial_dir(pose_dir) / "canonical_bed_frame.json",
        ],
        required=required,
    )


def resolve_stretch_reach_snapshot(
    pose_dir: Path, *, required: bool = False
) -> Path:
    pose_dir = Path(pose_dir)
    return first_existing(
        [
            pose_dir / "stretch_reach_snapshot.json",
            initial_dir(pose_dir) / "stretch_reach_snapshot.json",
        ],
        required=required,
    )


def uncover_write_dir(pose_dir: Path) -> Path:
    dest = uncover_dir(pose_dir)
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def session_pose_dir(session_dir: Path, pose_num: str | int) -> Path:
    return Path(session_dir) / "poses" / f"pose_{pose_num}"


def pose_pack_complete(directory: Path, *, require_body: bool = False) -> bool:
    directory = Path(directory)
    names = POSE_PACK_FILES + ((POSE_PACK_BODY,) if require_body else ())
    return all((directory / name).is_file() for name in names)


def _optional_current_session_dir() -> Path | None:
    try:
        from session_paths import resolve_session_dir_once

        return resolve_session_dir_once()
    except FileNotFoundError:
        return None


def resolve_body_info(
    pose_dir: Path,
    subject_dir: Path,
    *,
    session_dir: Path | None = None,
    pose_num: str | int | None = None,
    allow_subject: bool = False,
    required: bool = True,
) -> Path:
    """Trial copy, then this session's pose pack.

    ``subject_dir/body_info.pkl`` is not used unless ``allow_subject``
    (``--reuse-subject-body``). Silent subject fallback would leak another
    exp's diameters into this trial.
    """

    pose_dir = Path(pose_dir)
    subject_dir = Path(subject_dir)
    if session_dir is None:
        session_dir = _optional_current_session_dir()
    candidates = [pose_dir / POSE_PACK_BODY]
    if session_dir is not None:
        num = pose_num
        if num is None and pose_dir.name.startswith("pose_"):
            num = pose_dir.name.split("_")[1]
        if num is not None:
            candidates.append(session_pose_dir(session_dir, num) / POSE_PACK_BODY)
    if allow_subject:
        candidates.append(subject_dir / POSE_PACK_BODY)
    try:
        return first_existing(candidates, required=required)
    except FileNotFoundError:
        looked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Missing body_info.pkl (trial then this exp's poses/pose_n). "
            f"Looked in: {looked}. Measure diameters for this session, or pass "
            "--reuse-subject-body to copy the subject-level file."
        ) from None


def ensure_trial_body_info(
    pose_dir: Path,
    subject_dir: Path,
    *,
    session_dir: Path | None = None,
    pose_num: str | int | None = None,
    allow_subject: bool = False,
) -> Path | None:
    """Copy body diameters into the trial from this exp's pack.

    Subject-level ``body_info.pkl`` is copied only when ``allow_subject``.
    """

    pose_dir = Path(pose_dir)
    dest = pose_dir / POSE_PACK_BODY
    if dest.is_file():
        return dest
    try:
        src = resolve_body_info(
            pose_dir,
            subject_dir,
            session_dir=session_dir,
            pose_num=pose_num,
            allow_subject=allow_subject,
            required=True,
        )
    except FileNotFoundError:
        return None
    if src.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return dest


def trial_dirs_for_pose(
    subject_dir: Path, pose_num: str | int, *, exclude: Path | None = None
) -> list[Path]:
    subject = Path(subject_dir)
    prefix = f"pose_{pose_num}_TL"
    skip = exclude.resolve() if exclude is not None else None
    found: list[Path] = []
    if not subject.is_dir():
        return found
    for path in subject.iterdir():
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        if skip is not None and path.resolve() == skip:
            continue
        found.append(path)
    found.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return found


def resolve_pose_pack_source(
    subject_dir: Path,
    pose_num: str | int,
    *,
    exclude: Path | None = None,
) -> Path | None:
    """Newest complete sibling trial ``pose_<n>_TL…``. No extra folder."""

    for trial in trial_dirs_for_pose(subject_dir, pose_num, exclude=exclude):
        if pose_pack_complete(trial):
            return trial
    return None


def copy_pose_pack_files(
    src: Path, dest: Path, *, overwrite: bool = False
) -> list[str]:
    """Copy skeleton, diameters, origin, and pose viz. Not parking or PCD."""

    src = Path(src)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in POSE_PACK_COPY_FILES:
        source = src / name
        target = dest / name
        if not source.is_file():
            continue
        if target.is_file() and not overwrite:
            continue
        shutil.copy2(source, target)
        copied.append(name)
    return copied


def export_pose_pack_to_session(
    pose_dir: Path,
    session_dir: Path,
    pose_num: str | int,
    *,
    overwrite: bool = True,
) -> list[str]:
    """Write this trial's pose pack into ``session/poses/pose_<n>/``."""

    dest = session_pose_dir(session_dir, pose_num)
    return copy_pose_pack_files(pose_dir, dest, overwrite=overwrite)


def apply_pose_pack(
    pose_dir: Path,
    subject_dir: Path,
    pose_num: str | int,
    *,
    prefer: Path | None = None,
    session_dir: Path | None = None,
    copy_sibling: bool = True,
) -> tuple[Path | None, list[str]]:
    """Fill a new trial from the current session pose pack, or ``--reuse-human-pose``.

    Default reuse is session-scoped. Other-exp sibling trials are not used
    unless ``session_dir`` is None (legacy / no session).
    """

    pose_dir = Path(pose_dir)
    copied: list[str] = []
    used: Path | None = None
    if prefer is not None:
        source = Path(prefer).expanduser().resolve()
        if source.is_file():
            if source.name == "human_pose.pkl":
                dest = pose_dir / "human_pose.pkl"
                if not dest.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
                    copied.append("human_pose.pkl")
            source = source.parent
        if source.is_dir():
            copied.extend(copy_pose_pack_files(source, pose_dir, overwrite=False))
            used = source
    if not copy_sibling:
        return used, list(dict.fromkeys(copied))
    if session_dir is not None:
        fill = session_pose_dir(session_dir, pose_num)
        if pose_pack_complete(fill, require_body=True):
            copied.extend(copy_pose_pack_files(fill, pose_dir, overwrite=False))
            if used is None:
                used = fill
        return used, list(dict.fromkeys(copied))
    fill = resolve_pose_pack_source(subject_dir, pose_num, exclude=pose_dir)
    if fill is not None:
        copied.extend(copy_pose_pack_files(fill, pose_dir, overwrite=False))
        if used is None:
            used = fill
    return used, list(dict.fromkeys(copied))
