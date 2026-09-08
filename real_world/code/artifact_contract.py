"""Session / pose / capture artifact contracts.

New captures must not emit IMAGE-frame clouds as bed-frame, and must not
silently fall back to calibration filters when a session is frozen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from session_paths import atomic_write_json, sha256_file
from trial_config import MARKER_POLICIES, REGISTRATION_POLICIES

CAPTURE_CONTRACT_NAME = "capture_contract.json"
POSE_REQUIRED = (
    "human_pose.pkl",
    "body_info.pkl",
    "sim_origin_data.pkl",
    "uncovered_rgb.png",
    "canonical_bed_frame.json",
)
CAPTURE_ROLES = ("ceiling", "side_left", "side_right")


class ArtifactError(RuntimeError):
    """Contract violation. Safe to show the operator as-is."""


def normalize_marker_policy(value: object) -> str:
    text = str(value or "reuse_pack").strip()
    if text not in MARKER_POLICIES:
        raise ArtifactError(
            f"marker_policy={value!r} is not one of {MARKER_POLICIES}"
        )
    return text


def normalize_registration_policy(value: object) -> str:
    text = str(value or "keep_session").strip()
    if text not in REGISTRATION_POLICIES:
        raise ArtifactError(
            f"registration_policy={value!r} is not one of {REGISTRATION_POLICIES}"
        )
    return text


def require_file(path: Path, *, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"{label} missing: {path}")
    return path


def validate_pose_pack(pose_dir: Path, *, require_body: bool = True) -> list[str]:
    missing: list[str] = []
    names = POSE_REQUIRED if require_body else tuple(
        name for name in POSE_REQUIRED if name != "body_info.pkl"
    )
    for name in names:
        if not (Path(pose_dir) / name).is_file():
            missing.append(name)
    return missing


def assert_pose_pack(pose_dir: Path, *, require_body: bool = True) -> None:
    missing = validate_pose_pack(pose_dir, require_body=require_body)
    if missing:
        raise ArtifactError(
            f"{pose_dir} is missing pose pack files: {', '.join(missing)}"
        )


def assert_bed_frame_transform(transform: object, *, role: str) -> None:
    if transform is None:
        raise ArtifactError(
            f"REJECT: {role} has no T_bed_camera. "
            "IMAGE-frame PCD cannot be written as bed-frame output. "
            "Calibrate the camera or omit this role."
        )


def assert_session_filters(
    filters: Mapping[str, Path],
    *,
    roles: tuple[str, ...] = CAPTURE_ROLES,
    allow_calibration_fallback: bool = False,
) -> None:
    if allow_calibration_fallback:
        return
    missing = [
        role
        for role in roles
        if role not in filters or not Path(filters[role]).is_file()
    ]
    if missing:
        raise ArtifactError(
            "REJECT: session filter JSON missing for "
            + ", ".join(missing)
            + ". Do not fall back to calibration/. Freeze session/zed first."
        )


def role_paths(capture_dir: Path, role: str) -> dict[str, Path]:
    dest = Path(capture_dir)
    return {
        "rgb": dest / f"covered_rgb_{role}.png",
        "mask": dest / f"blanket_mask_{role}.png",
        "hsv_mask": dest / f"blanket_mask_{role}_hsv.png",
        "pcd": dest / f"pcd_filtered_{role}.pcd",
    }


def build_capture_contract(
    capture_dir: Path,
    *,
    metadata: Mapping[str, Any],
    session_hashes: Mapping[str, Any] | None = None,
    sam_weights: str | None = None,
    sam_device: str | None = None,
) -> dict[str, Any]:
    dest = Path(capture_dir)
    roles = dict(metadata.get("roles") or {})
    role_files: dict[str, Any] = {}
    for role, info in roles.items():
        paths = role_paths(dest, role)
        if info.get("T_bed_camera") is None:
            raise ArtifactError(
                f"REJECT: {role} metadata has no T_bed_camera; "
                "refusing to record an IMAGE-frame cloud as bed-frame."
            )
        role_files[role] = {
            "rgb": str(paths["rgb"]) if paths["rgb"].is_file() else None,
            "rgb_sha256": sha256_file(paths["rgb"]),
            "mask": str(paths["mask"]) if paths["mask"].is_file() else None,
            "mask_sha256": sha256_file(paths["mask"]),
            "hsv_mask": str(paths["hsv_mask"]) if paths["hsv_mask"].is_file() else None,
            "hsv_mask_sha256": sha256_file(paths["hsv_mask"]),
            "pcd": str(paths["pcd"]) if paths["pcd"].is_file() else None,
            "pcd_sha256": sha256_file(paths["pcd"]),
            "filter_json": info.get("filter_json"),
            "filter_sha256": sha256_file(Path(info["filter_json"]))
            if info.get("filter_json")
            else None,
            "T_bed_camera": info.get("T_bed_camera"),
            "mask_backend": info.get("mask_backend"),
            "hsv_px": info.get("hsv_px"),
            "mask_px": info.get("mask_px"),
            "num_points_bed": info.get("num_points_bed"),
        }
    merged = dest / "blanket_pcd.pcd"
    contract = {
        "schema": "robe.capture_contract.v1",
        "capture_dir": str(dest.resolve()),
        "frame": metadata.get("frame"),
        "mask_backend": metadata.get("mask_backend"),
        "extrinsics": metadata.get("extrinsics"),
        "extrinsics_sha256": sha256_file(Path(metadata["extrinsics"]))
        if metadata.get("extrinsics")
        else None,
        "roles": role_files,
        "merged_pcd": str(merged) if merged.is_file() else None,
        "merged_pcd_sha256": sha256_file(merged),
        "merged_points": (metadata.get("canonical") or {}).get("num_points_after_crop"),
        "canonical": metadata.get("canonical"),
        "merge": metadata.get("merge"),
        "session_hashes": dict(session_hashes or {}),
        "sam_weights": sam_weights,
        "sam_device": sam_device,
    }
    return contract


def write_capture_contract(capture_dir: Path, contract: Mapping[str, Any]) -> Path:
    dest = Path(capture_dir) / CAPTURE_CONTRACT_NAME
    atomic_write_json(dest, contract)
    return dest


def load_capture_contract(capture_dir: Path) -> dict[str, Any]:
    path = Path(capture_dir) / CAPTURE_CONTRACT_NAME
    if not path.is_file():
        raise ArtifactError(f"missing {path}")
    return json.loads(path.read_text())


def validate_capture_contract(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("frame") not in {"canonical_bed", "bed"}:
        errors.append(f"frame={contract.get('frame')!r} is not canonical_bed")
    roles = contract.get("roles") or {}
    if not roles:
        errors.append("no camera roles in capture contract")
    for role, info in roles.items():
        if not info.get("T_bed_camera"):
            errors.append(f"{role} missing T_bed_camera")
        if not info.get("rgb_sha256"):
            errors.append(f"{role} missing RGB")
        if not info.get("mask_sha256"):
            errors.append(f"{role} missing SAM/HSV mask")
        if not info.get("pcd_sha256"):
            errors.append(f"{role} missing per-view PCD")
        backend = str(info.get("mask_backend") or contract.get("mask_backend") or "")
        if backend == "sam2" and not info.get("hsv_mask_sha256"):
            errors.append(f"{role} SAM2 capture missing HSV seed mask")
    if not contract.get("merged_pcd_sha256"):
        errors.append("merged blanket_pcd.pcd missing")
    canonical = contract.get("canonical") or {}
    if not canonical.get("source") and not canonical.get("path"):
        if "source" not in canonical and "skipped" not in canonical:
            errors.append("canonical source not recorded")
    return errors


def assert_capture_contract(capture_dir: Path) -> dict[str, Any]:
    contract = load_capture_contract(capture_dir)
    errors = validate_capture_contract(contract)
    if errors:
        raise ArtifactError(
            f"{capture_dir}/{CAPTURE_CONTRACT_NAME}: " + "; ".join(errors)
        )
    return contract
