"""Shared CMA search-box and Stretch EE workspace limits.

Yesterday's working bedside run used:

* CMA policy box ``[-1, 1]^4`` (symmetric / sim)
* planner arm cap ``0.50 m`` (2 cm inside hardware ``0.52 m``)
* base/layout snapshot; reachability uses execution wrist-down
  ``link_grasp_center``, not the parking EE TF

``run_trial`` closed and open Recover loops reuse this module instead of
passing those flags by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from stretch_cartesian import StretchWorkspace
from stretch_reachability import (
    ReachabilityResult,
    check_bed_pull_reachability,
    required_wrist_extension_xy,
)

PLANNER_ARM_MAX_M = 0.50
HARDWARE_ARM_MAX_M = 0.52
# 1 mm numerical slack only. Over-max arm rejects; do not clip the pickle.
EXEC_ARM_SLACK_M = 0.001
# After WRIST_DOWN, live arm=0 EE must match the execution model this close.
LIVE_GEOMETRY_MATCH_M = 0.02
DEFAULT_CMA_BOUNDS = "symmetric"
BED_X_LIMITS = (-0.55, 0.55)
# Marker board long edge is ±0.925 m. Sim ACTION_SCALE y=1.05 let CMA
# place release past the foot (TL15 hit 1.038 m). Keep actions inside
# the mattress with a few centimetres of inset.
BED_Y_LIMITS = (-0.90, 0.90)
SNAPSHOT_NAME = "stretch_reach_snapshot.json"
CMA_BOUNDS_NAME = "stretch_cma_bounds.json"
# Same as recover_runtime.ACTION_SCALE; keep local to avoid importing torch.
DEFAULT_ACTION_SCALE = np.asarray([0.44, 1.05, 0.44, 1.05], dtype=np.float64)


def planner_workspace(arm_max_m: float = PLANNER_ARM_MAX_M) -> StretchWorkspace:
    """Workspace used by CMA and preflight (tighter than hardware)."""

    return StretchWorkspace(arm_max_m=float(arm_max_m))


def hardware_workspace() -> StretchWorkspace:
    return StretchWorkspace(arm_max_m=HARDWARE_ARM_MAX_M)


def cma_policy_bounds(mode: str = DEFAULT_CMA_BOUNDS) -> tuple[np.ndarray, np.ndarray]:
    """Normalized CMA box. ``asymmetric`` is the legacy real-world half-bed box."""

    mode = str(mode)
    if mode == "asymmetric":
        upper = np.ones(4, dtype=np.float64)
        lower = -np.asarray([0.0, 0.5, 0.0, 1.0], dtype=np.float64)
        return lower, upper
    if mode != "symmetric":
        raise ValueError(f"unknown CMA bounds mode: {mode}")
    ones = np.ones(4, dtype=np.float64)
    return -ones, ones


def clip_action_to_bounds(
    action: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    out = np.asarray(action, dtype=np.float64).copy()
    return np.minimum(np.maximum(out, lower), upper)


def snapshot_path(pose_dir: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    return Path(pose_dir).resolve() / SNAPSHOT_NAME


def load_reach_snapshot(path: Path) -> dict:
    import json

    path = Path(path)
    payload = json.loads(path.read_text())
    required = (
        "T_odom_layout",
        "T_odom_base",
        "ee_base",
        "wrist_extension",
        "joint_lift",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path} missing {missing}")
    return payload


@dataclass(frozen=True)
class CmaReachContext:
    snapshot: dict | None
    snapshot_path: Path | None
    workspace: StretchWorkspace
    frame: Any
    arm_max_m: float

    @property
    def enabled(self) -> bool:
        return self.snapshot is not None and self.frame is not None

    def check(self, action_bed: np.ndarray) -> ReachabilityResult:
        if not self.enabled:
            return ReachabilityResult(True, "no_snapshot", 0)
        return check_bed_pull_reachability(
            action_bed,
            frame=self.frame,
            snapshot=self.snapshot,
            workspace=self.workspace,
        )


def load_cma_reach(
    *,
    pose_dir: Path,
    snapshot_path_or_none: Path | None,
    arm_max_m: float = PLANNER_ARM_MAX_M,
    require: bool = False,
) -> CmaReachContext:
    """Load parking snapshot + canonical frame for CMA / preflight."""

    from canonical_bed import maybe_load_canonical_frame

    workspace = planner_workspace(arm_max_m)
    if snapshot_path_or_none is None:
        if require:
            raise FileNotFoundError(
                f"Stretch reach snapshot required under {pose_dir}"
            )
        return CmaReachContext(None, None, workspace, None, float(arm_max_m))
    path = Path(snapshot_path_or_none).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stretch reach snapshot missing: {path}")
    frame = maybe_load_canonical_frame(Path(pose_dir))
    if frame is None:
        raise RuntimeError(
            f"canonical_bed_frame.json required with stretch reach snapshot "
            f"({path})"
        )
    return CmaReachContext(
        load_reach_snapshot(path),
        path,
        workspace,
        frame,
        float(arm_max_m),
    )


def into_bed_canonical_x_limits(
    snapshot: dict,
    frame: Any,
    arm_max_m: float,
    y_limits: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Canonical X strip whose whole CMA Y rectangle stays under ``arm_max_m``.

    ``q_arm(x,y)`` is affine in canonical XY (fixed cloth Z, rigid T).
    Iso-q lines are therefore tilted when the base is not a perfect 90°
    park, so a strip taken only at the EE's along-bed Y leaks at the Y
    corners. Invert ``q=0`` and ``q=arm_max`` at the worst Y of ``y_limits``.
    """

    y_lo, y_hi = BED_Y_LIMITS if y_limits is None else y_limits
    y_lo, y_hi = float(min(y_lo, y_hi)), float(max(y_lo, y_hi))
    workspace = planner_workspace(arm_max_m)

    def q_arm(x: float, y: float) -> float:
        return required_wrist_extension_xy(
            (x, y),
            frame=frame,
            snapshot=snapshot,
            workspace=workspace,
        )

    q00 = q_arm(0.0, 0.0)
    a = q_arm(1.0, 0.0) - q00
    b = q_arm(0.0, 1.0) - q00
    if abs(a) < 1e-4:
        raise RuntimeError(
            "into-bed arm axis is not canonical X; "
            "park with +X_base along the bed and the arm toward the mattress."
        )
    q_far = float(arm_max_m)
    q_near = 0.0
    y_for_far = y_hi if b >= 0.0 else y_lo
    y_for_near = y_lo if b >= 0.0 else y_hi
    x_far = (q_far - b * y_for_far - q00) / a
    x_near = (q_near - b * y_for_near - q00) / a
    x_lo, x_hi = (min(x_near, x_far), max(x_near, x_far))
    for x in (x_lo, x_hi):
        for y in (y_lo, y_hi):
            action = np.array([x, y, x, y], dtype=np.float64)
            result = check_bed_pull_reachability(
                action,
                frame=frame,
                snapshot=snapshot,
                workspace=workspace,
            )
            if not result.reachable:
                raise RuntimeError(
                    "X-strip corner failed check_bed_pull_reachability "
                    f"({result.reason}). into_bed_canonical_x_limits is wrong."
                )
    return (x_lo, x_hi)


def restrict_cma_bounds_to_reach(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    snapshot: dict,
    frame: Any,
    arm_max_m: float,
    action_scale: np.ndarray | None = None,
    mirror_x: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect the CMA box with the execution-wrist-down X strip.

    This is the RoBE workspace restriction: grasp/release X stay in the
    arm stroke. CMA does not run a per-candidate trajectory check.
    """

    scale = (
        DEFAULT_ACTION_SCALE
        if action_scale is None
        else np.asarray(action_scale, dtype=np.float64).reshape(4)
    )
    out_lo = np.asarray(lower, dtype=np.float64).reshape(4).copy()
    out_hi = np.asarray(upper, dtype=np.float64).reshape(4).copy()
    for index in (1, 3):
        n_lo = float(BED_Y_LIMITS[0]) / float(scale[index])
        n_hi = float(BED_Y_LIMITS[1]) / float(scale[index])
        if n_lo > n_hi:
            n_lo, n_hi = n_hi, n_lo
        lo = max(float(out_lo[index]), n_lo)
        hi = min(float(out_hi[index]), n_hi)
        if lo < hi:
            out_lo[index] = lo
            out_hi[index] = hi
    y_lo = min(
        float(out_lo[1]) * float(scale[1]), float(out_lo[3]) * float(scale[3])
    )
    y_hi = max(
        float(out_hi[1]) * float(scale[1]), float(out_hi[3]) * float(scale[3])
    )
    x_lo, x_hi = into_bed_canonical_x_limits(
        snapshot, frame, arm_max_m, y_limits=(y_lo, y_hi)
    )
    if mirror_x:
        x_lo, x_hi = -x_hi, -x_lo
    n_lo = float(x_lo) / float(scale[0])
    n_hi = float(x_hi) / float(scale[0])
    if n_lo > n_hi:
        n_lo, n_hi = n_hi, n_lo
    for index in (0, 2):
        lo = max(float(out_lo[index]), n_lo)
        hi = min(float(out_hi[index]), n_hi)
        if lo < hi:
            out_lo[index] = lo
            out_hi[index] = hi
    return out_lo, out_hi


def snapshot_uses_live_tf(snapshot: dict | Path) -> bool:
    """True when the JSON was taken under executor live-TF broadcast."""

    if isinstance(snapshot, Path):
        if not snapshot.is_file():
            return False
        snapshot = json.loads(Path(snapshot).read_text())
    return bool(snapshot.get("tf_live"))


def write_restricted_cma_bounds(
    dest: Path,
    *,
    snapshot: dict,
    frame: Any,
    arm_max_m: float,
    bounds_mode: str = DEFAULT_CMA_BOUNDS,
    mirror_x: bool = True,
) -> dict:
    """Write the loop/CMA box next to the parking snapshot."""

    lower, upper = cma_policy_bounds(bounds_mode)
    rlo, rhi = restrict_cma_bounds_to_reach(
        lower,
        upper,
        snapshot=snapshot,
        frame=frame,
        arm_max_m=arm_max_m,
        action_scale=DEFAULT_ACTION_SCALE,
        mirror_x=mirror_x,
    )
    y_lo = min(float(rlo[1]), float(rlo[3])) * float(DEFAULT_ACTION_SCALE[1])
    y_hi = max(float(rhi[1]), float(rhi[3])) * float(DEFAULT_ACTION_SCALE[1])
    x_lo, x_hi = into_bed_canonical_x_limits(
        snapshot, frame, arm_max_m, y_limits=(y_lo, y_hi)
    )
    payload = {
        "source": "restrict_cma_bounds_to_reach",
        "ee_source": "execution_wrist_down",
        "mirror_x": bool(mirror_x),
        "arm_max_m": float(arm_max_m),
        "canonical_y_limits_m": [float(y_lo), float(y_hi)],
        "canonical_x_band_m": [float(x_lo), float(x_hi)],
        "bounds_min": rlo.tolist(),
        "bounds_max": rhi.tolist(),
    }
    dest = Path(dest)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def assert_selected_action_reachable(
    action_bed: np.ndarray,
    reach_ctx: CmaReachContext,
) -> ReachabilityResult:
    """One exact check after CMA. Failure means the X-strip is wrong."""

    if not reach_ctx.enabled:
        return ReachabilityResult(True, "no_snapshot", 0)
    result = reach_ctx.check(action_bed)
    if not result.reachable:
        raise RuntimeError(
            "Selected action is outside the execution workspace. "
            "The CMA X-strip does not match check_bed_pull_reachability "
            f"({result.reason}). Fix into_bed_canonical_x_limits. "
            "Do not clip the pickle."
        )
    return result
