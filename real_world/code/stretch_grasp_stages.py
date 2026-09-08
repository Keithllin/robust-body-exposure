"""Frozen BedPull / grasp-tune stage names and stop-after semantics.

``stop_after`` means the named stage has **finished successfully**, then
the executor returns. It never means “stop before entering”.
"""

from __future__ import annotations

from dataclasses import dataclass

STAGES = (
    "SAFE_LIFT",
    "RETRACT",
    "WRIST_DOWN",
    "SAFE_LIFT_RECHECK",
    "BASE_APPROACH",
    "ARM_APPROACH",
    "VERIFY_PREGRASP",
    "DESCEND_UNLATCH",
    "DESCEND_BREAKAWAY",
    "DESCEND_COARSE",
    "DESCEND_CONTACT",
    "GRASP_BUMP",
    "GRASP_OPEN",
    "GRASP_TAP2",
    "GRASP_CLOSE",
    "LIFTING",
    "PULLING",
    "RELEASING",
    "REVERSE_CONTACT",
    "REVERSE_BUMP",
    "REVERSE_OPEN",
    "REVERSE_TAP2",
    "REVERSE_CLOSE",
    "REVERSE_LIFT",
    "REVERSE_PULL",
    "REVERSE_PLACE",
    "RETRACTING",
    "RESET_EE",
    "DONE",
)

STOP_AFTER_ALLOWED = STAGES

# Result / fault codes. IntentionalStop is not FAULT_OK.
FAULT_OK = "OK"
FAULT_INTENTIONAL_STOP = "INTENTIONAL_STOP"
FAULT_GOAL_REJECT = "GOAL_REJECT"
FAULT_LIVE_GEOMETRY = "LIVE_GEOMETRY"
FAULT_WRIST_VERIFY = "WRIST_VERIFY"
FAULT_UNREACHABLE = "UNREACHABLE"
FAULT_PREFLIGHT = "PREFLIGHT"
FAULT_CANCELED = "CANCELED"
FAULT_CONTACT = "CONTACT"
FAULT_DRIVER = "DRIVER"
FAULT_DRY_RUN = "DRY_RUN"

ALLOWED_CONTROLLERS = ("streaming", "position")
LIVE_GEOMETRY_POLICIES = ("reject", "warn")

RECIPES = {
    "verify-only": "VERIFY_PREGRASP",
    "contact-only": "DESCEND_CONTACT",
    "grasp-only": "GRASP_CLOSE",
    "full-pull": "",
}


@dataclass(frozen=True)
class StageContract:
    name: str
    entry: tuple[str, ...]
    exit: tuple[str, ...]
    stop_ok: bool = True
    fault_codes: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()


STAGE_CONTRACTS: dict[str, StageContract] = {
    "SAFE_LIFT": StageContract(
        "SAFE_LIFT",
        entry=("goal accepted", "EE above cloth+margin"),
        exit=("joint_lift at approach plane",),
        side_effects=("joint_lift",),
    ),
    "RETRACT": StageContract(
        "RETRACT",
        entry=("SAFE_LIFT done",),
        exit=("arm <= approach.retract_arm_m",),
        side_effects=("joint_arm",),
    ),
    "WRIST_DOWN": StageContract(
        "WRIST_DOWN",
        entry=("arm clear of mast or will extend first",),
        exit=("yaw/pitch/roll match execution wrist",),
        fault_codes=(FAULT_WRIST_VERIFY,),
        side_effects=("joint_wrist_yaw", "joint_wrist_pitch", "joint_wrist_roll", "joint_arm"),
    ),
    "VERIFY_PREGRASP": StageContract(
        "VERIFY_PREGRASP",
        entry=("arm/base at hover",),
        exit=("grasp XY within pregrasp_reject_m",),
        fault_codes=(FAULT_PREFLIGHT,),
    ),
    "DESCEND_CONTACT": StageContract(
        "DESCEND_CONTACT",
        entry=("VERIFY_PREGRASP passed",),
        exit=("relative lift-effort unload",),
        fault_codes=(FAULT_CONTACT,),
        side_effects=("joint_lift",),
    ),
    "GRASP_CLOSE": StageContract(
        "GRASP_CLOSE",
        entry=("contact held",),
        exit=("gripper closed",),
        side_effects=("stretch_gripper",),
    ),
    "DONE": StageContract(
        "DONE",
        entry=("full pull completed",),
        exit=("retract + RESET_EE",),
        stop_ok=False,
        fault_codes=(FAULT_OK,),
    ),
}


class IntentionalStop(Exception):
    """Requested ``stop_after`` stage finished without fault."""

    def __init__(self, stage: str, detail: str = "", extra: dict | None = None):
        super().__init__(detail or stage)
        self.stage = stage
        self.detail = detail
        self.extra = extra or {}
        self.fault_code = FAULT_INTENTIONAL_STOP
INTENTIONAL_STOP_PREFIX = "INTENTIONAL_STOP_AFTER:"

# Snapshot at goal accept; the running goal never re-reads these live.
TUNE_PARAM_NAMES = (
    "pull.stop_after",
    "pull.recipe",
    "pull.dry_run",
    "pull.controller",
    "pull.clearance_above_bed_m",
    "live_geometry.policy",
    "pull.round_trip",
    "contact.threshold_mode",
    "contact.lift_effort",
    "contact.approach_effort",
    "contact.raise_effort",
    "contact.disable_effort",
    "contact.breakaway_m",
    "contact.effort_spike",
    "contact.max_descent_m",
    "contact.min_descent_m",
    "contact.cloth_margin_m",
    "contact.guard_band_m",
    "contact.lower_speed_m_s",
    "contact.lower_accel_m_s2",
    "contact.hover_positive_warn_pct",
    "contact.lift_effort_threshold",
    "contact.move_increment_m",
    "contact.lowest_allowed_m",
    "contact.cloth_below_estimate_m",
    "contact.log_trace",
    "contact.trace_path",
    "contact.coarse_standoff_m",
    "contact.coarse_speed_m_s",
    "contact.coarse_accel_m_s2",
    "contact.probe_step_m",
    "contact.probe_max_m",
    "contact.probe_speed_m_s",
    "contact.probe_accel_m_s2",
    "contact.baseline_settle_s",
    "contact.baseline_samples",
    "contact.window_samples",
    "contact.unload_drop_pct",
    "contact.unload_consecutive",
    "contact.compress_max_m",
    "contact.detect_above_cloth_m",
    "approach.retract_arm_m",
    "approach.clear_above_cloth_m",
    "workspace.arm_min_m",
    "workspace.arm_max_m",
    "workspace.lift_min_m",
    "workspace.lift_max_m",
    "approach.pregrasp_pass_m",
    "approach.pregrasp_reject_m",
    "approach.refine_passes",
    "grasp.lift_after_contact_m",
    "grasp.lower_after_open_m",
    "grasp.probe_extra_m",
    "grasp.pause_after_bump_s",
    "grasp.pause_after_probe_s",
    "grasp.pause_after_close_s",
    "grasp.gripper_open",
    "grasp.gripper_closed",
)

_RELEVANT = {
    "VERIFY_PREGRASP": (
        "approach.pregrasp_pass_m",
        "approach.pregrasp_reject_m",
        "approach.clear_above_cloth_m",
    ),
    "DESCEND_UNLATCH": ("contact.disable_effort",),
    "DESCEND_BREAKAWAY": (
        "contact.breakaway_m",
        "contact.disable_effort",
        "contact.lower_speed_m_s",
    ),
    "DESCEND_COARSE": (
        "contact.coarse_standoff_m",
        "contact.coarse_speed_m_s",
        "contact.coarse_accel_m_s2",
        "approach.clear_above_cloth_m",
    ),
    "DESCEND_CONTACT": (
        "contact.unload_drop_pct",
        "contact.unload_consecutive",
        "contact.baseline_settle_s",
        "contact.baseline_samples",
        "contact.move_increment_m",
        "contact.max_descent_m",
        "contact.lowest_allowed_m",
    ),
    "GRASP_BUMP": ("grasp.lift_after_contact_m",),
    "GRASP_OPEN": ("grasp.gripper_open", "grasp.pause_after_bump_s"),
    "GRASP_TAP2": ("grasp.lower_after_open_m",),
    "GRASP_CLOSE": ("grasp.gripper_closed", "grasp.pause_after_close_s"),
    "LIFTING": ("pull.clearance_above_bed_m", "contact.disable_effort"),
    "REVERSE_CONTACT": (
        "contact.lift_effort_threshold",
        "contact.move_increment_m",
        "contact.max_descent_m",
    ),
    "REVERSE_TAP2": ("grasp.lower_after_open_m",),
    "REVERSE_LIFT": ("pull.clearance_above_bed_m",),
    "REVERSE_PLACE": ("pull.round_trip", "grasp.gripper_open"),
}


def normalize_stop_after(value: object) -> str:
    """Empty / DONE = full pull. Else must be a frozen stage name."""

    text = "" if value is None else str(value).strip()
    if text in ("", "DONE", "none", "NONE", "false", "False"):
        return ""
    if text not in STOP_AFTER_ALLOWED:
        raise ValueError(
            f"pull.stop_after={text!r} is not a frozen stage. "
            f"Use one of {', '.join(STAGES)} or empty for full pull."
        )
    return text


def normalize_recipe(value: object) -> str:
    text = str(value or "full-pull").strip().lower()
    if text in ("", "full", "done"):
        text = "full-pull"
    if text not in RECIPES:
        raise ValueError(
            f"pull.recipe={value!r} is not a frozen recipe. "
            f"Use one of {', '.join(RECIPES)}."
        )
    return text


def recipe_stop_after(recipe: object) -> str:
    return RECIPES[normalize_recipe(recipe)]


def resolve_stop_after(*, recipe: object = "full-pull", stop_after: object = "") -> str:
    """Recipe wins when both are set; empty recipe falls back to stop_after."""

    recipe_name = normalize_recipe(recipe)
    recipe_stop = RECIPES[recipe_name]
    explicit = normalize_stop_after(stop_after)
    if recipe_name != "full-pull":
        return recipe_stop
    return explicit


def production_stop_after_ok(value: object) -> bool:
    """True when leftover tune stop_after will not truncate a real trial."""

    try:
        return normalize_stop_after(value) == ""
    except ValueError:
        return False


def should_stop_after(completed_stage: str, stop_after: str) -> bool:
    """True once ``completed_stage`` has finished and matches the request."""

    want = normalize_stop_after(stop_after)
    if not want:
        return False
    return completed_stage == want


def intentional_stop_message(stage: str, detail: str = "") -> str:
    token = f"{INTENTIONAL_STOP_PREFIX}{stage}"
    extra = str(detail).strip()
    return f"{token} {extra}".strip() if extra else token


def parse_intentional_stop(message: str) -> str | None:
    text = str(message or "")
    if INTENTIONAL_STOP_PREFIX not in text:
        return None
    rest = text.split(INTENTIONAL_STOP_PREFIX, 1)[1]
    stage = rest.split()[0] if rest.strip() else ""
    return stage or None


def relevant_params(stop_after: str) -> tuple[str, ...]:
    want = normalize_stop_after(stop_after)
    return _RELEVANT.get(want, TUNE_PARAM_NAMES[:8])


def motion_lines(stop_after: str) -> list[str]:
    want = normalize_stop_after(stop_after)
    lines = {
        "VERIFY_PREGRASP": [
            "park → pregrasp",
            "NO DESCEND",
            "NO GRIPPER CLOSE",
            "NO LIFT",
            "NO PULL",
        ],
        "DESCEND_UNLATCH": [
            "park → pregrasp → unlatch",
            "NO BREAKAWAY",
            "NO CONTACT",
            "NO GRIPPER CLOSE",
            "NO LIFT",
            "NO PULL",
        ],
        "DESCEND_BREAKAWAY": [
            "park → pregrasp → breakaway",
            "NO CONTACT FEEL",
            "NO GRIPPER CLOSE",
            "NO LIFT",
            "NO PULL",
        ],
        "DESCEND_COARSE": [
            "park → pregrasp (coarse skip)",
            "NO CONTACT",
            "NO GRIPPER CLOSE",
            "NO LIFT",
            "NO PULL",
        ],
        "DESCEND_CONTACT": [
            "park → pregrasp → relative lift-effort unload",
            "NO GRIPPER CLOSE",
            "NO LIFT",
            "NO PULL",
        ],
        "GRASP_BUMP": [
            "park → pregrasp → contact → bump",
            "NO GRIPPER CLOSE",
            "NO LIFT",
            "NO PULL",
        ],
        "GRASP_OPEN": [
            "park → pregrasp → contact → open",
            "NO TAP2",
            "NO LIFT",
            "NO PULL",
        ],
        "GRASP_TAP2": [
            "park → pregrasp → contact → tap2",
            "NO GRIPPER CLOSE",
            "NO LIFT",
            "NO PULL",
        ],
        "GRASP_CLOSE": [
            "park → pregrasp → grasp close",
            "NO LIFT",
            "NO PULL",
        ],
        "LIFTING": [
            "park → pregrasp → grasp → lift",
            "NO PULL",
        ],
        "PULLING": ["park → pregrasp → grasp → lift → pull", "NO RELEASE"],
        "RELEASING": ["park → full pull except retract"],
        "REVERSE_PLACE": [
            "uncover place → pick at place → place at uncover pick",
            "NO RETRACT",
        ],
        "RESET_EE": [
            "full pull → retract → wrist/gripper known pose",
            "NO further motion",
        ],
        "DONE": ["full BedPull including retract and EE reset"],
        "": ["full BedPull"],
    }
    return list(lines.get(want, ["park → requested stage", "NO PULL"]))


def parse_set_override(spec: str) -> tuple[str, str]:
    """Parse ``contact.effort_spike:=8`` into (name, value)."""

    raw = str(spec).strip()
    if ":=" in raw:
        name, value = raw.split(":=", 1)
    elif "=" in raw:
        name, value = raw.split("=", 1)
    else:
        raise ValueError(f"override must be name:=value, got {spec!r}")
    name = name.strip()
    value = value.strip()
    if not name or not value:
        raise ValueError(f"override must be name:=value, got {spec!r}")
    return name, value


def descend_budget_ok(*, dropped_m: float, max_descent_m: float) -> bool:
    remaining = float(max_descent_m) - float(dropped_m)
    return remaining > 0.01


def verify_allows_descend(*, e_g_m: float, reject_m: float) -> bool:
    return float(e_g_m) <= float(reject_m)
