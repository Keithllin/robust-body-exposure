"""Lower-until-contact: relative lift-effort unload vs a descending hover.

Contact is ``filtered`` dropping by ``unload_drop_pct`` of the
in-motion hover median, for ``unload_consecutive`` samples. The
pre-move hold spike is diagnostic only. Cloth Z and Funmap's
absolute 20 are not the stop. The mechanical floor is panic-only.
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

CONTACT_ERROR_CODE = 100
PATH_TOLERANCE_VIOLATED = -4
CLEAN_SURFACE_LIFT_EFFORT = 33.7
MIN_CONTACT_DROP_M = 0.008
THRESHOLD_MODE_DEFAULT = "robot_default"
THRESHOLD_MODE_CUSTOM = "custom_symmetric"
THRESHOLD_MODES = (THRESHOLD_MODE_DEFAULT, THRESHOLD_MODE_CUSTOM)

FUNMAP_LIFT_EFFORT_THRESHOLD = 20.0
FUNMAP_MOVE_INCREMENT_M = 0.008
FUNMAP_PERIOD_S = 0.2
FUNMAP_AVG_WINDOW = 3
HOVER_BASELINE_S = 0.5
HOVER_BASELINE_DT_S = 0.05

CONTACT_TRACE_COLUMNS = (
    "timestamp",
    "t_s",
    "q_lift",
    "z_ee",
    "effort",
    "effort_filtered",
    "baseline",
    "relative_drop",
    "funmap20_would_hit",
    "phase",
)


def traj_error_code(result) -> int:
    if result is None:
        return 0
    return int(getattr(result, "error_code", 0) or 0)


def is_force_contact(result) -> bool:
    return traj_error_code(result) == CONTACT_ERROR_CODE


def guarded_lower_ok(
    result, dropped_m: float, *, min_drop_m: float = MIN_CONTACT_DROP_M
) -> bool:
    return is_force_contact(result) and float(dropped_m) >= float(min_drop_m)


def guarded_too_sensitive(
    result, dropped_m: float, *, min_drop_m: float = MIN_CONTACT_DROP_M
) -> bool:
    return is_force_contact(result) and float(dropped_m) < float(min_drop_m)


def normalize_threshold_mode(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    if text in ("", "robot_default", "default"):
        return THRESHOLD_MODE_DEFAULT
    if text in ("custom_symmetric", "custom", "symmetric"):
        return THRESHOLD_MODE_CUSTOM
    raise ValueError(
        f"contact.threshold_mode={value!r} must be "
        f"{THRESHOLD_MODE_DEFAULT} or {THRESHOLD_MODE_CUSTOM}"
    )


def format_effort_pct(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{float(value):+.1f}"


def lift_contact_func(
    effort: float,
    av_effort: float,
    *,
    threshold: float = FUNMAP_LIFT_EFFORT_THRESHOLD,
) -> bool:
    """Diagnostic only: Funmap absolute 20. Production uses relative unload."""

    return float(effort) <= float(threshold) or float(av_effort) < float(threshold)


def relative_unload(
    baseline: float,
    filtered: float,
    *,
    drop_pct: float,
) -> bool:
    need = abs(float(baseline)) * float(drop_pct) / 100.0
    return relative_drop(baseline, filtered) >= need


def update_average_effort(
    av_effort: float | None,
    effort: float,
    *,
    window: int = FUNMAP_AVG_WINDOW,
) -> float:
    if av_effort is None:
        return float(effort)
    n = float(window)
    return ((n - 1.0) * float(av_effort) + float(effort)) / n


def passed_stopping_position(
    position: float,
    stopping_position: float,
    direction_sign: int,
) -> bool:
    """True once motion has gone past the safety floor (Funmap)."""

    difference = float(stopping_position) - float(position)
    if int(_sign(difference)) == int(direction_sign):
        return False
    return True


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def relative_drop(baseline: float, filtered: float) -> float:
    """Unload vs hover baseline (positive = effort fell)."""

    return float(baseline) - float(filtered)


def median_effort(samples: list[float]) -> float:
    if not samples:
        raise ValueError("median_effort needs at least one sample")
    return float(statistics.median(float(x) for x in samples))


def descent_floor_q(
    *,
    q_start: float,
    max_descent_m: float,
    lift_min_m: float,
    lowest_allowed_m: float = 0.0,
) -> float:
    """Mechanical floor only: start lift minus max descent."""

    floor = max(float(lift_min_m), float(q_start) - float(max_descent_m))
    if float(lowest_allowed_m) > 0.0:
        floor = max(floor, float(lowest_allowed_m))
    return floor


def contact_trace_row(
    *,
    t0: float,
    q_lift: float,
    z_ee: float,
    effort: float | None,
    effort_filtered: float | None,
    baseline: float | None,
    phase: str,
) -> dict:
    drop = None
    funmap20 = False
    if effort is not None and effort_filtered is not None:
        funmap20 = lift_contact_func(effort, effort_filtered)
    if baseline is not None and effort_filtered is not None:
        drop = relative_drop(baseline, effort_filtered)
    return {
        "timestamp": time.time(),
        "t_s": time.monotonic() - float(t0),
        "q_lift": float(q_lift),
        "z_ee": float(z_ee),
        "effort": None if effort is None else float(effort),
        "effort_filtered": (
            None if effort_filtered is None else float(effort_filtered)
        ),
        "baseline": None if baseline is None else float(baseline),
        "relative_drop": drop,
        "funmap20_would_hit": funmap20,
        "phase": str(phase),
    }


def write_contact_trace_csv(path: Path, rows: list[dict]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTACT_TRACE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp": f"{float(row['timestamp']):.3f}",
                    "t_s": f"{float(row['t_s']):.3f}",
                    "q_lift": f"{float(row['q_lift']):.4f}",
                    "z_ee": f"{float(row['z_ee']):.4f}",
                    "effort": _fmt_opt(row.get("effort"), 3),
                    "effort_filtered": _fmt_opt(row.get("effort_filtered"), 3),
                    "baseline": _fmt_opt(row.get("baseline"), 3),
                    "relative_drop": _fmt_opt(row.get("relative_drop"), 3),
                    "funmap20_would_hit": (
                        "1" if row.get("funmap20_would_hit") else "0"
                    ),
                    "phase": row.get("phase", ""),
                }
            )
    return dest


def _fmt_opt(value, digits: int) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


class LiftContactDetector:
    """Relative unload vs the descending-hover effort, not the hold spike.

    Pre-move samples are diagnostic only. After real downward travel the
    executor collects in-motion samples, then locks that median before
    any contact decision. Unarmed updates never report contact.
    """

    def __init__(
        self,
        *,
        unload_drop_pct: float = 15.0,
        unload_consecutive: int = 2,
        baseline_samples: int = 15,
        window: int = FUNMAP_AVG_WINDOW,
        threshold: float = FUNMAP_LIFT_EFFORT_THRESHOLD,
    ) -> None:
        self.unload_drop_pct = float(unload_drop_pct)
        self.unload_consecutive = max(1, int(unload_consecutive))
        self.baseline_samples = max(1, int(baseline_samples))
        self.window = int(window)
        self.threshold = float(threshold)
        self.av_effort: float | None = None
        self.baseline: float | None = None
        self._baseline_buf: list[float] = []
        self._below = 0
        self.in_contact = False
        self.contact_q: float | None = None
        self.contact_effort: float | None = None
        self.last_q: float | None = None
        self.last_effort: float | None = None
        self.active = False
        self.armed = False
        self.collecting_motion = False

    def reset(self) -> None:
        self.av_effort = None
        self.baseline = None
        self._baseline_buf = []
        self._below = 0
        self.in_contact = False
        self.contact_q = None
        self.contact_effort = None
        self.last_q = None
        self.last_effort = None
        self.armed = False
        self.collecting_motion = False

    def lock_baseline(self) -> None:
        if self.baseline is not None:
            return
        if self._baseline_buf:
            self.baseline = median_effort(
                self._baseline_buf[-self.baseline_samples :]
            )
        elif self.last_effort is not None:
            self.baseline = float(self.last_effort)

    def begin_motion_baseline(self) -> None:
        """Start collecting the descending-hover effort. Do not decide yet.

        The first 8 mm still rides the hold-spike (trace: +36 → +27).
        Relative unload uses the median after that transient.
        """

        self._baseline_buf = []
        self.baseline = None
        self.av_effort = None
        self._below = 0
        self.in_contact = False
        self.armed = False
        self.collecting_motion = True

    def has_motion_samples(self) -> bool:
        return bool(self._baseline_buf)

    def sample_median(self) -> float | None:
        if self._baseline_buf:
            return median_effort(self._baseline_buf[-self.baseline_samples :])
        if self.last_effort is not None:
            return float(self.last_effort)
        return None

    def lock_motion_baseline(self) -> None:
        if not self._baseline_buf:
            raise RuntimeError(
                "Cannot arm lift contact detector without in-motion effort"
            )
        self.lock_baseline()
        if self.baseline is None:
            raise RuntimeError("Cannot arm lift contact detector without effort")
        self.av_effort = float(self.baseline)
        self._below = 0
        self.collecting_motion = False
        self.armed = True

    def update(self, q: float, effort: float) -> bool:
        self.last_q = float(q)
        self.last_effort = float(effort)
        if not self.armed:
            self._baseline_buf.append(float(effort))
            return False
        self.av_effort = update_average_effort(
            self.av_effort, effort, window=self.window
        )
        if relative_unload(
            self.baseline, self.av_effort, drop_pct=self.unload_drop_pct
        ):
            self._below += 1
        else:
            self._below = 0
        if self._below >= self.unload_consecutive:
            self.in_contact = True
            self.contact_q = float(q)
            self.contact_effort = float(effort)
            return True
        return False


def effort_spike(current: float, baseline: float, spike: float) -> bool:
    """Deprecated."""

    return abs(float(current)) >= abs(float(baseline)) + float(spike)
