"""Execute-stage builders: preflight, /bed_pull, return-home. No CMA."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from stretch_grasp_stages import production_stop_after_ok
from trial_capture import _ros2_python, cma_snapshot_dest
from trial_context import TrialContext
from trial_layout import resolve_initial_pcd
from trial_plan import cma_reach_flags


def preflight_cmd(ctx: TrialContext, action_path: Path, label: str) -> list[str]:
    if label == "uncover":
        cloth_pcd = resolve_initial_pcd(ctx.pose_dir)
    else:
        cloth_pcd = ctx.intermediate_dir / "blanket_pcd.pcd"
    if not cloth_pcd.is_file():
        raise FileNotFoundError(f"REJECT {label}: missing execution blanket PCD {cloth_pcd}")
    cmd = [
        ctx.robe_py,
        str(ctx.code_dir / "preflight_bed_pull.py"),
        "--action",
        str(action_path),
        "--pose-dir",
        str(ctx.pose_dir),
        "--trial-id",
        str(ctx.args.trial_id or ctx.pose_dir.name),
        "--manifest",
        str(ctx.pose_dir / f"{label}_execution_manifest.json"),
        "--arm-max-m",
        str(ctx.args.arm_max_m),
        "--blanket-pcd",
        str(cloth_pcd),
    ]
    if label == "recover":
        cmd.extend(["--snap-off-cloth", "--write-snapped"])
    cmd.extend(cma_reach_flags(ctx))
    return cmd


def recover_grasp_was_snapped(ctx: TrialContext, label: str = "recover") -> bool:
    path = ctx.pose_dir / f"{label}_execution_manifest.json"
    if not path.is_file():
        return False
    import json

    check = json.loads(path.read_text()).get("grasp_cloth_check") or {}
    return bool(check.get("snapped"))


def leftover_stop_after_hint() -> str:
    return (
        "pull.stop_after is set on /bed_pull_executor (grasp-tune leftover). "
        "A production trial must run a full pull. Clear it with: "
        "ros2 param set /bed_pull_executor pull.stop_after \"\" "
        "then send /bed_pull again. Do not use tune_grasp for run_trial."
    )


def send_bed_pull_checks() -> str:
    leftover = leftover_stop_after_hint()
    return (
        "ok=0; "
        "for _i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do "
        "  if ros2 action info /bed_pull >/dev/null 2>&1; then ok=1; break; fi; "
        "  sleep 1; "
        "done; "
        "if [ \"$ok\" != 1 ]; then "
        f"echo {shlex.quote(bed_pull_missing_server_hint())} >&2; "
        "exit 1; fi; "
        "jt=$(ros2 action info /stretch_controller/follow_joint_trajectory "
        "2>/dev/null || true); "
        "servers=$(printf '%s\\n' \"$jt\" | sed -n 's/^Action servers: //p' | head -n1); "
        "if [ \"$servers\" = \"0\" ] || [ -z \"$servers\" ]; then "
        f"echo {shlex.quote(stretch_driver_missing_hint())} >&2; "
        "exit 1; fi; "
        "sa=$(ros2 param get /bed_pull_executor pull.stop_after 2>/dev/null "
        "| sed -n 's/.*value is: //p' | tr -d \"'\\\" \"); "
        "if [ -n \"$sa\" ] && [ \"$sa\" != \"DONE\" ]; then "
        f"echo {shlex.quote(leftover)} >&2; "
        "echo leftover pull.stop_after=$sa >&2; "
        "exit 1; fi; "
        "pregrasp=$(ros2 param get /bed_pull_executor pull.stop_after_pregrasp "
        "2>/dev/null || true); "
        "case \"$pregrasp\" in "
        "*[Tt]rue*) "
        f"echo {shlex.quote(leftover)} >&2; exit 1 ;; "
        "esac; "
        "recipe=$(ros2 param get /bed_pull_executor pull.recipe 2>/dev/null "
        "| sed -n 's/.*value is: //p' | tr -d \"'\\\" \"); "
        "if [ -n \"$recipe\" ] && [ \"$recipe\" != \"full-pull\" ]; then "
        f"echo {shlex.quote(leftover)} >&2; "
        "echo leftover pull.recipe=$recipe >&2; exit 1; fi; "
    )


def bed_pull_missing_server_hint() -> str:
    return (
        "BedPull server /bed_pull is not running. Start the executor on RCHI, "
        "not Stretch: source real_world/ros2/source_humble.sh && "
        "ros2 launch robe_stretch workstation_executor.launch.py "
        "(no trial paths; run_trial writes session/active_trial.json)."
    )


def stretch_driver_missing_hint() -> str:
    return (
        "stretch_driver is not visible (FollowJointTrajectory has 0 servers). "
        "On Stretch only, ROS_DOMAIN_ID=12: "
        "ros2 launch stretch_core stretch_driver.launch.py broadcast_odom_tf:=True "
        "and ros2 launch stretch_core d435i_high_resolution.launch.py."
    )


def send_bed_pull_cmd(ctx: TrialContext, action_path: Path, trial_id: str) -> list[str]:
    domain = os.environ.get("ROS_DOMAIN_ID", "12")
    setup = ctx.real_world_dir / "ros2" / "source_humble.sh"
    send = (
        "ros2 run robe_stretch send_bed_pull "
        f"--action {shlex.quote(str(action_path))} "
        f"--trial-id {shlex.quote(trial_id)}"
    )
    inner = (
        "set -eo pipefail; set +u; "
        f"source {shlex.quote(str(setup))}; "
        f"export ROS_DOMAIN_ID={shlex.quote(domain)}; "
        + send_bed_pull_checks()
        + send
    )
    return ["bash", "-c", inner]


def return_home_cmd(ctx: TrialContext) -> list[str]:
    return [
        "bash",
        "-lc",
        _ros2_python(
            ctx,
            ctx.code_dir / "return_uncover_home.py",
            f"--pose-dir {shlex.quote(str(ctx.pose_dir))}",
        ),
    ]


def preflight_must_stop(ok: bool, *, label: str) -> None:
    if ok:
        return
    raise SystemExit(
        f"{label} preflight rejected; robot did not execute. "
        "Do not continue to the next motion stage or pretend F1 is valid."
    )


def production_recipe_ok(recipe: str, stop_after: str) -> bool:
    if recipe not in ("", "full-pull"):
        return False
    return production_stop_after_ok(stop_after)
