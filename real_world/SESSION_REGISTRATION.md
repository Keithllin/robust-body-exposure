# Session calibration (three modules)

One experimental session = one `real_world/sessions/<id>/`. Trials reuse that
session. `canonical_bed_frame.json` and `sim_origin_data.pkl` stay on the
trial (capture / CMA). No session module reads a trial `sim_origin`.

```text
RARE          marker_layout / URDF
SESSION       3-ZED  +  Stretch origin  +  robot↔planning  +  PCD/filter
TRIAL         pose_<n>_TL<code>_<subid>/  (canonical / capture / CMA / pickle)
```

Same ``pose_num``, later TL: `run_trial` copies skeleton + `sim_origin` from
the newest sibling trial of that pose. No extra ``pose_<n>/`` folder. Not
session. Do not uncover again and do not redo 136. Three overlays (action /
PCD-on-RGB / EE-traj with live TF) must exist before YES.

After Recover, `run_trial` drives back to session ``stretch/uncover_home.json``
(first Uncover bedside of the day) with ``translate_mobile_base`` only.

Terms:

- **layout frame** — fixed `marker_layout.json` board
- **canonical bed frame** — per-trial tags 0,1,2,3
- **Stretch origin** — `T_odom_layout`
- **robot-planning registration** — ceiling 136 vs wrist TF (XY only)

## SOP

```bash
python real_world/sessions/new_session.py --name exp01
python real_world/sessions/calibrate_zed_system.py
python real_world/sessions/calibrate_stretch_origin.py
python real_world/sessions/register_robot_to_planning.py
python real_world/sessions/session_status.py
```

Accept only:

```text
READY FOR TRIAL: YES
```

Then start the executor (no trial paths) and `run_trial`:

```bash
# 1. doctor (zero motion)
python real_world/code/robe_ops.py doctor
# 2. validate the last trial or pose pack
python real_world/code/robe_ops.py validate --trial <pose_dir>
# 3. execute
source real_world/ros2/source_humble.sh
ros2 launch robe_stretch workstation_executor.launch.py
# other terminal
python real_world/run_trial.py --profile production --subject-id ... --pose-num ...
# on failure: robe_ops.py bundle --trial <pose_dir>  and handle the fault code
```

Named stages: `pose`, `initial-capture`, `uncover-plan`, `uncover-exec`,
`intermediate-capture`, `recover-plan`, `recover-exec`, `final-capture`,
`score`. Resume with `--resume latest` (or a trial dir). Do not pass
`--start-here > 0` with a new random TL / missing `--sub-id`.

`--profile production` is closed-loop, SAM2, random study TL, remeasure
diameters, reuse 0–3 / 136, reject live-geometry drift, and send `/bed_pull`.
CLI overrides are recorded on `trial_identity.json` as `debug_overrides`.

`run_trial` writes `session/active_trial.json` before `/bed_pull`. Do not
pass `canonical_frame_path` / `manifest_dir` / `layout_snapshot_path`.

Hardware / perception (ZED serials, extrinsics, exposure/gain, filters,
Stretch registration) come from the frozen session. CLI overrides are debug
only. `--allow-unfrozen-session` is the escape hatch; the default door is
READY.

`new_session.py` only creates dirs and `current`. It does not copy an old
`zed_extrinsics.json`.

## Module duties

1. `calibrate_zed_system.py` — 3 ZED → layout PnP, filter snapshot into
   `session/zed/pcd_config.json` (relative filter names), freeze ZED.
   Invalidates registration. Does not touch the robot. Plane-residual
   fusion is `run_trial --validate-fusion`, not this module.
2. `calibrate_stretch_origin.py` — D435i bed posts → raw `T_odom_layout`,
   auto-sync. Does not look at 136 or apply XY correction.
3. `register_robot_to_planning.py` — session ceiling calibration + current
   ceiling RGB + 136 + Stretch wrist TF + raw origin. XY only. Residual
   `< accept_xy_m` (0.02 m) → `SESSION REGISTRATION FROZEN`, else `REJECT`.
   No trial `sim_origin_data.pkl`.

`session_status.py` is a gate, not a fourth calibration module.

## Invalidation

```text
new ZED freeze  →  registration needs_alignment
new Stretch raw →  registration needs_alignment
```

`registration.json` stores `marker_layout` / `zed_extrinsics` /
`stretch_origin_raw` hashes. Any drift → not READY.

## Internals (do not chain by hand)

`calibrate_zed_markers.py`, `validate_zed_*`, `tune_zed_blanket_filter.py`,
`sample_origin`, `grab_ceiling_now.py`, `overlay_stretch_tf_ceiling.py`,
`correct_stretch_origin_xy.py`.

## Redo the session

Only if Stretch was shoved, `stretch_driver` restarted (odom reset), the
bed or marker rack moved, or D435i / ceiling ZED moved.
