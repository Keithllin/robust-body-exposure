# Stretch 3 ROS2 bring-up for RoBE

Workstation = ZED + policy. Stretch = localize + execute. Do not run
`stretch_zmq_host.py` at the same time as `stretch_driver`.

Policy metres are identity after the trial `canonical_bed_frame.json`.
Do not remirror the pickle. Do not scale `0.845 × 1.85` onto sim `0.88 × 2.10`.

## 1. Robot health (on the robot)

```bash
# Home / calibrated URDF
stretch_robot_home.py
ros2 run stretch_calibration check_head_calibration   # if installed

# Driver + head camera
ros2 launch stretch_core stretch_driver.launch.py
# new terminal
ros2 launch stretch_core d435i_high_resolution.launch.py

bash real_world/ros2/check_robot_health.sh
# optional 5 cm base move (position mode; not an e-stop)
ros2 run robe_stretch health_check --ros-args -- --move-5cm
```

Confirm in RViz that `camera_color_optical_frame` moves with head pan/tilt
and that `base_link` → camera TF is the calibrated URDF, not a default.

Primary e-stop: white hardware runstop on the head.
Secondary: `ros2 service call /runstop std_srvs/srv/SetBool "{data: true}"`.
`keyboard_teleop` is debug positioning only.

## 2. Network (robot + workstation)

Same `ROS_DOMAIN_ID=12`. Humble FastRTPS (do not force CycloneDDS on this
Stretch). Prefer a wired LAN. Campus WiFi drops DDS multicast, so both
machines source `dds_env.sh`: correct `ROS_IP` plus FastDDS unicast peers.

Do not pin Stretch `ROS_IP` to `192.168.100.5` (hotspot). That address is
not reachable from RCHI.

RCHI (every ROS terminal, including `run_trial.py` / executor):

```bash
source /path/to/robust-body-exposure/real_world/ros2/source_humble.sh
```

Stretch (bashrc sources `~/robe_ros2/dds_env.sh`). Restart driver with that
env — already-running `stretch_driver` keeps the old `ROS_IP`:

```bash
bash ~/robe_ros2/stretch_ros.sh restart
```

Override peers if DHCP changes: `ROBE_STRETCH_IP`, `ROBE_WORKSTATION_IP`.
Check from RCHI: `bash real_world/ros2/check_ros2_network.sh` (after sourcing
Humble). Expect `/stretch/joint_states` and FollowJointTrajectory. **Do not**
subscribe to `/camera/*` on RCHI — D435i stays on Stretch.

## 3. Humble on RCHI (policy computer)

Stretch runs **only** `stretch_driver` + head D435i. Camera pixels never
cross the LAN. Aiming and bed-post PnP run **on Stretch**
(`preview_origin` / `sample_origin`). Session start: copy that PnP snapshot,
then ceiling↔Stretch **XY registration**. Freeze
`stretch_origin_corrected.json` at the **subject/session** dir (shared by
all trials). See `real_world/SESSION_REGISTRATION.md`. RCHI
`action_executor` loads that file and never subscribes to image topics.

The BedPull client and `action_executor` run on RCHI.

This workstation uses conda env `robe-ros2` (RoboStack Humble) because apt
`/opt/ros/humble` needs sudo. Keep `empy=3.3.4` in that env (Humble rosidl).

```bash
# Workstation ROS env
source /path/to/robust-body-exposure/real_world/ros2/source_humble.sh

mkdir -p ~/ament_ws/src
ln -sfn /path/to/robust-body-exposure/real_world/ros2/robe_stretch_interfaces ~/ament_ws/src/
ln -sfn /path/to/robust-body-exposure/real_world/ros2/robe_stretch ~/ament_ws/src/
cd ~/ament_ws
colcon build --packages-select robe_stretch_interfaces robe_stretch
source install/setup.bash

# Executor on RCHI — no trial paths. Session origin from sessions/current.
# canonical_bed_frame + manifest_dir come from session/active_trial.json
# (run_trial writes it before /bed_pull). Do not pass layout_snapshot_path.
ros2 launch robe_stretch workstation_executor.launch.py
```

On Stretch: stop any `action_executor`. Driver + D435i:

```bash
bash ~/robe_ros2/stretch_ros.sh restart
# equivalent: source ~/robe_ros2/dds_env.sh, then the two stretch_core launches
```

Do not `execute_pull.launch.py` on either machine (it starts the robot driver).

## 4. Trial execution (workstation, conda `robe`)

Operator order: **doctor → validate → execute**. Failures are classified by
fault code (`GOAL_REJECT`, `LIVE_GEOMETRY`, `WRIST_VERIFY`, `PREFLIGHT`,
`INTENTIONAL_STOP`, …). Formal reach/cloth gate is always
`preflight_bed_pull.py` (not the weaker ROS preflight).

```bash
python real_world/code/robe_ops.py doctor
python real_world/code/robe_ops.py validate --trial <pose> --preflight
python real_world/code/robe_ops.py dry-run --trial <pose> --recipe verify-only
python real_world/run_trial.py --profile production --subject-id ... --pose-num ...
```

```bash
python real_world/run_trial.py ... --ros2-execute --validate-fusion

# or manually:
python real_world/code/overlay_coord_contract.py \
  --image <pose>/uncovered_rgb.png \
  --sim-origin <pose>/sim_origin_data.pkl \
  --pose-dir <pose> \
  --output <pose>/coord_contract.png

python real_world/code/preflight_bed_pull.py \
  --action <pose>/scaled_action.pkl \
  --pose-dir <pose>

python real_world/code/validate_zed_fusion.py --pose-dir <pose>/initial

ros2 run robe_stretch send_bed_pull \
  --action <pose>/scaled_action.pkl \
  --trial-id <id>
```

`scaled_action.pkl` is canonical bed metres. Preflight rejects; it never clips.
`--ros2-execute` sends `/bed_pull` from RCHI (source Humble first).

Executor sequence: load frozen session `stretch_origin_corrected.json` → require park
parallel to the long side (`+X_base` ≈ bed `+Y`; no `rotate_mobile_base`) →
Hello split: `translate_mobile_base` then re-query TF then `joint_arm`
(`-Y_base`) → guarded contact → pull with base+arm together. Driving the
base does **not** void the snapshot unless `stretch_driver` restarts.

## 5. Safety

| Mechanism | Role |
| --- | --- |
| Hardware runstop | Primary emergency stop |
| `/runstop` | Software e-stop |
| BedPull cancel | Cancels FollowJointTrajectory; runstop if that fails |
| keyboard_teleop | Debug only |

Contact effort is **lift-only** during lower/grasp (`custom_contact_thresholds=True`
so a 2-tuple is effort, not velocity). A contact trip during pull is an
execution failure, not a successful policy outcome.
