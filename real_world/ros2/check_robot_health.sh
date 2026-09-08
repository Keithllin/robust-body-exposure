#!/usr/bin/env bash
# Stretch 3 health runbook. Run on the robot after homing.
set -euo pipefail
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

echo "== Stretch health =="
echo "1. Home / calibrated URDF (once per boot):"
echo "     stretch_robot_home.py"
echo "     ros2 run stretch_calibration check_head_calibration"
echo "2. Primary e-stop: white hardware runstop on the head"
echo "   Secondary: ros2 service call /runstop std_srvs/srv/SetBool \"{data: true}\""
echo

if ! command -v ros2 >/dev/null; then
  echo "ros2 not on PATH; source Humble first" >&2
  exit 2
fi

echo "Waiting for /stretch/joint_states..."
if ! timeout 20 ros2 topic echo /stretch/joint_states --once >/dev/null; then
  echo "FAIL: stretch_driver not publishing. Launch:" >&2
  echo "  ros2 launch stretch_core stretch_driver.launch.py" >&2
  exit 1
fi
echo "OK: joint_states"

echo "Waiting for D435i /camera/color/image_raw..."
if ! timeout 20 ros2 topic hz /camera/color/image_raw --window 5 >/dev/null 2>&1; then
  echo "FAIL: no D435i. Launch:" >&2
  echo "  ros2 launch stretch_core d435i_high_resolution.launch.py" >&2
  exit 1
fi
echo "OK: D435i color"

if timeout 8 ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame >/tmp/robe_tf_echo.txt 2>/dev/null; then
  echo "OK: TF base_link → camera_color_optical_frame (use RViz to confirm it moves with head pan)"
else
  echo "WARN: tf2_echo timed out; confirm calibrated URDF, not a default camera TF"
fi

echo "Checking /runstop..."
if ros2 service list | grep -q '/runstop'; then
  echo "OK: /runstop"
else
  echo "FAIL: /runstop missing" >&2
  exit 1
fi

echo
echo "Optional 5 cm sanity (position mode), NOT an e-stop:"
echo "  ros2 run robe_stretch health_check --ros-args -- --move-5cm"
echo "keyboard_teleop is debug positioning only."
echo "Done."
