#!/usr/bin/env bash
# Bidirectional ROS2 Humble discovery. Run on the workstation with the
# robot driver already up. Prefer wired LAN.
#
# Usage:
#   ./check_ros2_network.sh              # 60s hz soak
#   ./check_ros2_network.sh --soak 600   # 10 min soak
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_SCRIPT_DIR}/dds_env.sh" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${_SCRIPT_DIR}/dds_env.sh"
  set -u
fi
unset _SCRIPT_DIR
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-12}"
if [[ -z "${RMW_IMPLEMENTATION:-}" ]]; then
  unset RMW_IMPLEMENTATION
fi

SOAK=60
if [[ "${1:-}" == "--soak" ]]; then
  SOAK="${2:-60}"
fi

echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID RMW=$RMW_IMPLEMENTATION soak=${SOAK}s"
if [[ -n "${ROS_IP:-}" ]]; then
  echo "ROS_IP=$ROS_IP"
fi
if [[ -n "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]]; then
  echo "FASTRTPS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE"
elif [[ -n "${CYCLONEDDS_URI:-}" ]]; then
  echo "CYCLONEDDS_URI=$CYCLONEDDS_URI"
else
  echo "If multicast is blocked, source dds_env.sh (FASTRTPS_DEFAULT_PROFILES_FILE)."
fi

if ! command -v ros2 >/dev/null; then
  echo "ros2 not on PATH; source Humble first" >&2
  exit 2
fi

echo "Topics:"
ros2 topic list || true

echo "Waiting for /stretch/joint_states (30s) workstation → robot..."
if timeout 30 ros2 topic echo /stretch/joint_states --once >/dev/null; then
  echo "OK: workstation sees robot joint_states"
else
  echo "FAIL: no joint_states. Check domain, multicast/Peers, and cabling." >&2
  exit 1
fi

echo "Checking reverse discovery (robot should see workstation /rosout)..."
timeout 15 ros2 topic hz /rosout --window 5 >/dev/null 2>&1 || true

echo "Soak ${SOAK}s on /stretch/joint_states (watch for discovery dropouts)..."
if ! timeout "${SOAK}" ros2 topic hz /stretch/joint_states; then
  echo "WARN: hz ended early (Ctrl-C or timeout)."
fi
echo "Done. Target soak for a trial day is 5–10 min with no dropouts."
