#!/usr/bin/env bash
# Source ROS 2 Humble on RCHI. Prefer /opt/ros/humble; else conda env robe-ros2.
# Humble setup.bash is not nounset-safe.
set +u
if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]] \
  && [[ -d "${HOME}/miniconda3/envs/robe-ros2" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate robe-ros2
else
  echo "ROS 2 Humble not found. Install /opt/ros/humble or conda env robe-ros2." >&2
  return 1 2>/dev/null || exit 1
fi
if [[ -f "${HOME}/ament_ws/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/ament_ws/install/setup.bash"
fi
_ROBE_ROS2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_ROBE_ROS2_DIR}/dds_env.sh"
_ROBE_REPO_ROOT="$(cd "${_ROBE_ROS2_DIR}/../.." && pwd)"
export ROBE_CODE_DIR="${ROBE_CODE_DIR:-${_ROBE_REPO_ROOT}/real_world/code}"
export PYTHONPATH="${ROBE_CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
unset _ROBE_ROS2_DIR _ROBE_REPO_ROOT
