#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ISAACSIM_ROOT:-}" ]]; then
  echo "Set ISAACSIM_ROOT to the Isaac Sim 5.1 installation." >&2
  exit 2
fi

isaac_uw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Isaac Sim embeds Python 3.11 and its own ROS 2 Jazzy bridge. Remove any
# system ROS/Python 3.12 paths before loading the matching internal runtime.
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION RMW_IMPLEMENTATION
# shellcheck disable=SC1090
set +u
source "$ISAACSIM_ROOT/setup_ros_env.sh"
set -u

exec "$ISAACSIM_ROOT/python.sh" \
  "$isaac_uw_root/run_underwater_sim.py" \
  --config "$isaac_uw_root/config/sim_params.json" \
  --overlay "$isaac_uw_root/config/ardusub_sitl.overlay.json" \
  "$@"
