#!/usr/bin/env bash
set -euo pipefail

isaac_uw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
install_parent="$(dirname "$isaac_uw_root")"

# README examples use /path/to/... placeholders. Treat those literal example
# values as unset so they cannot accidentally override automatic discovery.
for path_var in ARDUPILOT_ROOT ARDUPILOT_VENV ISAACSIM_ROOT OCEANSIM_ROOT MVM_PY_PATH; do
  path_value="${!path_var:-}"
  if [[ "$path_value" == /path/to/* ]]; then
    echo "Ignoring placeholder $path_var=$path_value"
    unset "$path_var"
  fi
done

find_latest_dir() {
  local pattern="$1"
  local candidate
  candidate="$(compgen -G "$pattern" | sort -V | tail -n 1 || true)"
  [[ -n "$candidate" ]] && printf '%s\n' "$candidate"
}

export ARDUPILOT_ROOT="${ARDUPILOT_ROOT:-$install_parent/ardupilot}"
export ARDUPILOT_VENV="${ARDUPILOT_VENV:-$install_parent/ardupilot-venv}"
isaac_default="$(find_latest_dir "$install_parent/isaac-sim-standalone-5.1.*")"
export ISAACSIM_ROOT="${ISAACSIM_ROOT:-$isaac_default}"
export OCEANSIM_ROOT="${OCEANSIM_ROOT:-${ISAACSIM_ROOT:-}/extsUser/OceanSim}"
export MVM_PY_PATH="${MVM_PY_PATH:-$HOME/graal_ws/marine_vehicle_models/underwater_vehicle_model/build_isaac_py311}"
export QGC_MAVLINK_PORT="${QGC_MAVLINK_PORT:-14550}"
export BRIDGE_MAVLINK_PORT="${BRIDGE_MAVLINK_PORT:-14551}"

require_dir() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" || ! -d "$value" ]]; then
    echo "$name is not a directory: ${value:-<unset>}" >&2
    echo "Export $name before running this script." >&2
    exit 2
  fi
}

require_dir ARDUPILOT_ROOT "$ARDUPILOT_ROOT"
require_dir ARDUPILOT_VENV "$ARDUPILOT_VENV"
require_dir ISAACSIM_ROOT "$ISAACSIM_ROOT"
require_dir OCEANSIM_ROOT "$OCEANSIM_ROOT"
require_dir MVM_PY_PATH "$MVM_PY_PATH"
for command_name in setsid ss; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is missing: $command_name" >&2
    exit 2
  fi
done

if [[ "${1:-}" == "--check" ]]; then
  printf 'ARDUPILOT_ROOT=%s\n' "$ARDUPILOT_ROOT"
  printf 'ARDUPILOT_VENV=%s\n' "$ARDUPILOT_VENV"
  printf 'ISAACSIM_ROOT=%s\n' "$ISAACSIM_ROOT"
  printf 'OCEANSIM_ROOT=%s\n' "$OCEANSIM_ROOT"
  printf 'MVM_PY_PATH=%s\n' "$MVM_PY_PATH"
  printf 'QGC_MAVLINK_PORT=%s\n' "$QGC_MAVLINK_PORT"
  printf 'BRIDGE_MAVLINK_PORT=%s\n' "$BRIDGE_MAVLINK_PORT"
  printf 'ISAAC_SENSOR_MAVLINK_PORT=14552 (internal)\n'
  echo "SITL stack paths: OK"
  exit 0
fi

isaac_pid=""
ardusub_pid=""
stopping=false

stop_stack() {
  if $stopping; then
    return
  fi
  stopping=true
  trap - INT TERM EXIT
  echo "Stopping ArduSub and Isaac Sim..."
  if [[ -n "$ardusub_pid" ]] && kill -0 "$ardusub_pid" 2>/dev/null; then
    kill -INT -- "-$ardusub_pid" 2>/dev/null || kill -INT "$ardusub_pid" 2>/dev/null || true
  fi
  if [[ -n "$isaac_pid" ]] && kill -0 "$isaac_pid" 2>/dev/null; then
    kill -INT -- "-$isaac_pid" 2>/dev/null || kill -INT "$isaac_pid" 2>/dev/null || true
  fi
  [[ -z "$ardusub_pid" ]] || wait "$ardusub_pid" 2>/dev/null || true
  [[ -z "$isaac_pid" ]] || wait "$isaac_pid" 2>/dev/null || true
}
trap stop_stack INT TERM EXIT

echo "Starting Isaac Sim from: $ISAACSIM_ROOT"
setsid "$isaac_uw_root/scripts/run_isaac_sitl.sh" &
isaac_pid=$!

startup_timeout_s="${SITL_STARTUP_TIMEOUT_S:-120}"
deadline=$((SECONDS + startup_timeout_s))
echo "Waiting for Isaac Sim to bind the ArduPilot JSON port 9002..."
while ! ss -H -lun 'sport = :9002' 2>/dev/null | grep -q .; do
  if ! kill -0 "$isaac_pid" 2>/dev/null; then
    wait "$isaac_pid" || true
    echo "Isaac Sim exited before opening UDP 9002." >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out after ${startup_timeout_s}s waiting for Isaac Sim UDP 9002." >&2
    exit 1
  fi
  sleep 1
done

echo "Starting ArduSub from: $ARDUPILOT_ROOT"
ARDUSUB_NON_INTERACTIVE=1 setsid "$isaac_uw_root/scripts/run_ardusub_sitl.sh" &
ardusub_pid=$!

echo "Isaac Sim and ArduSub are running. Press Ctrl-C to stop both."
set +e
wait -n "$isaac_pid" "$ardusub_pid"
status=$?
set -e
echo "One stack process exited with status $status."
exit "$status"
