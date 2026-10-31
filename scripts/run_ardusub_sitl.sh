#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ARDUPILOT_ROOT:-}" ]]; then
  echo "Set ARDUPILOT_ROOT to an unmodified ArduPilot checkout at tag Sub-4.7.2." >&2
  exit 2
fi

# Keep sim_vehicle.py on the host Python while allowing MAVProxy to live in an
# isolated virtual environment. By default, use a sibling "ardupilot-venv"
# when it exists; ARDUPILOT_VENV can point elsewhere.
ardupilot_venv="${ARDUPILOT_VENV:-$(dirname "$ARDUPILOT_ROOT")/ardupilot-venv}"
if [[ -x "$ardupilot_venv/bin/mavproxy.py" ]]; then
  export PATH="/usr/bin:$ardupilot_venv/bin:$PATH"
fi
if ! command -v mavproxy.py >/dev/null 2>&1; then
  echo "mavproxy.py was not found. Install MAVProxy in a venv and set ARDUPILOT_VENV to it." >&2
  exit 2
fi

isaac_uw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sitl_overlay="$isaac_uw_root/config/ardusub_sitl.overlay.json"
custom_location="$(python3 - "$sitl_overlay" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    home = json.load(stream)["control"]["ardusub_sitl"]["home"]

latitude = float(home["latitude_deg"])
longitude = float(home["longitude_deg"])
altitude = float(home["altitude_m"])
heading = float(home["heading_deg"])
if not -90.0 <= latitude <= 90.0:
    raise SystemExit("home.latitude_deg must be in [-90, 90]")
if not -180.0 <= longitude <= 180.0:
    raise SystemExit("home.longitude_deg must be in [-180, 180]")
heading %= 360.0
print(f"{latitude:.8f},{longitude:.8f},{altitude:.3f},{heading:.3f}")
PY
)"
echo "ArduSub global home: $custom_location"
expected_commit="fb773567896014da007bca58358704d4a0d8516f"
actual_commit="$(git -C "$ARDUPILOT_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "ArduPilot must be pinned to Sub-4.7.2 ($expected_commit); found $actual_commit." >&2
  exit 2
fi
if ! git -C "$ARDUPILOT_ROOT" diff --quiet || ! git -C "$ARDUPILOT_ROOT" diff --cached --quiet; then
  echo "ArduPilot checkout has tracked changes; refusing to run a patched firmware tree." >&2
  exit 2
fi

cd "$ARDUPILOT_ROOT"

# Public MAVLink endpoints. Port 14552 remains the private Isaac-to-ArduSub
# sensor-injection endpoint required by the DVL integration.
qgc_mavlink_port="${QGC_MAVLINK_PORT:-14550}"
bridge_mavlink_port="${BRIDGE_MAVLINK_PORT:-14551}"
for port_value in "$qgc_mavlink_port" "$bridge_mavlink_port"; do
  if [[ ! "$port_value" =~ ^[0-9]+$ ]] || (( port_value < 1 || port_value > 65535 )); then
    echo "MAVLink ports must be integers from 1 through 65535; got: $port_value" >&2
    exit 2
  fi
done
if [[ "$qgc_mavlink_port" == "$bridge_mavlink_port" ||
      "$qgc_mavlink_port" == "14552" || "$bridge_mavlink_port" == "14552" ]]; then
  echo "QGroundControl, ROS bridge, and internal Isaac port 14552 must be distinct." >&2
  exit 2
fi
echo "MAVLink QGroundControl endpoint: udp://127.0.0.1:$qgc_mavlink_port"
echo "MAVLink ROS 2 bridge endpoint: udp://127.0.0.1:$bridge_mavlink_port"
echo "MAVLink internal Isaac DVL endpoint: udp://127.0.0.1:14552"

sim_vehicle_args=(
  -v ArduSub -f vectored_6dof
  --model JSON:127.0.0.1
  --custom-location="$custom_location"
  --add-param-file="$isaac_uw_root/config/ardusub_isaac.parm"
  --out="udp:127.0.0.1:$qgc_mavlink_port"
  --out="udp:127.0.0.1:$bridge_mavlink_port"
  --out=udp:127.0.0.1:14552
  -w
)
if [[ "${ARDUSUB_NON_INTERACTIVE:-0}" == "1" ]]; then
  sim_vehicle_args+=(--mavproxy-args=--daemon)
else
  sim_vehicle_args+=(--console)
fi
exec ./Tools/autotest/sim_vehicle.py "${sim_vehicle_args[@]}"
