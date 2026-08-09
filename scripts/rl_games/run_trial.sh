#!/usr/bin/env bash
# Run one evaluation trial — PX4 is reused if already up for the same world.
#
# Usage:
#   bash run_trial.sh <seed> <trial_id> [method] [checkpoint]
#
# Examples:
#   bash run_trial.sh 0 0            # env 0, trial 0, method=pa
#   bash run_trial.sh 1 6 pa         # env 1, trial 1 (flat id 6)
#   bash run_trial.sh 2 12 pa /path/to/student.pth
#
# seed:      0 | 1 | 2  →  world iros_exp_<seed>
# trial_id:  flat index (seed 0 → 0-4, seed 1 → 5-9, seed 2 → 10-14)
# method:    label for result JSON (default: pa)
# checkpoint: .pth file (default: logs/dagger/student_latest.pth)
#
# Between trials of the SAME world, just run the script again — PX4 and
# Gazebo keep running, only the ROS2 eval nodes are restarted.
# To switch worlds, the script detects the change and restarts everything.

set -uo pipefail

SEED="${1:?'Usage: bash run_trial.sh <seed> <trial_id> [method] [checkpoint]'}"
TRIAL_ID="${2:?'Usage: bash run_trial.sh <seed> <trial_id> [method] [checkpoint]'}"
METHOD="${3:-pa}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PX4_DIR="${PX4_DIR:-${HOME}/PX4-Autopilot}"
WORLDS_DIR="${WORLDS_DIR:-${PX4_DIR}/Tools/simulation/gz/worlds}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/quick_test}"
CKPT="${4:-${REPO_ROOT}/logs/dagger/student_latest.pth}"
WORLD="${WORLD:-iros_exp_${SEED}}"

# ── Config ────────────────────────────────────────────────────────────────────
GOAL_EAST=12.0;  GOAL_NORTH=0.0;  GOAL_ALT=2.5
VEL_SCALE=0.375; D_REFLEXIVE=1.0; D_PREDICTIVE=2.0
NO_ICP="${NO_ICP:-false}"  # set NO_ICP=true to disable ICP speed scaling
ARM_DELAY=8        # seconds before first arm message
ARM_DURATION=25    # seconds to keep publishing arm — covers startup phase only
TRIAL_TIMEOUT=90   # 20s AUTO_TAKEOFF + 70s navigation budget
HARD_TIMEOUT=$(( TRIAL_TIMEOUT + 40 ))
PX4_SETTLE=30      # settle after PX4 starts (seconds)
PX4_BUILD_TIMEOUT=600  # handles first-time cmake/ninja build

# Set DENSE_WORLD=true for high-density obstacle environments
# (20 pillars + 8 walls packed into x∈[1.5,11.5] y∈[-4.5,4.5])
DENSE_WORLD="${DENSE_WORLD:-false}"

# State file — tracks which world PX4 is currently running for
PX4_WORLD_FILE="/tmp/px4_current_world_${USER}"
PX4_LOG="${RESULTS_DIR}/px4_${WORLD}.log"

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "${RESULTS_DIR}"
export GZ_SIM_RESOURCE_PATH="${PX4_DIR}/Tools/simulation/gz/models:${PX4_DIR}/Tools/simulation/gz/worlds${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"

_log() { echo "  [$(date '+%H:%M:%S')] $*"; }

# ── Check checkpoint ──────────────────────────────────────────────────────────
if [[ ! -f "${CKPT}" ]]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    exit 1
fi

echo ""
echo "  ┌─────────────────────────────────────────┐"
echo "  │  run_trial.sh  seed=${SEED}  trial_id=${TRIAL_ID}  method=${METHOD}"
echo "  │  world       : ${WORLD}"
echo "  │  checkpoint  : $(basename "${CKPT}")"
echo "  │  results     : ${RESULTS_DIR}/"
echo "  └─────────────────────────────────────────┘"
echo ""

# Skip if result already exists
RESULT_JSON="${RESULTS_DIR}/${METHOD}_trial_${TRIAL_ID}.json"
if [[ -f "${RESULT_JSON}" ]]; then
    _log "Result already exists: ${RESULT_JSON}"
    python3 - << PYEOF
import json
d = json.load(open("${RESULT_JSON}"))
print(f"  Result: {d.get('result','?')}  time={d.get('flight_time_s',0):.1f}s  orb={d.get('orb_mean',0):.0f}  lat={d.get('latency_mean_ms',0):.0f}ms")
PYEOF
    exit 0
fi

# ── Kill only ROS2 trial nodes (leaves PX4 + Gazebo running) ─────────────────
_kill_trial_nodes() {
    pkill -f "gz_eval_monitor"    2>/dev/null || true
    pkill -f "student_ros2_node"  2>/dev/null || true
    pkill -f "pointcloud_to_scan" 2>/dev/null || true
    pkill -f "parameter_bridge"   2>/dev/null || true
    pkill -f "ros_gz_bridge"      2>/dev/null || true
    sleep 2
}

# ── Kill everything including PX4 + Gazebo ───────────────────────────────────
_kill_all() {
    _log "Killing all processes (Gazebo, PX4, ROS2)..."
    _kill_trial_nodes
    pkill -9 -f "MicroXRCEAgent" 2>/dev/null || true
    pkill -9 -f "gz sim"         2>/dev/null || true
    pkill -9 -f "ruby.*gz"       2>/dev/null || true
    pkill -9 -f "gzserver"       2>/dev/null || true
    pkill -9 -f "px4"            2>/dev/null || true
    rm -f "${PX4_WORLD_FILE}"
    sleep 3
}

# ── Delete PX4 flight logs (ulg) to keep disk free ───────────────────────────
_clean_px4_logs() {
    local log_dir="${HOME}/PX4-Autopilot/build/px4_sitl_default/rootfs/log"
    if [[ -d "${log_dir}" ]]; then
        find "${log_dir}" -name "*.ulg" -delete 2>/dev/null || true
        _log "PX4 flight logs cleared."
    fi
}

trap '_log "Interrupted — cleaning up trial nodes."; _kill_trial_nodes; exit 1' INT TERM

# ── Decide: reuse PX4+Gazebo or restart ──────────────────────────────────────
CURRENT_WORLD="$(cat "${PX4_WORLD_FILE}" 2>/dev/null || true)"
PX4_ALIVE=false
GZ_ALIVE=false
pgrep -f "px4" > /dev/null 2>&1 && PX4_ALIVE=true
pgrep -f "gz sim\|ruby.*gz\|gzserver\|gz_agi" > /dev/null 2>&1 && GZ_ALIVE=true

if [[ "${PX4_ALIVE}" == "true" && "${GZ_ALIVE}" == "true" && "${CURRENT_WORLD}" == "${WORLD}" ]]; then
    _log "PX4 + Gazebo already running for ${WORLD} — reusing."
    _kill_trial_nodes

else
    if [[ "${PX4_ALIVE}" == "true" && "${GZ_ALIVE}" == "false" ]]; then
        _log "PX4 running but Gazebo is gone — restarting both."
    elif [[ "${CURRENT_WORLD}" != "${WORLD}" ]]; then
        _log "World changed (${CURRENT_WORLD} -> ${WORLD}) — restarting."
    fi
    # ── Restart MicroXRCEAgent + PX4 ─────────────────────────────────────────
    _kill_all

    _log "Starting MicroXRCEAgent..."
    MicroXRCEAgent udp4 -p 8888 >> "${RESULTS_DIR}/dds_agent.log" 2>&1 &
    sleep 2

    _log "Starting PX4 + Gazebo (world=${WORLD})..."
    _log "Build log: ${PX4_LOG}"

    # setsid → own process group so kill -9 -PGID cleans up the whole tree.
    # env -i → avoid conda protobuf conflicts; ROS_DOMAIN_ID passed explicitly
    # so PX4's uxrce_dds_client, MicroXRCEAgent, and ROS2 nodes share the same domain.
    setsid env -i \
        HOME="${HOME}" \
        USER="${USER:-$(whoami)}" \
        TERM="xterm" \
        PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        DISPLAY="${DISPLAY:-:0}" \
        XAUTHORITY="${XAUTHORITY:-${HOME}/.Xauthority}" \
        GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH}" \
        ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
        __NV_PRIME_RENDER_OFFLOAD=1 \
        __GLX_VENDOR_LIBRARY_NAME=nvidia \
        bash -c "
            source /opt/ros/humble/setup.bash 2>/dev/null || true
            cd '${PX4_DIR}'
            PX4_GZ_WORLD='${WORLD}' PX4_GZ_MODEL_POSE='1,1,0.5,0,0,0' \
            make px4_sitl gz_agi_drone_depth
        " > "${PX4_LOG}" 2>&1 &
    echo "${WORLD}" > "${PX4_WORLD_FILE}"

    # Wait for PX4 binary to actually start (not just cmake/ninja)
    _log "Waiting for PX4 to start (first build can take ${PX4_BUILD_TIMEOUT}s)..."
    PX4_READY=0
    for _i in $(seq 1 $(( PX4_BUILD_TIMEOUT / 5 ))); do
        grep -q "uxrce_dds_client.*init UDP" "${PX4_LOG}" 2>/dev/null \
            && PX4_READY=1 && break
        sleep 5
    done
    if (( PX4_READY == 0 )); then
        _log "ERROR: PX4 never started — check ${PX4_LOG}"
        exit 1
    fi

    _log "PX4 running. Settling ${PX4_SETTLE}s for Gazebo + DDS..."
    sleep "${PX4_SETTLE}"
    _clean_px4_logs   # remove ulg written during startup
fi

# ── Generate world SDF if missing ────────────────────────────────────────────
SDF="${WORLDS_DIR}/${WORLD}.sdf"
if [[ ! -f "${SDF}" ]]; then
    _log "Generating world SDF: ${SDF}  (dense=${DENSE_WORLD})"
    _dense_flag=""
    [[ "${DENSE_WORLD}" == "true" ]] && _dense_flag="--dense"
    python3 "${SCRIPT_DIR}/gz_world_gen.py" \
        --seed "${SEED}" ${_dense_flag} --output "${SDF}"
fi

# ── Launch ROS2 eval stack ────────────────────────────────────────────────────
LAUNCH_LOG="${RESULTS_DIR}/launch_seed${SEED}_t${TRIAL_ID}.log"
STEP_LOG="${RESULTS_DIR}/${METHOD}_trial_${TRIAL_ID}_steps.csv"
_log "Launching ROS2 eval stack..."

ros2 launch "${SCRIPT_DIR}/gz_eval.launch.py" \
    method:="${METHOD}" \
    checkpoint:="${CKPT}" \
    seed:="${SEED}" \
    trial_id:="${TRIAL_ID}" \
    goal_east:="${GOAL_EAST}" \
    goal_north:="${GOAL_NORTH}" \
    goal_alt:="${GOAL_ALT}" \
    worlds_dir:="${WORLDS_DIR}" \
    result_dir:="${RESULTS_DIR}" \
    timeout:="${TRIAL_TIMEOUT}" \
    vel_scale:="${VEL_SCALE}" \
    d_reflexive:="${D_REFLEXIVE}" \
    d_predictive:="${D_PREDICTIVE}" \
    no_icp:="${NO_ICP}" \
    trial_log:="${STEP_LOG}" \
    rviz:=true \
    2>&1 | tee "${LAUNCH_LOG}" &
LAUNCH_PID=$!

# ── Auto-arm: 1 Hz for ARM_DURATION seconds only (startup phase) ─────────────
(
    sleep "${ARM_DELAY}"
    _log "Publishing /arm_message @ 1 Hz for ${ARM_DURATION}s (startup phase)..."
    timeout "${ARM_DURATION}" \
        ros2 topic pub --rate 1 /arm_message std_msgs/msg/Bool "data: true" || true
    _log "Arm phase complete."
) &
ARM_PID=$!

# ── Wait for monitor to finish ────────────────────────────────────────────────
_log "Waiting for trial result (hard timeout ${HARD_TIMEOUT}s)..."
WAIT_START=$(date +%s)
while kill -0 "${LAUNCH_PID}" 2>/dev/null; do
    sleep 2
    (( $(date +%s) - WAIT_START > HARD_TIMEOUT )) && { _log "Hard timeout."; break; }
done
ELAPSED=$(( $(date +%s) - WAIT_START ))
_log "Trial ended after ${ELAPSED}s."

# ── Tear down trial nodes (PX4 stays running for next trial) ─────────────────
kill "${ARM_PID}"    2>/dev/null || true
kill "${LAUNCH_PID}" 2>/dev/null || true
wait "${LAUNCH_PID}" 2>/dev/null || true
_kill_trial_nodes
_clean_px4_logs
# Remove full step CSV (1-2 MB/trial) — crash buffer _last5s.csv is enough for analysis
rm -f "${STEP_LOG}"

# ── Report ────────────────────────────────────────────────────────────────────
if [[ -f "${RESULT_JSON}" ]]; then
    python3 - << PYEOF
import json
d = json.load(open("${RESULT_JSON}"))
r   = d.get("result", "?")
t   = d.get("flight_time_s", 0)
orb = d.get("orb_mean", 0)
lat = d.get("latency_mean_ms", 0)
print(f"\n  Result : {r}")
print(f"  Time   : {t:.1f}s")
print(f"  ORB    : {orb:.0f}")
print(f"  Latency: {lat:.0f}ms")
print(f"  File   : ${RESULT_JSON}")
PYEOF
else
    _log "No result file written — check ${LAUNCH_LOG}"
fi
echo ""
