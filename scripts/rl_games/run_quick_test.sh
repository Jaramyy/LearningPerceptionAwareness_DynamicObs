#!/usr/bin/env bash
# Fully automated experiment loop: 3 environments × 5 trials each.
#
# Each trial:
#   1. Kill any leftover Gazebo / PX4 / ROS2 processes
#   2. Start PX4 + Gazebo in a clean env (no conda protobuf contamination)
#   3. Wait GZ_SETTLE seconds for Gazebo to stabilise
#   4. Launch ROS2 eval stack (bridge + student + monitor)
#   5. Publish /arm_message at 1 Hz until the drone arms and takes off
#   6. Wait for the monitor to exit (success / collision / timeout)
#   7. Report result and move to the next trial
#
# Usage
#   bash run_quick_test.sh [method] [checkpoint_path]

set -uo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="${PX4_DIR:-${HOME}/PX4-Autopilot}"
WORLDS_DIR="${WORLDS_DIR:-${PX4_DIR}/Tools/simulation/gz/worlds}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/../../results/quick_test}"
WORLD_PREFIX="iros_exp"

N_ENVS=3
N_TRIALS=5

GOAL_EAST=12.0
GOAL_NORTH=0.0
GOAL_ALT=2.5

VEL_SCALE=0.375
D_REFLEXIVE=1.0
D_PREDICTIVE=2.0
NO_ICP="${NO_ICP:-false}"  # set NO_ICP=true to disable ICP speed scaling

# Set DENSE_WORLD=true for high-density obstacle environments
# (20 pillars + 8 walls packed into x∈[1.5,11.5] y∈[-4.5,4.5])
DENSE_WORLD="${DENSE_WORLD:-false}"

PX4_BUILD_TIMEOUT=600  # seconds to wait for PX4 to finish building + start
PX4_SETTLE=30          # seconds to wait after PX4 starts, for Gazebo + DDS to stabilize
ARM_DELAY=8
TRIAL_TIMEOUT=90       # 20s AUTO_TAKEOFF + 70s navigation budget
HARD_TIMEOUT=$(( TRIAL_TIMEOUT + 40 ))

# ══════════════════════════════════════════════════════════════════════════════
# Arguments
# ══════════════════════════════════════════════════════════════════════════════
METHOD="${1:-pa}"
CKPT_DIR="${CKPT_DIR:-${HOME}/thesis_ws/IsaacLabExtensionTemplate/logs/dagger}"
CKPT_PATH="${2:-${CKPT_DIR}/student_latest.pth}"

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[ERROR] Checkpoint not found: ${CKPT_PATH}"
    echo "        Usage: bash run_quick_test.sh <method> /path/to/checkpoint.pth"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

export GZ_SIM_RESOURCE_PATH="${PX4_DIR}/Tools/simulation/gz/models:${PX4_DIR}/Tools/simulation/gz/worlds${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"

echo "======================================================"
echo "  Quick Test (fully automated)"
echo "  ${N_ENVS} envs × ${N_TRIALS} trials = $(( N_ENVS * N_TRIALS )) total"
echo "  Method     : ${METHOD}"
echo "  Checkpoint : ${CKPT_PATH}"
echo "  Results    : ${RESULTS_DIR}"
echo "======================================================"

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

_log() { echo "  [$(date '+%H:%M:%S')] $*"; }

_start_dds_agent() {
    # Kill any stale MicroXRCEAgent first — after pkill -9 px4 the old client
    # doesn't send a disconnect, leaving the agent holding stale session state.
    # Restarting gives a clean slate for each trial.
    pkill -f "MicroXRCEAgent" 2>/dev/null || true
    sleep 1
    _log "Starting MicroXRCEAgent udp4 -p 8888..."
    MicroXRCEAgent udp4 -p 8888 >> "${RESULTS_DIR}/dds_agent.log" 2>&1 &
    DDS_PID=$!
    sleep 2
    _log "MicroXRCEAgent started (PID=${DDS_PID})."
}

_kill_all() {
    _log "Killing Gazebo / PX4 / ROS2 nodes..."
    pkill -9 -f "gz sim"              2>/dev/null || true
    pkill -9 -f "ruby.*gz"            2>/dev/null || true
    pkill -9 -f "gzserver"            2>/dev/null || true
    pkill -9 -f "px4"                 2>/dev/null || true
    pkill -9 -f "gz_eval_monitor"     2>/dev/null || true
    pkill -9 -f "student_ros2_node"   2>/dev/null || true
    pkill -9 -f "pointcloud_to_scan"  2>/dev/null || true
    pkill -9 -f "parameter_bridge"    2>/dev/null || true
    pkill -9 -f "ros_gz_bridge"       2>/dev/null || true
    sleep 3
    _log "All processes killed."
}

_clean_px4_logs() {
    local log_dir="${HOME}/PX4-Autopilot/build/px4_sitl_default/rootfs/log"
    find "${log_dir}" -name "*.ulg" -delete 2>/dev/null || true
}

trap '_kill_all; pkill -f MicroXRCEAgent 2>/dev/null || true; echo "Aborted."; exit 1' INT TERM

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Generate missing worlds
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Checking worlds ──"
for seed in $(seq 0 $((N_ENVS - 1))); do
    SDF="${WORLDS_DIR}/${WORLD_PREFIX}_${seed}.sdf"
    if [[ ! -f "${SDF}" ]]; then
        echo "  Generating world ${seed}  (dense=${DENSE_WORLD})..."
        _dense_flag=""
        [[ "${DENSE_WORLD}" == "true" ]] && _dense_flag="--dense"
        python3 "${SCRIPT_DIR}/gz_world_gen.py" \
            --seed "${seed}" ${_dense_flag} --output "${SDF}"
    else
        echo "  World ${seed}: OK"
    fi
done

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Trial loop
# ══════════════════════════════════════════════════════════════════════════════
TOTAL=0; PASS=0; FAIL=0; TIMEOUT_COUNT=0

for seed in $(seq 0 $((N_ENVS - 1))); do
    WORLD="${WORLD_PREFIX}_${seed}"
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Environment ${seed}  —  world: ${WORLD}"
    echo "══════════════════════════════════════════════════════"

    for trial in $(seq 0 $((N_TRIALS - 1))); do

        TRIAL_ID=$(( seed * N_TRIALS + trial ))
        RESULT_JSON="${RESULTS_DIR}/${METHOD}_trial_${TRIAL_ID}.json"

        if [[ -f "${RESULT_JSON}" ]]; then
            _log "[env${seed} trial${trial}] already done — skipping."
            TOTAL=$(( TOTAL + 1 ))
            continue
        fi

        echo ""
        echo "  ────────────────────────────────────────────────"
        echo "  env=${seed}  trial=${trial}  id=${TRIAL_ID}"
        echo "  ────────────────────────────────────────────────"

        # ── 1. Kill everything and restart DDS agent ──────────────────
        _kill_all
        _start_dds_agent

        # ── 2. Start PX4 + Gazebo ─────────────────────────────────────
        # NOTE: do NOT use $(...) to call this — that would capture all
        # PX4 output into a variable and block forever.
        # PX4 output goes to a log file; tail shows it in the terminal.
        PX4_LOG="${RESULTS_DIR}/px4_env${seed}_t${trial}.log"
        _log "Starting PX4 + Gazebo (world=${WORLD})..."
        _log "PX4 log: ${PX4_LOG}"

        # setsid puts PX4 in its own process group so we can kill the whole
        # tree with kill -9 -PX4_PID (negative PID = kill group).
        # Pass ROS_DOMAIN_ID through env -i so PX4's uxrce_dds_client and
        # MicroXRCEAgent use the same DDS domain as the student node.
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
                PX4_GZ_WORLD='${WORLD}' \
                PX4_GZ_MODEL_POSE='1,1,0.5,0,0,0' \
                make px4_sitl gz_agi_drone_depth
            " > "${PX4_LOG}" 2>&1 &
        PX4_PGID=$!   # setsid makes this the process-group leader

        # Stream PX4 log to terminal with prefix (background tail)
        tail -f "${PX4_LOG}" 2>/dev/null | sed 's/^/    [PX4] /' &
        TAIL_PID=$!

        # Wait for PX4 binary to actually start (not just cmake/ninja build).
        # Marker: "uxrce_dds_client] init UDP" appears once PX4 runs.
        # Timeout of PX4_BUILD_TIMEOUT handles first-time compilation.
        _log "Waiting for PX4 to start (build may take up to ${PX4_BUILD_TIMEOUT}s)..."
        PX4_READY=0
        for _i in $(seq 1 $(( PX4_BUILD_TIMEOUT / 5 ))); do
            if grep -q "uxrce_dds_client.*init UDP" "${PX4_LOG}" 2>/dev/null; then
                PX4_READY=1
                break
            fi
            sleep 5
        done

        if (( PX4_READY == 0 )); then
            _log "ERROR: PX4 never started within ${PX4_BUILD_TIMEOUT}s — skipping trial."
            kill "${TAIL_PID}"        2>/dev/null || true
            kill -9 -"${PX4_PGID}"   2>/dev/null || true
            FAIL=$(( FAIL + 1 ))
            TOTAL=$(( TOTAL + 1 ))
            continue
        fi
        _log "PX4 running. Settling ${PX4_SETTLE}s for Gazebo + DDS bridge..."
        sleep "${PX4_SETTLE}"

        # Diagnostic: show which /fmu topics are visible in ROS2
        _log "FMU topics visible:"
        ros2 topic list 2>/dev/null | grep fmu | sed 's/^/    /' || _log "  (none — check MicroXRCEAgent)"

        # ── 3. Launch ROS2 eval stack ─────────────────────────────────
        _log "Launching ROS2 eval stack..."
        LAUNCH_LOG="${RESULTS_DIR}/launch_env${seed}_t${trial}.log"
        STEP_LOG="${RESULTS_DIR}/${METHOD}_trial_${TRIAL_ID}_steps.csv"

        ros2 launch "${SCRIPT_DIR}/gz_eval.launch.py" \
            method:="${METHOD}" \
            checkpoint:="${CKPT_PATH}" \
            seed:="${seed}" \
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
        _log "ROS2 stack PID=${LAUNCH_PID}  log: ${LAUNCH_LOG}"

        # ── 4. Auto-arm at 1 Hz ───────────────────────────────────────
        (
            sleep "${ARM_DELAY}"
            echo "    [ARM] publishing /arm_message @ 1 Hz..."
            ros2 topic pub --rate 1 /arm_message std_msgs/msg/Bool "data: true" || true
        ) &
        ARM_PID=$!
        _log "Auto-arm will fire in ${ARM_DELAY}s (ARM_PID=${ARM_PID})"

        # ── 5. Wait for monitor to finish ─────────────────────────────
        _log "Waiting for trial to complete (timeout=${TRIAL_TIMEOUT}s)..."
        WAIT_START=$(date +%s)
        while kill -0 "${LAUNCH_PID}" 2>/dev/null; do
            sleep 2
            ELAPSED=$(( $(date +%s) - WAIT_START ))
            if (( ELAPSED > HARD_TIMEOUT )); then
                _log "Hard timeout (${HARD_TIMEOUT}s) — ending trial."
                break
            fi
        done
        _log "Trial finished after $(( $(date +%s) - WAIT_START ))s."

        # ── 6. Tear down ──────────────────────────────────────────────
        kill "${ARM_PID}"    2>/dev/null || true
        kill "${TAIL_PID}"   2>/dev/null || true
        kill "${LAUNCH_PID}" 2>/dev/null || true
        kill -9 -"${PX4_PGID}" 2>/dev/null || true  # kill entire PX4 process group
        wait "${LAUNCH_PID}" 2>/dev/null || true
        _clean_px4_logs

        # ── 7. Report ─────────────────────────────────────────────────
        TOTAL=$(( TOTAL + 1 ))
        if [[ -f "${RESULT_JSON}" ]]; then
            RLINE=$(python3 - << PYEOF 2>/dev/null || echo "parse-error"
import json, sys
d = json.load(open("${RESULT_JSON}"))
r   = d.get("result", "?")
orb = round(d.get("orb_mean", 0), 1)
lat = d.get("latency_mean_ms", "?")
print(f"{r}  orb={orb}  lat={lat}ms", end="")
PYEOF
)
            _log "RESULT [env${seed} trial${trial}]: ${RLINE}"
            case "${RLINE}" in
                success*)  PASS=$(( PASS + 1 )) ;;
                timeout*)  TIMEOUT_COUNT=$(( TIMEOUT_COUNT + 1 )) ;;
                *)         FAIL=$(( FAIL + 1 )) ;;
            esac
        else
            _log "RESULT [env${seed} trial${trial}]: NO RESULT — check ${LAUNCH_LOG}"
            FAIL=$(( FAIL + 1 ))
        fi

        sleep 2
    done   # trial loop
done   # env loop

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
_kill_all

echo ""
echo "══════════════════════════════════════════════"
echo "  Experiment complete"
printf "  Total    : %d / %d\n" "${TOTAL}" "$(( N_ENVS * N_TRIALS ))"
printf "  Success  : %d\n" "${PASS}"
printf "  Collision: %d\n" "${FAIL}"
printf "  Timeout  : %d\n" "${TIMEOUT_COUNT}"
echo "  Results  : ${RESULTS_DIR}/"
echo "══════════════════════════════════════════════"
