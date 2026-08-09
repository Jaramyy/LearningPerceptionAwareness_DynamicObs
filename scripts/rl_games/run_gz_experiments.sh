#!/usr/bin/env bash
# Run all IROS Gazebo simulation evaluation trials.
#
# For each method × seed this script:
#   1. Generates the textured SDF world (once, if not already present)
#   2. Starts PX4 SITL + Gazebo in a background process
#   3. Waits for Gazebo to stabilise
#   4. Runs the ROS2 eval launch (bridge + student + monitor)
#   5. Waits for the monitor to finish (success / collision / timeout)
#   6. Kills PX4 + Gazebo cleanly before the next trial
#
# After all trials, runs aggregate_results.py to print Table I & II from
# the paper.
#
# Usage
# -----
#   bash run_gz_experiments.sh [pa|nopa|navrl|panther|all] [N_TRIALS]
#
# Examples
#   bash run_gz_experiments.sh pa 20      # 20 trials of the PA method
#   bash run_gz_experiments.sh all 20     # 20 × 4 = 80 trials (all methods)
#
# Prerequisites
# -------------
#   • PX4-Autopilot at ~/PX4-Autopilot (built with gz_agi_drone_depth target)
#   • ROS2 Humble sourced
#   • Checkpoints under CKPT_DIR (set below)
#   • python3 with rclpy, px4_msgs, sensor_msgs, cv2 (opencv-python)
#
# Environment overrides
# ---------------------
#   PX4_DIR     path to PX4-Autopilot      (default ~/PX4-Autopilot)
#   CKPT_DIR    path to checkpoint dir      (default ~/thesis_ws/IsaacLabExtensionTemplate/logs/dagger)
#   WORLDS_DIR  where SDF worlds are stored (default $PX4_DIR/Tools/simulation/gz/worlds)
#   RESULTS_DIR where JSON results go        (default ./results)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="${PX4_DIR:-${HOME}/PX4-Autopilot}"
CKPT_DIR="${CKPT_DIR:-${HOME}/thesis_ws/IsaacLabExtensionTemplate/logs/dagger}"
WORLDS_DIR="${WORLDS_DIR:-${PX4_DIR}/Tools/simulation/gz/worlds}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/../../results}"
WORLD_PREFIX="iros_exp"

# Checkpoint files per method.
# Right now there is one trained student (student_latest.pth = PA policy).
# When you train No-PA / NAV-RL baselines, save them as student_nopa_latest.pth etc.
declare -A CKPT=(
    [pa]="${CKPT_DIR}/student_latest.pth"
    [nopa]="${CKPT_DIR}/student_nopa_latest.pth"
    [navrl]="${CKPT_DIR}/student_navrl_latest.pth"
    [panther]="${CKPT_DIR}/student_latest.pth"      # PANTHER uses same student + its own nav
)

# Navigation goal (PX4 NED, matches world generator GOAL_ENU=(12,0,1.5))
GOAL_EAST=12.0
GOAL_NORTH=0.0
GOAL_ALT=1.5

# ICP and velocity parameters
VEL_SCALE=0.5
D_REFLEXIVE=1.5
D_PREDICTIVE=4.5

# Trial timeout (seconds)
TIMEOUT=60

# Number of pillars / walls per world (match gz_world_gen.py defaults)
N_PILLARS=8
N_WALLS=3

# PX4 settle time (seconds to wait after launching Gazebo)
GZ_SETTLE=8

# ── Parse arguments ───────────────────────────────────────────────────────────
METHOD_ARG="${1:-pa}"
N_TRIALS="${2:-20}"

if [[ "${METHOD_ARG}" == "all" ]]; then
    METHODS=(pa nopa navrl panther)
else
    METHODS=("${METHOD_ARG}")
fi

echo "=================================================="
echo " IROS Gazebo Evaluation"
echo " Methods   : ${METHODS[*]}"
echo " N trials  : ${N_TRIALS}"
echo " Worlds dir: ${WORLDS_DIR}"
echo " Results   : ${RESULTS_DIR}"
echo "=================================================="

# ── Step 1: Generate worlds (skip if already present) ────────────────────────
echo ""
echo "── Generating worlds (seed 0 to $((N_TRIALS-1))) ──"
for seed in $(seq 0 $((N_TRIALS - 1))); do
    SDF_PATH="${WORLDS_DIR}/${WORLD_PREFIX}_${seed}.sdf"
    if [[ ! -f "${SDF_PATH}" ]]; then
        python3 "${SCRIPT_DIR}/gz_world_gen.py" \
            --seed "${seed}" \
            --n_pillars "${N_PILLARS}" \
            --n_walls "${N_WALLS}" \
            --output "${SDF_PATH}"
    else
        echo "  World ${seed} already exists, skipping."
    fi
done
echo "All worlds ready."

# ── Guard: remove conda-polluted CMakeCache before first build ───────────────
# Root cause: if cmake was configured while isaaclab_pa conda was active it
# caches conda include paths (protobuf 6.33.6) → version mismatch with
# gz-msgs10 headers (generated for protobuf 3.12.4) → compile error.
# Fix: delete CMakeCache.txt so cmake re-detects with the system env below.
PX4_CMAKE_CACHE="${PX4_DIR}/build/px4_sitl_default/CMakeCache.txt"
if [[ -f "${PX4_CMAKE_CACHE}" ]] && grep -q "miniconda\|anaconda\|conda" "${PX4_CMAKE_CACHE}"; then
    echo "[FIX] CMakeCache contains conda paths — removing to force clean re-configure."
    rm "${PX4_CMAKE_CACHE}"
fi

# ── Clean environment for PX4 builds ─────────────────────────────────────────
# Strip conda from PATH / CPATH / LD_LIBRARY_PATH so the system protobuf
# headers (3.12.4, matching gz-msgs10) are found first during compilation.
_clean_path() {
    echo "${1}" | tr ':' '\n' | grep -v "/miniconda\|/anaconda\|/conda" | paste -sd:
}
_PX4_PATH=$(_clean_path "${PATH}")
_PX4_CPATH=$(_clean_path "${CPATH:-}")
_PX4_LD=$(_clean_path "${LD_LIBRARY_PATH:-}")
_PX4_CMAKE_PREFIX=$(_clean_path "${CMAKE_PREFIX_PATH:-}")

# ── Helper: kill background PX4 + Gazebo ────────────────────────────────────
_kill_sim() {
    echo "  Stopping simulation..."
    # Kill PX4 SITL
    pkill -f "px4" 2>/dev/null || true
    # Kill Gazebo
    pkill -f "gz sim"  2>/dev/null || true
    pkill -f "gzserver" 2>/dev/null || true
    # Kill leftover ROS2 nodes from this launch
    pkill -f "gz_eval_monitor" 2>/dev/null || true
    pkill -f "student_ros2_node_icp" 2>/dev/null || true
    pkill -f "pointcloud_to_scan" 2>/dev/null || true
    pkill -f "parameter_bridge" 2>/dev/null || true
    sleep 2
}

# Clean up on ctrl+c
trap '_kill_sim; echo "Aborted."; exit 1' INT TERM

# ── Step 2: Run trials ───────────────────────────────────────────────────────
for METHOD in "${METHODS[@]}"; do
    CKPT_PATH="${CKPT[${METHOD}]}"

    if [[ ! -f "${CKPT_PATH}" ]]; then
        echo "[WARN] Checkpoint not found for method '${METHOD}': ${CKPT_PATH}"
        echo "       Skipping method ${METHOD}."
        continue
    fi

    METHOD_RESULTS="${RESULTS_DIR}/${METHOD}"
    mkdir -p "${METHOD_RESULTS}"

    echo ""
    echo "══ Method: ${METHOD}  (${N_TRIALS} trials) ══"

    for trial in $(seq 0 $((N_TRIALS - 1))); do
        seed=${trial}    # one unique world per trial
        RESULT_JSON="${METHOD_RESULTS}/${METHOD}_trial_${trial}.json"

        if [[ -f "${RESULT_JSON}" ]]; then
            echo "  Trial ${trial}: result already exists, skipping."
            continue
        fi

        echo ""
        echo "  ── Trial ${trial} / ${N_TRIALS} (seed=${seed}) ──"

        # ── Launch PX4 SITL + Gazebo (conda paths stripped) ─────────
        # env -i was too aggressive (dropped GZ_SIM_RESOURCE_PATH).
        # Instead export cleaned path vars into a subshell so Gazebo can
        # still locate its models while conda protobuf headers stay out.
        echo "  Starting Gazebo with world ${WORLD_PREFIX}_${seed}..."
        (
            export PATH="${_PX4_PATH}"
            export CPATH="${_PX4_CPATH}"
            export LD_LIBRARY_PATH="${_PX4_LD}"
            export CMAKE_PREFIX_PATH="${_PX4_CMAKE_PREFIX}"
            unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PYTHON_EXE \
                  CONDA_EXE CONDA_SHLVL _CE_CONDA _CE_M
            # Ensure Gazebo can find PX4 models for drone spawning
            export GZ_SIM_RESOURCE_PATH="${PX4_DIR}/Tools/simulation/gz/models:${PX4_DIR}/Tools/simulation/gz/worlds${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"
            cd "${PX4_DIR}"
            PX4_GZ_WORLD="${WORLD_PREFIX}_${seed}" \
            PX4_GZ_MODEL_POSE="1,1,0.5,0,0,0" \
            make px4_sitl gz_agi_drone_depth \
                2>&1 | sed 's/^/  [PX4] /'
        ) &
        PX4_PID=$!

        echo "  Waiting ${GZ_SETTLE}s for Gazebo to settle..."
        sleep "${GZ_SETTLE}"

        # ── Launch ROS2 stack + eval monitor ─────────────────────────
        echo "  Starting ROS2 eval stack..."
        ROS2_LAUNCH_LOG="${METHOD_RESULTS}/launch_trial_${trial}.log"
        ros2 launch "${SCRIPT_DIR}/gz_eval.launch.py" \
            method:="${METHOD}" \
            checkpoint:="${CKPT_PATH}" \
            seed:="${seed}" \
            trial_id:="${trial}" \
            goal_east:="${GOAL_EAST}" \
            goal_north:="${GOAL_NORTH}" \
            goal_alt:="${GOAL_ALT}" \
            worlds_dir:="${WORLDS_DIR}" \
            result_dir:="${METHOD_RESULTS}" \
            timeout:="${TIMEOUT}" \
            vel_scale:="${VEL_SCALE}" \
            d_reflexive:="${D_REFLEXIVE}" \
            d_predictive:="${D_PREDICTIVE}" \
            rviz:=false \
            2>&1 | tee "${ROS2_LAUNCH_LOG}" &
        LAUNCH_PID=$!

        # Wait for the monitor to finish (it exits when trial ends)
        # Add a hard timeout = trial timeout + startup overhead
        HARD_TIMEOUT=$(( TIMEOUT + 30 ))
        WAIT_START=$(date +%s)
        while kill -0 "${LAUNCH_PID}" 2>/dev/null; do
            sleep 2
            NOW=$(date +%s)
            ELAPSED=$(( NOW - WAIT_START ))
            if (( ELAPSED > HARD_TIMEOUT )); then
                echo "  [WARN] Hard timeout reached for trial ${trial}."
                break
            fi
        done

        # ── Tear down ─────────────────────────────────────────────────
        kill "${LAUNCH_PID}" 2>/dev/null || true
        wait "${LAUNCH_PID}" 2>/dev/null || true
        kill "${PX4_PID}"    2>/dev/null || true
        _kill_sim

        # Report result
        if [[ -f "${RESULT_JSON}" ]]; then
            RESULT=$(python3 -c "import json; d=json.load(open('${RESULT_JSON}')); \
                print(d['result'], 'orb='+str(round(d['orb_mean'],1)), \
                'lat='+str(d['latency_mean_ms'])+'ms')")
            echo "  Trial ${trial}: ${RESULT}"
        else
            echo "  Trial ${trial}: [no result written — check launch log]"
        fi

        sleep 3   # brief pause before next trial
    done
done

# ── Step 3: Aggregate results ────────────────────────────────────────────────
echo ""
echo "══ Aggregating results ══"
python3 "${SCRIPT_DIR}/gz_aggregate_results.py" \
    --results_dir "${RESULTS_DIR}" \
    --methods "${METHODS[@]}" \
    --n_trials "${N_TRIALS}" \
    2>&1 || echo "[WARN] Aggregator not found or failed — see ${RESULTS_DIR}/ for raw JSONs."

echo ""
echo "Done. Results in: ${RESULTS_DIR}/"
