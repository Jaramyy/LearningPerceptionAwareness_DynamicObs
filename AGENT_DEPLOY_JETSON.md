# Agent Handoff — PA Student Policy on Real Drone (Jetson Orin)

**You are a Claude agent running on a Jetson Orin companion computer.**  
Your job is to deploy the student policy, arm the drone, run one flight to the goal, and report the result.  
Read this entire file before taking any action.

---

## 0. What You Are Deploying

- **Policy:** `student_ros2_node_icp.py` — a 16-D → 4-D MLP velocity controller with MODANC ICP speed adaptation
- **Checkpoint:** `logs/dagger/student_latest.pth` — 200-iter DAgger distilled from a PA teacher, 44.5% sim success
- **Control mode:** PX4 Offboard — velocity setpoints at 100 Hz over DDS/uXRCE
- **Perception:** front-facing 2-D LiDAR, 5 angular sectors, ±90°

The policy runs on the **Jetson Orin GPU** (`--device cuda`). No Isaac Sim, no ROS2 launch file needed.

---

## 1. Clone the Repo (first time only)

```bash
# HTTPS (recommended — no SSH key needed on Jetson)
git clone https://github.com/Jaramyy/LearningPerceptionAwareness_DynamicObs.git \
    ~/thesis_ws/IsaacLabExtensionTemplate

# SSH (if your Jetson has a GitHub key configured)
git clone git@github.com:Jaramyy/LearningPerceptionAwareness_DynamicObs.git \
    ~/thesis_ws/IsaacLabExtensionTemplate
```

**Verify:**
```bash
ls ~/thesis_ws/IsaacLabExtensionTemplate/scripts/rl_games/student_ros2_node_icp.py
```

If the repo already exists, pull latest:
```bash
cd ~/thesis_ws/IsaacLabExtensionTemplate && git pull
```

---

## 2. Copy the Student Checkpoint

The checkpoint is **not in git** (too large). Copy it from the dev machine:

```bash
# Run this on the DEV MACHINE (not the Jetson):
scp logs/dagger/student_latest.pth \
    jetson:~/thesis_ws/IsaacLabExtensionTemplate/logs/dagger/student_latest.pth

# OR from Jetson, pull from dev machine:
mkdir -p ~/thesis_ws/IsaacLabExtensionTemplate/logs/dagger
scp devmachine:~/thesis_ws/IsaacLabExtensionTemplate/logs/dagger/student_latest.pth \
    ~/thesis_ws/IsaacLabExtensionTemplate/logs/dagger/
```

**Verify on Jetson:**
```bash
REPO=~/thesis_ws/IsaacLabExtensionTemplate
CKPT=$REPO/logs/dagger/student_latest.pth
NODE=$REPO/scripts/rl_games/student_ros2_node_icp.py

test -f "$CKPT" && echo "CHECKPOINT OK" || echo "CHECKPOINT MISSING"
test -f "$NODE" && echo "NODE OK"       || echo "NODE MISSING"
```

**Expected:** both print `OK`. Do not proceed until both pass.

---

## 3. Environment Check

Run each block and verify the expected output before proceeding.

### 3a. ROS2
```bash
source /opt/ros/humble/setup.bash
ros2 --version
```
**Expected:** `ros2cli 0.18.x` or similar Humble version.

### 3b. px4_msgs
```bash
source /opt/ros/humble/setup.bash
ros2 interface show px4_msgs/msg/VehicleLocalPosition | head -5
```
**Expected:** field definitions starting with `# Vehicle local position ...`  
**If missing:** `sudo apt install ros-humble-px4-msgs` OR build from source matching firmware version.

### 3c. PyTorch with CUDA (Jetson GPU)

The policy **must** run on the Jetson Orin GPU. Check first:
```bash
python3 -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```
**Expected:** version string + `CUDA: True` + `Orin` in device name.

**If torch is missing or CUDA is False — install the JetPack-compatible wheel:**

```bash
# Step 1: check your JetPack version
cat /etc/nv_tegra_release | head -1
# or: dpkg -l | grep jetpack

# Step 2a — JetPack 6.x (L4T 36.x, Ubuntu 22.04):
pip3 install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# Step 2b — JetPack 5.x (L4T 35.x, Ubuntu 20.04):
# Download the wheel from https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# Then install:
pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

After install, re-run the check. **Do not proceed if CUDA is False.**

### 3d. Other Python deps
```bash
python3 -c "import rclpy, sensor_msgs, geometry_msgs, visualization_msgs, tf2_ros; print('deps OK')"
```
**Expected:** `deps OK`

---

## 4. Hardware Connections (verify before powering motors)

| Connection | Check command | Expected |
|-----------|---------------|----------|
| PX4 → Jetson serial | `ls /dev/ttyUSB* /dev/ttyACM* /dev/ttyTHS*` | At least one device |
| LiDAR → Jetson USB | `ls /dev/ttyUSB*` or `ls /dev/rplidar` | Device present |
| Jetson power | `cat /sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power0_input 2>/dev/null \| head -1` | Non-zero |

---

## 5. Start MicroXRCEAgent (PX4 ↔ ROS2 DDS Bridge)

This bridges PX4 uXRCE-DDS to ROS2 topics. Must be running before the student node.

```bash
# Serial connection (most common with Jetson UART):
MicroXRCEAgent serial --dev /dev/ttyTHS0 -b 921600 &

# OR: USB-serial adapter:
MicroXRCEAgent serial --dev /dev/ttyUSB0 -b 921600 &

# OR: UDP (if PX4 connected over WiFi/ethernet):
MicroXRCEAgent udp4 -p 8888 &
```

**Verify it is working (wait ~5 seconds after start):**
```bash
source /opt/ros/humble/setup.bash
ros2 topic list | grep fmu
```
**Expected:** lines including `/fmu/out/vehicle_local_position`, `/fmu/out/vehicle_attitude`, `/fmu/out/vehicle_status`  
**If empty:** check baud rate, port name, and that PX4 has `uxrce_dds_client` running (`nsh> uxrce_dds_client status`)

---

## 6. Start LiDAR Driver

### RPLIDAR (most common):
```bash
# Install if needed: sudo apt install ros-humble-rplidar-ros
source /opt/ros/humble/setup.bash
ros2 launch rplidar_ros rplidar_a2_launch.py \
    serial_port:=/dev/ttyUSB0 \
    frame_id:=laser \
    scan_topic:=/scan &
sleep 3
```

### Hokuyo UST-10LX:
```bash
source /opt/ros/humble/setup.bash
ros2 launch urg_node2 urg_node2.launch.py \
    serial_port:=/dev/ttyACM0 \
    topic_name:=/scan &
sleep 3
```

### Verify LiDAR:
```bash
ros2 topic hz /scan
```
**Expected:** ~10–15 Hz for RPLIDAR A2, ~40 Hz for Hokuyo  

```bash
ros2 topic echo /scan --once | grep -E "angle_min|angle_max|ranges"
```
**Expected:**
- `angle_min` ≤ −1.57 (−90°)
- `angle_max` ≥ +1.57 (+90°)
- `ranges` array with 360+ values

**If angle range is insufficient:** the policy's lateral sectors will be empty. Do not fly.

---

## 7. Verify PX4 Sensor Data

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /fmu/out/vehicle_local_position --once 2>/dev/null | grep -E "x:|y:|z:|vx:|vy:|vz:"
ros2 topic echo /fmu/out/vehicle_attitude --once 2>/dev/null | grep "q:"
ros2 topic echo /fmu/out/vehicle_status --once 2>/dev/null | grep -E "pre_flight_checks_pass|nav_state|arming_state"
```
**Expected:**
- Position x/y/z: small numbers relative to home (metres)
- Attitude q: 4-element quaternion close to (1,0,0,0) if level
- `pre_flight_checks_pass: true` ← this is **critical**. If false, do not attempt arming.

```bash
ros2 topic hz /fmu/out/vehicle_local_position
```
**Expected:** ~50 Hz

---

## 8. Set Flight Parameters

Edit these values to match the real environment before running the node:

```bash
# Distance from takeoff point to goal, in NED frame (North = straight ahead of drone)
GOAL_NORTH=5.0   # metres forward (North)
GOAL_EAST=0.0    # metres right   (East)
GOAL_ALT=1.5     # metres above takeoff altitude

# Safety: start conservative, raise after first successful flight
VEL_SCALE=0.3    # 0.3 × 3.0 m/s = 0.9 m/s max. Raise to 0.5 after validation.

# ICP speed adaptation thresholds
D_REFLEXIVE=1.0   # m — full stop trigger (reflexive brake)
D_PREDICTIVE=2.0  # m — predictive slow-down begins

# Log path for crash buffer
TRIAL_LOG=~/results/real_pa_trial_0_steps.csv
mkdir -p ~/results
```

---

## 9. Launch the Student Node

Open a new terminal (or `screen`/`tmux` session) for the node so you can see its logs:

```bash
source /opt/ros/humble/setup.bash
cd ~/thesis_ws/IsaacLabExtensionTemplate

python3 scripts/rl_games/student_ros2_node_icp.py \
    --checkpoint logs/dagger/student_latest.pth \
    --device      cuda \
    --goal_north  $GOAL_NORTH \
    --goal_east   $GOAL_EAST \
    --goal_alt    $GOAL_ALT \
    --takeoff_alt $GOAL_ALT \
    --vel_scale   $VEL_SCALE \
    --lidar_topic /scan \
    --d_reflexive  $D_REFLEXIVE \
    --d_predictive $D_PREDICTIVE \
    --w1_init  0.5 \
    --w1_max   5.0 \
    --gamma_s  0.5 \
    --k_s      0.5 \
    --trial_log $TRIAL_LOG
```

**Expected startup output (first 5 lines):**
```
[INFO] Device      : cuda (Orin)
[INFO] Checkpoint  : /home/.../logs/dagger/student_latest.pth
[INFO] Obs dim     : 16
[INFO] Hidden dims : [256, 128]
[INFO] DAgger iter : 199
```

**If `Device: cpu` appears instead of `cuda`:** CUDA is not available. Stop, fix PyTorch install (Section 3c), then restart.

**Wait for this line in the output before arming:**
```
[diag] fsm=IDLE flight_check=True arm_req=False nav=...
```

**If you see `flight_check=False` repeatedly:** PX4 preflight check is failing. Check:
- GPS fix (outdoor) or EKF source (indoor with mocap/optical flow)
- Barometer sanity
- IMU calibration
- Check `nsh> commander status` on PX4 console

---

## 10. Arm and Fly

In a **separate terminal**:

```bash
source /opt/ros/humble/setup.bash

# Arm the drone
ros2 topic pub --once /arm_message std_msgs/msg/Bool "data: true"
```

**Observe the student node terminal. Expected FSM progression:**
```
-> ARMING
-> TAKEOFF
-> LOITER (waiting for hover)
-> OFFBOARD  student policy active
```

The drone will:
1. Auto-takeoff to `--takeoff_alt` via PX4 NAV_TAKEOFF
2. Hover in LOITER until stable
3. Switch to OFFBOARD and execute the student policy
4. Navigate toward the goal

**To disarm / abort at any time:**
```bash
ros2 topic pub --once /arm_message std_msgs/msg/Bool "data: false"
```
Then immediately switch RC to Position or Land mode as backup.

---

## 11. Monitor During Flight

Watch these topics in real time to detect problems:

```bash
# ICP speed status — [d_fwd, d_fwd, speed_scale, X0, X1, W1, 1.0]
ros2 topic echo /student/icp_status

# Pipeline latency — [scan_age_ms, policy_ms, icp_ms, total_ms]
ros2 topic echo /student/latency
```

**Warning thresholds (GPU inference is fast — policy_ms should be < 2 ms on Orin):**
| Metric | Warn | Abort |
|--------|------|-------|
| `scan_age_ms` (latency[0]) | > 200 ms | > 500 ms — LiDAR stale, ICP blind |
| `speed_scale` (icp_status[2]) | < 0.3 | = 0.0 — full stop, obstacle very close |
| `policy_ms` (latency[1]) | > 5 ms | > 20 ms — GPU stall or thermal throttle |
| `W1` (icp_status[5]) | > 3.0 | — predictive gain high, will slow strongly |

**If `scan_age_ms` > 500:** LiDAR driver may have crashed. Check:
```bash
ros2 topic hz /scan
```
If dead → restart LiDAR driver → **abort the trial first**, then restart.

**Student node logs every 1.5 s:**
```
[  150] OFFBOARD | dist=3.42m | alt=1.51m(err=-0.01) | yaw=+5deg |
        act=[yaw=-0.02 vx=+0.87 vy=+0.03 vz=+0.00] |
        vel_NED=(+0.87,+0.03,+0.00) | ICP[fwd=4.12m spd=1.00 X0=0.00 W1=0.12]
```

---

## 12. After the Flight

The node writes a crash buffer automatically on Ctrl-C or shutdown:
```
[crash_log] 500 rows → ~/results/real_pa_trial_0_steps_last5s.csv
```

To stop cleanly:
```bash
# In the node terminal:
Ctrl-C
```

**Retrieve the log:**
```bash
ls -lh ~/results/real_pa_trial_0_steps*.csv
head -5 ~/results/real_pa_trial_0_steps_last5s.csv
```

Columns: `t_mono, pos_N, pos_E, alt_m, vel_N, vel_E, vel_D, yaw_deg, d_fwd_m, d_all_m, speed_scale, W1, X0, X1, b_right, b_slR, b_ctr, b_slL, b_left, a_yaw, a_vx, a_vy, a_vz, fsm, lat_scan_ms, lat_policy_ms, lat_icp_ms, lat_total_ms`

---

## 13. Copy Files to/from Jetson (from dev machine)

```bash
# Dev machine → Jetson: copy checkpoint
scp logs/dagger/student_latest.pth          jetson:~/thesis_ws/IsaacLabExtensionTemplate/logs/dagger/
scp scripts/rl_games/student_ros2_node_icp.py jetson:~/thesis_ws/IsaacLabExtensionTemplate/scripts/rl_games/

# Jetson → dev machine: retrieve logs after flight
scp jetson:~/results/real_pa_trial_*_last5s.csv results/real/
```

---

## 14. Abort Conditions — Stop Immediately If:

| Condition | Action |
|-----------|--------|
| `flight_check=False` in node output | Do not arm. Fix PX4 preflight errors. |
| `scan_age_ms > 500` during flight | Kill node → switch RC to Position mode |
| Drone attitude > 45° from level | RC override to stabilise, then land |
| `speed_scale = 0` sustained > 3 s (stuck obstacle) | Kill node → RC Land |
| Drone drifts significantly sideways | EKF likely glitching — RC override |
| Student node crashes (Python exception) | RC takes over automatically (PX4 loses offboard) |
| Battery voltage low alert from PX4 | Land immediately |

When in doubt, RC override first, debug after.

---

## 15. Known Limitations for Real Deployment

1. **No absolute heading sense** — student outputs ego-centric velocity, rotated to world by PX4 yaw. Yaw drift = navigation drift.
2. **LiDAR is front 180° only** — rear obstacles are invisible to the policy. ICP emergency brake watches all 5 sectors but cannot steer away from behind.
3. **No goal-reached detection** — the node never commands land. Hover manually then land via RC or `/goal_pose` retarget.
4. **Altitude from PX4 EKF only** — indoors without VIO/mocap, altitude will drift. Fly low (1.5 m) to limit crash damage.
5. **First sim eval: 44.5% success** — trained in simulation. Expect sim-to-real gap. Start with `--vel_scale 0.3` and open space.

---

## 16. Quick-Start Checklist (Agent Use)

Run through in order. Only proceed to the next step if current step passes.

- [ ] Repo cloned: `ls ~/thesis_ws/IsaacLabExtensionTemplate/scripts/rl_games/student_ros2_node_icp.py`
- [ ] `CHECKPOINT OK` — `test -f logs/dagger/student_latest.pth && echo OK`
- [ ] `ros2 --version` shows Humble
- [ ] `px4_msgs` interface resolves
- [ ] `torch.cuda.is_available()` returns `True` on Jetson
- [ ] MicroXRCEAgent running, `/fmu/out/vehicle_local_position` visible at ~50 Hz
- [ ] LiDAR `/scan` at ≥10 Hz, angle range ≥ ±90°
- [ ] `pre_flight_checks_pass: true` from `/fmu/out/vehicle_status`
- [ ] Node started with `--device cuda`, startup shows `Device: cuda (Orin)`
- [ ] `flight_check=True` in node diag output
- [ ] `GOAL_NORTH/EAST/ALT` set to real-world values
- [ ] RC transmitter in hand, mode switch configured
- [ ] Arm published, FSM reaches `OFFBOARD`
- [ ] `policy_ms < 5` in `/student/latency` — GPU is working
- [ ] Ctrl-C after landing, retrieve `_last5s.csv`
