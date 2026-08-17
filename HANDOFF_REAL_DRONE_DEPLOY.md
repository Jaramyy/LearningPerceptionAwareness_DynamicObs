# Real-Drone Deployment Handoff — PA Student Policy

**For:** deploying agent  
**Checkpoint:** `logs/dagger/student_latest.pth` (DAgger iter 199, beta=0.05)  
**Node:** `scripts/rl_games/student_ros2_node_icp.py`  
**Sim eval:** 44.5% success / 0% collision @ 200 episodes (200-iter DAgger, PA teacher)

---

## 1. Hardware Requirements

| Component | Spec |
|-----------|------|
| Autopilot | PX4 v1.14+ (FMU-v5 or Pixhawk 6C tested in sim) |
| Companion | Any Linux PC / Jetson with ROS2 Humble, DDS bridge |
| LiDAR | 2-D planar scan, **±90° minimum FOV**, mounted flat at drone CoG height |
| Frame convention | PX4 FRD body, NED world |

The student uses only **5 angular sectors of the front 180°** of the LaserScan — a single-plane 2-D LiDAR suffices (RPLIDAR A2/A3, Hokuyo UST-10LX, etc.).

---

## 2. Software Dependencies

```
ROS2 Humble
px4_msgs  (must match PX4 firmware version)
sensor_msgs, geometry_msgs, visualization_msgs, tf2_ros, std_msgs
python3 torch  (CPU only is fine — the model is tiny: 256-128 MLP)
```

No Isaac Sim, no rl_games, no GPU needed at inference time.

---

## 3. Checkpoint Contents

File: `logs/dagger/student_latest.pth`

```
student           — StudentPolicy state dict  (MLP: 16→256→128→4, ELU+Tanh)
normalizer        — RunningNormalizer buffers (mean, var, count; dim=16)
optimizer         — Adam state (not needed at inference)
dagger_iter       — 199
beta              — 0.05  (converged: <5% teacher mixing at end of training)
student_obs_dim   — 16
action_dim        — 4
method            — "pa"
```

The `normalizer` mean/var **must** be loaded and used at inference — the policy is sensitive to unnormalized inputs.

---

## 4. Observation Layout (16-D, must match exactly)

| Idx | Field | Source | Notes |
|-----|-------|--------|-------|
| 0:3 | `root_lin_vel_b` | PX4 `VehicleLocalPosition` → FLU body | vx,vy,vz in FLU (m/s) |
| 3:6 | `root_ang_vel_b` | PX4 `SensorCombined` gyro → FLU body | roll,pitch,yaw in FLU (rad/s) |
| 6:9 | `unit_desired_pos_b` | GPS + goal → FLU body | unit vector, normalized |
| 9   | `desired_dist_2d` | horizontal dist to goal (m) | not normalized |
| 10  | `desired_dist_z` | vertical dist, **positive = goal above** (m) | NED: pD − gD |
| 11:16 | LiDAR sectors 0–4 | LaserScan `/scan` | min-range / 5.0, clipped [0,1] |

### LiDAR Sector Angles (FLU body frame: − = right, + = left)

| Idx | Sector | Angle range |
|-----|--------|-------------|
| 11 | Sector 0 | −90° → −54° (hard right) |
| 12 | Sector 1 | −54° → −18° (soft right) |
| 13 | Sector 2 | −18° → +18° (**forward**, ICP uses this) |
| 14 | Sector 3 | +18° → +54° (soft left) |
| 15 | Sector 4 | +54° → +90° (hard left) |

`value = min_range_in_sector / 5.0`  →  0 = contact, 1 = free (≥5 m)

### Frame Conversion (PX4 → Student)

```
FRD → FLU:   ang_vel_flu = (gyro_x, -gyro_y, -gyro_z)
NED → FLU:   q_inv = conjugate(q_frd_ned)
              v_frd = rotate(q_inv, v_ned)
              v_flu = (v_frd.x, -v_frd.y, -v_frd.z)
```

---

## 5. Action Layout (4-D, Tanh output → scaled)

| Idx | Field | Scale | NED mapping |
|-----|-------|-------|-------------|
| 0 | `yaw_rate` FLU (CCW+) | × 4.14 rad/s | negate → PX4 yawspeed (CW+ in NED) |
| 1 | `vx_b` FLU (forward) | × `MAX_VEL` | rotated to NED N/E via PX4 yaw |
| 2 | `vy_b` FLU (left) | × `MAX_VEL` | rotated to NED N/E via PX4 yaw |
| 3 | `vz_b` FLU (up) | × `MAX_VEL` | negate → NED vD |

`MAX_VEL = 3.0 m/s` (training). Use `--vel_scale 0.5` for first flights → 1.5 m/s cap.

Body→world rotation (using PX4 NED yaw `ψ`, CW-from-North):
```
v_north = vx_b * cos(ψ) + vy_b * sin(ψ)
v_east  = vx_b * sin(ψ) − vy_b * cos(ψ)
```

---

## 6. ROS2 Topics

### Subscribed by the node

| Topic | Type | Source |
|-------|------|--------|
| `/fmu/out/vehicle_local_position` | `px4_msgs/VehicleLocalPosition` | PX4 DDS bridge |
| `/fmu/out/vehicle_attitude` | `px4_msgs/VehicleAttitude` | PX4 DDS bridge |
| `/fmu/out/sensor_combined` | `px4_msgs/SensorCombined` | PX4 DDS bridge |
| `/fmu/out/vehicle_status` | `px4_msgs/VehicleStatus` | PX4 DDS bridge |
| `/scan` | `sensor_msgs/LaserScan` | LiDAR driver (default, override with `--lidar_topic`) |
| `/arm_message` | `std_msgs/Bool` | Operator trigger |
| `/goal_pose` | `geometry_msgs/PoseStamped` | RViz 2D Nav Goal (optional, in-flight retargeting) |

### Published by the node

| Topic | Type | Purpose |
|-------|------|---------|
| `/fmu/in/offboard_control_mode` | `px4_msgs/OffboardControlMode` | Keep PX4 offboard alive |
| `/fmu/in/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` | Velocity setpoints (NED) |
| `/fmu/in/vehicle_command` | `px4_msgs/VehicleCommand` | Arm / mode commands |
| `/student/icp_status` | `std_msgs/Float32MultiArray` | `[d_fwd, d_fwd, speed_scale, X0, X1, W1, 1.0]` |
| `/student/latency` | `std_msgs/Float32MultiArray` | `[scan_age_ms, policy_ms, icp_ms, total_ms]` |
| `/student/markers` | `visualization_msgs/MarkerArray` | RViz velocity + goal arrows |
| `/tf` (map→base_link) | TF2 | RViz visualization |

---

## 7. Launch Command

```bash
# First flight — conservative speed, balanced ICP
python3 scripts/rl_games/student_ros2_node_icp.py \
    --checkpoint logs/dagger/student_latest.pth \
    --goal_north 5.0 \
    --goal_east  0.0 \
    --goal_alt   1.5 \
    --takeoff_alt 1.5 \
    --vel_scale  0.5 \
    --lidar_topic /scan \
    --d_reflexive  1.0 \
    --d_predictive 2.0 \
    --w1_init  0.5 \
    --w1_max   5.0 \
    --gamma_s  0.5 \
    --k_s      0.5 \
    --trial_log results/real_trial_0_steps.csv
```

Adjust `--goal_north/east` to the NED offset from the takeoff spot to the target.

---

## 8. Arming Procedure (node FSM)

The node runs an internal 5-state FSM: `IDLE → ARMING → TAKEOFF → LOITER → OFFBOARD`.

```
State       Trigger
IDLE        flight_check=True AND arm message received
ARMING      sends arm command repeatedly
TAKEOFF     sends VEHICLE_CMD_NAV_TAKEOFF to --takeoff_alt
LOITER      waits for AUTO_LOITER nav state (hover stable)
OFFBOARD    student policy active; sends velocity setpoints at 100 Hz
```

**To arm:**
```bash
ros2 topic pub --once /arm_message std_msgs/msg/Bool "data: true"
```

**To disarm (kill policy):**
```bash
ros2 topic pub --once /arm_message std_msgs/msg/Bool "data: false"
# Then switch RC to Position or Land mode as backup
```

The node **never** disarms the drone autonomously — always have RC override ready.

---

## 9. Adaptive Speed Control (MODANC ICP, IROS 2025)

The ICP speed controller wraps the policy output. It reads sector 2 (forward, idx 13 in obs).

```
d_fwd = obs[13] * 5.0  (metres)

X1 = (d_predictive - d_fwd) / (d_predictive - d_reflexive)   clamped ≥ 0
X0 = X1  if X1 ≥ 1.0  else  0                                (reflexive)

W1 update:  ΔW1 = μs·ΔX0·X1 − γs·(ks + 0.05·X1)·W1²·dt
speed_scale = max(0, 1 − (W0·X0 + W1·X1))

vx_body  *= speed_scale
vy_body   capped at max(speed_scale, 0.3) × MAX_VEL  (keeps dodge authority)
```

Recommended starting params for indoor/narrow space:
- `--d_reflexive 1.0 --d_predictive 2.0`  
- For wide outdoor space: `--d_reflexive 1.0 --d_predictive 4.0`

---

## 10. Safety Checklist Before First Flight

- [ ] LiDAR scan visible on `/scan` at ≥10 Hz, angle_min ≤ −π/2, angle_max ≥ +π/2
- [ ] PX4 DDS bridge running (`MicroXRCEAgent` or `px4_ros_com`)
- [ ] `/fmu/out/vehicle_local_position` arriving (check with `ros2 topic hz`)
- [ ] `flight_check=True` visible in node startup logs before arming
- [ ] RC transmitter in hand, configured for Position/Altitude hold on switch
- [ ] `--vel_scale 0.5` (≤1.5 m/s) for first flight; raise to 0.7 after validation
- [ ] Clear 5 m radius around takeoff point, 10 m path to goal
- [ ] `--trial_log` path set so crash buffer is written on Ctrl-C

---

## 11. Known Limitations

- **Altitude**: The student has no barometer/sonar input. Altitude hold relies entirely on PX4 velocity control mode. If EKF altitude drifts, the drone may descend into the ground.
- **Yaw**: Student cannot sense absolute heading — it outputs ego-centric commands that are rotated by the current PX4 yaw. Yaw estimation errors degrade navigation.
- **LiDAR blind spot**: Only front 180° is used. Obstacles from behind are invisible to the policy; ICP emergency brake uses all 5 sectors but the policy itself cannot react to rear threats.
- **Goal reached**: The node has no automatic land command — command a landing via RC or publish a `/goal_pose` at the drone's current position to hold hover, then land manually.
- **Sim-to-real gap**: Trained at max 3 m/s with PX4 SITL dynamics. First test at `--vel_scale 0.3` (0.9 m/s) to verify attitude/velocity tracking matches sim assumptions.
