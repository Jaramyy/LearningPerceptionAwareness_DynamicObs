"""Standalone ROS2 node: student policy with real PX4 sensor observations.

No Isaac Lab required — observations are built from PX4 uXRCE-DDS topics and a
ROS2 LiDAR topic.  Works identically for Gazebo sim2sim and real hardware.

Observation layout (19D, must match dagger_distill.py):
  [0:3]   root_lin_vel_b      linear velocity in FLU body frame (m/s)
  [3:6]   root_ang_vel_b      angular velocity in FLU body frame (rad/s)
  [6:9]   projected_gravity_b unit gravity vector in FLU body frame
  [9:12]  unit_desired_pos_b  unit vector to goal in FLU body frame
  [12]    desired_dist_2d     horizontal distance to goal (m)
  [13]    desired_dist_z      vertical distance to goal, + = goal above (m)
  [14:19] 5 front LiDAR beams angles -11,-5,+1,+7,+13 deg relative to forward (m)

PX4 ROS2 topics consumed:
  /fmu/out/vehicle_local_position   (position + velocity in NED)
  /fmu/out/vehicle_attitude          (quaternion FRD body -> NED world)
  /fmu/out/sensor_combined           (gyro_rad[3] = angular velocity in FRD body frame)
  <--lidar_topic>                    (sensor_msgs/LaserScan, horizontal 2-D)

Frame conventions:
  PX4 body : FRD  (Forward-Right-Down)
  Isaac/Student body : FLU  (Forward-Left-Up)
  PX4 world : NED  (North-East-Down)
  Goal input: North / East / altitude-above-home (converted to NED internally)

Setup (Gazebo sim2sim):
  Terminal 1: cd PX4-Autopilot && make px4_sitl gz_x500_lidar
  Terminal 2: MicroXRCEAgent udp4 -p 8888
  Terminal 3 (LiDAR bridge):
      ros2 run ros_gz_bridge parameter_bridge /lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan
  Terminal 4 (this node):
      python3 scripts/rl_games/student_ros2_node.py \\
          --checkpoint logs/dagger/student_latest.pth \\
          --goal_north 5.0 --goal_east 0.0 --goal_alt 1.5
  Terminal 5 (arm):
      ros2 topic pub --once /arm_message std_msgs/msg/Bool "data: true"
"""

import argparse
import math
import os
import threading

import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy,
    QoSHistoryPolicy, QoSDurabilityPolicy,
)

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    VehicleAttitude,
    VehicleLocalPosition,
    SensorCombined,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

# ── Obs constants (must match dagger_distill.py) ─────────────────────────────
STUDENT_OBS_DIM = 19
ACTION_DIM = 4
LIDAR_RANGE = 5.0           # training max range (m)
BEAM_ANGLES_DEG = [-11.0, -5.0, 1.0, 7.0, 13.0]   # beams 28-32 at ±12° FOV

# Scale factors (match QuadcopterEnvCfg)
MAX_VELOCITY = 4.0    # m/s
MAX_YAW_RATE = 3.14   # rad/s


# ── Network definitions (identical to dagger_distill.py) ─────────────────────

class RunningNormalizer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0.0))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) / (self.var + self.eps).sqrt()


class StudentPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _infer_hidden_dims(sd: dict, action_dim: int) -> list[int]:
    return [
        v.shape[0] for k, v in sd.items()
        if k.endswith(".weight") and "net" in k and v.shape[0] != action_dim
    ]


# ── Quaternion math (pure Python, no extra deps) ─────────────────────────────

def _quat_apply(q_wxyz: tuple, v: tuple) -> tuple:
    """Rotate vector v by unit quaternion q: v' = q * v * q^-1."""
    w, qx, qy, qz = q_wxyz
    vx, vy, vz = v
    # Cross product q_vec x v
    cx = qy * vz - qz * vy
    cy = qz * vx - qx * vz
    cz = qx * vy - qy * vx
    # Rodrigues rotation
    return (
        vx + 2 * w * cx + 2 * (qy * cz - qz * cy),
        vy + 2 * w * cy + 2 * (qz * cx - qx * cz),
        vz + 2 * w * cz + 2 * (qx * cy - qy * cx),
    )


def _ned_to_flu(q_frd_ned_wxyz: tuple, v_ned: tuple) -> tuple:
    """Transform vector from NED world frame to FLU body frame.

    q_frd_ned: rotation quaternion from FRD body -> NED world (PX4 convention).
    Returns vector in FLU body frame.
    """
    w, x, y, z = q_frd_ned_wxyz
    # Inverse (conjugate for unit quaternion): rotates NED -> FRD body
    q_inv = (w, -x, -y, -z)
    v_frd = _quat_apply(q_inv, v_ned)
    # FRD -> FLU: negate Y (right->left) and Z (down->up)
    return (v_frd[0], -v_frd[1], -v_frd[2])


def _norm3(v: tuple) -> float:
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def _normalize3(v: tuple, eps: float = 1e-6) -> tuple:
    n = max(_norm3(v), eps)
    return (v[0] / n, v[1] / n, v[2] / n)


# ── LiDAR beam extraction ────────────────────────────────────────────────────

def _extract_front_beams(msg: LaserScan, angles_deg: list[float], max_range: float) -> list[float]:
    """Extract range values at specified body-frame angles from a LaserScan."""
    beams = []
    n = len(msg.ranges)
    for deg in angles_deg:
        rad = math.radians(deg)
        # Nearest index in the scan
        idx = round((rad - msg.angle_min) / msg.angle_increment)
        idx = max(0, min(n - 1, idx))
        r = msg.ranges[idx]
        if math.isnan(r) or math.isinf(r) or r <= 0.0:
            r = max_range
        beams.append(min(r, max_range))
    return beams


# ── Sensor state container ────────────────────────────────────────────────────

class _State:
    """Thread-safe cache for the latest sensor readings."""

    def __init__(self):
        self._lock = threading.Lock()
        self.pos_ned = (0.0, 0.0, 0.0)         # North, East, Down (m)
        self.vel_ned = (0.0, 0.0, 0.0)         # vN, vE, vD (m/s)
        self.att_q = (1.0, 0.0, 0.0, 0.0)      # w,x,y,z  FRD->NED
        self.ang_vel_frd = (0.0, 0.0, 0.0)     # rad/s FRD body
        self.lidar_msg: LaserScan | None = None
        self.px4_yaw = 0.0                      # NED yaw for FSM logging
        # Freshness flags
        self.has_pos = False
        self.has_att = False
        self.has_ang_vel = False
        self.has_lidar = False

    def update_local_pos(self, msg: VehicleLocalPosition):
        with self._lock:
            self.pos_ned = (msg.x, msg.y, msg.z)
            self.vel_ned = (msg.vx, msg.vy, msg.vz)
            self.has_pos = True

    def update_attitude(self, msg: VehicleAttitude):
        with self._lock:
            q = msg.q
            self.att_q = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            w, x, y, z = self.att_q
            # NED yaw (CW from North): standard ZYX Euler for FRD→NED quaternion
            # denominator = 1-2*(y²+z²), NOT 1-2*(x²+y²)
            self.px4_yaw = math.atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
            self.has_att = True

    def update_ang_vel(self, msg: SensorCombined):
        with self._lock:
            g = msg.gyro_rad
            self.ang_vel_frd = (float(g[0]), float(g[1]), float(g[2]))
            self.has_ang_vel = True

    def update_lidar(self, msg: LaserScan):
        with self._lock:
            self.lidar_msg = msg
            self.has_lidar = True

    def snapshot(self):
        with self._lock:
            return (
                self.pos_ned, self.vel_ned, self.att_q,
                self.ang_vel_frd, self.lidar_msg, self.px4_yaw,
                self.has_pos and self.has_att and self.has_ang_vel and self.has_lidar,
            )


# ── Main ROS2 node ────────────────────────────────────────────────────────────

class StudentOffboardNode(Node):

    def __init__(self, student: StudentPolicy, normalizer: RunningNormalizer,
                 goal_ned: tuple, vel_scale: float, args):
        super().__init__("student_offboard")

        self._student = student
        self._normalizer = normalizer
        self._goal_ned = goal_ned          # (goal_N, goal_E, goal_D)
        self._eff_max_vel = MAX_VELOCITY * vel_scale
        self._eff_max_yaw = MAX_YAW_RATE * vel_scale
        self._lidar_topic = args.lidar_topic
        self._max_vz_down = args.max_vz        # downward NED velocity cap (m/s)
        # Altitude floor in NED D: don't go below (goal_alt - 0.5m)
        # NED D is negative altitude, so floor_D = goal_D + 0.5  (less negative = lower altitude)
        self._alt_floor_D = goal_ned[2] + 0.5  # e.g. goal at -1.5m → floor at -1.0m (1.0m above gnd)
        self._state = _State()

        # PX4 arming state
        self._fsm_state = "IDLE"
        self._fsm_cnt = 0
        self._arm_request = False
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arm_state = VehicleStatus.ARMING_STATE_DISARMED
        self.flight_check = False
        self.failsafe = False
        self._takeoff_alt = args.takeoff_alt

        # Velocity setpoints (NED), updated each control cycle
        self._v_ned = (0.0, 0.0, 0.0)
        self._yawspeed = 0.0

        px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscriptions
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position",
            lambda m: self._state.update_local_pos(m), px4_qos)
        self.create_subscription(
            VehicleAttitude, "/fmu/out/vehicle_attitude",
            lambda m: self._state.update_attitude(m), px4_qos)
        self.create_subscription(
            SensorCombined, "/fmu/out/sensor_combined",
            lambda m: self._state.update_ang_vel(m), px4_qos)
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status",
            self._status_cb, px4_qos)
        self.create_subscription(
            LaserScan, self._lidar_topic,
            lambda m: self._state.update_lidar(m), sensor_qos)
        self.create_subscription(
            Bool, "/arm_message", self._arm_cb, px4_qos)

        # Publishers
        self._pub_offboard = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", px4_qos)
        self._pub_traj = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", px4_qos)
        self._pub_cmd = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10)

        # Control timer
        self._step = 0
        self.create_timer(1.0 / 75.0, self._control_cb)

        self.get_logger().info(
            f"StudentOffboard ready | goal_NED={goal_ned} | "
            f"max_vel={self._eff_max_vel:.1f} m/s | "
            f"lidar={self._lidar_topic}"
        )
        self.get_logger().info(
            "Arm: ros2 topic pub --once /arm_message std_msgs/msg/Bool \"data: true\""
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _status_cb(self, msg: VehicleStatus):
        if msg.nav_state != self.nav_state:
            self.get_logger().info(f"NAV_STATE -> {msg.nav_state}")
        if msg.arming_state != self.arm_state:
            self.get_logger().info(f"ARM_STATE -> {msg.arming_state}")
        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state
        self.flight_check = msg.pre_flight_checks_pass
        self.failsafe = msg.failsafe

    def _arm_cb(self, msg: Bool):
        self._arm_request = msg.data
        self.get_logger().info(f"Arm request: {msg.data}")

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_obs(self, pos, vel_ned, q, ang_vel_frd, lidar_msg) -> torch.Tensor | None:
        """Build 19D student obs from sensor readings. Returns (1,19) tensor."""
        if lidar_msg is None:
            return None

        # [0:3] Linear velocity in FLU body frame
        vel_flu = _ned_to_flu(q, vel_ned)

        # [3:6] Angular velocity in FLU body frame (FRD: negate Y and Z)
        ang_flu = (ang_vel_frd[0], -ang_vel_frd[1], -ang_vel_frd[2])

        # [6:9] Projected gravity in FLU body frame
        # Gravity in NED = unit down vector = (0, 0, +1) in NED
        grav_flu = _ned_to_flu(q, (0.0, 0.0, 1.0))
        grav_flu = _normalize3(grav_flu)

        # [9:12] Unit vector from drone to goal in FLU body frame
        gN, gE, gD = self._goal_ned
        pN, pE, pD = pos
        dN, dE, dD = gN - pN, gE - pE, gD - pD
        goal_flu = _ned_to_flu(q, (dN, dE, dD))
        unit_goal_flu = _normalize3(goal_flu)

        # [12] 2D horizontal distance (m)
        dist_2d = math.sqrt(dN**2 + dE**2)

        # [13] Vertical distance: positive = goal is above drone
        # In NED: D decreases as altitude increases, so goal above means gD < pD
        dist_z = pD - gD   # positive when goal is above (NED D is negative altitude)

        # [14:19] Front 5 LiDAR beams normalized to [0,1] — matches Isaac training obs
        beams = [b / LIDAR_RANGE for b in _extract_front_beams(lidar_msg, BEAM_ANGLES_DEG, LIDAR_RANGE)]

        obs_list = (
            list(vel_flu)
            + list(ang_flu)
            + list(grav_flu)
            + list(unit_goal_flu)
            + [dist_2d, dist_z]
            + beams
        )
        return torch.tensor(obs_list, dtype=torch.float32).unsqueeze(0)  # (1, 19)

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_cb(self):
        pos, vel_ned, q, ang_vel_frd, lidar_msg, px4_yaw, ready = self._state.snapshot()

        # Always tick FSM and publish to keep the offboard stream alive
        self._tick_fsm()

        if not ready:
            self._publish_zero()
            return

        obs = self._build_obs(pos, vel_ned, q, ang_vel_frd, lidar_msg)
        if obs is None:
            self._publish_zero()
            return

        with torch.inference_mode():
            norm_obs = self._normalizer.normalize(obs)
            actions = self._student(norm_obs)   # (1, 4) in [-1, 1]

        yaw_rate_flu = float(actions[0, 0]) * self._eff_max_yaw
        vx_b = float(actions[0, 1]) * self._eff_max_vel
        vy_b = float(actions[0, 2]) * self._eff_max_vel
        vz_b = float(actions[0, 3]) * self._eff_max_vel

        # Body FLU → world NEUp using NED CW yaw:
        #   forward unit = (cos yaw_cw, sin yaw_cw) in (N, E)
        #   left    unit = (sin yaw_cw, -cos yaw_cw) in (N, E)
        # Verified: N-facing(0°): fwd→N✓, left→W✓
        #           E-facing(90°): fwd→E✓, left→N✓
        cos_y = math.cos(px4_yaw)
        sin_y = math.sin(px4_yaw)
        vx_world = vx_b * cos_y + vy_b * sin_y   # North (NEUp)
        vy_world = vx_b * sin_y - vy_b * cos_y   # East  (NEUp)

        # NEUp → NED with clamping
        v_ned_n = max(-self._eff_max_vel, min(self._eff_max_vel, vx_world))
        v_ned_e = max(-self._eff_max_vel, min(self._eff_max_vel, vy_world))
        # Vertical: cap downward at max_vz (conservative) but allow full upward speed
        v_ned_d_raw = -vz_b
        v_ned_d = max(-self._eff_max_vel, min(self._max_vz_down, v_ned_d_raw))
        # Altitude floor: if below goal_alt-0.5m, refuse further descent
        pD = pos[2]
        if pD > self._alt_floor_D and v_ned_d > 0.0:
            v_ned_d = 0.0
        yawspeed_ned = max(-self._eff_max_yaw, min(self._eff_max_yaw, -yaw_rate_flu))

        self._v_ned = (v_ned_n, v_ned_e, v_ned_d)
        self._yawspeed = yawspeed_ned

        if self._fsm_state == "OFFBOARD":
            self._publish_velocity(v_ned_n, v_ned_e, v_ned_d, yawspeed_ned)
        else:
            self._publish_zero()

        if self._step % 150 == 0:
            pN, pE, pD = pos
            gN, gE, gD = self._goal_ned
            dist = math.sqrt((gN - pN)**2 + (gE - pE)**2 + (gD - pD)**2)
            v_ned_n, v_ned_e, v_ned_d = self._v_ned
            self.get_logger().info(
                f"[{self._step:6d}] {self._fsm_state:8s} | "
                f"vel_NED=({v_ned_n:+.2f},{v_ned_e:+.2f},{v_ned_d:+.2f}) | "
                f"yaw_spd={yawspeed_ned:+.2f} | "
                f"dist={dist:.2f}m | "
                f"beams(norm)={[f'{b:.2f}' for b in obs[0, 14:].tolist()]}"
            )
        self._step += 1

    # ── Arming FSM (same as sim2sim_px4.py) ──────────────────────────────────

    def _tick_fsm(self):
        match self._fsm_state:
            case "IDLE":
                if self.flight_check and self._arm_request:
                    self._fsm_state = "ARMING"
                    self._fsm_cnt = 0
                    self.get_logger().info("-> ARMING")
            case "ARMING":
                self._send_offboard_mode()
                self._send_zero_traj()
                self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                if self.arm_state == VehicleStatus.ARMING_STATE_ARMED and self._fsm_cnt > 10:
                    self._fsm_state = "TAKEOFF"
                    self._fsm_cnt = 0
                    self.get_logger().info("-> TAKEOFF")
                if not self.flight_check:
                    self._fsm_state = "IDLE"
            case "TAKEOFF":
                self._send_offboard_mode()
                self._send_zero_traj()
                self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
                               param1=1.0, param7=self._takeoff_alt)
                if self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                    self._fsm_state = "LOITER"
                    self._fsm_cnt = 0
                    self.get_logger().info("-> LOITER (waiting for hover)")
                if not self.flight_check:
                    self._fsm_state = "IDLE"
            case "LOITER":
                self._send_offboard_mode()
                self._send_zero_traj()
                self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                if self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                    self._fsm_state = "OFFBOARD"
                    self._fsm_cnt = 0
                    self.get_logger().info("-> OFFBOARD  student policy active")
                if not self.flight_check:
                    self._fsm_state = "IDLE"
            case "OFFBOARD":
                if (not self.flight_check
                        or self.arm_state != VehicleStatus.ARMING_STATE_ARMED
                        or self.failsafe):
                    self._fsm_state = "IDLE"
                    self.get_logger().warn("Offboard lost — returning to IDLE")
                else:
                    self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        if self.arm_state != VehicleStatus.ARMING_STATE_ARMED:
            self._arm_request = False
        self._fsm_cnt += 1

    # ── Publishers ────────────────────────────────────────────────────────────

    def _send_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self._pub_offboard.publish(msg)

    def _publish_zero(self):
        self._send_offboard_mode()
        self._send_zero_traj()

    def _send_zero_traj(self):
        msg = TrajectorySetpoint()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.position[0] = float("nan")
        msg.position[1] = float("nan")
        msg.position[2] = float("nan")
        msg.velocity[0] = 0.0
        msg.velocity[1] = 0.0
        msg.velocity[2] = 0.0
        msg.yaw = float("nan")
        msg.yawspeed = 0.0
        self._pub_traj.publish(msg)

    def _publish_velocity(self, vn: float, ve: float, vd: float, yawspeed: float):
        self._send_offboard_mode()
        ts = self.get_clock().now().nanoseconds // 1000
        msg = TrajectorySetpoint()
        msg.timestamp = ts
        msg.position[0] = float("nan")
        msg.position[1] = float("nan")
        msg.position[2] = float("nan")
        msg.acceleration[0] = float("nan")
        msg.acceleration[1] = float("nan")
        msg.acceleration[2] = float("nan")
        msg.velocity[0] = vn
        msg.velocity[1] = ve
        msg.velocity[2] = vd
        msg.yaw = float("nan")
        msg.yawspeed = yawspeed
        self._pub_traj.publish(msg)

    def _send_cmd(self, command, param1: float = 0.0, param2: float = 0.0,
                  param7: float = 0.0):
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.param7 = param7
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self._pub_cmd.publish(msg)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Student policy ROS2 offboard node")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to student .pth from dagger_distill.py")
    parser.add_argument("--goal_north", type=float, default=5.0,
                        help="Goal North of takeoff point (m)")
    parser.add_argument("--goal_east", type=float, default=0.0,
                        help="Goal East of takeoff point (m)")
    parser.add_argument("--goal_alt", type=float, default=1.5,
                        help="Goal altitude above takeoff point (m)")
    parser.add_argument("--takeoff_alt", type=float, default=1.5,
                        help="PX4 auto-takeoff altitude (m). Set equal to goal_alt.")
    parser.add_argument("--vel_scale", type=float, default=0.5,
                        help="Scale factor on MAX_VELOCITY (0.5 = 2 m/s max).")
    parser.add_argument("--max_vz", type=float, default=0.3,
                        help="Max downward NED velocity the policy can command (m/s). "
                             "Lower = safer near goal altitude. Default 0.3 m/s.")
    parser.add_argument("--lidar_topic", type=str, default="/scan",
                        help="ROS2 LaserScan topic from LiDAR.")
    args = parser.parse_args()

    # Load student checkpoint
    ckpt_path = os.path.abspath(args.checkpoint)
    print(f"[INFO] Checkpoint  : {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    obs_dim = ckpt.get("student_obs_dim", STUDENT_OBS_DIM)
    action_dim = ckpt.get("action_dim", ACTION_DIM)
    hidden_dims = _infer_hidden_dims(ckpt["student"], action_dim)

    print(f"[INFO] Obs dim     : {obs_dim}")
    print(f"[INFO] Hidden dims : {hidden_dims}")
    print(f"[INFO] DAgger iter : {ckpt.get('dagger_iter', '?')}")
    print(f"[INFO] Final beta  : {ckpt.get('beta', float('nan')):.4f}")

    student = StudentPolicy(obs_dim, action_dim, hidden_dims)
    normalizer = RunningNormalizer(obs_dim)
    student.load_state_dict(ckpt["student"])
    normalizer.load_state_dict(ckpt["normalizer"])
    student.eval()

    # Goal in NED: (North, East, Down)
    # Down = negative altitude (NED convention: positive D = below home)
    goal_ned = (args.goal_north, args.goal_east, -args.goal_alt)
    print(f"[INFO] Goal NED    : {goal_ned}  (N={args.goal_north}, E={args.goal_east}, alt={args.goal_alt}m)")
    print(f"[INFO] vel_scale   : {args.vel_scale}  (max {MAX_VELOCITY * args.vel_scale:.1f} m/s)")
    print(f"[INFO] LiDAR topic : {args.lidar_topic}")

    rclpy.init()
    node = StudentOffboardNode(student, normalizer, goal_ned, args.vel_scale, args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("[INFO] Node shut down.")


if __name__ == "__main__":
    main()
