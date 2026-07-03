"""
Sim2Sim: Isaac Lab (DAgger student policy, 19D obs) → PX4 SITL via ROS2 velocity offboard.

Data flow:
  Isaac Lab  ── physics + LiDAR simulation  (1 env, 77D obs)
  extract    ── 77D → 19D  (base state + 5 front beams, no privileged obs)
  student    ── StudentPolicy MLP: 19D → 4D action
  ROS2       ── TrajectorySetpoint (velocity mode) → PX4 SITL

Frame conventions:  (identical to sim2sim_px4.py)
  Isaac body : FLU  (Forward-Left-Up)
  PX4 world  : NED  (North-East-Down)
  Policy output is body-frame FLU; converted to world NED before publishing.

Setup:
  Terminal 1 — PX4 SITL:
      cd PX4-Autopilot && make px4_sitl gz_x500

  Terminal 2 — MicroXRCE-DDS agent:
      MicroXRCEAgent udp4 -p 8888

  Terminal 3 — this script:
      ./isaaclab.sh -p scripts/rl_games/sim2sim_student_px4.py \\
          --task Isaac-Agile-Lidar-Vel-PA-v0 \\
          --checkpoint logs/dagger/student_latest.pth \\
          --goal_x 5.0 --goal_y 0.0 --goal_z 1.5

  Terminal 4 — arm:
      ros2 topic pub --once /arm_message std_msgs/msg/Bool "data: true"
"""

"""Launch Isaac Sim first."""

import argparse
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Sim2Sim: DAgger student policy -> PX4 velocity offboard via ROS2"
)
parser.add_argument("--task", type=str, default="Isaac-Agile-Lidar-Vel-PA-v0")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to student .pth saved by dagger_distill.py")
parser.add_argument("--goal_x", type=float, default=5.0)
parser.add_argument("--goal_y", type=float, default=0.0)
parser.add_argument("--goal_z", type=float, default=1.5)
parser.add_argument("--takeoff_alt", type=float, default=1.5,
                    help="PX4 auto-takeoff altitude (m). Must match goal_z for level flight.")
parser.add_argument("--vel_scale", type=float, default=0.5,
                    help="Scale applied to MAX_VELOCITY for deployment (0.5 = 2 m/s, 1.0 = 4 m/s).")
parser.add_argument("--max_steps", type=int, default=100_000)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1   # sim2sim always uses a single environment

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""All other imports after Isaac Sim is launched."""

from typing import Any, cast

import gymnasium as gym
import torch
import torch.nn as nn

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import PerceptionAwareDrone.tasks  # noqa: F401

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    VehicleAttitude,
)
from std_msgs.msg import Bool

# ── Obs layout (must match dagger_distill.py) ───────────────────────────────
LIDAR_START        = 14
LIDAR_END          = 74
FRONT_BEAM_INDICES = [28, 29, 30, 31, 32]   # ±12° FOV around forward
STUDENT_OBS_DIM    = 14 + len(FRONT_BEAM_INDICES)  # 19
ACTION_DIM         = 4

# Match QuadcopterEnvCfg velocity limits
MAX_VELOCITY = 4.0    # m/s
MAX_YAW_RATE = 3.14   # rad/s


def extract_student_obs(full_obs: torch.Tensor) -> torch.Tensor:
    """Slice 19D student obs from 77D teacher obs tensor."""
    base  = full_obs[:, :LIDAR_START]
    front = full_obs[:, LIDAR_START:LIDAR_END][:, FRONT_BEAM_INDICES]
    return torch.cat([base, front], dim=-1)


# ── Network definitions (identical to dagger_distill.py) ────────────────────

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


def _infer_hidden_dims(student_sd: dict, action_dim: int) -> list[int]:
    return [
        v.shape[0]
        for k, v in student_sd.items()
        if k.endswith(".weight") and "net" in k and v.shape[0] != action_dim
    ]


def _reset_isaac_state(isaac_env, goal_pos_w: torch.Tensor):
    """Pin the Isaac Lab drone to (0, 0, goal_z) with zero velocity and fix the goal.

    This keeps Isaac Lab's internal state consistent with PX4's physical state so
    that desired_dist_z = 0 when PX4 is already at takeoff altitude, and the
    goal direction in obs always points at the real target.
    """
    device = isaac_env.device
    root_state = isaac_env._robot.data.default_root_state.clone()  # (1, 13)
    # Place Isaac drone directly below goal at goal altitude so dist_z ~ 0
    root_state[0, 0] = 0.0               # x
    root_state[0, 1] = 0.0               # y
    root_state[0, 2] = float(goal_pos_w[2])   # z = goal altitude
    root_state[0, 3] = 1.0               # qw  (level orientation)
    root_state[0, 4:7] = 0.0            # qx qy qz
    root_state[0, 7:] = 0.0             # zero velocity
    isaac_env._robot.write_root_pose_to_sim(root_state[:, :7])
    isaac_env._robot.write_root_velocity_to_sim(root_state[:, 7:])
    isaac_env._desired_pos_w[0] = goal_pos_w


# ── ROS2 node (identical to sim2sim_px4.py) ──────────────────────────────────

class VelocityOffboardNode(Node):
    """
    ROS2 node that manages arming FSM and publishes NED velocity setpoints to PX4.
    States: IDLE -> ARMING -> TAKEOFF -> LOITER -> OFFBOARD
    """

    def __init__(self, takeoff_alt: float = 1.5):
        super().__init__("isaac_student_offboard")
        self._takeoff_alt = takeoff_alt

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self._status_cb, qos)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self._attitude_cb, qos)
        self.create_subscription(Bool, "/arm_message", self._arm_msg_cb, qos)

        self._pub_offboard = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self._pub_traj = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)
        self._pub_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)

        self.state       = "IDLE"
        self._cnt        = 0
        self._arm_request = False
        self.nav_state   = VehicleStatus.NAVIGATION_STATE_MAX
        self.arm_state   = VehicleStatus.ARMING_STATE_DISARMED
        self.flight_check = False
        self.failsafe    = False
        self.px4_yaw     = 0.0   # NED yaw (rad, CW from North)

        self._v_ned_n  = 0.0
        self._v_ned_e  = 0.0
        self._v_ned_d  = 0.0
        self._yawspeed = 0.0

        self.get_logger().info("VelocityOffboardNode ready. "
                               "Publish True to /arm_message to start.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _arm_msg_cb(self, msg: Bool):
        self._arm_request = msg.data
        self.get_logger().info(f"Arm request: {msg.data}")

    def _status_cb(self, msg: VehicleStatus):
        if msg.nav_state  != self.nav_state:
            self.get_logger().info(f"NAV_STATE  -> {msg.nav_state}")
        if msg.arming_state != self.arm_state:
            self.get_logger().info(f"ARM_STATE  -> {msg.arming_state}")
        self.nav_state    = msg.nav_state
        self.arm_state    = msg.arming_state
        self.flight_check = msg.pre_flight_checks_pass
        self.failsafe     = msg.failsafe

    def _attitude_cb(self, msg: VehicleAttitude):
        # PX4 quaternion: q[0]=w, q[1]=x, q[2]=y, q[3]=z (FRD body, NED world)
        q = msg.q
        self.px4_yaw = -(math.atan2(
            2.0 * (q[3] * q[0] + q[1] * q[2]),
            1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1]),
        ))

    # ── FSM ──────────────────────────────────────────────────────────────────

    def tick_state_machine(self):
        match self.state:
            case "IDLE":
                if self.flight_check and self._arm_request:
                    self.state = "ARMING"
                    self._cnt  = 0
                    self.get_logger().info("-> ARMING")

            case "ARMING":
                self._send_offboard_mode()
                self._send_zero_traj()
                self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                if self.arm_state == VehicleStatus.ARMING_STATE_ARMED and self._cnt > 10:
                    self.state = "TAKEOFF"
                    self._cnt  = 0
                    self.get_logger().info("-> TAKEOFF")
                if not self.flight_check:
                    self.state = "IDLE"

            case "TAKEOFF":
                self._send_offboard_mode()
                self._send_zero_traj()
                self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
                               param1=1.0, param7=self._takeoff_alt)
                if self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                    self.state = "LOITER"
                    self._cnt  = 0
                    self.get_logger().info("-> LOITER (waiting for hover)")
                if not self.flight_check:
                    self.state = "IDLE"

            case "LOITER":
                self._send_offboard_mode()
                self._send_zero_traj()
                self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                if self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                    self.state = "OFFBOARD"
                    self._cnt  = 0
                    self.get_logger().info("-> OFFBOARD  student policy active")
                if not self.flight_check:
                    self.state = "IDLE"

            case "OFFBOARD":
                if (not self.flight_check
                        or self.arm_state != VehicleStatus.ARMING_STATE_ARMED
                        or self.failsafe):
                    self.state = "IDLE"
                    self.get_logger().warn("Offboard lost — returning to IDLE")
                else:
                    self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

        if self.arm_state != VehicleStatus.ARMING_STATE_ARMED:
            self._arm_request = False

        self._cnt += 1

    @property
    def is_offboard(self) -> bool:
        return self.state == "OFFBOARD"

    # ── Velocity publishing ───────────────────────────────────────────────────

    def set_velocity(self, v_n: float, v_e: float, v_d: float, yawspeed: float):
        self._v_ned_n  = v_n
        self._v_ned_e  = v_e
        self._v_ned_d  = v_d
        self._yawspeed = yawspeed

    def publish_velocity(self):
        self._send_offboard_mode()
        ts  = self.get_clock().now().nanoseconds // 1000
        msg = TrajectorySetpoint()
        msg.timestamp        = ts
        msg.position[0]      = float("nan")
        msg.position[1]      = float("nan")
        msg.position[2]      = float("nan")
        msg.acceleration[0]  = float("nan")
        msg.acceleration[1]  = float("nan")
        msg.acceleration[2]  = float("nan")
        msg.velocity[0]      = self._v_ned_n
        msg.velocity[1]      = self._v_ned_e
        msg.velocity[2]      = self._v_ned_d
        msg.yaw              = float("nan")
        msg.yawspeed         = self._yawspeed
        self._pub_traj.publish(msg)

    def _send_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp    = self.get_clock().now().nanoseconds // 1000
        msg.position     = False
        msg.velocity     = True
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        self._pub_offboard.publish(msg)

    def _send_zero_traj(self):
        msg = TrajectorySetpoint()
        msg.timestamp   = self.get_clock().now().nanoseconds // 1000
        msg.position[0] = float("nan")
        msg.position[1] = float("nan")
        msg.position[2] = float("nan")
        msg.velocity[0] = 0.0
        msg.velocity[1] = 0.0
        msg.velocity[2] = 0.0
        msg.yaw         = float("nan")
        msg.yawspeed    = 0.0
        self._pub_traj.publish(msg)

    def _send_cmd(self, command, param1: float = 0.0, param2: float = 0.0, param7: float = 0.0):
        msg = VehicleCommand()
        msg.timestamp        = self.get_clock().now().nanoseconds // 1000
        msg.command          = command
        msg.param1           = param1
        msg.param2           = param2
        msg.param7           = param7
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        self._pub_cmd.publish(msg)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Load student checkpoint
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Student checkpoint : {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    obs_dim = ckpt.get("student_obs_dim", STUDENT_OBS_DIM)
    action_dim = ckpt.get("action_dim", ACTION_DIM)
    hidden_dims = _infer_hidden_dims(ckpt["student"], action_dim)

    print(f"[INFO] Obs dim       : {obs_dim}")
    print(f"[INFO] Hidden dims   : {hidden_dims}")
    print(f"[INFO] DAgger iter   : {ckpt.get('dagger_iter', '?')}")
    print(f"[INFO] Final beta    : {ckpt.get('beta', float('nan')):.4f}")

    if obs_dim != STUDENT_OBS_DIM:
        print(f"[WARN] Checkpoint obs_dim={obs_dim} != STUDENT_OBS_DIM={STUDENT_OBS_DIM}. "
              "Using checkpoint value.")

    # 2. Isaac Lab environment (provides physics + LiDAR obs)
    env_cfg   = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cast(dict[str, Any], load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point"))
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env_raw.unwrapped, DirectMARLEnv):
        env_raw = multi_agent_to_single_agent(env_raw)
    env = RlGamesVecEnvWrapper(env_raw, rl_device, clip_obs, clip_actions)

    device   = torch.device(rl_device)
    isaac_env = env_raw.unwrapped
    print(f"[INFO] Device: {device}")

    # 3. Student policy + normalizer
    student    = StudentPolicy(obs_dim, action_dim, hidden_dims).to(device)
    normalizer = RunningNormalizer(obs_dim).to(device)
    student.load_state_dict(ckpt["student"])
    normalizer.load_state_dict(ckpt["normalizer"])
    student.eval()

    # 4. Goal position (world frame, Z-up)
    goal_pos_w = torch.tensor(
        [args_cli.goal_x, args_cli.goal_y, args_cli.goal_z],
        device=device, dtype=torch.float32,
    )

    # 5. ROS2
    rclpy.init()
    ros_node = VelocityOffboardNode(takeoff_alt=args_cli.takeoff_alt)

    # 6. Reset env and pin goal
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    isaac_env._desired_pos_w[0] = goal_pos_w

    eff_max_vel = MAX_VELOCITY * args_cli.vel_scale
    eff_max_yaw = MAX_YAW_RATE * args_cli.vel_scale

    print(f"\n[INFO] Goal         : {goal_pos_w.cpu().numpy()}")
    print(f"[INFO] Loop rate    : 75 Hz")
    print(f"[INFO] vel_scale    : {args_cli.vel_scale}  (effective max {eff_max_vel:.1f} m/s)")
    print("[INFO] Arm: ros2 topic pub --once /arm_message std_msgs/msg/Bool \"data: true\"")
    print()

    # Sync Isaac drone to (0, 0, goal_z) so desired_dist_z=0 at takeoff
    _reset_isaac_state(isaac_env, goal_pos_w)

    loop_dt = 1.0 / 75.0   # env dt = 1/150 s, decimation = 2
    step = 0

    with torch.inference_mode():
        while simulation_app.is_running() and step < args_cli.max_steps:
            t0 = time.time()

            # ── Student inference ─────────────────────────────────────────────
            full_obs    = obs.float()                     # (1, 77)
            student_obs = extract_student_obs(full_obs)   # (1, 19)
            norm_obs    = normalizer.normalize(student_obs)
            actions     = student(norm_obs)               # (1, 4)  in [-1, 1]

            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

            # Re-sync Isaac state after every episode reset so goal dir stays valid
            if dones.any():
                _reset_isaac_state(isaac_env, goal_pos_w)

            # ── Action → NED velocity ──────────────────────────────────────────
            #
            # Policy output layout (same as teacher):
            #   actions[0, 0] = yaw_rate_norm  (FLU CCW positive)
            #   actions[0, 1] = vx_norm        (forward, body +X)
            #   actions[0, 2] = vy_norm        (left,    body +Y)
            #   actions[0, 3] = vz_norm        (up,      body +Z)
            #
            yaw_rate_flu = float(actions[0, 0]) * eff_max_yaw
            vx_b         = float(actions[0, 1]) * eff_max_vel
            vy_b         = float(actions[0, 2]) * eff_max_vel
            vz_b         = float(actions[0, 3]) * eff_max_vel

            # Body FLU → world FLU (rotate by Isaac heading)
            yaw_isaac = float(isaac_env._robot.data.heading_w[0])
            cos_y     = math.cos(yaw_isaac)
            sin_y     = math.sin(yaw_isaac)
            vx_world  = vx_b * cos_y - vy_b * sin_y
            vy_world  = vx_b * sin_y + vy_b * cos_y

            # World FLU (Z-up) → PX4 NED, then clamp to safe limits
            v_ned_n = max(-eff_max_vel, min(eff_max_vel, vx_world))
            v_ned_e = max(-eff_max_vel, min(eff_max_vel, vy_world))
            v_ned_d = max(-eff_max_vel, min(eff_max_vel, -vz_b))
            yawspeed_ned = max(-eff_max_yaw, min(eff_max_yaw, -yaw_rate_flu))

            ros_node.set_velocity(v_ned_n, v_ned_e, v_ned_d, yawspeed_ned)

            # ── ROS2 spin + FSM ───────────────────────────────────────────────
            rclpy.spin_once(ros_node, timeout_sec=0.0)
            ros_node.tick_state_machine()
            ros_node.publish_velocity()

            # ── Logging (every ~2 s) ──────────────────────────────────────────
            if step % 150 == 0:
                drone_pos = isaac_env._robot.data.root_pos_w[0].cpu()
                dist = (goal_pos_w.cpu() - drone_pos).norm().item()
                ros_node.get_logger().info(
                    f"[{step:6d}] {ros_node.state:8s} | "
                    f"vel_NED=({v_ned_n:+.2f}, {v_ned_e:+.2f}, {v_ned_d:+.2f}) m/s | "
                    f"yaw_spd={yawspeed_ned:+.2f} rad/s | "
                    f"hdg={math.degrees(yaw_isaac):+.1f}° | "
                    f"dist_goal={dist:.2f} m"
                )

            step += 1

            # Rate-limit to 75 Hz
            elapsed = time.time() - t0
            if elapsed < loop_dt:
                time.sleep(loop_dt - elapsed)

    print(f"\n[INFO] Finished after {step} steps.")
    env.close()
    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
