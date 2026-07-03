"""
Sim2Sim: Isaac Lab (rl_games PA policy) → PX4 SITL via ROS2 velocity offboard.

Data flow:
  Isaac Lab  ── physics + LiDAR simulation (1 env)
  rl_games   ── policy: obs → [yaw_rate_norm, vx_norm, vy_norm, vz_norm]
  ROS2       ── TrajectorySetpoint (velocity mode) → PX4 SITL

Frame conventions:
  Isaac body : FLU  (Forward-Left-Up)
  PX4 world  : NED  (North-East-Down)
  policy output is body-frame FLU, this script converts to world NED

Run:
  Terminal 1 — PX4 SITL:
      cd PX4-Autopilot && make px4_sitl gz_x500

  Terminal 2 — MicroXRCE-DDS agent:
      MicroXRCEAgent udp4 -p 8888

  Terminal 3 — this script:
      ./isaaclab.sh -p scripts/rl_games/sim2sim_px4.py \\
          --task Isaac-Agile-Lidar-Vel-PA-v0 \\
          --goal_x 5.0 --goal_y 0.0 --goal_z 1.5

  Terminal 4 — arm:
      ros2 topic pub --once /arm_message std_msgs/msg/Bool "data: true"
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Sim2Sim: Isaac Lab PA policy -> PX4 velocity offboard via ROS2"
)
parser.add_argument("--task", type=str, default="Isaac-Agile-Lidar-Vel-PA-v0")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to .pth file. Loads best checkpoint from logs/ if omitted.")
parser.add_argument("--goal_x", type=float, default=5.0)
parser.add_argument("--goal_y", type=float, default=0.0)
parser.add_argument("--goal_z", type=float, default=1.5)
parser.add_argument("--takeoff_alt", type=float, default=1.5,
                    help="PX4 auto-takeoff altitude (m)")
parser.add_argument("--max_steps", type=int, default=100000)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1  # sim2sim is always single env

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""All other imports after Isaac Sim is launched."""

import gymnasium as gym
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from typing import Any, cast

from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

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

# Match these to QuadcopterEnvCfg
MAX_VELOCITY = 4.0   # m/s
MAX_YAW_RATE = 3.14  # rad/s


class VelocityOffboardNode(Node):
    """
    Self-contained ROS2 node that:
      1. Manages arming FSM (IDLE -> ARMING -> TAKEOFF -> LOITER -> OFFBOARD)
      2. Publishes OffboardControlMode (velocity=True)
      3. Publishes TrajectorySetpoint (velocity + yawspeed in NED)
    """

    def __init__(self, takeoff_alt: float = 1.5):
        super().__init__("isaac_velocity_offboard")
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

        self.state = "IDLE"
        self._cnt = 0
        self._arm_request = False
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arm_state = VehicleStatus.ARMING_STATE_DISARMED
        self.flight_check = False
        self.failsafe = False
        self.px4_yaw = 0.0   # NED yaw (rad, CW from North)

        # Velocity setpoint in NED — updated each step via set_velocity()
        self._v_ned_n = 0.0
        self._v_ned_e = 0.0
        self._v_ned_d = 0.0
        self._yawspeed = 0.0   # NED: positive = CW = turn right

        self.get_logger().info("VelocityOffboardNode ready. Publish True to /arm_message to start.")

    def _arm_msg_cb(self, msg: Bool):
        self._arm_request = msg.data
        self.get_logger().info(f"Arm request: {msg.data}")

    def _status_cb(self, msg: VehicleStatus):
        if msg.nav_state != self.nav_state:
            self.get_logger().info(f"NAV_STATE -> {msg.nav_state}")
        if msg.arming_state != self.arm_state:
            self.get_logger().info(f"ARM_STATE -> {msg.arming_state}")
        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state
        self.flight_check = msg.pre_flight_checks_pass
        self.failsafe = msg.failsafe

    def _attitude_cb(self, msg: VehicleAttitude):
        # PX4 quaternion: q[0]=w, q[1]=x, q[2]=y, q[3]=z (FRD body, NED world)
        # Same formula used in velocity_control.py
        q = msg.q
        self.px4_yaw = -(math.atan2(
            2.0 * (q[3] * q[0] + q[1] * q[2]),
            1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1])
        ))

    def tick_state_machine(self):
        """Drive arming FSM one step. Call on every sim loop iteration."""
        match self.state:
            case "IDLE":
                if self.flight_check and self._arm_request:
                    self.state = "ARMING"
                    self._cnt = 0
                    self.get_logger().info("-> ARMING")

            case "ARMING":
                self._send_offboard_mode()
                self._send_zero_traj()
                self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                if self.arm_state == VehicleStatus.ARMING_STATE_ARMED and self._cnt > 10:
                    self.state = "TAKEOFF"
                    self._cnt = 0
                    self.get_logger().info("-> TAKEOFF")
                if not self.flight_check:
                    self.state = "IDLE"

            case "TAKEOFF":
                self._send_offboard_mode()
                self._send_zero_traj()
                self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param1=1.0, param7=self._takeoff_alt)
                if self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                    self.state = "LOITER"
                    self._cnt = 0
                    self.get_logger().info("-> LOITER (waiting for hover)")
                if not self.flight_check:
                    self.state = "IDLE"

            case "LOITER":
                self._send_offboard_mode()
                self._send_zero_traj()
                self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                if self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                    self.state = "OFFBOARD"
                    self._cnt = 0
                    self.get_logger().info("-> OFFBOARD  policy active")
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

    def set_velocity(self, v_ned_n: float, v_ned_e: float, v_ned_d: float, yawspeed: float):
        """Store NED velocity setpoint; published on next publish_velocity() call."""
        self._v_ned_n = v_ned_n
        self._v_ned_e = v_ned_e
        self._v_ned_d = v_ned_d
        self._yawspeed = yawspeed

    def publish_velocity(self):
        """Send OffboardControlMode + TrajectorySetpoint with stored velocity."""
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
        msg.velocity[0] = self._v_ned_n
        msg.velocity[1] = self._v_ned_e
        msg.velocity[2] = self._v_ned_d
        msg.yaw = float("nan")
        msg.yawspeed = self._yawspeed
        self._pub_traj.publish(msg)

    def _send_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self._pub_offboard.publish(msg)

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

    def _send_cmd(self, command, param1=0.0, param2=0.0, param7=0.0):
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


def main():
    # 1. Task and agent config
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cast(dict[str, Any], load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point"))

    # 2. Find checkpoint
    if args_cli.checkpoint is not None:
        resume_path = args_cli.checkpoint
    else:
        log_root = os.path.abspath(
            os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
        )
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root, run_dir, checkpoint_file, other_dirs=["nn"])
    print(f"[INFO] Checkpoint: {resume_path}")

    # 3. Isaac Lab environment
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env_raw.unwrapped, DirectMARLEnv):
        env_raw = multi_agent_to_single_agent(env_raw)
    env = RlGamesVecEnvWrapper(env_raw, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
    )

    # 4. rl_games agent (obs normalization handled internally by .pth)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = 1

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # 5. Isaac env internals
    isaac_env = env_raw.unwrapped
    device = getattr(isaac_env, "device")
    print(f"[INFO] Device: {device}")

    goal_pos_w = torch.tensor(
        [args_cli.goal_x, args_cli.goal_y, args_cli.goal_z],
        device=device, dtype=torch.float32,
    )

    # 6. ROS2
    rclpy.init()
    ros_node = VelocityOffboardNode(takeoff_alt=args_cli.takeoff_alt)

    # 7. Reset env and pin goal
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    isaac_env._desired_pos_w[0] = goal_pos_w
    print(f"[INFO] Goal: {goal_pos_w.cpu().numpy()}")

    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    print("\n[INFO] Sim2sim running at 75 Hz.")
    print("[INFO] Arm: ros2 topic pub --once /arm_message std_msgs/msg/Bool \"data: true\"")

    loop_dt = 1.0 / 75.0  # 75 Hz  (env dt=1/150, decimation=2)
    step = 0

    while simulation_app.is_running() and step < args_cli.max_steps:
        t0 = time.time()

        # 8a. Policy step in Isaac
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)
            actions: torch.Tensor = agent.get_action(obs_t, is_deterministic=True)  # type: ignore[assignment]
            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

        # 8b. Re-pin goal after episode reset
        if dones.any():
            isaac_env._desired_pos_w[0] = goal_pos_w
            if agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0

        # 8c. Convert policy actions -> NED velocity setpoint
        #
        # Policy output layout:
        #   actions[0, 0] = yaw_rate_norm  (FLU CCW positive)
        #   actions[0, 1] = vx_norm        (forward, body +X)
        #   actions[0, 2] = vy_norm        (left,    body +Y)
        #   actions[0, 3] = vz_norm        (up,      body +Z)
        #
        # Step 1: scale to physical units
        yaw_rate_flu = float(actions[0, 0]) * MAX_YAW_RATE  # rad/s, CCW
        vx_b = float(actions[0, 1]) * MAX_VELOCITY           # forward m/s
        vy_b = float(actions[0, 2]) * MAX_VELOCITY           # left    m/s
        vz_b = float(actions[0, 3]) * MAX_VELOCITY           # up      m/s

        # Step 2: body FLU -> world FLU using Isaac heading
        #   heading_w = CCW angle from world +X (math convention)
        yaw_isaac = float(isaac_env._robot.data.heading_w[0])
        cos_y = math.cos(yaw_isaac)
        sin_y = math.sin(yaw_isaac)
        vx_world = vx_b * cos_y - vy_b * sin_y
        vy_world = vx_b * sin_y + vy_b * cos_y
        vz_world = vz_b

        # Step 3: Isaac world (Z-up) -> PX4 NED
        #   Assumes Isaac world X = NED North, Isaac world Y = NED East
        v_ned_n = vx_world    # North
        v_ned_e = vy_world    # East
        v_ned_d = -vz_world   # Down  (negate: FLU up -> NED down)

        # Step 4: yaw rate FLU CCW -> NED CW (negate)
        yawspeed_ned = -yaw_rate_flu

        ros_node.set_velocity(v_ned_n, v_ned_e, v_ned_d, yawspeed_ned)

        # 8d. Spin ROS2 (non-blocking, processes incoming callbacks)
        rclpy.spin_once(ros_node, timeout_sec=0.0)

        # 8e. Drive arming FSM then publish velocity (always, to keep stream alive)
        ros_node.tick_state_machine()
        ros_node.publish_velocity()

        # 8f. Log every ~2 s
        if step % 150 == 0:
            ros_node.get_logger().info(
                f"[{step:6d}] {ros_node.state:8s} | "
                f"vel_NED=({v_ned_n:.2f}, {v_ned_e:.2f}, {v_ned_d:.2f}) m/s | "
                f"yawspd={yawspeed_ned:.2f} rad/s | "
                f"heading={math.degrees(yaw_isaac):.1f} deg"
            )

        step += 1

        # 8g. Rate limit to 75 Hz
        remaining = loop_dt - (time.time() - t0)
        if remaining > 0.0:
            time.sleep(remaining)

    print(f"\n[INFO] Finished after {step} steps.")
    env.close()
    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
