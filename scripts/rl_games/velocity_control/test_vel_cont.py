# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a quadcopter.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/quadcopter.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates how to simulate a quadcopter."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip
from PerceptionAwareDrone.tasks.agile_quadcopter.robot.agileDrone import AGILE_CFG

import torch
import torch.nn as nn
import torch.optim as optim
import os

from datetime import datetime
import csv
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import matrix_from_quat
import time


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(
        dt=1 / 100,
        device=args_cli.device, 
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,)
    )

    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.0, 1.0, 2.0], target=[0.0, 0.0, 1.0])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robots
    robot_cfg = AGILE_CFG.replace(prim_path="/World/AgileDrone")
    # robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg.spawn.func(
        "/World/AgileDrone", robot_cfg.spawn, translation=robot_cfg.init_state.pos
    )

    # create handles for the robots
    robot = Articulation(robot_cfg)

    # Play the simulator
    sim.reset()


    # Fetch relevant parameters to make the quadcopter hover in place
    body_id = robot.find_bodies("base_link")[0]
    # robot_mass = robot.root_physx_view.get_masses().sum()
    robot_mass = robot.root_physx_view.get_masses()[0].sum()
    print(f"[INFO]: Robot mass: {robot_mass:.2f} kg")
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    print(f"[INFO]: Simulation DT: {sim_dt:.4f} seconds")
    sim_time = 0.0
    count = 0
    # Simulate physics
    desired_pos = torch.tensor(
        [0.0, 0.0 , 1.0], device=sim.device, dtype=torch.float32
    )  # desired position in world frame

    forces = torch.zeros(1, 3, device=sim.device)
    torques = torch.zeros_like(forces)
    lidar = torch.ones(robot.num_instances, 60, device=sim.device) * 4.9
    
    taus: list[float] = (0.0001, 0.0001, 0.0001, 0.0001)
    """Time constants for each motor."""
            
    init: list[float] = (2572.5, 2572.5, 2572.5, 2572.5)
    """Initial angular velocities for each motor in rad/s."""
            
    max_rate: list[float] = (50000.0, 50000.0, 50000.0, 50000.0)
    """Maximum rate of change of angular velocities for each motor in rad/s^2."""
            
    min_rate: list[float] = (-50000.0, -50000.0, -50000.0, -50000.0)
    """Minimum rate of change of angular velocities for each motor in rad/s^2."""
            
    use_motor_model: bool = False
    """Flag to determine if motor delay is bypassed."""


    alloc_matrix = Allocation(num_envs=1, device=sim.device)
    motor = Motor(
        num_envs=1,
        taus=taus,
        init=init,
        max_rate=max_rate,
        min_rate=min_rate,
        dt=sim_dt,
        use=use_motor_model,
        device=sim.device,
    )

    # Create controllers
    pos_controller = PositionController(k_p=5.2, k_d=0.4, max_vel=10.0, device=sim.device)
    vel_controller = VelocityController(k_p=10.0, k_d=1.2, max_force=50.0, device=sim.device)
    yaw_controller = YawController(k_p=4.0, k_d=0.8, max_torque=0.5, device=sim.device)

    desired_pos = torch.tensor([[0.0, 0.0, 2.0]], device=sim.device)
    desired_yaw = torch.tensor([0.0], device=sim.device)  # 0 rad (facing forward)

    while simulation_app.is_running():
        with torch.inference_mode():
            

            # Read current states
            state_w = {
                "pos_w": robot.data.root_pos_w,
                "lin_vel_w": robot.data.root_lin_vel_w,
                "rot_w_b": matrix_from_quat(robot.data.root_link_quat_w),
            }
            ang_vel_b = robot.data.root_ang_vel_b

            # --- Position Control ---
            # target_vel_b = pos_controller(state_w, desired_pos)

            # --- Velocity Control (for thrust & XY) ---
            state_vel = {"lin_vel_b": robot.data.root_lin_vel_b, "ang_vel_b": ang_vel_b}
            # target_vel_4d = torch.cat([target_vel_b, torch.zeros((1, 1), device=sim.device)], dim=-1)
            force_b, torque_b = vel_controller(state_vel, target_vel_4d)

            # --- Yaw Control ---
            # yaw_torque = yaw_controller(matrix_from_quat(robot.data.root_link_quat_w), ang_vel_b, desired_yaw)
            # torque_b[:, 2:3] = yaw_torque  # replace yaw torque

            # --- Force + Motor Allocation ---
            total_thrust = force_b[:, 2] + robot_mass * (-1*gravity)
            # processed_actions = torch.cat([total_thrust.unsqueeze(-1), torque_b], dim=-1).unsqueeze(0)

            # omega_guess = torch.sqrt(torch.abs(processed_actions / alloc_matrix._thrust_coeff))
            # omega_ref = torch.clamp(omega_guess.squeeze(0), 0.0, alloc_matrix.get_omega_max())
            # omega_real = motor.compute(omega_ref)
            # processed_actions = alloc_matrix.compute(omega_real)

            # forces[:, 2] = processed_actions[0, 0]
            # torques[:, :] = processed_actions[0, 1:]
            forces[:, 2] = total_thrust
            torques[:, :] = torque_b


            # --- Publish debug data for PlotJuggler ---
            pid_publisher.publish_vec(pid_publisher.pub_error, (target_vel_b - robot.data.root_lin_vel_b).squeeze().cpu())
            pid_publisher.publish_vec(pid_publisher.pub_cmd, force_b.squeeze().cpu())
            pid_publisher.publish_vec(pid_publisher.pub_vel, robot.data.root_lin_vel_b.squeeze().cpu())
            pid_publisher.publish_vec(pid_publisher.pub_target, target_vel_b.squeeze().cpu())

            # --- Apply control ---
            robot.set_external_force_and_torque(forces, torques, body_ids=body_id)
            robot.write_data_to_sim()
            sim.step()
            sim_time += sim_dt
            robot.update(sim_dt)



class Allocation:
    def __init__(self, num_envs, device="cuda", dtype=torch.float32):
        """
        Initializes the allocation matrix for a quadrotor for multiple environments.

        Parameters:
        - num_envs (int): Number of environments
        - arm_length (float): Distance from the center to the rotor
        - thrust_coeff (float): Rotor thrust constant
        - drag_coeff (float): Rotor torque constant
        - device (str): 'cpu' or 'cuda'
        - dtype (torch.dtype): Desired tensor dtype
        """
        
        arm_length: float = 0.130 #0.035
        """Length of the arms of the drone in meters."""
        
        drag_coef: float = 1.5e-9
        """Drag torque coefficient."""
        
        thrust_coef: float = 5.327e-7 #2.25e-7
        """Thrust coefficient.
        Calculated with 5145 rad/s max angular velociy, thrust to weight: 4, mass: 0.6076 kg and gravity: 9.81 m/s^2.
        thrust_coef = (4 * 0.6076 * 9.81) / (4 * 5145**2) = 2.25e-7."""
        
        self.omega_max: float = 5145.0
        """Maximum angular velocity of the drone motors in rad/s.
        Calculated with 1950KV motor, with 6S LiPo battery with 4.2V per cell.
        1950 * 6 * 4.2 = 49,140 RPM ~= 5145 rad/s."""
        

        sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        A = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [arm_length * sqrt2_inv, -arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [-arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [drag_coef, -drag_coef, drag_coef, -drag_coef],
            ],
            dtype=dtype,
            device=device,
        )
        self._allocation_matrix = A.unsqueeze(0).repeat(num_envs, 1, 1)
        self._thrust_coeff = thrust_coef

    def compute(self, omega):
        """
        Computes the total thrust and body torques given the rotor angular velocities.

        Parameters:
        - omega (torch.Tensor): Tensor of shape (num_envs, 4) representing rotor angular velocities

        Returns:
        - thrust_torque (torch.Tensor): Tensor of shape (num_envs, 4)
        """

        
        thrusts_ref = self._thrust_coeff * omega**2
        thrusts_ref_batched = thrusts_ref.unsqueeze(0)
        # print(f"Thrusts reference: {thrusts_ref.unsqueeze(-1).shape}, Allocation matrix: {self._allocation_matrix.shape}")
        # thrust_torque = torch.bmm(self._allocation_matrix, thrusts_ref_batched.unsqueeze(-1)).squeeze(-1)
        # Ensure thrusts_ref_batched is 3D
        
        if thrusts_ref_batched.ndim == 1:
            # [4] → [1, 4, 1]
            thrusts_ref_batched = thrusts_ref_batched.unsqueeze(0).unsqueeze(-1)
        elif thrusts_ref_batched.ndim == 2:
            # [1, 4] or [B, 4] → [B, 4, 1]
            thrusts_ref_batched = thrusts_ref_batched.unsqueeze(-1)
        elif thrusts_ref_batched.ndim == 3 and thrusts_ref_batched.shape[1] == 1:
            # [B, 1, 4] → [B, 4, 1]
            thrusts_ref_batched = thrusts_ref_batched.transpose(1, 2)

        thrust_torque = torch.bmm(self._allocation_matrix, thrusts_ref_batched).squeeze(-1)

        
        # thrust_torque = torch.matmul(thrusts_ref, self._allocation_matrix.T) 
        return thrust_torque
    
    def get_omega_max(self):
        """
        Returns the maximum angular velocity of the motors.

        Returns:
        - omega_max (float): Maximum angular velocity in rad/s.
        """
        return self.omega_max


class Motor:
    def __init__(self, num_envs, taus, init, max_rate, min_rate, dt, use, device="cpu", dtype=torch.float32):
        """
        Initializes the motor model.

        Parameters:
        - num_envs: Number of envs.
        - taus: (4,) Tensor or list specifying time constants per motor.
        - init: (4,) Tensor or list specifying the initial omega per motor. (rad/s)
        - max_rate: (4,) Tensor or list specifying max rate of change of omega per motor. (rad/s^2)
        - min_rate: (4,) Tensor or list specifying min rate of change of omega per motor. (rad/s^2)
        - dt: Time step for integration.
        - use: Boolean indicating whether to use motor dynamics.
        - device: 'cpu' or 'cuda' for tensor operations.
        - dtype: Data type for tensors.
        """
        self.num_envs = num_envs
        self.num_motors = len(taus)
        self.dt = dt
        self.use = use
        self.init = init
        self.device = device
        self.dtype = dtype

        self.omega = torch.tensor(init, device=device).expand(num_envs, -1).clone()  # (num_envs, num_motors)

        # Convert to tensors and expand for all drones
        self.tau = torch.tensor(taus, device=device).expand(num_envs, -1)  # (num_envs, num_motors)
        self.max_rate = torch.tensor(max_rate, device=device).expand(num_envs, -1)  # (num_envs, num_motors)
        self.min_rate = torch.tensor(min_rate, device=device).expand(num_envs, -1)  # (num_envs, num_motors)

    def compute(self, omega_ref):
        """
        Computes the new omega values based on reference omega and motor dynamics.

        Parameters:
        - omega_ref: (num_envs, num_motors) Tensor of reference omega values.

        Returns:
        - omega: (num_envs, num_motors) Tensor of updated omega values.
        """

        if not self.use:
            self.omega = omega_ref
            return self.omega

        # Compute omega rate using first-order motor dynamics
        omega_rate = (1.0 / self.tau) * (omega_ref - self.omega)  # (num_envs, num_motors)
        omega_rate = omega_rate.clamp(self.min_rate, self.max_rate)

        # Integrate
        self.omega += self.dt * omega_rate
        return self.omega

    def reset(self, env_ids):
        """
        Resets the motor model to initial conditions.
        """
        self.omega[env_ids] = torch.tensor(self.init, device=self.device, dtype=self.dtype).expand(len(env_ids), -1)

class VelocityController:
    def __init__(self, k_p=2.0, k_d=0.5, max_force=15.0, max_torque=1.0, device="cuda"):
        """
        Simple velocity + yaw rate PD controller.
        Controls linear velocity and yaw angular velocity.
        """
        self.k_p = k_p
        self.k_d = k_d
        self.max_force = max_force
        self.max_torque = max_torque
        self.device = device

    def __call__(self, state, target_vel):
        """
        Args:
            state: dict with 'lin_vel_b', 'ang_vel_b'
            target_vel: (num_envs, 4) tensor [vx, vy, vz, yaw_rate]
        Returns:
            forces: (num_envs, 3)
            torques: (num_envs, 3)
        """
        lin_vel_b = state["lin_vel_b"]
        ang_vel_b = state["ang_vel_b"]

        # Split velocity and yaw rate targets
        target_lin_vel = target_vel[..., :3]
        target_yaw_rate = target_vel[..., 3:4]

        # Linear velocity control
        lin_error = target_lin_vel - lin_vel_b
        lin_acc_cmd = self.k_p * lin_error - self.k_d * lin_vel_b
        force_b = torch.clamp(lin_acc_cmd, -self.max_force, self.max_force)

        # Yaw rate control (around body z)
        yaw_error = target_yaw_rate - ang_vel_b[..., 2:3]
        torque_b = torch.zeros_like(lin_vel_b)
        torque_b[..., 2:3] = torch.clamp(self.k_p * yaw_error, -self.max_torque, self.max_torque)

        return force_b, torque_b
    
class PIDPlotPublisher(Node):
    def __init__(self, name="vel_pid_pub"):
        super().__init__(name)
        self.pub_error = self.create_publisher(Vector3, "/vel_pid/error", 10)
        self.pub_cmd = self.create_publisher(Vector3, "/vel_pid/command", 10)
        self.pub_vel = self.create_publisher(Vector3, "/drone/velocity", 10)
        self.pub_target = self.create_publisher(Vector3, "/drone/target_velocity", 10)

    def publish_vec(self, pub, vec):
        msg = Vector3()
        msg.x = float(vec[0])
        msg.y = float(vec[1])
        msg.z = float(vec[2])
        pub.publish(msg)

class PositionController:
    def __init__(self, k_p=1.0, k_d=0.3, max_vel=2.0, device="cuda"):
        """
        Simple PD position controller that outputs desired body-frame velocity.
        Args:
            k_p: proportional gain for position error
            k_d: derivative gain for velocity damping
            max_vel: max output velocity [m/s]
        """
        self.k_p = k_p
        self.k_d = k_d
        self.max_vel = max_vel
        self.device = device

    def __call__(self, state, target_pos):
        """
        Args:
            state: dict with 'pos_w', 'lin_vel_w' (world frame)
            target_pos: (num_envs, 3) desired position in world frame
        Returns:
            target_vel_b: (num_envs, 3) desired velocity in body frame
        """
        pos_w = state["pos_w"]
        lin_vel_w = state["lin_vel_w"]

        pos_error = target_pos - pos_w
        vel_cmd_w = self.k_p * pos_error - self.k_d * lin_vel_w
        vel_cmd_w = torch.clamp(vel_cmd_w, -self.max_vel, self.max_vel)

        # Convert to body frame using rotation matrix
        rot_w_b = state["rot_w_b"]  # shape: (num_envs, 3, 3)
        target_vel_b = torch.bmm(rot_w_b.transpose(1, 2), vel_cmd_w.unsqueeze(-1)).squeeze(-1)

        return target_vel_b
class YawController:
    def __init__(self, k_p=3.0, k_d=0.5, max_torque=0.5, device="cuda"):
        """
        Simple PD controller for yaw orientation.
        Args:
            k_p: proportional gain for yaw error
            k_d: derivative gain for yaw rate damping
        """
        self.k_p = k_p
        self.k_d = k_d
        self.max_torque = max_torque
        self.device = device

    def __call__(self, current_rot, current_ang_vel, desired_yaw):
        """
        Args:
            current_rot: (num_envs, 3, 3) rotation matrices world->body
            current_ang_vel: (num_envs, 3) angular velocity in body frame
            desired_yaw: (num_envs,) desired yaw in radians
        Returns:
            yaw_torque: (num_envs, 1)
        """
        # Extract current yaw angle from rotation matrix
        current_yaw = torch.atan2(current_rot[:, 1, 0], current_rot[:, 0, 0])  # atan2(R21, R11)
        yaw_error = desired_yaw - current_yaw

        # Wrap error to [-pi, pi]
        yaw_error = (yaw_error + torch.pi) % (2 * torch.pi) - torch.pi

        # PD control on yaw
        yaw_rate = current_ang_vel[:, 2]
        yaw_torque = self.k_p * yaw_error - self.k_d * yaw_rate

        yaw_torque = torch.clamp(yaw_torque, -self.max_torque, self.max_torque)
        return yaw_torque.unsqueeze(-1)

if __name__ == "__main__":
    # run the main function
    
    rclpy.init()
    pid_publisher = PIDPlotPublisher()
    
    main()
    
    # close sim app
    simulation_app.close()
