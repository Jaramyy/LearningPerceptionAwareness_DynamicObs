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

import time

class StudentPolicy(nn.Module):
    def __init__(self, input_size=71, hidden_size=64, output_size=4):
        super(StudentPolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.05)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.05)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(0.05)
        self.out = nn.Linear(hidden_size, output_size)
        self.elu = nn.ELU()

    def forward(self, x):
        x1 = self.elu(self.fc1(x))
        x1 = self.dropout1(x1)

        x2 = self.elu(self.fc2(x1))
        x2 = self.dropout2(x2)

        x3 = self.elu(self.fc3(x2))
        x3 = self.dropout3(x3)

        out = self.out(x3)

        return out  # self.fc3(x)

    def loss(self):
        return nn.MSELoss()


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

    checkpoint_dir = os.path.join(
        "/home/jaramy/ros2_ws/src/ROS2_PX4_Offboard_Example/px4_offboard",
        "checkpoints",
    )
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    student_policy = StudentPolicy()
    student_policy.to(torch.device("cuda"))
    checkpoint = torch.load(checkpoint_path)
    student_policy.load_state_dict(checkpoint["model_state_dict"])
    student_policy.eval()

    # with torch.no_grad():
    #     action = student_policy(inputs)
    #     action = action.clone().clamp(-1.0, 1.0)
    #     action = action.cpu().numpy()

    # Fetch relevant parameters to make the quadcopter hover in place
    # prop_body_ids = robot.find_bodies("m.*_prop")[0]
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
        [10.0, 4.0 , 1.5], device=sim.device, dtype=torch.float32
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

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            
            lin_vel = robot.data.root_lin_vel_b
            ang_vel = robot.data.root_ang_vel_b
            robot_pos = robot.data.root_state_w[:, :3]
            
            desired_pos_b, _ = subtract_frame_transforms(
                robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], desired_pos
            )
            
            desired_dist = desired_pos_b.norm(dim=-1, keepdim=True)
            unit_desird_pos_b = desired_pos_b / (desired_dist + 1e-6)
            
            # print(f"Desired distance: {desired_distance}")
            desired_dist_2d = desired_pos_b[:, :2].norm(dim=-1, keepdim=True)
            desired_dist_z = desired_pos_b[:, 2].unsqueeze(1)

            print(f"desired_dist_2d: {desired_dist_2d}")
            print(f"Desired distance z: {desired_dist_z}")
            
            # print(f"Lidar readings: {lidar}")
            # print(
            #     f"Linear velocity: {lin_vel.shape}, Angular velocity: {ang_vel.shape}, "
            #     f"Unit desired position in body frame: {unit_desird_pos_b.shape}, "
            #     f"Desired distance 2D: {desired_dist_2d.shape}, Desired distance Z: {desired_dist_z.shape}"
            #     f" Lidar readings: {lidar.shape}"
            # )
            input_tensor = torch.cat(
                (
                    lin_vel[0],  # linear velocity in body frame
                    ang_vel[0],  # angular velocity in body frame
                    unit_desird_pos_b[0],  # unit vector towards desired position
                    desired_dist_2d[0],  # distance to desired position
                    desired_dist_z[0],  # distance to desired position in z-axis
                    lidar[0],  # lidar readings
                ),
            )

            action = student_policy(input_tensor)
            # action = action.clone().clamp(-1.0, 1.0)

            # print(f"Action: {action}")

            # print(
            #     f"Robot position: {robot_pos}, Linear velocity: {lin_vel}, Angular velocity: {ang_vel}"
            # )

            # reset
            if count % 1000 == 0:
                # reset counters
                sim_time = 0.0
                count = 0
                # reset dof state
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos,
                    robot.data.default_joint_vel,
                )
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
                robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
                robot.reset()
                # reset command
                print(">>>>>>>> Reset!")
            # apply action to the robot (make the robot float in place)
            # forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
            # print(robot.num_instances)
    
            
            # forces[:, 2] = robot_mass * gravity * 8.0
            # scaled_actions = (action + 1.0) / 2.0
            scaled_actions = (action + 1.0) / 2.0


            omega_ref = alloc_matrix.get_omega_max() * scaled_actions
            omega_real = motor.compute(omega_ref)

            processed_actions = alloc_matrix.compute(omega_real)
            # print(f"Processed actions: {processed_actions.squeeze(-1).shape}")

            forces[:, 2] = processed_actions[0, 0]
            torques[:, :] = processed_actions[0, 1:]

            # print(f"forces: {forces}, torques: {torques}")

            # forces[:, 2] = ((action[0] + 1.0) / 2.0) * robot_mass * gravity * 5.0
            # torques[:, 0] = action[1] * 0.7
            # torques[:, 1] = action[2] * 0.7
            # torques[:, 2] = action[3] * 0.7


            robot.set_external_force_and_torque(forces, torques, body_ids=body_id)
            robot.write_data_to_sim()
            # perform step
            sim.step()
            # update sim-time
            sim_time += sim_dt
            count += 1
            # update buffers
    
            inference_time = 0.03
            elapsed_time = time.time() - start_time
            remaining_time = inference_time - elapsed_time
            if remaining_time > 0.0:
                # print(f"Sleeping for {remaining_time:.4f} seconds to maintain inference time.")
                time.sleep(remaining_time)

            robot.update(sim_dt)

            loop_duration = time.time() - start_time
            # print(f"Loop time: {loop_duration:.4f} seconds, Frequency: {1.0 / loop_duration:.2f} Hz")



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
        thrust_torque = torch.bmm(self._allocation_matrix, thrusts_ref_batched.unsqueeze(-1)).squeeze(-1)

        
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

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
