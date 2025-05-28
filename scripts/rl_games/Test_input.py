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
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
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
        [8.0, 0.5, 1.0], device=sim.device, dtype=torch.float32
    )  # desired position in world frame

    forces = torch.zeros(1, 3, device=sim.device)
    torques = torch.zeros_like(forces)
    while simulation_app.is_running():

        lin_vel = robot.data.root_lin_vel_b
        ang_vel = robot.data.root_ang_vel_b
        robot_pos = robot.data.root_state_w[:, :3]
        
        desired_pos_b, _ = subtract_frame_transforms(
            robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], desired_pos
        )
        err_pose = desired_pos_b
        # print(f"Error pose: {err_pose}")
        # norm_err = torch.norm(err_pose)
        # unit_desired = err_pose / (norm_err + 1e-6)

        desired_dist = err_pose.norm(dim=-1, keepdim=True)
        unit_desird_pos_b = err_pose / (desired_dist + 1e-6)
        
        # print(f"Desired distance: {desired_distance}")
        desired_dist_2d = err_pose[:, :2].norm(dim=-1, keepdim=True)
        desired_distance_z = err_pose[: , 2]
        # print(f"Desired distance z: {desired_distance_z}")

        lidar = torch.ones(robot.num_instances, 60, device=sim.device) * 5.0
        # print(f"Lidar readings: {lidar}")

        input_tensor = torch.cat(
            (
                lin_vel[0],  # linear velocity in body frame
                ang_vel[0],  # angular velocity in body frame
                unit_desird_pos_b[0],  # unit vector towards desired position
                torch.tensor([desired_dist_2d], device=sim.device),  # distance to desired position
                desired_distance_z,  # distance to desired position in z-axis
                lidar[0],  # lidar readings
            ),
        )

        action = student_policy(input_tensor)
        action = action.clone().clamp(-1.0, 1.0)

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

        forces[:, 2] = ((action[0] + 1.0) / 2.0) * robot_mass * gravity * 5.0
        torques[:, 0] = action[1] * 0.9
        torques[:, 1] = action[2] * 0.9
        torques[:, 2] = action[3] * 0.9


        robot.set_external_force_and_torque(forces, torques, body_ids=body_id)
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        robot.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
