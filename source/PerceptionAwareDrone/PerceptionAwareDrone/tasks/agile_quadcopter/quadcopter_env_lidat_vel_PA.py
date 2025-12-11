# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_apply_yaw, quat_from_angle_axis, euler_xyz_from_quat

##
# Pre-defined configs
##
from .robot.agileDrone import AGILE_CFG    # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

#terrain
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG, OBSTACLE_RAND_POS

import isaaclab.envs.mdp as mdp    
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

# viewpoint
from isaaclab.envs.ui  import ViewportCameraController
from isaaclab.envs import ViewerCfg

# sensor
from isaaclab.sensors import RayCasterCfg, RayCaster, patterns
from isaaclab.sensors import Imu, ImuCfg

#lidar 
import einops
from isaacsim.util.debug_draw import _debug_draw

#markers
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

#goal
import numpy as np

@configclass
class EventCfg:
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.50, 0.5),
            "operation": "add",
        },
    )

class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    observation_space = 12
    # observation_space = 17 #with 5 beams lidar
    # observation_space = 12
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow
    
    # viewer = ViewerCfg(eye=(-19.8, -23.8, 11.5), lookat=(-24.0, -8.5, -1.7), origin_type='env', env_index=2015)
    viewer = ViewerCfg(eye=(-19.8, -23.8, 11.5), lookat=(-24.0, -8.5, -1.7), origin_type='env')

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/150,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # flat_terrain = False  # for generator terrain
    flat_terrain = True
    if flat_terrain:
        # for flat and emtry terrain
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
    else:
        # for custom terrain
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            max_init_terrain_level=9,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            # visual_material=sim_utils.MdlFileCfg(
            #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            #     project_uvw=True,
            # ),
            debug_vis=True,
        )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    
    # events
    # events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = AGILE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # sensor 
    lidar_sensor = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),
        attach_yaw_only=False,
        # pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(10.0, 20.0), horizontal_fov_range=(-50.0, 50.0),horizontal_res=1.67),     #For limited fov
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(10.0, 20.0), horizontal_fov_range=(-179.0, 179.0),horizontal_res=6.0),      #For full fov 
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    thrust_to_weight = 5.0
    moment_scale = 0.7

    # reward scales
    lin_vel_reward_scale = -0.2
    ang_vel_reward_scale = -0.003
    distance_to_goal_reward_scale = 60.0
    action_rate_reward_scale = -0.5

    #max velocity
    max_velocity = 4.0  # m/s
    max_yaw_rate = 3.14  # rad/s



class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.previous_action = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._lin_vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._yaw_vel_cmd = torch.zeros(self.num_envs, 1, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "action_rate",
            ]
        }
       # Get specific body indices
        self._body_id = self._robot.find_bodies("base_link")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        all_inertia_tensor = self._robot.root_physx_view.get_inertias()[0] # shape (num_envs, 3, 3)
        robot_inertia = torch.sum(all_inertia_tensor,dim=0)
        print("Inertia tensor of the robot:", robot_inertia)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # Ixx = 0.00072172
        Ixx = 7.6191e-04
        # Iyy = 0.00088563
        Iyy = 8.9651e-04
        # Izz = 0.0012558
        Izz = 1.2983e-03
        robot_inertia = torch.diag(torch.tensor([Ixx, Iyy, Izz], device=self.device))

        self.velocity_controller = GeometricVelocityController(
            num_env=self.num_envs,
            mass=self._robot_mass,
            inertia=robot_inertia,
            device=self.device,
        )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # self._actions = actions.clone().clamp(-1.0, 1.0)
        # self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        # self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

        self._actions = actions.clone().clamp(-1.0, 1.0)  
        # self._actions = torch.ones_like(actions)   # just for testing 
        # self._actions[:, 1:] = self._actions[:, 1:]*0.0
        # print(f"Actions received: {self._actions}")   

        # fake_action = torch.zeros_like(self._actions)
        # fake_action[:, 0] = 0.5
        # fake_action[:, 1:] = 1.0
        

        self._yaw_vel_cmd[:, 0] = self._actions[:, 0] * self.cfg.max_yaw_rate # rad/s
        self._lin_vel_cmd[:, :] = self._actions[:, 1:] * self.cfg.max_velocity  # m/s
        # self._lin_vel_cmd[:, :] = fake_action[:, 1:] * self.cfg.max_velocity  # m/s
        # self._yaw_vel_cmd[:, 0] = fake_action[:, 0] * self.cfg.max_yaw_rate # rad/s

        
        
        #TODO: Feed input to controller and CHECK!!
        thrust, moment = self.velocity_controller.update_velocity_only(
            quat_w=self._robot.data.root_link_quat_w,
            vel_w=self._robot.data.root_lin_vel_w,
            omega_b=self._robot.data.root_ang_vel_b,
            desired_vel_w=self._lin_vel_cmd[:, :],
            desired_yaw_rate=self._yaw_vel_cmd[:, 0],
        )
        # print(f" thrust: {thrust}, moment: {moment}")
        self._thrust[:, :] = 0.0
        self._moment[:, :] = 0.0
        self._thrust[:, 0, 2] = thrust
        self._moment[:, 0, :] = moment

        

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
        

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self.previous_action), dim=1)

        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "action_rate": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        self.previous_action = self._actions.clone()

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.3, self._robot.data.root_pos_w[:, 2] > 4.0)
        
        uprightness = self._robot.data.projected_gravity_b[:, 2] >= 0.0
        died = died | uprightness
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)



class GeometricVelocityController:

    def __init__(self, num_env, mass, inertia, device):
        self.num_env = num_env
        self.mass = mass

        # self.J = inertia.unsqueeze(0).to(device)    # (1,3,3)
        self.J_expand = inertia.unsqueeze(0).expand(self.num_env, 3, 3).to(device)
        self.device = device
        

        # Position gains
        self.k_p = 8.0 
        self.k_d = 25.0

        self.k_d_xy = 22.0
        self.k_d_z = 80.0
        
        # FOR 1/200
        # self.k_p = 16.0 # WORKED at 16.0
        # self.k_d = 180.0

        # Attitude gains
        self.kR = 2.5 #4.5
        self.kW = 0.1 #0.3

        # Gravity
        self.g = torch.tensor([0., 0., -9.81], device=device).unsqueeze(0)


    # --------------------
    # Utility functions
    # --------------------
    def vee_map(self, M):
        # Extract [M32 - M23, M13 - M31, M21 - M12]
        return torch.stack((
            M[:, 2, 1] - M[:, 1, 2],
            M[:, 0, 2] - M[:, 2, 0],
            M[:, 1, 0] - M[:, 0, 1],
        ), dim=1)

    def matrix_from_quat(self, q):
        """ IsaacLab format: q = (w,x,y,z) """
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.zeros((q.shape[0], 3, 3), device=q.device)

        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)

        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - x * w)

        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)

        return R

    def quat_rotate(self, q, v):
        """Rotate vector v by quaternion q."""
        R = self.matrix_from_quat(q)
        return torch.bmm(R, v.unsqueeze(2)).squeeze(2)

    # ----------------------------------------
    # MAIN CONTROL STEP
    # ----------------------------------------
    def update(
        self,
        pos_w, vel_w, quat_w, omega_b,
        desired_pos_w, desired_vel_w,
        desired_yaw, desired_yaw_rate
    ):
        # ---------------------------
        # 1) POSITION + VELOCITY ERRORS
        # ---------------------------
        pos_err = desired_pos_w - desired_pos_w
        vel_err = desired_vel_w - vel_w

        # ---------------------------
        # 2) DESIRED ACCELERATION
        # ---------------------------
        # acc_des = self.k_d * vel_err + self.g
        acc_des = torch.zeros_like(vel_w)
        acc_des[:, 0] = self.k_d_xy * vel_err[:, 0]
        acc_des[:, 1] = self.k_d_xy * vel_err[:, 1]
        acc_des[:, 2] = self.k_d_z * vel_err[:, 2] + self.g[:, 2]

        # ---------------------------
        # 3) DESIRED BODY Z (b3c)
        # ---------------------------
        b3_c = acc_des / torch.norm(acc_des, dim=1, keepdim=True)

        # ---------------------------
        # 4) DESIRED BODY X,Y FROM YAW
        # ---------------------------
        cy = torch.cos(desired_yaw)
        sy = torch.sin(desired_yaw)

        b1_des = torch.tensor([[cy, sy, 0.0]], device=self.device)

        b2_c = torch.cross(b3_c, b1_des, dim=1)
        b2_c = b2_c / torch.norm(b2_c, dim=1, keepdim=True)

        b1_c = torch.cross(b2_c, b3_c, dim=1)

        # ---------------------------
        # 5) DESIRED ROTATION MATRIX Rd
        # ---------------------------
        Rd = torch.zeros((1, 3, 3), device=self.device)
        Rd[:, :, 0] = b1_c
        Rd[:, :, 1] = b2_c
        Rd[:, :, 2] = b3_c

        # ---------------------------
        # 6) CURRENT ROTATION MATRIX R
        # ---------------------------
        R = self.matrix_from_quat(quat_w)

        # ---------------------------
        # 7) THRUST COMMAND
        # ---------------------------
        # Project desired force onto current body z axis
        b3 = R[:, :, 2]
        thrust = self.mass * torch.sum(acc_des * b3, dim=1)

        # ---------------------------
        # 8) ROTATION ERROR
        # ---------------------------
        rotation_error = 0.5 * self.vee_map(
            torch.bmm(Rd.transpose(1, 2), R) -
            torch.bmm(R.transpose(1, 2), Rd)
        )

        # ---------------------------
        # 9) DESIRED BODY RATES
        # ---------------------------
        omega_c = torch.zeros((1, 3), device=self.device)
        omega_c[:, 2] = desired_yaw_rate

        RRT = torch.bmm(R.transpose(1, 2), Rd)
        omega_c_body = torch.bmm(RRT, omega_c.unsqueeze(2)).squeeze(2)

        omega_err = omega_b - omega_c_body
        print(f"Omega error: {omega_err}")

        # ---------------------------
        # 10) FEEDFORWARD TERM (Ω × JΩc)
        # ---------------------------
        J_omega_c = torch.bmm(self.J, omega_c.unsqueeze(2)).squeeze(2)
        feedforward = torch.cross(omega_b, J_omega_c, dim=1)

        # ---------------------------
        # 11) TORQUE COMMAND
        # ---------------------------
        torque = -self.kR * rotation_error - self.kW * omega_err + feedforward

        return thrust, torque
    
    def update_velocity_only(
        self,
        vel_w, quat_w, omega_b,
        desired_vel_w,
        desired_yaw_rate
    ):
        """
        Velocity-only control with yaw-rate tracking.
        Inputs:
            vel_w          : current linear velocity (1,3)
            quat_w         : current orientation quaternion (1,4)
            omega_b        : current body angular velocity (1,3)
            desired_vel_w  : commanded linear velocity (1,3)
            desired_yaw_rate : commanded body-frame yaw-rate (rad/s)
        Outputs:
            thrust  : scalar thrust along body z-axis
            torque  : body-frame torque (1,3)
        """
        # Ensure batch dims
        # vel_w = vel_w.view(1,3)
        # quat_w = quat_w.view(1,4)
        # omega_b = omega_b.view(1,3)
        # desired_vel_w = desired_vel_w.view(1,3)

        # ---------------------------
        # 1) VELOCITY ERROR
        # ---------------------------
        vel_err = desired_vel_w - vel_w

        # ---------------------------
        # 2) DESIRED ACCELERATION (velocity-only)
        # ---------------------------
        acc_des = torch.zeros_like(vel_w)  # force command
        acc_des[:, 0] = self.k_d_xy * vel_err[:, 0]
        acc_des[:, 1] = self.k_d_xy * vel_err[:, 1]
        acc_des[:, 2] = self.k_d_z * vel_err[:, 2] + self.g[:, 2]  # include gravity

        # ---------------------------
        # 3) BODY Z (thrust direction)
        # ---------------------------
        b3_c = acc_des / (torch.norm(acc_des, dim=1, keepdim=True))

        # ---------------------------
        # 4) DESIRED BODY X/Y (arbitrary, yaw free)
        # ---------------------------
        # current yaw from quaternion
        # w, x, y, z = quat_w[:,0], quat_w[:,1], quat_w[:,2], quat_w[:,3]
        
        euler = euler_xyz_from_quat(quat_w)
        # print("Quat shape: ", len(euler))
        # print("Euler angles (rpy) shape: ", euler[0])

        if isinstance(euler, tuple):
            # Each tensor shape: (1,) → stack to (3,) → then batch to (env,3)
            self.current_euler = torch.stack([e.squeeze(-1) for e in euler], dim=1)

        # print("Current euler angles (rpy): ", self.current_euler.shape)
        
        cy = torch.cos(self.current_euler[:, 2])
        sy = torch.sin(self.current_euler[:, 2])

        # cp = torch.cos(self.current_euler[:, 1])
        # sp = torch.sin(self.current_euler[:, 1])

        # cr = torch.cos(self.current_euler[:, 0])
        # sr = torch.sin(self.current_euler[:, 0])

        # b1_des = torch.tensor([[cy, sy, torch.zeros_like(cy)]], device=self.device)
        b1_des = torch.stack((cy, sy, torch.zeros_like(cy)), dim=1)  # shape (B,3)

        # b1_des = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)  # yaw-free reference
        b2_c = torch.cross(b3_c, b1_des, dim=1)
        b2_c /= (torch.norm(b2_c, dim=1, keepdim=True))
        b1_c = torch.cross(b2_c, b3_c, dim=1)

        

        # ---------------------------
        # 5) DESIRED ROTATION MATRIX Rd
        # ---------------------------
        # Rd = torch.zeros((1,3,3), device=self.device)
        # Rd[:,:,0] = b1_c
        # Rd[:,:,1] = b2_c
        # Rd[:,:,2] = b3_c
        Rd = torch.stack((b1_c, b2_c, b3_c), dim=2)

        # ---------------------------
        # 6) CURRENT ROTATION MATRIX R
        # ---------------------------
        R = self.matrix_from_quat(quat_w)

        # ---------------------------
        # 7) THRUST
        # ---------------------------
        b3 = R[:, :, 2]  # current body z-axis
        thrust = self.mass * torch.sum(acc_des * b3, dim=1)

        # ---------------------------
        # 8) ROTATION ERROR (roll/pitch only, yaw ignored)
        # ---------------------------
        rotation_error = 0.5 * self.vee_map(
            torch.bmm(Rd.transpose(1,2), R) - torch.bmm(R.transpose(1,2), Rd)
        )
        rotation_error[:,2] = 0.0  # zero yaw error

        # ---------------------------
        # 9) DESIRED BODY RATES
        # ---------------------------
        omega_c = torch.zeros_like(omega_b)
        # print("Omega command shape: ", omega_c.shape)
        # print("Desired yaw rate shape: ", desired_yaw_rate.shape)
        omega_c[:, 2] = desired_yaw_rate[:]  # yaw-rate tracking

        omega_err = omega_b - omega_c

        # ---------------------------
        # 10) FEEDFORWARD
        # ---------------------------
        
        J_omega_c = torch.bmm(self.J_expand, omega_c.unsqueeze(2)).squeeze(2)
        feedforward = torch.cross(omega_b, J_omega_c, dim=1)

        # ---------------------------
        # 11) TORQUE
        # ---------------------------
        torque = -self.kR * rotation_error - self.kW * omega_err + feedforward

        return thrust, torque

