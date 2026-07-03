#python3 scripts/rl_games/train.py --task Isaac-Agile-Lidar-Vel-PA-v0 --num_envs 4096

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
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg,  HfDiscreteObstaclesTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_apply_yaw, quat_from_angle_axis, euler_xyz_from_quat, quat_apply_inverse

##z
# Pre-defined configs
##
from .robot.agileDrone import AGILE_CFG    # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

#terrain
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

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
# from isaacsim.util.debug_draw import _debug_draw

#markers
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

#goal
import numpy as np
from .utility.lee_velocity_controller.vel_controller import GeometricVelocityController

PUSH_LIN_VEL = 0.3  # m/s
PUSH_ANG_VEL = 0.3  # rad/s
@configclass
class EventCfg:

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.90, 1.10),
            "operation": "scale",
        },
    )

    noise_com_pos = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 15.0),
        params={
            "velocity_range": {
                "x": (-PUSH_LIN_VEL, PUSH_LIN_VEL),
                "y": (-PUSH_LIN_VEL, PUSH_LIN_VEL),
                "z": (-PUSH_LIN_VEL, PUSH_LIN_VEL),
                "roll": (-PUSH_ANG_VEL, PUSH_ANG_VEL),
                "pitch": (-PUSH_ANG_VEL, PUSH_ANG_VEL),
                "yaw": (-PUSH_ANG_VEL, PUSH_ANG_VEL),
            }
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-3.5, -1.5),
                "y": (-0.5, 0.5),
                "z": (3.5, 1.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
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
    # observation_space = 12 #without lidar
    # observation_space = 17 #with 5 beams lidar
    # observation_space = 12
    observation_space = 12 + 60 + 2 + 1 + 2  # vel(3)+ang(3)+grav(3)+goal_dir(3)+dist2d(1)+distz(1) + lidar(60) + nearest_obs_dir_b(2) + potential(1)
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

    flat_terrain = False  # for generator terrain
    # flat_terrain = True  # for flat terrain
    if flat_terrain:
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
        map_range = [20.0, 20.0, 4.5]
        num_obstacles = 100
        terrain = TerrainImporterCfg(
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(map_range[0]*2, map_range[1]*2),
                border_width=5.0,
                num_rows=1,
                num_cols=1,
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=num_obstacles,
                        obstacle_height_mode="fixed",
                        obstacle_width_range=(0.4, 1.1),
                        obstacle_height_range=[3.0, 6.0],
                        platform_width=0.0,
                    ),
                },
            ),
            visual_material=None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    
    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = AGILE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # sensor 
    lidar_sensor = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),
        # attach_yaw_only=False,
        ray_alignment='base',
        # pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(10.0, 20.0), horizontal_fov_range=(-50.0, 50.0),horizontal_res=1.67),     #For limited fov
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(10.0, 20.0), horizontal_fov_range=(-179.0, 179.0),horizontal_res=6.0),      #For full fov 
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    thrust_to_weight = 5.0
    moment_scale = 0.7

    # reward scales
    # lin_vel_reward_scale = -0.5
    ang_vel_reward_scale = -0.005         # was -0.02 — created -9.64 episode penalty, too dominant
    distance_to_goal_reward_scale = 100.0  # raised from 80 — goal must dominate over safety penalties
    action_rate_reward_scale = -0.01
    velocity_direction = 20.0
    head_tracking = 30.0
    reward_safety_static = 25.0
    proximity_penalty_scale = 60.0
    collision_penalty_scale = 15.0
    goal_reach_bonus = 20.0
    vel_repulsion_scale = 0.0             # disabled — taught drone to face AWAY from obstacles (PA alignment -0.252)
                                          # proximity_penalty + collision_terminal handle avoidance without destroying PA
    potential_field_PA = 40.0            # raised from 30 — needs to compete with head_tracking_PA=25 for obstacle facing
    head_tracking_PA = 20.0              # slightly reduced from 25 — was dominating over PA facing reward

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
        self.height_range = torch.zeros(self.num_envs, 2 , device=self.device)


                #lidar
        self.lidar_resolution = (60)
        self.lidar_range = 5.0


        self.my_visualizer = self.define_markers()
        self.robot_visualizer = self.define_robot_markers()
        self.nearest_obs_visualizer = self.define_nearest_obs_markers()

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                # "rew_lin_vel",
                "rew_ang_vel",
                "rew_distance_to_goal",
                "rew_action_rate",
                "rew_velocity_dir",
                # "rew_head_tracking",
                "rew_height_penalty",
                "rew_reward_safety_static",
                "rew_proximity_penalty",
                "rew_collision_terminal",
                "rew_goal_reach",
                "rew_potential_field_PA",
                "rew_heading_tracking_PA",
                "rew_vel_repulsion",
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

        self._lidar_sensor = RayCaster(self.cfg.lidar_sensor)
        self.scene.sensors["lidar_sensor"] = self._lidar_sensor

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

        thrust, moment = self.velocity_controller.update_velocity_only_edit(
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

        desired_dist = desired_pos_b.norm(dim=-1, keepdim=True)
        unit_desird_pos_b = desired_pos_b / (desired_dist + 1e-6)

        desired_dist_2d = desired_pos_b[:, :2].norm(dim=-1, keepdim=True)
        desired_dist_z = desired_pos_b[:, 2].unsqueeze(1)

        # lidar distances — clamp distance (not vector components)
        raw_vecs = self._lidar_sensor.data.ray_hits_w - self._lidar_sensor.data.pos_w.unsqueeze(1)
        raw_dists = torch.nan_to_num(raw_vecs.norm(dim=-1), nan=self.lidar_range, posinf=self.lidar_range)
        self.lidar_scan = raw_dists.clamp_max(self.lidar_range).reshape(self.num_envs, 1, self.lidar_resolution)
        lidar_scan_norm = self.lidar_scan.squeeze(1) / self.lidar_range  # normalised to [0, 1]

        # nearest obstacle — correct: find min distance, then get direction
        env_idx = torch.arange(self.num_envs, device=self.device)
        closest_idx = torch.argmin(raw_dists, dim=1)
        nearest_dist = raw_dists[env_idx, closest_idx].clamp_max(self.lidar_range)
        nearest_vec_w = raw_vecs[env_idx, closest_idx]  # world-frame vector to nearest obstacle
        nearest_vec_w_safe = torch.nan_to_num(nearest_vec_w, nan=0.0, posinf=self.lidar_range, neginf=-self.lidar_range)
        nearest_vec_w_norm = nearest_vec_w_safe / (nearest_vec_w_safe.norm(dim=-1, keepdim=True).clamp(min=0.1))
        # transform to body frame so policy knows WHERE to steer away from
        nearest_vec_b = quat_apply_inverse(self._robot.data.root_quat_w, nearest_vec_w_norm)
        nearest_vec_b = torch.nan_to_num(nearest_vec_b, nan=0.0)

        sigma = 1.5  # wider activation: PA behaviour starts at ~2-3 m, not ~0.7 m
        gaussian_factor = 1 / (0.1 * torch.sqrt(2 * torch.tensor(torch.pi)))
        potential = 0.25 * gaussian_factor * torch.exp(-nearest_dist**2 / (2 * sigma**2))

        # obs size: 3+3+3+3+1+1+60+2+1 = 77
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,    # 3
                self._robot.data.root_ang_vel_b,    # 3
                self._robot.data.projected_gravity_b,  # 3
                unit_desird_pos_b,                  # 3
                desired_dist_2d,                    # 1
                desired_dist_z,                     # 1
                lidar_scan_norm,                    # 60  (normalised 0–1)
                nearest_vec_b[:, :2],               # 2   (body-frame XY direction to nearest obstacle)
                potential.unsqueeze(-1),            # 1
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
        # exp(-dist/4): gives gradient at 8m(0.14), 6m(0.22), 4m(0.37), 2m(0.61), 0m(1.0)
        # tanh(dist/2.5) was flat at 6-8m (0.016) — policy had no signal to push past 6m
        distance_to_goal_mapped = torch.exp(-distance_to_goal / 4.0)
        
        relative_err_pos_w = self._desired_pos_w - self._robot.data.root_pos_w
        unit_relative_err_pos = relative_err_pos_w / (relative_err_pos_w.norm(dim=-1, keepdim=True) + 1e-6)
        rew_vel_dir_w = self._robot.data.root_lin_vel_w * unit_relative_err_pos
        rew_vel_dir_w = torch.sum(rew_vel_dir_w, dim=-1)

        # heading tracking reward
        self.ref_heading = torch.atan2(relative_err_pos_w[:, 1], relative_err_pos_w[:, 0])  # radian
        self.robot_heading = self._robot.data.heading_w
        self.angle_diff = self.ref_heading - self.robot_heading
        head_tracking_path_rew = 0.7 - torch.tanh(torch.abs(self.angle_diff) / 0.9)
        

        # debug visualization
        ref_heading_marker_orientations = quat_from_angle_axis(self.ref_heading, torch.tensor([0.0, 0.0, 1.0], device=self.device))
        self.robot_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=ref_heading_marker_orientations)

        robot_heading_marker_orientations = quat_from_angle_axis(self.robot_heading, torch.tensor([0.0, 0.0, 1.0], device=self.device))
        self.my_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=robot_heading_marker_orientations)
        # self.robot_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=self._robot.data.root_quat_w)
        
        # height penalty
        clipped_z = torch.clamp(
                    self._robot.data.root_pos_w[:, 2],
                    self.height_range[:, 0] ,  # allow small tolerance
                    self.height_range[:, 1]
                )
        penalty_height = ((self._robot.data.root_pos_w[:, 2] - clipped_z) ** 2)  # shape (num_envs, 1)

        # lidar safety reward — mean clearance (global open-space reward)
        reward_safety_static = 1 - torch.tanh((self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=2).squeeze(1)

        # quadratic proximity penalty — zero at 1.5m, reaches 1.0 at contact
        # quadratic form means the gradient is steep near collision, giving strong avoidance signal
        min_beam = self.lidar_scan.min(dim=2).values.squeeze(1).clamp(min=0.0, max=self.lidar_range)
        safe_dist = 1.5
        proximity_penalty = torch.clamp(1.0 - min_beam / safe_dist, min=0.0) ** 2
        # hard collision terminal penalty — flat large penalty when a beam hits collision threshold
        collision_terminal = (min_beam < 0.35).float()  # slightly wider than termination threshold (0.3)

        # nearest obstacle — correct: compute distances first, then get direction vector
        raw_vecs_rew = self._lidar_sensor.data.ray_hits_w - self._lidar_sensor.data.pos_w.unsqueeze(1)
        dists = torch.nan_to_num(raw_vecs_rew.norm(dim=-1), nan=self.lidar_range, posinf=self.lidar_range)
        closest_idx = torch.argmin(dists, dim=1)
        env_idx = torch.arange(self.num_envs, device=self.device)

        nearest_vec = raw_vecs_rew[env_idx, closest_idx]
        nearest_dist = dists[env_idx, closest_idx].clamp_max(self.lidar_range)
        
        robot_heading_vector = quat_apply_yaw(
            self._robot.data.root_state_w[:, 3:7].to(torch.float32),
            torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1),
        )
        #check if robot still in potential field range, if not set nearest vec to goal direction
        # in_field_range = nearest_dist < (self.lidar_range * 2.0)
        # nearest_vec = torch.where(
        #     in_field_range.unsqueeze(-1),
        #     nearest_vec,
        #     -unit_relative_err_pos * 5.0,  # scale to match typical obstacle distance
        # )
        nearest_angle = torch.atan2(nearest_vec[:, 1], nearest_vec[:, 0])
        nearest_quat = quat_from_angle_axis(nearest_angle, torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1))

        self.nearest_obs_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=nearest_quat)

        nearest_vec_norm = nearest_vec / (nearest_vec.norm(dim=-1, keepdim=True) + 1e-6)
        robot_vec_heading_norm = robot_heading_vector / (robot_heading_vector.norm(dim=-1, keepdim=True) + 1e-6)
        # cosine_similarity = torch.dot(nearest_vec_norm, robot_vec_heading_norm)/torch.norm(nearest_vec_norm)*torch.norm(robot_vec_heading_norm)
        cosine_similarity = torch.nn.functional.cosine_similarity(
            nearest_vec_norm,
            robot_vec_heading_norm,
            dim=1,
            eps=1e-8,
        )

        cos_facing_obstacle = torch.sum(robot_heading_vector * nearest_vec_norm, dim=1)  # in [-1, 1]
        # dot_vec = torch.abs(torch.sum(robot_heading_vector * nearest_vec, dim=1))
        # sigma = 0.7  # Standard deviation of Gaussian function
        # gaussian_factor = 1 / (0.1 * torch.sqrt(2 * torch.tensor(torch.pi)))  # Precomputed constant
        # potential = 0.5 * gaussian_factor * torch.exp(-nearest_dist**2 / (2 * sigma**2))
        sigma = 1.5  # wider activation: PA behaviour starts at ~2-3 m, not ~0.7 m
        gaussian_factor = 1 / (0.1 * torch.sqrt(2 * torch.tensor(torch.pi)))
        potential = 0.25 * gaussian_factor * torch.exp(-nearest_dist**2 / (2 * sigma**2))
        
        # Raised-cosine: maps cosine_similarity ∈ [-1, 1] → [0, 1].
        # Gradient = -sin(θ)/2 throughout, so the policy always knows LEFT vs RIGHT:
        #   obstacle to RIGHT (cos=0 currently) → turning right increases cos → reward increases ✓
        #   obstacle to LEFT  (cos=0 currently) → turning left  increases cos → reward increases ✓
        # torch.abs() was the bug — it made left and right symmetric → policy converged on one side.
        # Minimum is 0 (not -1) so rear obstacles don't create a hard "spin around" penalty.
        rew_potential_pa = potential * (1.0 + cosine_similarity) / 2.0

        # Keep goal-heading reward PURE — no obstacle blending.
        # The two rewards compete independently:
        #   far from obstacles:  head_tracking_PA × head_tracking_path_rew dominates → faces goal
        #   near obstacle:       potential_field_PA × raised_cosine adds a directional pull toward obstacle
        # Blending them with abs(cos) caused left/right ambiguity. Separate rewards have no ambiguity.
        rew_heading_reward_blended = head_tracking_path_rew

        # velocity repulsion — penalise flying toward the nearest obstacle when close,
        # but NOT when the obstacle is in the same direction as the goal (drone must pass through)
        nearest_vec_safe = torch.nan_to_num(nearest_vec, nan=0.0, posinf=self.lidar_range, neginf=-self.lidar_range)
        nearest_vec_norm = nearest_vec_safe / (nearest_vec_safe.norm(dim=-1, keepdim=True).clamp(min=0.1))
        vel_w_safe = self._robot.data.root_lin_vel_w.clamp(-50.0, 50.0)
        vel_toward_obs = torch.sum(vel_w_safe * nearest_vec_norm, dim=1)
        # suppress penalty when obstacle is roughly in goal direction (alignment > 0.7 → obstacle is blocking goal)
        obs_goal_alignment = torch.sum(nearest_vec_norm * unit_relative_err_pos, dim=1)
        repulsion_mask = (obs_goal_alignment < 0.7).float()
        rew_vel_repulsion = -potential * vel_toward_obs.clamp(min=0) * repulsion_mask

        # sparse goal-reach bonus — flat reward (no step_dt) so it registers even on the final step
        goal_reached = (distance_to_goal < 1.5).float()

        rewards = {
            # "rew_lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "rew_ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "rew_distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "rew_action_rate": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "rew_velocity_dir": rew_vel_dir_w * self.cfg.velocity_direction * self.step_dt,
            "rew_height_penalty": -penalty_height * 10.0 * self.step_dt,
            "rew_reward_safety_static": reward_safety_static * self.cfg.reward_safety_static * self.step_dt,
            "rew_proximity_penalty": -proximity_penalty * self.cfg.proximity_penalty_scale * self.step_dt,
            "rew_collision_terminal": -collision_terminal * self.cfg.collision_penalty_scale,
            "rew_goal_reach": goal_reached * self.cfg.goal_reach_bonus,  # flat sparse bonus — no step_dt
            "rew_potential_field_PA": rew_potential_pa * self.cfg.potential_field_PA * self.step_dt,
            "rew_heading_tracking_PA": rew_heading_reward_blended * self.cfg.head_tracking_PA * self.step_dt,
            "rew_vel_repulsion": rew_vel_repulsion * self.cfg.vel_repulsion_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        self.previous_action = self._actions.clone()

        # Logging — clamp each term so NaN never reaches TensorBoard
        for key, value in rewards.items():
            self._episode_sums[key] += torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.3, self._robot.data.root_pos_w[:, 2] > 4.0)
        
        uprightness = self._robot.data.projected_gravity_b[:, 2] >= 0.0

        static_collision = einops.reduce(self.lidar_scan, "n 1 w -> n 1", "min") < 0.3 # 0.4  # 0.3 collision radius
        # reach_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1) < 0.15

        relative_err_pos_w = self._desired_pos_w - self._robot.data.root_pos_w
        ref_heading = torch.atan2(relative_err_pos_w[:, 1], relative_err_pos_w[:, 0])  # radian
        # wrap to [-π, π] — without this, angle_diff ≈ 2π for goals near ±180° gives false termination
        angle_diff = ref_heading - self._robot.data.heading_w
        angle_diff = (angle_diff + torch.pi) % (2 * torch.pi) - torch.pi
        opposite_direction_heading = torch.abs(angle_diff) > 2.6180  # 150 degrees

        # died = died | uprightness | static_collision.squeeze(1) | reach_goal | opposite_direction_heading
        evaluate_mode = True
        if evaluate_mode:
            died = died | static_collision.squeeze(1) | opposite_direction_heading
        else:
            died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 8.0)
            died = died
        # died = died | static_collision.squeeze(1) | opposite_direction_heading

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
        self.previous_action[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-8.0, 8.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(1.5, 1.6)

        desired_heights = self._desired_pos_w[:, 2]
        margin = 0.20  # was 0.05 — 10cm band caused persistent height penalty; 40cm gives realistic flying room
        self.height_range = torch.stack([
            desired_heights - margin,
            desired_heights + margin
        ], dim=-1)

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
        
    def define_markers(self) -> VisualizationMarkers:
        """Define markers with various different shapes."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                # "frame": sim_utils.UsdFileCfg(
                #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                #     scale=(0.5, 0.5, 0.5),
                # ),
                "arrow_x": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.1, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)
    
    def define_robot_markers(self) -> VisualizationMarkers:
        """Define markers with various different shapes."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                # "frame": sim_utils.UsdFileCfg(
                #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                #     scale=(0.5, 0.5, 0.5),
                # ),
                "arrow_x": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.1, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)
    
    def define_nearest_obs_markers(self) -> VisualizationMarkers:
        """Define markers with various different shapes."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                # "frame": sim_utils.UsdFileCfg(
                #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                #     scale=(0.5, 0.5, 0.5),
                # ),
                "arrow_x": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.1, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)