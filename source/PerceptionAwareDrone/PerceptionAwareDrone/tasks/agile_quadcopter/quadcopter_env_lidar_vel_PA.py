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
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_apply_yaw, quat_from_angle_axis, euler_xyz_from_quat

##z
# Pre-defined configs
##
from .robot.agileDrone import AGILE_CFG    # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

#terrain
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG#, OBSTACLE_RAND_POS

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

from .utility.noisemodel import NoiseModel
from .utility.lee_velocity_controller.vel_controller import GeometricVelocityController
from .utility.utilitymath import sampleUniformQuatwithTilt, sampleCenterQuatwithTilt

from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_euler,
    matrix_from_quat,
    normalize,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
)

from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg


PUSH_LIN_VEL = 0.3  # m/s
PUSH_ANG_VEL = 0.3  # rad/s
@configclass
class EventCfg:

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        # params={
        #     "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        #     "mass_distribution_params": (-0.2, 0.2),
        #     "operation": "add",
        # },
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
    
    # push_robot = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.2),
    #     params={
    #         "force_range": (-0.5, 0.5),
    #         "torque_range": (-0.05, 0.05),
    #     },
    # )
    
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
    evaluate_mode = False
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    # observation_space = 12 #without lidar
    # observation_space = 17 #with 5 beams lidar
    # observation_space = 12
    observation_space = 9 + 60 + 1 + 2 + 4  # with 60 beams lidar + potential field + last action 
    state_space = 0
    debug_vis = True

    # # # # Noise Configuration
    add_noise = True
    noiseCfg = {
        "root_pos": {
            "type": "uniform",
            "dim": 3,
            "mean": 0.005,
            "std": 0.005,
            "clip": 0.3,
        },
        "root_quat": {
            "type": "uniform",
            "dim": 3,
            "mean": torch.pi * (0.5 / 180.0),
            "std": torch.pi * (1.0 / 180.0),
            "clip": 0.3,
        },
        "lin_vel": {
            "type": "uniform",
            "dim": 3,
            "mean": 0.005,
            "std": 0.005,
            "clip": 3.0,
        },
        "ang_vel": {
            "type": "uniform",
            "dim": 3,
            "mean": 0.005,
            "std": 0.005,
            "clip": 3.0,
        },
        "gravity": {
            "type": "uniform",
            "dim": 3,
            "mean": 0.0,
            "std": 0.05,
            "clip": 0.1,
        },

        "lidar_pose": {
            "type": "uniform",
            "dim": 3,
            "mean": 0.005,
            "std": 0.005,
            "clip": 0.05,
        },

    }


    ui_window_class_type = QuadcopterEnvWindow
    
    # viewer = ViewerCfg(eye=(-19.8, -23.8, 11.5), lookat=(-24.0, -8.5, -1.7), origin_type='env', env_index=2015)
    viewer = ViewerCfg(eye=(-19.8, -23.8, 11.5), lookat=(-24.0, -8.5, -1.7), origin_type='env')

    # simulation
    sim: SimulationCfg = SimulationCfg(
        # dt=1/150,
        dt=1/200,
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
    # flat_terrain = True
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
        # terrain = TerrainImporterCfg(
        #     prim_path="/World/ground",
        #     terrain_type="generator",
        #     terrain_generator=ROUGH_TERRAINS_CFG,
        #     max_init_terrain_level=9,
        #     collision_group=-1,
        #     physics_material=sim_utils.RigidBodyMaterialCfg(
        #         friction_combine_mode="multiply",
        #         restitution_combine_mode="multiply",
        #         static_friction=1.0,
        #         dynamic_friction=1.0,
        #     ),
        #     # visual_material=sim_utils.MdlFileCfg(
        #     #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
        #     #     project_uvw=True,
        #     # ),
        #     debug_vis=True,
        # )
        map_range = [20.0, 20.0, 4.5]
        num_obstacles = 150
        terrain = TerrainImporterCfg(
            # num_envs=self.num_envs,
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
                        # obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
                        platform_width=0.0,
                    ),
                },
            ),
            visual_material = None,
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

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",  # Bind to the robot root link
        history_length=1,
        update_period=0,  # Update every physics step
        track_air_time=True,
        # track_contact_points=True,
        debug_vis=True,
        # filter_prim_paths_expr=["/World/ground"],  # Only track contacts with the ground
        # filter_prim_paths_expr=[terrain.prim_path],  # Only track contacts with the ground
    )

    thrust_to_weight = 5.0
    moment_scale = 0.7
    contact_force_threshold = 0.1

    # reward scales
    lin_vel_reward_scale = -0.0002
    ang_vel_reward_scale = -0.001

    distance_to_goal_reward_scale = 50.0
    action_rate_reward_scale = -0.01 #-0.05
    velocity_direction = 25.0
    head_tracking = 30.0
    # reward_safety_static = 32.0
    potential_field_PA = 25.0 #32.0
    head_tracking_PA = 32.0

    #max velocity
    max_velocity = 3.5 #2.5  # 2.0  # m/s
    max_yaw_rate = 6.28 # 6.28  # rad/s



    #NEW Reward parameters
    # rew_angular_to_goal_scale = 0.5
    died_reward_scale = 0.5
    reach_goal_reward_timeout_scale = 0.01
    reach_goal_reward_scale = 0.5

    velocity_direction_reward_scale = 6.0  #2.00
    distance_to_goal_reward_scale = 9.0
    # angular_to_goal_reward_scale = 5.0
    head_tracking_reward_scale = 2.0
    height_penalty_scale = -0.5
    reward_safety_static_scale = 6.0  # 4.0
    thrust_power_penalty_scale = -0.5

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        # self.previous_action = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.previous_action = torch.zeros(self.num_envs, 3, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._lin_vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._yaw_vel_cmd = torch.zeros(self.num_envs, 1, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.height_range = torch.zeros(self.num_envs, 2 , device=self.device)


        #lidar
        self.lidar_resolution = (60)
        self.lidar_range = 5.0

        # if headless mode is off, setup debug visualization
        # if not self.headless:
        self.my_visualizer = self.define_markers()
        self.robot_visualizer = self.define_robot_markers()
        self.nearest_obs_visualizer = self.define_nearest_obs_markers()

        # noise model
        self.noiseModel = NoiseModel(cfg.noiseCfg, device=self.device, num_envs=self.num_envs)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                # "rew_lin_vel",
                # "rew_ang_vel",
                # "rew_thrust_power",
                # "rew_distance_to_goal",
                # "rew_action_rate",
                # "rew_velocity_dir",
                # "rew_head_tracking",
                # "rew_height_penalty",
                # "rew_reward_safety_static",
                # "rew_potential_field_PA",
                # "rew_heading_tracking_PA",
                # "died",
                # "rew_reach_goal",
                # "rew_reach_goal_timeout",
                # "rew_angular_to_goal",
                # "rew_head_tracking_path",
                # "flip_penalty", 
                # "rew_heading_stability",
                # "rew_stop",
                "distance",
                "progress",
                "velocity_dir",
                "heading",
                "collision",
                "lin_vel",
                "ang_vel",
                "flip",
                "reach_goal",
            ]
        }
       # Get specific body indices
        self._body_id = self._robot.find_bodies("base_link")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        all_inertia_tensor = self._robot.root_physx_view.get_inertias()[0] # shape (num_envs, 3, 3)
        robot_inertia = torch.sum(all_inertia_tensor,dim=0)
        print("Inertia tensor of the robot1:", robot_inertia)    

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        
        # Ixx = 7.6191e-04
        Ixx = robot_inertia[0]
        # Iyy = 8.9651e-04
        Iyy = robot_inertia[4]
        # Izz = 1.2983e-03
        Izz = robot_inertia[8]
        
        input_robot_inertia = torch.diag(torch.tensor([ robot_inertia[0], robot_inertia[4],  robot_inertia[8]], device=self.device))
        print("Inertia tensor of the robot2:", input_robot_inertia)

        self.velocity_controller = GeometricVelocityController(
            num_env=self.num_envs,
            mass=self._robot_mass,
            inertia=input_robot_inertia,
            device=self.device,
        )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._lidar_sensor = RayCaster(self.cfg.lidar_sensor)
        self.scene.sensors["lidar_sensor"] = self._lidar_sensor

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

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

        self._last_actions = self._actions.clone().clamp(-1.0, 1.0)
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # =====================================================
        # 1. Distance to goal
        # =====================================================
        relative_goal = self._desired_pos_w - self._robot.data.root_pos_w

        distance_to_goal = torch.linalg.norm(
            relative_goal,
            dim=1,
            keepdim=True,
        )

        # =====================================================
        # 2. Smooth slowdown near goal
        # =====================================================
        SLOWDOWN_DIST = 2.0
        MIN_SPEED_SCALE = 0.05

        # linear slowdown
        speed_scale = torch.clamp(
            distance_to_goal / SLOWDOWN_DIST,
            min=MIN_SPEED_SCALE,
            max=1.0,
        )

        # smoother profile
        speed_scale = torch.sqrt(speed_scale)

        # =====================================================
        # 3. Convert actions -> commands
        # =====================================================

        # yaw command also slows down near goal
        self._yaw_vel_cmd[:, 0] = (
            self._actions[:, 0]
            * self.cfg.max_yaw_rate
            * speed_scale.squeeze(-1)
        )

        # translational velocity slows near goal
        self._lin_vel_cmd[:, :] = (
            self._actions[:, 1:]
            * self.cfg.max_velocity
            * speed_scale
        )

        # =====================================================
        # 4. Extra stabilization zone
        # =====================================================
        HOLD_DIST = 0.05 #15

        near_goal = (distance_to_goal < HOLD_DIST).float()

        # very close -> almost hover
        self._lin_vel_cmd *= (1.0 - 0.9 * near_goal)
        self._yaw_vel_cmd[:, 0] *= (1.0 - 0.9 * near_goal.squeeze(-1))

        # =====================================================
        # 5. Controller
        # =====================================================
        thrust, moment = self.velocity_controller.update_velocity_only_edit(
            quat_w=self._robot.data.root_link_quat_w,
            vel_w=self._robot.data.root_lin_vel_w,
            omega_b=self._robot.data.root_ang_vel_b,
            desired_vel_w=self._lin_vel_cmd,
            desired_yaw_rate=self._yaw_vel_cmd[:, 0],
        )

        self._thrust[:, :] = 0.0
        self._moment[:, :] = 0.0

        self._thrust[:, 0, 2] = thrust
        self._moment[:, 0, :] = moment

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
        

    def _get_observations(self) -> dict:
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        root_lin_vel_w = self._robot.data.root_lin_vel_w
        root_ang_vel_b = self._robot.data.root_ang_vel_b
        projected_gravity_w = self._robot.data.GRAVITY_VEC_W
        lidar_sensor_pos_w = self._lidar_sensor.data.pos_w

        
        if self.cfg.add_noise:
            if "lin_vel" in self.noiseModel.params:
                root_lin_vel_w = self.noiseModel.apply(root_lin_vel_w, "lin_vel")
            if "ang_vel" in self.noiseModel.params:
                root_ang_vel_b = self.noiseModel.apply(root_ang_vel_b, "ang_vel")
            if "gravity" in self.noiseModel.params:
                noise = torch.empty(self.num_envs, 3, device=self.device).uniform_(
                    -self.cfg.noiseCfg["gravity"]["std"], self.cfg.noiseCfg["gravity"]["std"]
                )
                projected_gravity_w += noise
                projected_gravity_w = normalize(projected_gravity_w)
            if "root_pos" in self.noiseModel.params:
                root_pos_w = self.noiseModel.apply(root_pos_w, "root_pos")
            if "root_quat" in self.noiseModel.params:
                axis = torch.rand(self.num_envs, 3, device=self.device)
                axis = axis / axis.norm(dim=-1, keepdim=True)
                angle = torch.empty(self.num_envs, 1, device=self.device).uniform_(
                    -self.cfg.noiseCfg["root_quat"]["std"], self.cfg.noiseCfg["root_quat"]["std"]
                )
                box_quat = quat_from_angle_axis(angle.squeeze(1), axis)
                root_quat_w = quat_mul(box_quat, root_quat_w)
            
            if "lidar_pose" in self.noiseModel.params:
                lidar_sensor_pos_w = self.noiseModel.apply(lidar_sensor_pos_w, "lidar_pose")
                
        
        root_lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        projected_gravity_b = quat_apply_inverse(root_quat_w, projected_gravity_w)
        
        desired_pos_b, _ = subtract_frame_transforms(
            root_pos_w, root_quat_w, self._desired_pos_w, self._desired_quat_w
        )

        desired_dist = desired_pos_b.norm(dim=-1, keepdim=True)
        unit_desird_pos_b = desired_pos_b / (desired_dist + 1e-6)

        desired_dist_2d = desired_pos_b[:, :2].norm(dim=-1, keepdim=True)
        desired_dist_z = desired_pos_b[:, 2].unsqueeze(1)

        self.lidar_scan = (
            (
                self._lidar_sensor.data.ray_hits_w
                - lidar_sensor_pos_w.unsqueeze(1)
            )
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, self.lidar_resolution)
        )

        # lidar potential field
        vec_to_obstacles = (self._lidar_sensor.data.ray_hits_w - lidar_sensor_pos_w.unsqueeze(1)) .clamp_max(self.lidar_range)
        dists_to_obstacle = vec_to_obstacles.norm(dim=-1)
        closest_idx = torch.argmin(dists_to_obstacle, dim=1)
        env_idx = torch.arange(vec_to_obstacles.shape[0])
        nearest_dist = dists_to_obstacle[env_idx, closest_idx]
        
        sigma = 3.0 #0.9  # Standard deviation of Gaussian function
        gaussian_factor = 1 / (0.1 * torch.sqrt(2 * torch.tensor(torch.pi)))  # Precomputed constant
        potential = 0.25 * gaussian_factor * torch.exp(-nearest_dist**2 / (2 * sigma**2))


        obs = torch.cat(
            [
                # self._robot.data.root_lin_vel_b,
                root_lin_vel_b,
                # self._robot.data.root_ang_vel_b,
                root_ang_vel_b,
                # self._robot.data.projected_gravity_b,
                # projected_gravity_b,
                # desired_pos_b,
                unit_desird_pos_b,  # 3
                desired_dist_2d,  # 1
                desired_dist_z,  # 1
                self.lidar_scan.squeeze(1),
                potential.unsqueeze(-1),
                self._last_actions,
            ],
            dim=-1,
        )
        # observations = {"policy": obs}

        states = self._get_states()
        # states = torch.clamp(states, -clip_obs, clip_obs)
        observations = {"policy": obs, "critic": states}

        return observations
    
    def _get_states(self):
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )

        desired_dist = desired_pos_b.norm(dim=-1, keepdim=True)
        unit_desird_pos_b = desired_pos_b / (desired_dist + 1e-6)

        desired_dist_2d = desired_pos_b[:, :2].norm(dim=-1, keepdim=True)
        desired_dist_z = desired_pos_b[:, 2].unsqueeze(1)

        lidar_scan = (
            (
                self._lidar_sensor.data.ray_hits_w
                - self._lidar_sensor.data.pos_w.unsqueeze(1)
            )
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, self.lidar_resolution)
        )

        # lidar potential field
        vec_to_obstacles = (self._lidar_sensor.data.ray_hits_w - self._lidar_sensor.data.pos_w.unsqueeze(1)).clamp_max(self.lidar_range)
        dists_to_obstacle = vec_to_obstacles.norm(dim=-1)
        closest_idx = torch.argmin(dists_to_obstacle, dim=1)
        env_idx = torch.arange(vec_to_obstacles.shape[0])
        nearest_dist = dists_to_obstacle[env_idx, closest_idx]
        
        sigma = 3.0 #0.9  # Standard deviation of Gaussian function
        gaussian_factor = 1 / (0.1 * torch.sqrt(2 * torch.tensor(torch.pi)))  # Precomputed constant
        potential = 0.25 * gaussian_factor * torch.exp(-nearest_dist**2 / (2 * sigma**2))

        # print("projected gravity", self._robot.data.projected_gravity_b[0])
        # print("norm proj", self._robot.data.projected_gravity_b[0].norm())
        # if self._robot.data.projected_gravity_b[0][2] < 0.2:
        #     print("Robot is close to upside down!")

        # proj_gravity = self._robot.data.projected_gravity_b
        # yaw = torch.atan2(
        #     proj_gravity[:, 1],
        #     proj_gravity[:, 0]
        # )

        # quat = quat_from_angle_axis(
        #     yaw,
        #     torch.tensor(
        #         [0.0, 0.0, 1.0],
        #         device=self.device,
        #         dtype=torch.float32,
        #     )
        # )

        # self.my_visualizer.visualize(
        #     translations=self._robot.data.root_pos_w,
        #     orientations=quat,
        # )

        states = torch.cat(
            (
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                # self._robot.data.projected_gravity_b,
                desired_pos_b,
                unit_desird_pos_b,
                desired_dist_2d,
                desired_dist_z,
                lidar_scan.squeeze(1),
                potential.unsqueeze(-1),
                self._last_actions,
            ),
            dim=-1,
        )
        return states
#####################################################################################
    def _get_rewards(self) -> torch.Tensor:
        pose_err, rot_err = compute_pose_error(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
            self._desired_quat_w,
        )
        self._position_error = pose_err
        self._angle_error = rot_err

        distance_to_goal = torch.linalg.norm(self._position_error, dim=1,)


        # ==========================================================
        # Goal reward
        # ==========================================================
        rew_distance_to_goal = 1 - torch.tanh(distance_to_goal / 0.8)

        # ==========================================================
        # Progress reward
        # ==========================================================
        if not hasattr(self, "_prev_distance_to_goal"):
            self._prev_distance_to_goal = distance_to_goal.clone()

        rew_progress = (self._prev_distance_to_goal - distance_to_goal)

        self._prev_distance_to_goal = distance_to_goal.clone()


        # ==========================================================
        # Velocity
        # ==========================================================
        lin_vel_norm = torch.linalg.norm(
            self._robot.data.root_lin_vel_b,
            dim=1,
        )

        ang_vel_norm = torch.linalg.norm(
            self._robot.data.root_ang_vel_b,
            dim=1,
        )

        rew_lin_vel_penalty = lin_vel_norm ** 2
        rew_ang_vel_penalty = ang_vel_norm ** 2

        # ==========================================================
        # Goal direction reward
        # ==========================================================
        relative_err_pos_w = self._desired_pos_w - self._robot.data.root_pos_w
        unit_relative_err_pos = (relative_err_pos_w / (relative_err_pos_w.norm(dim=-1, keepdim=True,) + 1e-6))

        rew_vel_dir_w = (self._robot.data.root_lin_vel_w * unit_relative_err_pos).sum(dim=-1)
        rew_vel_dir_w = torch.clamp(rew_vel_dir_w, min=0.0,)


        # ==========================================================
        # LIDAR FIELD
        # ==========================================================
        # reward_safety_static = 1 - torch.tanh((self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=2).squeeze(1)
        vec_to_obstacles = (self._lidar_sensor.data.ray_hits_w - self._lidar_sensor.data.pos_w.unsqueeze(1)).clamp_max(self.lidar_range)
        dists = vec_to_obstacles.norm(dim=-1)

        nearest_dist = dists.min(dim=1)[0]
        # ==========================================================
        # Collision penalty
        # ==========================================================
        # collision_margin = 0.25

        # collision_penalty = torch.relu(
        #     collision_margin - nearest_dist
        # )

        collision_penalty = 1 - torch.tanh((nearest_dist))
        
        # ==========================================================
        # Flip penalty
        # ==========================================================
        # upside_down = (
        #     self._robot.data.projected_gravity_b[:, 2]
        #     < 0.0
        # )

        # flip_penalty = upside_down.float()

        # ==========================================================
        # Reach goal
        # ==========================================================
        reach_goal = (
            distance_to_goal < 0.10
        )

        reach_goal_reward = (
            reach_goal.float()
            * 10.0
        )

        # ==========================================================
        # Potential field
        # ==========================================================
        potential = torch.exp(
            -nearest_dist / 0.7
        )

        # ==========================================================
        # Weighted obstacle direction
        # ==========================================================
        weights = torch.exp(-dists / 0.7)
        # print("weights:", weights)
        # print("vec_to_obstacles:", vec_to_obstacles)
        obstacle_vector = (vec_to_obstacles * weights.unsqueeze(-1)).sum(dim=1)
        # print("obstacle_vector before normalization:", obstacle_vector)
        obstacle_vector = (obstacle_vector / (obstacle_vector.norm(dim=-1, keepdim=True,) + 1e-6))
        # print("obstacle_vector after normalization:", obstacle_vector)  

        # ==========================================================
        # Robot heading vector
        # ==========================================================
        robot_heading_vector = quat_apply_yaw( self._robot.data.root_state_w[:, 3:7].float(), torch.tensor([1.0, 0.0, 0.0], device=self.device,).repeat(self.num_envs, 1))
        robot_heading_vector = (robot_heading_vector / (robot_heading_vector.norm(dim=-1, keepdim=True,) + 1e-6))

        # ==========================================================
        # Goal heading reward
        # ==========================================================
        goal_heading_cos = torch.nn.functional.cosine_similarity(robot_heading_vector, unit_relative_err_pos, dim=1,)
        goal_heading_reward = (goal_heading_cos + 1.0) * 0.5

        # ==========================================================
        # Obstacle heading reward
        # ==========================================================
        obstacle_heading_cos = (torch.nn.functional.cosine_similarity(robot_heading_vector, obstacle_vector, dim=1,))
        obstacle_heading_reward = (obstacle_heading_cos + 1.0) * 0.5

        # ==========================================================
        # Adaptive heading reward
        # ==========================================================
        rew_heading = ((1.0 - potential) * goal_heading_reward + potential* obstacle_heading_reward)
        
        # ==========================================================
        # Visualization
        # ==========================================================
        obstacle_angle = torch.atan2(
            obstacle_vector[:, 1],
            obstacle_vector[:, 0],
        )

        obstacle_quat = quat_from_angle_axis(
            obstacle_angle,
            torch.tensor(
                [0.0, 0.0, 1.0],
                device=self.device,
            ).repeat(self.num_envs, 1),
        )

        self.nearest_obs_visualizer.visualize(
            translations=self._robot.data.root_pos_w,
            orientations=obstacle_quat,
        )


        # ==========================================================
        # Rewards
        # ==========================================================
        rewards = {
            "distance":
                rew_distance_to_goal
                * self.cfg.distance_to_goal_reward_scale
                * self.step_dt,

            "velocity_dir":
                rew_vel_dir_w
                * self.cfg.velocity_direction_reward_scale
                * self.step_dt,

            "heading":
                rew_heading
                * self.cfg.head_tracking_reward_scale
                * self.step_dt,

            "collision":
                -collision_penalty
                * 15.0
                * self.step_dt,

            "lin_vel":
                -rew_lin_vel_penalty
                * 0.01
                * self.step_dt,

            "ang_vel":
                -rew_ang_vel_penalty
                * 0.01
                * self.step_dt,

            "reach_goal":
                reach_goal_reward
                * self.cfg.reach_goal_reward_scale
                * self.step_dt,
        }

        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return total_reward
        

#####################################################################################BEST REWARD FUNCTION
    # def _get_rewards(self) -> torch.Tensor:

    #     pose_err, rot_err = compute_pose_error(
    #         self._robot.data.root_pos_w,
    #         self._robot.data.root_quat_w,
    #         self._desired_pos_w,
    #         self._desired_quat_w,
    #     )

    #     self._position_error = pose_err
    #     self._angle_error = rot_err

    #     # --- Distance ---
    #     distance_to_goal = torch.linalg.norm(self._position_error, dim=1)
    #     rew_distance_to_goal = 1 - torch.tanh(distance_to_goal / 0.8)

    #     # --- Angular error ---
    #     angular_to_goal = torch.linalg.norm(self._angle_error, dim=1)
    #     distance_weight = 1 - torch.tanh(distance_to_goal / 0.8)
    #     rew_angular_to_goal = (1 - torch.tanh(angular_to_goal)) * distance_weight

    #     # --- Velocity norms ---
    #     lin_vel_norm = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=1)
    #     ang_vel_norm = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)

    #     lin_vel_clamp = torch.clamp(lin_vel_norm, max=10.0)
    #     ang_vel_clamp = torch.clamp(ang_vel_norm, max=10.0)

    #     # --- Base velocity penalties ---
    #     rew_lin_vel = torch.square(torch.exp(0.6 * lin_vel_clamp) - 1.0)
    #     rew_ang_vel_far = torch.square(torch.exp(0.4 * ang_vel_clamp) - 1.0)

    #     # =========================
    #     # 🔥 KEY FIX: GOAL REGION LOGIC
    #     # =========================
    #     DIST_TH = 0.3
    #     near_goal = (distance_to_goal < DIST_TH).float()
    #     far_goal = 1.0 - near_goal

    #     # --- Strong angular damping near goal ---
    #     rew_ang_vel_near = ang_vel_norm**2 * 5.0
    #     rew_ang_vel = far_goal * rew_ang_vel_far + near_goal * rew_ang_vel_near

    #     # =========================
    #     # Heading tracking (ONLY FAR)
    #     # =========================
    #     relative_err_pos_w = self._desired_pos_w - self._robot.data.root_pos_w
    #     unit_relative_err_pos = relative_err_pos_w / (
    #         relative_err_pos_w.norm(dim=-1, keepdim=True) + 1e-6
    #     )

    #     # Velocity direction reward
    #     rew_vel_dir_w = self._robot.data.root_lin_vel_w * unit_relative_err_pos
    #     rew_vel_dir_w = torch.sum(rew_vel_dir_w, dim=-1)
    #     rew_vel_dir_w = torch.clamp(rew_vel_dir_w, min=0.0)

    #     # Heading tracking (DISABLED near goal)
    #     self.ref_heading = torch.atan2(
    #         relative_err_pos_w[:, 1], relative_err_pos_w[:, 0]
    #     )
    #     self.robot_heading = self._robot.data.heading_w
    #     self.angle_diff = self.ref_heading - self.robot_heading

    #     head_tracking = 0.7 - torch.tanh(torch.abs(self.angle_diff) / 0.9)
    #     head_tracking = head_tracking * far_goal  # 🔥 critical fix

    #     # =========================
    #     # Heading stabilization (ONLY NEAR)
    #     # =========================
    #     heading_stability = torch.exp(-2.0 * ang_vel_norm)
    #     heading_stability = heading_stability * near_goal

    #     # =========================
    #     # Stop-and-hold reward
    #     # =========================
    #     stop_reward = (
    #         torch.exp(-3 * lin_vel_norm)
    #         * torch.exp(-3 * ang_vel_norm)
    #         * torch.exp(-2 * distance_to_goal)
    #     )

    #     # =========================
    #     # Reach goal condition
    #     # =========================
    #     ANG_VEL_TH = 0.2
    #     LIN_VEL_TH = 0.1
    #     POS_TH = 0.1
    #     ANG_TH = 0.1

    #     reach_goal = torch.logical_and(
    #         torch.logical_and(ang_vel_norm < ANG_VEL_TH, lin_vel_norm < LIN_VEL_TH),
    #         # torch.logical_and(angular_to_goal < ANG_TH, distance_to_goal < POS_TH),
    #         distance_to_goal < POS_TH,
    #     )

    #     reach_goal = torch.logical_and(
    #         reach_goal,
    #         self.episode_length_buf > (
    #             self.max_episode_length - (2.0 / (self.cfg.sim.dt * self.cfg.decimation))
    #         ),
    #     )

    #     self._reach_goal = reach_goal.to(torch.float32) * self.reset_time_outs.to(torch.float32)

    #     reach_goal_reward_timeout = (
    #         self.reset_time_outs.to(torch.float32)
    #         * reach_goal.to(torch.float32)
    #         * self.max_episode_length_s
    #     )

    #     reach_goal_reward = torch.zeros_like(reach_goal_reward_timeout)
    #     reach_goal_reward += torch.exp(-2 * distance_to_goal / POS_TH) * reach_goal.float()
    #     reach_goal_reward += torch.exp(-2 * angular_to_goal / ANG_TH) * reach_goal.float()

    #     # =========================
    #     # Height penalty
    #     # =========================
    #     clipped_z = torch.clamp(
    #         self._robot.data.root_pos_w[:, 2],
    #         self.height_range[:, 0],
    #         self.height_range[:, 1],
    #     )
    #     penalty_height = (torch.abs(self._robot.data.root_pos_w[:, 2] - clipped_z)) ** 2

    #     # =========================
    #     # Flip penalty
    #     # =========================
    #     uprightness = self._robot.data.projected_gravity_b[:, 2] >= 0.0
    #     flip_penalty = torch.where(
    #         uprightness,
    #         torch.tensor(0.0, device=self.device),
    #         torch.tensor(5.0, device=self.device),
    #     )


    #     # lidar safety reward
    #     reward_safety_static = 1 - torch.tanh((self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=2).squeeze(1)


    #     # debug visualization
    #     ref_heading_marker_orientations = self._desired_quat_w
    #     self.robot_visualizer.visualize(translations=self._desired_pos_w, orientations=ref_heading_marker_orientations)

    #     robot_heading_marker_orientations = self._robot.data.root_quat_w
    #     self.my_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=robot_heading_marker_orientations)
    #     # self.robot_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=self._robot.data.root_quat_w)


    #     # thrust penalty reward
    #     thrust_power = self._thrust[:, 0, 2]
    #     rew_thrust_power = torch.tanh(thrust_power / 0.6)
    #     # print(f"thrust_power: {thrust_power.mean().item():.4f}, rew_thrust_power: {rew_thrust_power.mean().item():.4f}")

        
    #     # =========================
    #     # FINAL REWARD DICT
    #     # =========================
    #     rewards = {
    #         "rew_distance_to_goal": rew_distance_to_goal * self.cfg.distance_to_goal_reward_scale * self.step_dt,
    #         # "rew_angular_to_goal": rew_angular_to_goal * self.cfg.angular_to_goal_reward_scale * self.step_dt,
    #         "rew_lin_vel": rew_lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
    #         "rew_ang_vel": rew_ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
    #         "rew_velocity_dir": rew_vel_dir_w * self.cfg.velocity_direction_reward_scale * self.step_dt,
    #         "rew_head_tracking": head_tracking * self.cfg.head_tracking_reward_scale * self.step_dt,
    #         "rew_heading_stability": heading_stability * 3.0 * self.step_dt,
    #         "rew_stop": stop_reward * 5.0 * self.step_dt,
    #         "rew_reach_goal": reach_goal_reward * self.cfg.reach_goal_reward_scale * self.step_dt,
    #         "rew_reach_goal_timeout": reach_goal_reward_timeout * self.cfg.reach_goal_reward_timeout_scale * self.step_dt,
    #         "rew_height_penalty": penalty_height * self.cfg.height_penalty_scale * self.step_dt,
    #         "flip_penalty": flip_penalty * self.cfg.died_reward_scale * self.step_dt,
    #         "rew_reward_safety_static": reward_safety_static * self.cfg.reward_safety_static_scale * self.step_dt,
    #         "rew_thrust_power": rew_thrust_power * self.cfg.thrust_power_penalty_scale * self.step_dt,
    #     }

    #     total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

    #     # Logging
    #     for key, value in rewards.items():
    #         self._episode_sums[key] += value

    #     return total_reward
    

###############################################################################################################
        
    # def _get_rewards(self) -> torch.Tensor:
        
    #     pose_err, rot_err = compute_pose_error(
    #         self._robot.data.root_pos_w,
    #         self._robot.data.root_quat_w,
    #         self._desired_pos_w,
    #         self._desired_quat_w,
    #         # quat_from_euler_xyz(torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device)),
    #     )

    #     self._position_error = pose_err
    #     self._angle_error = rot_err
    # #     distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
    # #     distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        
    #     distance_to_goal = torch.linalg.norm(self._position_error , dim=1)
    #     # print(f"distance_to_goal: {distance_to_goal}")
    #     rew_distance_to_goal = 1 - torch.tanh(distance_to_goal / 0.8)
    #     # print(f"distance_to_goal: {distance_to_goal.shape}, rew_distance_to_goal: {rew_distance_to_goal.shape}")

    #     distance_to_goal_weight = 1 - torch.tanh(distance_to_goal / 0.8)
    #     angular_to_goal = torch.linalg.norm(self._angle_error, dim=1)
    #     rew_angular_to_goal = (1 - torch.tanh(angular_to_goal)) * distance_to_goal_weight
    #     # print(f"rew_angular_to_goal: {rew_angular_to_goal.shape}")

    #     lin_vel_norm = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=1)
    #     lin_vel_norm_clamp = torch.clamp(lin_vel_norm, max=10.0)
    #     rew_lin_vel = torch.square(torch.exp(0.6 * lin_vel_norm_clamp) - 1.0)
    #     # print(f"lin_vel_norm: {lin_vel_norm.shape}, rew_lin_vel: {rew_lin_vel.shape}")

    #     ang_vel_norm = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)
    #     ang_vel_norm_clamp = torch.clamp(ang_vel_norm, max=10.0)
    #     rew_ang_vel = torch.square(torch.exp(0.4 * ang_vel_norm_clamp) - 1.0)

    #     # rew_thrust_power = self._thrust[:, 0, 2] #TODO: change to actual thrust
    #     # print(f"rew_thrust_power: {rew_thrust_power.shape}")

    #     ANG_VEL_TH = 0.2
    #     LIN_VEL_TH = 0.1
    #     POS_TH = 0.1
    #     ANG_TH = 0.1

    #     reach_goal = torch.logical_and(
    #         torch.logical_and(ang_vel_norm < ANG_VEL_TH, lin_vel_norm < LIN_VEL_TH),
    #         torch.logical_and(angular_to_goal < ANG_TH, distance_to_goal < POS_TH),
    #     )
    #     # print("reach_goal:", reach_goal.sum())
    #     reach_goal = torch.logical_and(
    #         reach_goal,
    #         self.episode_length_buf > (self.max_episode_length - (2.0 / (self.cfg.sim.dt * self.cfg.decimation))),
    #     )
    #     # self._reach_goal_state = reach_goal
    #     self._reach_goal = reach_goal.to(torch.float32) * self.reset_time_outs.to(torch.float32)
    #     # self._reach_goal_count += self._reach_goal
    #     reach_goal_reward_timeout = (
    #         self.reset_time_outs.to(torch.float32) * reach_goal.to(torch.float32) * self.max_episode_length_s
    #     ) 
        
    #     reach_goal_reward = torch.zeros_like(reach_goal_reward_timeout)
    #     # reach_goal_reward += reach_goal.to(torch.float32) * self.step_dt * 1.0
    #     reach_goal_reward += torch.exp(-2 * distance_to_goal / POS_TH) * reach_goal.to(torch.float32)
    #     reach_goal_reward += torch.exp(-2 * angular_to_goal / ANG_TH) * reach_goal.to(torch.float32) 
    #     reach_goal_reward += torch.exp(-2 * lin_vel_norm / LIN_VEL_TH) * reach_goal.to(torch.float32) 
    #     reach_goal_reward += torch.exp(-2 * ang_vel_norm / ANG_VEL_TH) * reach_goal.to(torch.float32) 
    #     reach_goal_reward = reach_goal_reward

    #     # reward for velocity in the direction of the goal
    #     relative_err_pos_w = self._desired_pos_w - self._robot.data.root_pos_w
    #     unit_relative_err_pos = relative_err_pos_w / (relative_err_pos_w.norm(dim=-1, keepdim=True) + 1e-6)

    #     # unit_relative_err_pos = self._position_error / (self._position_error.norm(dim=-1, keepdim=True) + 1e-6)
    #     rew_vel_dir_w = self._robot.data.root_lin_vel_w * unit_relative_err_pos
    #     rew_vel_dir_w = torch.sum(rew_vel_dir_w, dim=-1)
    #     rew_vel_dir_w = torch.clamp(rew_vel_dir_w, min=0.0)
    #     # print(f"rew_vel_dir_w: {rew_vel_dir_w.shape}")

    #     # heading tracking reward
    #     self.ref_heading = torch.atan2(relative_err_pos_w[:, 1], relative_err_pos_w[:, 0])  # radian
    #     self.robot_heading = self._robot.data.heading_w
    #     self.angle_diff = self.ref_heading - self.robot_heading
    #     head_tracking_path_rew = 0.7 - torch.tanh(torch.abs(self.angle_diff) / 0.9)

    #     # height penalty
    #     clipped_z = torch.clamp(
    #                 self._robot.data.root_pos_w[:, 2],
    #                 self.height_range[:, 0] ,  # allow small tolerance
    #                 self.height_range[:, 1]
    #             )
    #     penalty_height = ((self._robot.data.root_pos_w[:, 2] - clipped_z) ** 2)  # shape (num_envs, 1)

    #     # debug visualization
    #     ref_heading_marker_orientations = self._desired_quat_w
    #     self.robot_visualizer.visualize(translations=self._desired_pos_w, orientations=ref_heading_marker_orientations)

    #     robot_heading_marker_orientations = self._robot.data.root_quat_w
    #     self.my_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=robot_heading_marker_orientations)
    #     # self.robot_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=self._robot.data.root_quat_w)

    #     #flip penalty
    #     uprightness = self._robot.data.projected_gravity_b[:, 2] >= 0.0
    #     # if it upside down, apply penalty
    #     flip_penalty = torch.where(uprightness, torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
    #     # flip_penalty = torch.where(uprightness[:, 2] < 0, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))



    #     rewards = {
    #         "rew_lin_vel": rew_lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
    #         "rew_ang_vel": rew_ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
    #         "rew_reach_goal": reach_goal_reward * self.cfg.reach_goal_reward_scale * self.step_dt,
    #         "rew_reach_goal_timeout": reach_goal_reward_timeout * self.cfg.reach_goal_reward_timeout_scale * self.step_dt ,
    #         "rew_distance_to_goal": rew_distance_to_goal * self.cfg.distance_to_goal_reward_scale * self.step_dt,
    #         # "rew_thrust_power": rew_thrust_power * self.cfg.thrust_power_scale * self.step_dt,
    #         "rew_angular_to_goal": rew_angular_to_goal * self.cfg.angular_to_goal_reward_scale * self.step_dt,
    #         "rew_velocity_dir": rew_vel_dir_w * self.cfg.velocity_direction_reward_scale * self.step_dt,
    #         "rew_head_tracking_path": head_tracking_path_rew * self.cfg.head_tracking_reward_scale * self.step_dt,
    #         "rew_height_penalty": penalty_height.squeeze(-1) * self.cfg.height_penalty_scale * self.step_dt,
    #         "flip_penalty": flip_penalty * self.cfg.died_reward_scale * self.step_dt,
    #     }
    #     total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
    #     # total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
    #     # Logging 
    #     for key, value in rewards.items():
    #                 self._episode_sums[key] += value
                    
    #     return total_reward






    # def _get_rewards(self) -> torch.Tensor:
    #     lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
    #     ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
    #     action_rate = torch.sum(torch.square(self._actions - self._last_actions), dim=1)

    #     # action_rate = torch.sum(torch.square(self._robot.data.root_lin_vel_w - self.previous_action), dim=1)

    #     distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
    #     distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        
    #     relative_err_pos_w = self._desired_pos_w - self._robot.data.root_pos_w
    #     unit_relative_err_pos = relative_err_pos_w / (relative_err_pos_w.norm(dim=-1, keepdim=True) + 1e-6)
    #     rew_vel_dir_w = self._robot.data.root_lin_vel_w * unit_relative_err_pos
    #     rew_vel_dir_w = torch.sum(rew_vel_dir_w, dim=-1)

    #     # heading tracking reward
    #     self.ref_heading = torch.atan2(relative_err_pos_w[:, 1], relative_err_pos_w[:, 0])  # radian
    #     self.robot_heading = self._robot.data.heading_w
    #     self.angle_diff = self.ref_heading - self.robot_heading
    #     head_tracking_path_rew = 0.7 - torch.tanh(torch.abs(self.angle_diff) / 0.9)
        

    #     # debug visualization
    #     ref_heading_marker_orientations = quat_from_angle_axis(self.ref_heading, torch.tensor([0.0, 0.0, 1.0], device=self.device))
    #     self.robot_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=ref_heading_marker_orientations)

    #     robot_heading_marker_orientations = quat_from_angle_axis(self.robot_heading, torch.tensor([0.0, 0.0, 1.0], device=self.device))
    #     self.my_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=robot_heading_marker_orientations)
    #     # self.robot_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=self._robot.data.root_quat_w)
        
    #     # height penalty
    #     clipped_z = torch.clamp(
    #                 self._robot.data.root_pos_w[:, 2],
    #                 self.height_range[:, 0] ,  # allow small tolerance
    #                 self.height_range[:, 1]
    #             )
    #     penalty_height = ((self._robot.data.root_pos_w[:, 2] - clipped_z) ** 2)  # shape (num_envs, 1)

    #     # lidar safety reward
    #     reward_safety_static = 1 - torch.tanh((self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=2).squeeze(1)


    #     # lidar potential field adapt heading reward
    #     vec_to_obstacles = (self._lidar_sensor.data.ray_hits_w - self._lidar_sensor.data.pos_w.unsqueeze(1)).clamp_max(2.5)
    #     dists = vec_to_obstacles.norm(dim=-1)
    #     closest_idx = torch.argmin(dists, dim=1)
    #     env_idx = torch.arange(vec_to_obstacles.shape[0])

    #     nearest_vec = vec_to_obsta/cles[env_idx, closest_idx]
    #     nearest_dist = dists[env_idx, closest_idx]
        
    #     robot_heading_vector = quat_apply_yaw(
    #         self._robot.data.root_state_w[:, 3:7].to(torch.float32),
    #         torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1),
    #     )
    #     #check if robot still in potential field range, if not set nearest vec to goal direction
    #     # in_field_range = nearest_dist < (self.lidar_range * 2.0)
    #     # nearest_vec = torch.where(
    #     #     in_field_range.unsqueeze(-1),
    #     #     nearest_vec,
    #     #     -unit_relative_err_pos * 5.0,  # scale to match typical obstacle distance
    #     # )
    #     nearest_angle = torch.atan2(nearest_vec[:, 1], nearest_vec[:, 0])
    #     nearest_quat = quat_from_angle_axis(nearest_angle, torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1))

    #     self.nearest_obs_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=nearest_quat)

    #     nearest_vec_norm = nearest_vec / (nearest_vec.norm(dim=-1, keepdim=True) + 1e-6)
    #     robot_vec_heading_norm = robot_heading_vector / (robot_heading_vector.norm(dim=-1, keepdim=True) + 1e-6)
    #     # cosine_similarity = torch.dot(nearest_vec_norm, robot_vec_heading_norm)/torch.norm(nearest_vec_norm)*torch.norm(robot_vec_heading_norm)
    #     cosine_similarity = torch.nn.functional.cosine_similarity(
    #         nearest_vec_norm,
    #         robot_vec_heading_norm,
    #         dim=1,
    #         eps=1e-8,
    #     )
    #     sigma = 3.0 #0.9  # Standard deviation of Gaussian function
    #     gaussian_factor = 1 / (0.1 * torch.sqrt(2 * torch.tensor(torch.pi)))  # Precomputed constant
    #     potential = 0.25 * gaussian_factor * torch.exp(-nearest_dist**2 / (2 * sigma**2))
    #     rew_potential_pa = (2*potential) * (1*cosine_similarity)
    #     rew_heading_reward_blended = (1 - potential) * head_tracking_path_rew +  potential * torch.abs(cosine_similarity)

    #     # penalty_shaking_roll_pitch = 

    #     rewards = {
    #         # "rew_lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
    #         # "rew_ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
    #         "rew_distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
    #         "rew_action_rate": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
    #         "rew_velocity_dir": rew_vel_dir_w * self.cfg.velocity_direction * self.step_dt,
    #         # "rew_head_tracking": head_tracking_path_rew * self.cfg.head_tracking * self.step_dt,
    #         "rew_height_penalty": -penalty_height * 0.5 * self.step_dt,
    #         "rew_reward_safety_static": reward_safety_static * self.cfg.reward_safety_static * self.step_dt,
    #         "rew_potential_field_PA": rew_potential_pa * self.cfg.potential_field_PA * self.step_dt,
    #         "rew_heading_tracking_PA": rew_heading_reward_blended * self.cfg.head_tracking_PA * self.step_dt,


    #     }
    #     total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

    #     # Early termination penalty
    #     # die_reward = self.reset_terminated.to(torch.float32) * self.cfg.died_reward_scale * self.max_episode_length_s
    #     # total_reward += die_reward
    #     # rewards["died"] = die_reward

    #     # # timeout reward with reaching goal
    #     # ang_vel_norm = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=-1)
    #     # lin_vel_norm = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=-1)

    #     # ANG_VEL_TH = 0.02
    #     # LIN_VEL_TH = 0.01
    #     # POS_TH = 0.02
    #     # ANG_TH = 0.05
    #     # reach_goal = torch.logical_and(
    #     #     torch.logical_and(ang_vel_norm < ANG_VEL_TH, lin_vel_norm < LIN_VEL_TH),
    #     #     torch.logical_and(angular_to_goal < ANG_TH, distance_to_goal < POS_TH),
    #     # )
    #     # reach_goal = torch.logical_and(
    #     #     reach_goal,
    #     #     self.episode_length_buf > (self.max_episode_length - (2.0 / (self.cfg.sim.dt * self.cfg.decimation))),
    #     # )
    #     # self._reach_goal_state = reach_goal
    #     # self._reach_goal = reach_goal.to(torch.float32) * self.reset_time_outs.to(torch.float32)
    #     # self._reach_goal_count += self._reach_goal
    #     # reach_goal_reward_timeout = (
    #     #     self.reset_time_outs.to(torch.float32) * reach_goal.to(torch.float32) * self.max_episode_length_s
    #     # ) * self.cfg.reach_goal_reward_timeout_scale
    #     # reach_goal_reward = torch.zeros_like(reach_goal_reward_timeout)
    #     # # reach_goal_reward += reach_goal.to(torch.float32) * self.step_dt * 1.0
    #     # reach_goal_reward += torch.exp(-2 * distance_to_goal / POS_TH) * reach_goal.to(torch.float32) * self.step_dt
    #     # reach_goal_reward += torch.exp(-2 * angular_to_goal / ANG_TH) * reach_goal.to(torch.float32) * self.step_dt
    #     # reach_goal_reward += torch.exp(-2 * lin_vel_norm / LIN_VEL_TH) * reach_goal.to(torch.float32) * self.step_dt
    #     # reach_goal_reward += torch.exp(-2 * ang_vel_norm / ANG_VEL_TH) * reach_goal.to(torch.float32) * self.step_dt
    #     # reach_goal_reward = reach_goal_reward * self.cfg.reach_goal_reward_scale


    #     # self.previous_action = self._actions.clone()
    #     # self.previous_action = self._robot.data.root_lin_vel_w.clone()


    #     # Logging
    #     for key, value in rewards.items():
    #         self._episode_sums[key] += value
            
    #     return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.35, self._robot.data.root_pos_w[:, 2] > 10.0)
        
        # uprightness = self._robot.data.projected_gravity_b[:, 2] > 0.0
        uprightness = self._robot.data.projected_gravity_b[:, 2] >= 0.0
        # print(f"uprightness: {self._robot.data.projected_gravity_b[:, :] }")
        # check if the drone upside down by looking at the z component of the projected gravity in the body frame. If it's negative, it means the drone is upside down.
        # uprightness = quat_apply_inverse(self._robot.data.root_quat_w,  self._robot.data.GRAVITY_VEC_W)
        # uprightness = uprightness[:, 2] >= 0.0

        static_collision = einops.reduce(self.lidar_scan, "n 1 w -> n 1", "min") < 0.32  # 0.3 collision radius
        # reach_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1) < 0.15

        relative_err_pos_w = self._desired_pos_w - self._robot.data.root_pos_w
        ref_heading = torch.atan2(relative_err_pos_w[:, 1], relative_err_pos_w[:, 0])  # radian
        angle_diff = ref_heading - self._robot.data.heading_w
        opposite_direction_heading = torch.abs(angle_diff) >  0.8  #45 degress # 90 degrees

        crash = (
            torch.linalg.norm(self._contact_sensor.data.net_forces_w.squeeze(1), dim=-1)
            > self.cfg.contact_force_threshold
        )

        # died = died | uprightness | static_collision.squeeze(1) | reach_goal | opposite_direction_heading
        # died = died | uprightness | static_collision.squeeze(1) | opposite_direction_heading
        # died = died | static_collision.squeeze(1) | opposite_direction_heading
        died = died | static_collision.squeeze(1) | crash #| uprightness
        
        if self.cfg.evaluate_mode:
            died = torch.zeros_like(time_out, dtype=torch.bool)
        
        return died, time_out
    
    def random_yaw_quaternion(self, num_envs, device):
        yaw_angles = (torch.rand(num_envs, device=device) - 0.5) * 2 * torch.pi
        return torch.zeros(num_envs, device=device)
    
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
        self._contact_sensor.reset(env_ids)
        self.noiseModel.reset(env_ids)

        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self.previous_action[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0


        
        # Sample new commands
        if self.cfg.evaluate_mode:
            self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-0.1, 0.1)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(2.0, 2.1)
        else:
            self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-10.0, 10.0)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(1.2, 1.8)
            
            body_ang = torch.pi / 180.0 * 0.0
            ang_range = body_ang 
            self._desired_quat_w[env_ids] = sampleUniformQuatwithTilt(torch.tensor(ang_range), len(env_ids)).to(self.device)

        # self._desired_quat_w[env_ids] = self.random_yaw_quaternion(len(env_ids), self.device)
        # dir_to_goal = self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids]
        # yaw = torch.atan2(dir_to_goal[:, 1], dir_to_goal[:, 0])
        # self._desired_quat_w[env_ids] = quat_from_euler_xyz(
        #     torch.zeros_like(yaw),
        #     torch.zeros_like(yaw),
        #     yaw
        # )
        

        desired_heights = self._desired_pos_w[:, 2]
        margin = 0.05
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
                #     scale=(0.2, 0.2, 0.2),
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
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.1, 0.1, 0.1),
                ),
                # "arrow_x": sim_utils.UsdFileCfg(
                #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                #     scale=(0.1, 0.1, 1.0),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                # ),
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



