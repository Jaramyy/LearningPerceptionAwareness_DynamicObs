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

##
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

# ------------------------------------------------------------------
# Avoidance zone parameters (used in both reward and heading blend)
# ------------------------------------------------------------------
AVOIDANCE_DIST  = 2.5   # (m) start transitioning heading toward obstacle
FULL_AVOID_DIST = 0.8   # (m) fully face obstacle within this range


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
    evaluate_mode = False
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    observation_space = 9 + 60 + 1 + 2 + 4  # with 60 beams lidar + potential field + last action
    state_space = 0
    debug_vis = True

    # Noise Configuration
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

    viewer = ViewerCfg(eye=(-19.8, -23.8, 11.5), lookat=(-24.0, -8.5, -1.7), origin_type='env')

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    flat_terrain = False
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
        ray_alignment='base',
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(10.0, 20.0),
            horizontal_fov_range=(-179.0, 179.0),
            horizontal_res=6.0,
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        history_length=1,
        update_period=0,
        track_air_time=True,
        debug_vis=True,
    )

    thrust_to_weight = 5.0
    moment_scale = 0.7
    contact_force_threshold = 0.1

    # reward scales
    lin_vel_reward_scale = -0.0002
    ang_vel_reward_scale = -0.001

    distance_to_goal_reward_scale = 50.0
    action_rate_reward_scale = -0.01
    velocity_direction = 25.0
    head_tracking = 30.0
    potential_field_PA = 25.0
    head_tracking_PA = 32.0

    # max velocity
    max_velocity = 3.5
    max_yaw_rate = 6.28

    # NEW Reward parameters
    died_reward_scale = 0.5
    reach_goal_reward_timeout_scale = 0.01
    reach_goal_reward_scale = 0.5

    velocity_direction_reward_scale = 6.0
    distance_to_goal_reward_scale = 9.0
    head_tracking_reward_scale = 5.0
    height_penalty_scale = -0.5
    reward_safety_static_scale = 6.0
    thrust_power_penalty_scale = -0.5


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.previous_action = torch.zeros(self.num_envs, 3, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._lin_vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._yaw_vel_cmd = torch.zeros(self.num_envs, 1, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.height_range = torch.zeros(self.num_envs, 2, device=self.device)

        # lidar
        self.lidar_resolution = 60
        self.lidar_range = 5.0

        self.my_visualizer = self.define_markers()
        self.robot_visualizer = self.define_robot_markers()
        self.nearest_obs_visualizer = self.define_nearest_obs_markers()

        # noise model
        self.noiseModel = NoiseModel(cfg.noiseCfg, device=self.device, num_envs=self.num_envs)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
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

        self._body_id = self._robot.find_bodies("base_link")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        all_inertia_tensor = self._robot.root_physx_view.get_inertias()[0]
        robot_inertia = torch.sum(all_inertia_tensor, dim=0)
        print("Inertia tensor of the robot1:", robot_inertia)

        self.set_debug_vis(self.cfg.debug_vis)

        input_robot_inertia = torch.diag(torch.tensor(
            [robot_inertia[0], robot_inertia[4], robot_inertia[8]],
            device=self.device,
        ))
        print("Inertia tensor of the robot2:", input_robot_inertia)

        self.velocity_controller = GeometricVelocityController(
            num_env=self.num_envs,
            mass=self._robot_mass,
            inertia=input_robot_inertia,
            device=self.device,
        )

    # ------------------------------------------------------------------
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
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):

        self._last_actions = self._actions.clone().clamp(-1.0, 1.0)
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # 1. Distance to goal
        relative_goal = self._desired_pos_w - self._robot.data.root_pos_w
        distance_to_goal = torch.linalg.norm(relative_goal, dim=1, keepdim=True)

        # 2. Smooth slowdown near goal
        SLOWDOWN_DIST = 2.0
        MIN_SPEED_SCALE = 0.05
        speed_scale = torch.clamp(distance_to_goal / SLOWDOWN_DIST, min=MIN_SPEED_SCALE, max=1.0)
        speed_scale = torch.sqrt(speed_scale)

        # 3. Convert actions -> commands
        self._yaw_vel_cmd[:, 0] = (
            self._actions[:, 0] * self.cfg.max_yaw_rate * speed_scale.squeeze(-1)
        )
        self._lin_vel_cmd[:, :] = (
            self._actions[:, 1:] * self.cfg.max_velocity * speed_scale
        )

        # 4. Extra stabilisation zone
        HOLD_DIST = 0.05
        near_goal = (distance_to_goal < HOLD_DIST).float()
        self._lin_vel_cmd *= (1.0 - 0.9 * near_goal)
        self._yaw_vel_cmd[:, 0] *= (1.0 - 0.9 * near_goal.squeeze(-1))

        # 5. Controller
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

    # ------------------------------------------------------------------
    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    # ------------------------------------------------------------------
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
        # root_lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        root_ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_b)   # kept as-is from original

        desired_pos_b, _ = subtract_frame_transforms(
            root_pos_w, root_quat_w, self._desired_pos_w, self._desired_quat_w
        )

        desired_dist = desired_pos_b.norm(dim=-1, keepdim=True)
        unit_desird_pos_b = desired_pos_b / (desired_dist + 1e-6)

        desired_dist_2d = desired_pos_b[:, :2].norm(dim=-1, keepdim=True)
        desired_dist_z = desired_pos_b[:, 2].unsqueeze(1)

        self.lidar_scan = (
            (self._lidar_sensor.data.ray_hits_w - lidar_sensor_pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, self.lidar_resolution)
        )

        # lidar potential field
        vec_to_obstacles = (
            self._lidar_sensor.data.ray_hits_w - lidar_sensor_pos_w.unsqueeze(1)
        ).clamp_max(self.lidar_range)
        dists_to_obstacle = vec_to_obstacles.norm(dim=-1)
        closest_idx = torch.argmin(dists_to_obstacle, dim=1)
        env_idx = torch.arange(vec_to_obstacles.shape[0])
        nearest_dist = dists_to_obstacle[env_idx, closest_idx]

        sigma = 3.0
        gaussian_factor = 1 / (0.1 * torch.sqrt(2 * torch.tensor(torch.pi)))
        potential = 0.25 * gaussian_factor * torch.exp(-nearest_dist**2 / (2 * sigma**2))

        obs = torch.cat(
            [
                root_lin_vel_b,
                root_ang_vel_b,
                unit_desird_pos_b,   # 3
                desired_dist_2d,     # 1
                desired_dist_z,      # 1
                self.lidar_scan.squeeze(1),
                potential.unsqueeze(-1),
                self._last_actions,
            ],
            dim=-1,
        )

        states = self._get_states()
        observations = {"policy": obs, "critic": states}
        return observations

    # ------------------------------------------------------------------
    def _get_states(self):
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )

        desired_dist = desired_pos_b.norm(dim=-1, keepdim=True)
        unit_desird_pos_b = desired_pos_b / (desired_dist + 1e-6)

        desired_dist_2d = desired_pos_b[:, :2].norm(dim=-1, keepdim=True)
        desired_dist_z = desired_pos_b[:, 2].unsqueeze(1)

        lidar_scan = (
            (self._lidar_sensor.data.ray_hits_w - self._lidar_sensor.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, self.lidar_resolution)
        )

        vec_to_obstacles = (
            self._lidar_sensor.data.ray_hits_w - self._lidar_sensor.data.pos_w.unsqueeze(1)
        ).clamp_max(self.lidar_range)
        dists_to_obstacle = vec_to_obstacles.norm(dim=-1)
        closest_idx = torch.argmin(dists_to_obstacle, dim=1)
        env_idx = torch.arange(vec_to_obstacles.shape[0])
        nearest_dist = dists_to_obstacle[env_idx, closest_idx]

        sigma = 3.0
        gaussian_factor = 1 / (0.1 * torch.sqrt(2 * torch.tensor(torch.pi)))
        potential = 0.25 * gaussian_factor * torch.exp(-nearest_dist**2 / (2 * sigma**2))

        states = torch.cat(
            (
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
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

    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        pose_err, rot_err = compute_pose_error(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
            self._desired_quat_w,
        )
        self._position_error = pose_err
        self._angle_error = rot_err

        distance_to_goal = torch.linalg.norm(self._position_error, dim=1)

        # ==========================================================
        # Goal reward
        # ==========================================================
        rew_distance_to_goal = 1 - torch.tanh(distance_to_goal / 0.8)

        # ==========================================================
        # Progress reward
        # ==========================================================
        if not hasattr(self, "_prev_distance_to_goal"):
            self._prev_distance_to_goal = distance_to_goal.clone()

        rew_progress = self._prev_distance_to_goal - distance_to_goal
        self._prev_distance_to_goal = distance_to_goal.clone()

        # ==========================================================
        # Velocity penalties
        # ==========================================================
        lin_vel_norm = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=1)
        ang_vel_norm = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)

        rew_lin_vel_penalty = lin_vel_norm ** 2
        rew_ang_vel_penalty = ang_vel_norm ** 2

        # ==========================================================
        # Goal direction reward
        # ==========================================================
        relative_err_pos_w = self._desired_pos_w - self._robot.data.root_pos_w
        unit_relative_err_pos = relative_err_pos_w / (
            relative_err_pos_w.norm(dim=-1, keepdim=True) + 1e-6
        )

        rew_vel_dir_w = (self._robot.data.root_lin_vel_w * unit_relative_err_pos).sum(dim=-1)
        rew_vel_dir_w = torch.clamp(rew_vel_dir_w, min=0.0)

        # ==========================================================
        # LiDAR: vectors and distances to obstacles
        # ==========================================================
        vec_to_obstacles = (
            self._lidar_sensor.data.ray_hits_w
            - self._lidar_sensor.data.pos_w.unsqueeze(1)
        ).clamp_max(self.lidar_range)                    # (N, R, 3)

        dists = vec_to_obstacles.norm(dim=-1)            # (N, R)
        nearest_dist = dists.min(dim=1)[0]               # (N,)

        # ==========================================================
        # Collision penalty
        # ==========================================================
        collision_penalty = (nearest_dist / 0.4).clamp(0.0, 1.0)

        # ==========================================================
        # Reach goal
        # ==========================================================
        reach_goal = distance_to_goal < 0.10
        reach_goal_reward = reach_goal.float() * 10.0

        # ==========================================================
        # Robot heading vector (yaw-only, XY plane)
        # ==========================================================
        robot_heading_w = quat_apply_yaw(
            self._robot.data.root_state_w[:, 3:7].float(),
            torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )                                                # (N, 3)
        robot_heading_w = robot_heading_w / (robot_heading_w.norm(dim=-1, keepdim=True) + 1e-6)

        # Project to XY only (avoid pitch response from vertical obstacles)
        robot_heading_xy = robot_heading_w.clone()
        robot_heading_xy[:, 2] = 0.0
        robot_heading_xy = robot_heading_xy / (robot_heading_xy.norm(dim=-1, keepdim=True) + 1e-6)

        # ==========================================================
        # Goal-facing heading reward
        # ==========================================================
        goal_cos = torch.nn.functional.cosine_similarity(
            robot_heading_xy, unit_relative_err_pos, dim=1
        )
        goal_heading_reward = (goal_cos + 1.0) * 0.5     # [0, 1]

        # ==========================================================
        # Obstacle-facing heading reward  (nearest single obstacle)
        # Drone faces TOWARD the threat so lidar keeps it centred.
        # ==========================================================
        env_idx = torch.arange(self.num_envs, device=self.device)
        nearest_idx = torch.argmin(dists, dim=1)                  # (N,)
        nearest_vec = vec_to_obstacles[env_idx, nearest_idx]      # (N, 3)

        # XY only — vertical pillars shouldn't tilt the heading target
        nearest_vec_xy = nearest_vec.clone()
        nearest_vec_xy[:, 2] = 0.0
        nearest_vec_xy = nearest_vec_xy / (nearest_vec_xy.norm(dim=-1, keepdim=True) + 1e-6)

        obstacle_cos = torch.nn.functional.cosine_similarity(
            robot_heading_xy, nearest_vec_xy, dim=1
        )
        obstacle_facing_reward = (obstacle_cos + 1.0) * 0.5       # [0, 1]

        # ==========================================================
        # Adaptive blend: alpha ramps 0→1 as drone enters avoidance zone
        #
        #   nearest_dist >= AVOIDANCE_DIST  →  alpha = 0  (face goal)
        #   nearest_dist <= FULL_AVOID_DIST →  alpha = 1  (face obstacle)
        # ==========================================================
        alpha = (
            (AVOIDANCE_DIST - nearest_dist) / (AVOIDANCE_DIST - FULL_AVOID_DIST)
        ).clamp(0.0, 1.0)                                          # (N,)

        rew_heading = (1.0 - alpha) * goal_heading_reward + alpha * obstacle_facing_reward

        # ==========================================================
        # Visualization: arrow points toward nearest obstacle
        # ==========================================================
        obstacle_angle = torch.atan2(nearest_vec_xy[:, 1], nearest_vec_xy[:, 0])
        obstacle_quat = quat_from_angle_axis(
            obstacle_angle,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.nearest_obs_visualizer.visualize(
            translations=self._robot.data.root_pos_w,
            orientations=obstacle_quat,
        )

        # ==========================================================
        # Reward dict
        # ==========================================================
        rewards = {
            "distance":
                rew_distance_to_goal
                * self.cfg.distance_to_goal_reward_scale
                * self.step_dt,

            "progress":
                rew_progress
                * self.cfg.velocity_direction_reward_scale  # reuse a reasonable scale
                * self.step_dt,

            "velocity_dir":
                rew_vel_dir_w
                * self.cfg.velocity_direction_reward_scale
                * self.step_dt,

            # Heading reward is boosted inside the avoidance zone so the
            # policy gets a stronger signal exactly when it matters most.
            "heading":
                rew_heading
                * self.cfg.head_tracking_reward_scale
                * (1.0 + alpha * 2.0)          # scale: 5 (far) → 15 (close)
                * self.step_dt,

            "collision":
                collision_penalty
                * 0.5
                * self.step_dt,

            "lin_vel":
                -rew_lin_vel_penalty
                * 0.001
                * self.step_dt,

            "ang_vel":
                -rew_ang_vel_penalty
                * 0.001
                * self.step_dt,

            "reach_goal":
                reach_goal_reward
                * self.cfg.reach_goal_reward_scale
                * self.step_dt,
        }

        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging — "flip" key kept in episode_sums but not computed here;
        # zero-fill so the logger doesn't complain.
        for key, value in rewards.items():
            if key in self._episode_sums:
                self._episode_sums[key] += value
        # "flip" was in the original episode_sums list; keep it zeroed
        # (re-add a flip penalty here if needed in future)

        return total_reward

    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.35,
            self._robot.data.root_pos_w[:, 2] > 10.0,
        )

        static_collision = (
            einops.reduce(self.lidar_scan, "n 1 w -> n 1", "min") < 0.32
        )

        crash = (
            torch.linalg.norm(self._contact_sensor.data.net_forces_w.squeeze(1), dim=-1)
            > self.cfg.contact_force_threshold
        )

        uprightness_threshold = 0.0
        projected_gravity_b = self._robot.data.projected_gravity_b
        # print("Projected gravity in body frame:", projected_gravity_b)
        flipped = projected_gravity_b[:, 2] > -uprightness_threshold
        

        died = died | static_collision.squeeze(1) | crash #| flipped

        if self.cfg.evaluate_mode:
            died = torch.zeros_like(time_out, dtype=torch.bool)

        return died, time_out

    # ------------------------------------------------------------------
    def random_yaw_quaternion(self, num_envs, device):
        yaw_angles = (torch.rand(num_envs, device=device) - 0.5) * 2 * torch.pi
        return torch.zeros(num_envs, device=device)

    # ------------------------------------------------------------------
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
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self.previous_action[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0

        # Sample new commands
        if self.cfg.evaluate_mode:
            self._desired_pos_w[env_ids, :2] = (
                torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-0.1, 0.1)
            )
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] = (
                torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(2.0, 2.1)
            )
        else:
            self._desired_pos_w[env_ids, :2] = (
                torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-10.0, 10.0)
            )
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] = (
                torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(1.2, 1.8)
            )

            body_ang = torch.pi / 180.0 * 0.0
            ang_range = body_ang
            self._desired_quat_w[env_ids] = sampleUniformQuatwithTilt(
                torch.tensor(ang_range), len(env_ids)
            ).to(self.device)

        desired_heights = self._desired_pos_w[:, 2]
        margin = 0.05
        self.height_range = torch.stack(
            [desired_heights - margin, desired_heights + margin], dim=-1
        )

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

    # ------------------------------------------------------------------
    def define_markers(self) -> VisualizationMarkers:
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                "arrow_x": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.1, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)

    def define_robot_markers(self) -> VisualizationMarkers:
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.1, 0.1, 0.1),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)

    def define_nearest_obs_markers(self) -> VisualizationMarkers:
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                "arrow_x": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.1, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)