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
from isaaclab.utils.math import subtract_frame_transforms, quat_apply_yaw, quat_from_angle_axis

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
    # start_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.9, 1.1)},
    #     },
    # )
    # push_force_body = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "force": (0, 0, 0.5),
    #         "torque": (0, 0, 0),
    #         "operation": "add",
    #     },

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
    observation_space = 75
    # observation_space = 17 #with 5 beams lidar
    # observation_space = 12
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow
    
    # viewer = ViewerCfg(eye=(-19.8, -23.8, 11.5), lookat=(-24.0, -8.5, -1.7), origin_type='env', env_index=2015)
    viewer = ViewerCfg(eye=(-19.8, -23.8, 11.5), lookat=(-24.0, -8.5, -1.7), origin_type='env')

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
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
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = AGILE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # sensor 
    lidar_sensor = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),
        attach_yaw_only=False,
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(10.0, 20.0), horizontal_fov_range=(-50.0, 50.0),horizontal_res=1.67),     
        # pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(10.0, 20.0), horizontal_fov_range=(-48.0, 48.0),horizontal_res=3.0),   
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    thrust_to_weight = 5.0
    moment_scale = 0.7

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.05
    distance_to_goal_reward_scale = 40.0
    action_rate_reward_scale = -0.5
    velocity_direction = 15.0
    reward_safety_static = 30.0#15.0
    head_tracking = 15.0
    potential_tracking = 65.0#60.0
    penalty_height = - 0.1


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_range = torch.zeros(self.num_envs, 2 , device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "action_rate",
                "rew_velocity_dir",
                "reward_safety_static",
                "head_tracking",
                "potential_tracking",
                "penalty_height",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("base_link")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # history
        self.previous_action = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

        #lidar
        self.lidar_resolution = (60)
        self.lidar_range = 5.0

        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.draw.clear_lines()

        self.my_visualizer = self.define_markers()
        self.robot_visualizer = self.define_robot_markers()

        self._reach_goal_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
    
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
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
        

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
    
        desired_dist = desired_pos_b.norm(dim=-1, keepdim=True)
        unit_desird_pos_b = desired_pos_b / (desired_dist + 1e-6)

        desired_dist_2d = desired_pos_b[:, :2].norm(dim=-1, keepdim=True)
        desired_dist_z = desired_pos_b[:, 2].unsqueeze(1)
        # print("unit_desird_pos_b", unit_desird_pos_b)
        

        # print("desired_pos_w", self._desired_pos_w[2015])
        # print("root_pos_w", self._robot.data.root_pos_w[2015])
        # print("desired_dist", desired_dist[2015])
        # print("desired_pos_b", desired_pos_b[2015])

        self.lidar_scan = (
            (
                self._lidar_sensor.data.ray_hits_w
                - self._lidar_sensor.data.pos_w.unsqueeze(1)
            )
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, self.lidar_resolution)
        )

        # lidar potential field
        vec_to_obstacles = (self._lidar_sensor.data.ray_hits_w - self._lidar_sensor.data.pos_w.unsqueeze(1)) .clamp_max(self.lidar_range * 2.0)
        dists_to_obstacle = vec_to_obstacles.norm(dim=-1)
        closest_idx = torch.argmin(dists_to_obstacle, dim=1)
        env_idx = torch.arange(vec_to_obstacles.shape[0])
        nearest_dist = dists_to_obstacle[env_idx, closest_idx]
        
        sigma = 0.7  # Standard deviation of Gaussian function
        gaussian_factor = 1 / (0.1 * torch.sqrt(2 * torch.tensor(torch.pi)))  # Precomputed constant
        potential = 0.5 * gaussian_factor * torch.exp(-nearest_dist**2 / (2 * sigma**2))
        
        # print(self.lidar_scan.squeeze(1)[2015])
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,  # 3
                self._robot.data.root_ang_vel_b,  # 3
                self._robot.data.projected_gravity_b,  # 3
                unit_desird_pos_b,  # 3
                desired_dist_2d,  # 1
                desired_dist_z,  # 1
                self.lidar_scan.squeeze(1),  # 60
                potential.unsqueeze(1),  # 1
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.5)

        # action rate reward
        # print(torch.square(self._actions - self.previous_action))
        action_rate = torch.sum(torch.square(self._actions - self.previous_action), dim=1)

        # target velocity direction reward
        relative_err_pos = self._desired_pos_w - self._robot.data.root_pos_w
        unit_relative_err_pos = relative_err_pos / (relative_err_pos.norm(dim=-1, keepdim=True) + 1e-6)
        rew_vel_dir = self._robot.data.root_lin_vel_w * unit_relative_err_pos
        rew_vel_dir = torch.sum(rew_vel_dir, dim=-1)
        # print("reward vel = ", rew_vel_dir)
        # print("reward vel shape = ", rew_vel_dir.shape)

        # lidar safety reward
        reward_safety_static = 1 - torch.tanh((self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=2).squeeze(1)

        # lidar potential field adapt heading reward
        vec_to_obstacles = (self._lidar_sensor.data.ray_hits_w - self._lidar_sensor.data.pos_w.unsqueeze(1)).clamp_max(self.lidar_range * 2.0)
        dists = vec_to_obstacles.norm(dim=-1)
        closest_idx = torch.argmin(dists, dim=1)
        env_idx = torch.arange(vec_to_obstacles.shape[0])

        nearest_vec = vec_to_obstacles[env_idx, closest_idx]
        nearest_dist = dists[env_idx, closest_idx]
        
        nearest_angle = torch.atan2(nearest_vec[:, 1], nearest_vec[:, 0])
        nearest_quat = quat_from_angle_axis(nearest_angle, torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1))

        robot_heading_vector = quat_apply_yaw(
            self._robot.data.root_state_w[:, 3:7].to(torch.float32),
            torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1),
        )

        dot_vec = torch.abs(torch.sum(robot_heading_vector * nearest_vec, dim=1))
        
        sigma = 0.7  # Standard deviation of Gaussian function
        gaussian_factor = 1 / (0.1 * torch.sqrt(2 * torch.tensor(torch.pi)))  # Precomputed constant
        potential = 0.5 * gaussian_factor * torch.exp(-nearest_dist**2 / (2 * sigma**2))
        # print("potential ", potential[2015])
        potential_rew = potential * dot_vec

        self.my_visualizer.visualize(translations=self._lidar_sensor.data.pos_w, orientations=nearest_quat)
        self.robot_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=self._robot.data.root_quat_w)
        
        # heading tracking reward
        self.ref_heading = torch.atan2(relative_err_pos[:, 1], relative_err_pos[:, 0])  # radian
        self.robot_heading = self._robot.data.heading_w
        self.angle_diff = self.ref_heading - self.robot_heading
        head_tracking_path_rew = 1 - torch.tanh(torch.abs(self.angle_diff) / 0.5)

        # penalty for too hight or too low
        clipped_z = torch.clamp(
            self._robot.data.root_pos_w[:, 2],
            self.height_range[:, 0] - 0.2,  # allow small tolerance
            self.height_range[:, 1] + 0.2
        )
        penalty_height = ((self._robot.data.root_pos_w[:, 2] - clipped_z) ** 2)  # shape (num_envs, 1)
        # print("shape of panalty", penalty_height.shape)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "action_rate": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "rew_velocity_dir": rew_vel_dir * self.cfg.velocity_direction * self.step_dt,
            "reward_safety_static": reward_safety_static * self.cfg.reward_safety_static * self.step_dt,
            "head_tracking": head_tracking_path_rew * self.cfg.head_tracking * self.step_dt,
            "potential_tracking": potential_rew * self.cfg.potential_tracking * self.step_dt,
            "penalty_height": penalty_height * self.cfg.penalty_height * self.step_dt,
        }
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        self.previous_action = self._actions.clone()

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.3, self._robot.data.root_pos_w[:, 2] > 3.0)
        
        self._time_limit = 100
        reach_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1) < 0.25
        self._reach_goal_counter = self._reach_goal_counter + reach_goal.int()
        self._reach_goal_exceed = self._reach_goal_counter > self._time_limit

        # died = died | reach_goal
        # print("dead shape", died.shape)

        # print("lidar dist",einops.reduce(self.lidar_scan, "n 1 w -> n 1", "min"))
        static_collision = einops.reduce(self.lidar_scan, "n 1 w -> n 1", "min") < 0.1  # 0.3 collision radius
        # print("static_collision", static_collision.squeeze(1))
        # num_collisions = static_collision.sum()
        uprightness = self._robot.data.projected_gravity_b[:, 2] >= 0.0

        # died = died | static_collision.squeeze(1) | limit_vel | uprightness | self._reach_goal_exceed
        died = died | static_collision.squeeze(1) | uprightness | self._reach_goal_exceed
        
        # print("dead shape2 ", died.shape)
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
            # self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
            self.episode_length_buf = torch.zeros_like(self.episode_length_buf)


        self._actions[env_ids] = 0.0
        self._reach_goal_counter[env_ids] = 0
        
        # Sample new commands
        box_size = 15.0
        half_box = box_size / 2.0
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-17.0, 17.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(1.0, 1.5)

        desired_heights = self._desired_pos_w[:, 2]
        margin = 0.5
        self.height_range = torch.stack([
            desired_heights - margin,
            desired_heights + margin
        ], dim=-1)
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        default_root_state[:, 0] += self._terrain.env_origins[env_ids, 0] - 15.0  # offset
        default_root_state[:, 1] += self._terrain.env_origins[env_ids, 1]
        default_root_state[:, 2] += self._terrain.env_origins[env_ids, 2]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
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


