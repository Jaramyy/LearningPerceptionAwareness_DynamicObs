# Train:  ./isaaclab.sh -p scripts/rl_games/train.py --task Isaac-Agile-Lidar-Vel-PA-DynObs-v0 --num_envs 4096
# Play:   ./isaaclab.sh -p scripts/rl_games/play.py  --task Isaac-Agile-Lidar-Vel-PA-DynObs-v0 --checkpoint <path>
#
# Obstacle-dense variant of quadcopter_env_lidar_vel_PA.py.
# Uses 8×8=64 terrain tiles, each with 15 randomly placed obstacles (HfDiscreteObstaclesTerrainCfg).
# The 4096 training envs are distributed across 64 unique obstacle layouts (~64 envs per layout).
# Obs/action space is IDENTICAL to the PA env (77D obs, 4D action) so the
# same teacher checkpoint and DAgger student can be used without changes.
#
# NOTE: obstacles within a given env are fixed for the training run — per-episode
# repositioning of LiDAR-visible obstacles requires BVH rebuild which Isaac Lab < 5.0
# does not support (RayCaster enforces a single mesh prim).

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
from isaaclab.utils.math import (
    subtract_frame_transforms,
    quat_apply_yaw,
    quat_from_angle_axis,
    quat_apply_inverse,
)

from .robot.agileDrone import AGILE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.envs import ViewerCfg
from isaaclab.sensors import RayCasterCfg, RayCaster, patterns

import einops
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .utility.lee_velocity_controller.vel_controller import GeometricVelocityController


PUSH_LIN_VEL = 0.3
PUSH_ANG_VEL = 0.3


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
    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env — identical to PA env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    # vel(3)+ang(3)+grav(3)+goal_dir(3)+dist2d(1)+distz(1) + lidar(60) + nearest_obs_dir_b(2) + potential(1)
    observation_space = 77
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow
    viewer = ViewerCfg(eye=(-19.8, -23.8, 11.5), lookat=(-24.0, -8.5, -1.7), origin_type="env")

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 150,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Flat generator terrain — avoids loading external Nucleus USDs (which fail offline).
    # num_obstacles=0 produces a flat heightfield; env_origins are 40 m apart (tile size),
    # so LiDAR (5 m range) cannot see a neighbouring env's cylinders.
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        env_spacing=0.0,          # overwritten by scene.env_spacing at runtime
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(40.0, 40.0),
            border_width=5.0,
            num_rows=8,           # 8×8 = 64 tile varieties → 64 unique obstacle layouts
            num_cols=8,           # each of the 4096 envs is assigned to one of 64 tiles
            horizontal_scale=0.2,
            vertical_scale=0.1,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    horizontal_scale=0.2,
                    vertical_scale=0.1,
                    border_width=0.0,
                    num_obstacles=15,
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
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    events: EventCfg = EventCfg()
    robot: ArticulationCfg = AGILE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # LiDAR
    lidar_sensor: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(10.0, 20.0),
            horizontal_fov_range=(-179.0, 179.0),
            horizontal_res=6.0,
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Physics / reward — identical to PA env
    thrust_to_weight = 5.0
    moment_scale = 0.7

    ang_vel_reward_scale = -0.005
    distance_to_goal_reward_scale = 100.0
    action_rate_reward_scale = -0.01
    velocity_direction = 20.0
    head_tracking = 30.0
    reward_safety_static = 25.0
    proximity_penalty_scale = 60.0
    collision_penalty_scale = 15.0
    goal_reach_bonus = 20.0
    vel_repulsion_scale = 0.0
    potential_field_PA = 40.0
    head_tracking_PA = 20.0

    max_velocity = 4.0
    max_yaw_rate = 3.14


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.previous_action = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._lin_vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._yaw_vel_cmd = torch.zeros(self.num_envs, 1, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_range = torch.zeros(self.num_envs, 2, device=self.device)

        self.lidar_resolution = 60
        self.lidar_range = 5.0

        self.my_visualizer = self.define_markers()
        self.robot_visualizer = self.define_robot_markers()
        self.nearest_obs_visualizer = self.define_nearest_obs_markers()

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "rew_ang_vel",
                "rew_distance_to_goal",
                "rew_action_rate",
                "rew_velocity_dir",
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

        self._body_id = self._robot.find_bodies("base_link")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        robot_inertia = torch.diag(torch.tensor(
            [7.6191e-04, 8.9651e-04, 1.2983e-03], device=self.device
        ))

        self.set_debug_vis(self.cfg.debug_vis)

        self.velocity_controller = GeometricVelocityController(
            num_env=self.num_envs,
            mass=self._robot_mass,
            inertia=robot_inertia,
            device=self.device,
        )

    def _setup_scene(self):
        # 1. Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # 2. LiDAR
        self._lidar_sensor = RayCaster(self.cfg.lidar_sensor)
        self.scene.sensors["lidar_sensor"] = self._lidar_sensor

        # 3. Terrain — must come before rigid objects so env_origins exist at clone time
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 4. Clone
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._yaw_vel_cmd[:, 0] = self._actions[:, 0] * self.cfg.max_yaw_rate
        self._lin_vel_cmd[:, :] = self._actions[:, 1:] * self.cfg.max_velocity

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
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        desired_dist = desired_pos_b.norm(dim=-1, keepdim=True)
        unit_desired_pos_b = desired_pos_b / (desired_dist + 1e-6)
        desired_dist_2d = desired_pos_b[:, :2].norm(dim=-1, keepdim=True)
        desired_dist_z = desired_pos_b[:, 2].unsqueeze(1)

        raw_vecs = self._lidar_sensor.data.ray_hits_w - self._lidar_sensor.data.pos_w.unsqueeze(1)
        raw_dists = torch.nan_to_num(raw_vecs.norm(dim=-1), nan=self.lidar_range, posinf=self.lidar_range)
        self.lidar_scan = raw_dists.clamp_max(self.lidar_range).reshape(self.num_envs, 1, self.lidar_resolution)
        lidar_scan_norm = self.lidar_scan.squeeze(1) / self.lidar_range

        env_idx = torch.arange(self.num_envs, device=self.device)
        closest_idx = torch.argmin(raw_dists, dim=1)
        nearest_dist = raw_dists[env_idx, closest_idx].clamp_max(self.lidar_range)
        nearest_vec_w = raw_vecs[env_idx, closest_idx]
        nearest_vec_w_safe = torch.nan_to_num(nearest_vec_w, nan=0.0, posinf=self.lidar_range, neginf=-self.lidar_range)
        nearest_vec_w_norm = nearest_vec_w_safe / (nearest_vec_w_safe.norm(dim=-1, keepdim=True).clamp(min=0.1))
        nearest_vec_b = quat_apply_inverse(self._robot.data.root_quat_w, nearest_vec_w_norm)
        nearest_vec_b = torch.nan_to_num(nearest_vec_b, nan=0.0)

        sigma = 1.5
        gaussian_factor = 1 / (0.1 * torch.sqrt(2 * torch.tensor(torch.pi)))
        potential = 0.25 * gaussian_factor * torch.exp(-nearest_dist**2 / (2 * sigma**2))

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,       # 3
                self._robot.data.root_ang_vel_b,       # 3
                self._robot.data.projected_gravity_b,  # 3
                unit_desired_pos_b,                    # 3
                desired_dist_2d,                       # 1
                desired_dist_z,                        # 1
                lidar_scan_norm,                       # 60
                nearest_vec_b[:, :2],                  # 2
                potential.unsqueeze(-1),               # 1
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self.previous_action), dim=1)

        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = torch.exp(-distance_to_goal / 4.0)

        relative_err_pos_w = self._desired_pos_w - self._robot.data.root_pos_w
        unit_relative_err_pos = relative_err_pos_w / (relative_err_pos_w.norm(dim=-1, keepdim=True) + 1e-6)
        rew_vel_dir_w = torch.sum(self._robot.data.root_lin_vel_w * unit_relative_err_pos, dim=-1)

        self.ref_heading = torch.atan2(relative_err_pos_w[:, 1], relative_err_pos_w[:, 0])
        self.robot_heading = self._robot.data.heading_w
        self.angle_diff = self.ref_heading - self.robot_heading
        head_tracking_path_rew = 0.7 - torch.tanh(torch.abs(self.angle_diff) / 0.9)

        # debug arrows
        ref_h_quat = quat_from_angle_axis(self.ref_heading, torch.tensor([0.0, 0.0, 1.0], device=self.device))
        self.robot_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=ref_h_quat)
        rob_h_quat = quat_from_angle_axis(self.robot_heading, torch.tensor([0.0, 0.0, 1.0], device=self.device))
        self.my_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=rob_h_quat)

        # height penalty
        clipped_z = torch.clamp(self._robot.data.root_pos_w[:, 2], self.height_range[:, 0], self.height_range[:, 1])
        penalty_height = (self._robot.data.root_pos_w[:, 2] - clipped_z) ** 2

        reward_safety_static = 1 - torch.tanh(
            (self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)
        ).mean(dim=2).squeeze(1)

        min_beam = self.lidar_scan.min(dim=2).values.squeeze(1).clamp(min=0.0, max=self.lidar_range)
        safe_dist = 1.5
        proximity_penalty = torch.clamp(1.0 - min_beam / safe_dist, min=0.0) ** 2
        collision_terminal = (min_beam < 0.35).float()

        raw_vecs_rew = self._lidar_sensor.data.ray_hits_w - self._lidar_sensor.data.pos_w.unsqueeze(1)
        dists = torch.nan_to_num(raw_vecs_rew.norm(dim=-1), nan=self.lidar_range, posinf=self.lidar_range)
        closest_idx = torch.argmin(dists, dim=1)
        env_idx = torch.arange(self.num_envs, device=self.device)

        nearest_vec = raw_vecs_rew[env_idx, closest_idx]
        nearest_dist = dists[env_idx, closest_idx].clamp_max(self.lidar_range)

        robot_heading_vector = quat_apply_yaw(
            self._robot.data.root_state_w[:, 3:7].float(),
            torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1),
        )

        nearest_angle = torch.atan2(nearest_vec[:, 1], nearest_vec[:, 0])
        nearest_quat = quat_from_angle_axis(
            nearest_angle,
            torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1),
        )
        self.nearest_obs_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=nearest_quat)

        nearest_vec_norm = nearest_vec / (nearest_vec.norm(dim=-1, keepdim=True) + 1e-6)
        robot_vec_norm = robot_heading_vector / (robot_heading_vector.norm(dim=-1, keepdim=True) + 1e-6)
        cosine_similarity = torch.nn.functional.cosine_similarity(nearest_vec_norm, robot_vec_norm, dim=1, eps=1e-8)

        sigma = 1.5
        gaussian_factor = 1 / (0.1 * torch.sqrt(2 * torch.tensor(torch.pi)))
        potential = 0.25 * gaussian_factor * torch.exp(-nearest_dist**2 / (2 * sigma**2))
        rew_potential_pa = potential * (1.0 + cosine_similarity) / 2.0

        nearest_vec_safe = torch.nan_to_num(nearest_vec, nan=0.0, posinf=self.lidar_range, neginf=-self.lidar_range)
        nearest_vec_norm2 = nearest_vec_safe / (nearest_vec_safe.norm(dim=-1, keepdim=True).clamp(min=0.1))
        vel_w_safe = self._robot.data.root_lin_vel_w.clamp(-50.0, 50.0)
        vel_toward_obs = torch.sum(vel_w_safe * nearest_vec_norm2, dim=1)
        obs_goal_alignment = torch.sum(nearest_vec_norm2 * unit_relative_err_pos, dim=1)
        repulsion_mask = (obs_goal_alignment < 0.7).float()
        rew_vel_repulsion = -potential * vel_toward_obs.clamp(min=0) * repulsion_mask

        goal_reached = (distance_to_goal < 1.5).float()

        rewards = {
            "rew_ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "rew_distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "rew_action_rate": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "rew_velocity_dir": rew_vel_dir_w * self.cfg.velocity_direction * self.step_dt,
            "rew_height_penalty": -penalty_height * 10.0 * self.step_dt,
            "rew_reward_safety_static": reward_safety_static * self.cfg.reward_safety_static * self.step_dt,
            "rew_proximity_penalty": -proximity_penalty * self.cfg.proximity_penalty_scale * self.step_dt,
            "rew_collision_terminal": -collision_terminal * self.cfg.collision_penalty_scale,
            "rew_goal_reach": goal_reached * self.cfg.goal_reach_bonus,
            "rew_potential_field_PA": rew_potential_pa * self.cfg.potential_field_PA * self.step_dt,
            "rew_heading_tracking_PA": head_tracking_path_rew * self.cfg.head_tracking_PA * self.step_dt,
            "rew_vel_repulsion": rew_vel_repulsion * self.cfg.vel_repulsion_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        self.previous_action = self._actions.clone()

        for key, value in rewards.items():
            self._episode_sums[key] += torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.3,
            self._robot.data.root_pos_w[:, 2] > 4.0,
        )

        static_collision = einops.reduce(self.lidar_scan, "n 1 w -> n 1", "min") < 0.3

        relative_err_pos_w = self._desired_pos_w - self._robot.data.root_pos_w
        ref_heading = torch.atan2(relative_err_pos_w[:, 1], relative_err_pos_w[:, 0])
        angle_diff = ref_heading - self._robot.data.heading_w
        angle_diff = (angle_diff + torch.pi) % (2 * torch.pi) - torch.pi
        opposite_direction_heading = torch.abs(angle_diff) > 2.618

        died = died | static_collision.squeeze(1) | opposite_direction_heading
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_dist = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras: dict = {}
        for key in self._episode_sums.keys():
            extras["Episode_Reward/" + key] = (
                torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = extras
        self.extras["log"].update({
            "Episode_Termination/died": torch.count_nonzero(self.reset_terminated[env_ids]).item(),
            "Episode_Termination/time_out": torch.count_nonzero(self.reset_time_outs[env_ids]).item(),
            "Metrics/final_distance_to_goal": final_dist.item(),
        })

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self.previous_action[env_ids] = 0.0

        # Goal
        self._desired_pos_w[env_ids, :2] = (
            torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-8.0, 8.0)
            + self._terrain.env_origins[env_ids, :2]
        )
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(1.5, 1.6)

        desired_heights = self._desired_pos_w[:, 2]
        margin = 0.20
        self.height_range = torch.stack([desired_heights - margin, desired_heights + margin], dim=-1)

        # Robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ── Debug visualisation ──────────────────────────────────────────────────

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

    def define_markers(self) -> VisualizationMarkers:
        """Goal-direction arrow (RED)."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/DynObs/GoalDirMarkers",
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
        """Robot current-heading arrow (GREEN)."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/DynObs/HeadingMarkers",
            markers={
                "arrow_x": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.1, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)

    def define_nearest_obs_markers(self) -> VisualizationMarkers:
        """Nearest-obstacle direction arrow (BLUE)."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/DynObs/NearestObsMarkers",
            markers={
                "arrow_x": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.1, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)
