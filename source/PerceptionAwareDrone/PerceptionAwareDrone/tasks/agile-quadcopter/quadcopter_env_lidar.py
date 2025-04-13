# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Claus

# from __future__ import annotations

# import gymnasium as gym
# import torch

# import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation, ArticulationCfg
# from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
# from isaaclab.envs.ui import BaseEnvWindow
# from isaaclab.markers import VisualizationMarkers
# from isaaclab.scene import InteractiveSceneCfg
# from isaaclab.sim import SimulationCfg
# from isaaclab.terrains import TerrainImporterCfg
# from isaaclab.utils import configclass
# from isaaclab.utils.math import subtract_frame_transforms, quat_apply_yaw, euler_xyz_from_quat


# ##
# # Pre-defined configs
# ##
# # from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
# from .robot.agileDrone import AGILE_CFG  # isort: skip
# from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


# # Guiding path generation
# import matplotlib.pyplot as plt
# from scipy.interpolate import CubicSpline
# from isaacsim.util.debug_draw import _debug_draw

# # sensor
# from isaaclab.sensors import RayCasterCfg, RayCaster, patterns
# from isaaclab.sensors import Imu, ImuCfg

# #terrain
# from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG, OBSTACLE_RAND_POS

# import isaaclab.envs.mdp as mdp    
# from isaaclab.managers import EventTermCfg as EventTerm
# from isaaclab.managers import SceneEntityCfg

# # viewpoint
# from isaaclab.envs.ui  import ViewportCameraController
# from isaaclab.envs import ViewerCfg

# # lidar
# import einops


# @configclass
# class EventCfg:
#     add_base_mass = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
#             "mass_distribution_params": (-0.50, 0.5),
#             "operation": "add",
#         },
#     )
#     # start_position = EventTerm(
#     #     func=mdp.reset_root_state_uniform,
#     #     mode="startup",
#     #     params={
#     #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
#     #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.9, 1.1)},
#     #     },
#     # )
#     # push_force_body = EventTerm(
#     #     func=mdp.apply_external_force_torque,
#     #     mode="interval",
#     #     params={
#     #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
#     #         "force": (0, 0, 0.5),
#     #         "torque": (0, 0, 0),
#     #         "operation": "add",
#     #     },
    


# class QuadcopterEnvWindow(BaseEnvWindow):
#     """Window manager for the Quadcopter environment."""

#     def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
#         """Initialize the window.

#         Args:
#             env: The environment object.
#             window_name: The name of the window. Defaults to "IsaacLab".
#         """
#         # initialize base window
#         super().__init__(env, window_name)
#         # add custom UI elements
#         with self.ui_window_elements["main_vstack"]:
#             with self.ui_window_elements["debug_frame"]:
#                 with self.ui_window_elements["debug_vstack"]:
#                     # add command manager visualization
#                     self._create_debug_vis_ui_element("targets", self.env)
        
        

# @configclass
# class QuadcopterEnvCfg(DirectRLEnvCfg):
#     # env
#     episode_length_s = 30.0
#     decimation = 2
#     action_space = 4
#     # observation_space = 12
#     observation_space = 15  # add the guilding path
#     # observation_space = 18  # add the guilding path + attitude
#     # observation_space = 20  # add the guilding path + attitude + lidar

#     state_space = 0
#     debug_vis = True

#     ui_window_class_type = QuadcopterEnvWindow

#     viewer = ViewerCfg(eye=(-19.8, -23.8, 11.5), lookat=(-24.0, -8.5, -1.7), origin_type='env', env_index=2015)
    
#     # simulation
#     sim: SimulationCfg = SimulationCfg(
#         dt=1 / 100,
#         render_interval=decimation,
#         physics_material=sim_utils.RigidBodyMaterialCfg(
#             friction_combine_mode="multiply",
#             restitution_combine_mode="multiply",
#             static_friction=1.0,
#             dynamic_friction=1.0,
#             restitution=0.0,
#         ),
#     ) 

#     flat_terrain = False  # for generator terrain
#     # flat_terrain = True
#     if flat_terrain:
#         # for flat and emtry terrain
#         terrain = TerrainImporterCfg(
#             prim_path="/World/ground",
#             terrain_type="plane",
#             collision_group=-1,
#             physics_material=sim_utils.RigidBodyMaterialCfg(
#                 friction_combine_mode="multiply",
#                 restitution_combine_mode="multiply",
#                 static_friction=1.0,
#                 dynamic_friction=1.0,
#                 restitution=0.0,
#             ),
#             debug_vis=False,
#         )
#     else:
#         # for custom terrain
#         terrain = TerrainImporterCfg(
#             prim_path="/World/ground",
#             terrain_type="generator",
#             terrain_generator=ROUGH_TERRAINS_CFG,
#             max_init_terrain_level=9,
#             collision_group=-1,
#             physics_material=sim_utils.RigidBodyMaterialCfg(
#                 friction_combine_mode="multiply",
#                 restitution_combine_mode="multiply",
#                 static_friction=1.0,
#                 dynamic_friction=1.0,
#             ),
#             # visual_material=sim_utils.MdlFileCfg(
#             #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
#             #     project_uvw=True,
#             # ),
#             debug_vis=True,
#         )

#     # scene
#     scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.5, replicate_physics=True)

#     # events
#     events: EventCfg = EventCfg()

#     # robot
#     robot: ArticulationCfg = AGILE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

#     # sensor 
#     lidar_sensor = RayCasterCfg(
#         prim_path="/World/envs/env_.*/Robot/base_link",
#         offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),
#         attach_yaw_only=False,
#         pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(10.0, 20.0), horizontal_fov_range=(-90.0, 90.0),horizontal_res=36.0),     
#         # pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(10.0, 20.0), horizontal_fov_range=(-48.0, 48.0),horizontal_res=3.0),   
#         debug_vis=False,
#         mesh_prim_paths=["/World/ground"],
#     )


#     # spec for agile drone
#     thrust_to_weight = 5.0
#     moment_scale = 0.35
#     # thrust_to_weight = 1.9
#     # moment_scale = 0.01

#     # reward scales
#     lin_vel_reward_scale = -0.5
#     # lin_vel_reward_scale = -0.05
#     ang_vel_reward_scale = -0.05
#     # ang_vel_reward_scale = -0.05

#     distance_to_goal_reward_scale = 1.5  # 15.0
#     distance_to_guide_reward_scale = 2.0  # 20.0
#     # heading_tracking_reward_scale = -0.5
#     heading_tracking_reward_scale = 1.5  # 15.0
#     potential_rew_scale = 1.5  # 15.0  #9.5  

    
# def normalize_angle(x):
#     return torch.atan2(torch.sin(x), torch.cos(x))

# class QuadcopterEnv(DirectRLEnv):
#     cfg: QuadcopterEnvCfg

#     def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
#         super().__init__(cfg, render_mode, **kwargs)

#         # Total thrust and moment applied to the base of the quadcopter
#         self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
#         self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
#         self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
#         # Goal position
#         self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

#         # Logging
#         self._episode_sums = {
#             key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
#             for key in [
#                 "lin_vel",
#                 "ang_vel",
#                 "distance_to_goal",
#                 # "distance_to_guide",
#                 # "heading_tracking",
#                 # "potential_rew",
#                 "action_rate",
#             ]
#         }
#         # Get specific body indices
#         self._body_id = self._robot.find_bodies("base_link")[0] 
#         self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
#         print("\n\n robot mass", self._robot_mass)
#         self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
#         self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

#         # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
#         self.set_debug_vis(self.cfg.debug_vis)

        

#         ## Guiding path generation
#         # self.future_traj_step = 4
#         # self.target_pos = torch.zeros(self.num_envs, self.future_traj_step, 3, device=self.device)
#         # self.guilding_target = torch.zeros(self.num_envs, self.future_traj_step, 3, device=self.device)
#         # self.distance_to_guide = torch.zeros(self.num_envs, device=self.device)

#         # self.guilding_planner = PotentialFieldPlanner(env_origins=self._terrain.env_origins, num_envs = self.num_envs, device=self.device)
#         # self.guilding_planner = PotentialFieldPlanner(env_origins=self._terrain.env_origins, num_envs = self.num_envs, obstacles=OBSTACLE_RAND_POS , device=self.device)


#         # self._desired_goal = torch.zeros(self.num_envs, 3, device=self.device)
        
#         # self.goal_pos = torch.tensor((0.2, 20.0, 1.0), dtype=torch.float32, device=self.device)
#         # self._desired_goal = self.goal_pos.repeat(self.num_envs, 1) + self._terrain.env_origins[:, :]


#         # self.target_path = self.guilding_planner.run(start=(0, 0, 1), goal=self.goal_pos)
        
#         self.previous_action = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

#         self.lidar_range = 5.0


#     def _setup_scene(self):
#         self._robot = Articulation(self.cfg.robot)
#         self.scene.articulations["robot"] = self._robot

#         self._lidar_sensor = RayCaster(self.cfg.lidar_sensor)
#         self.scene.sensors["lidar_sensor"] = self._lidar_sensor

#         self.cfg.terrain.num_envs = self.scene.cfg.num_envs
#         self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
#         self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
#         # clone and replicate
#         self.scene.clone_environments(copy_from_source=False)
#         # add lights
#         light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
#         light_cfg.func("/World/Light", light_cfg)



#     def _pre_physics_step(self, actions: torch.Tensor):
#         self._actions = actions.clone().clamp(-1.0, 1.0)
#         self.previous_action = self._actions.clone()
#         self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
#         self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

#     def _apply_action(self):
#         self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

#     def _get_observations(self) -> dict:
#         # self.target_pos = self.guilding_planner.compute_shortest_traj(input_path = self.target_path, eps_pro=self.episode_length_buf , steps = self.future_traj_step, step_size = 5)
#         # self.target_pos = self.guilding_planner.compute_shortest_traj(input_path=self.target_path, eps_pro=self.episode_length_buf , steps=self.future_traj_step, step_size=5)
#         # print("target_pos_0 = ", self.target_pos[0, :, :].shape)
#         # self.guilding_target = self.guilding_planner.compute_shortest_traj(input_path=self.target_path, eps_pro=self.episode_length_buf , steps=self.future_traj_step, step_size=5)
#         # original goal command
#         desired_pos_b, _ = subtract_frame_transforms(
#             self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
#         )
#         # _____________________
#         # desired_pos_b, _ = subtract_frame_transforms(
#         #     self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_goal
#         # )
#         # guilding_pos_b, _ = subtract_frame_transforms(
#         #     self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self.guilding_target[:, 0, :]
#         # )

#         # self.robot_orientaion = self._robot.data.root_quat_w
#         # self.robot_orientaion_euler = euler_xyz_from_quat(self._robot.data.root_quat_w)
#         # self.robot_orientaion_euler = torch.stack(self.robot_orientaion_euler, dim=1)

#         self.drone_pos = self._robot.data.root_state_w[:, 0:3]
        
        
#         self.lidar_resolution = (5)
#         self.lidar_scan = ((self._lidar_sensor.data.ray_hits_w - self._lidar_sensor.data.pos_w.unsqueeze(1)).norm(dim=-1).clamp_max(self.lidar_range).reshape(self.num_envs, 1, self.lidar_resolution))
#         print(self.lidar_scan.squeeze(1)[2015])



#         # with_orient = False
#         # if with_orient:
#         #     obs = torch.cat(
#         #         [
#         #             self._robot.data.root_lin_vel_b,
#         #             self._robot.data.root_ang_vel_b,
#         #             desired_pos_b,
#         #             self.robot_orientaion_euler,
#         #             self._robot.data.projected_gravity_b,
#         #             guilding_pos_b,
#         #         ],
#         #         dim=-1,
#         #     )
#         # else:
#         obs = torch.cat(
#             [
#                 # self._robot.data.root_lin_vel_b,
#                 # self._robot.data.root_ang_vel_b,
#                 # self._robot.data.projected_gravity_b,
#                 # desired_pos_b,
#                 # guilding_pos_b,
#                 self.drone_pos,
#                 self._robot.data.root_lin_vel_b,
#                 self._robot.data.root_ang_vel_b,
#                 self._robot.data.projected_gravity_b,
#                 desired_pos_b,
#                 # guilding_pos_b,
#                 # self.lidar_scan.squeeze(1),
                
#             ],
#             dim=-1,
#         )
#         observations = {"policy": obs}
#         return observations

#     def _get_rewards(self) -> torch.Tensor:
#         # print("target_pos_0 = ", self.target_pos.shape)

#         lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
#         ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

#         distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
#         # distance_to_goal = torch.linalg.norm(self._desired_goal - self._robot.data.root_pos_w, dim=1)
#         distance_to_goal_mapped_rew = 1 - torch.tanh(distance_to_goal / 0.8)
#         # print("distance_to_goal_mapped shape",distance_to_goal_mapped.shape)

#         self.robot_heading_vector = quat_apply_yaw(self._robot.data.root_state_w[:, 3:7].to(torch.float32), torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1))
#         # print("robot heading vec shape = ", robot_heading_vector[:5])
#         # robot_pos_local = self._robot.data.root_pos_w - self._terrain.env_origins
#         # potential_rew = self.guilding_planner.potentialReward(robot_pos=robot_pos_local, robot_heading_vec=self.robot_heading_vector)
        
#         # self.guilding_target = self.guilding_planner.compute_shortest_traj(input_path=self.target_path, eps_pro=self.episode_length_buf , steps=self.future_traj_step, step_size=5)
#         # self.distance_to_guide = torch.linalg.norm(self.guilding_target[:, 0, :] - self._robot.data.root_pos_w[:, :], dim=1)
#         # guilding_path_rew = 1 - torch.tanh(self.distance_to_guide / 0.4)
#         # print("guilding_path_rew shape",guilding_path_rew.shape)

#         # self.next_point_path = self.guilding_target[:, 1, :2] - self.guilding_target[:, 0, :2]   # next point - current point (4096,2) {x,y}
#         # self.ref_heading = torch.atan2(self.next_point_path[:, 1], self.next_point_path[:, 0])  # radian
#         # self.robot_heading = self._robot.data.heading_w
#         # self.angle_diff = self.ref_heading - self.robot_heading
#         # angle_diff = self.ref_heading - self.robot_heading
#         # head_tracking_path_rew = torch.tanh(torch.abs(angle_diff) / 0.8)
#         # head_tracking_path_rew = 1 - torch.tanh(torch.abs(self.angle_diff) / 0.8)

#         action_rate = torch.sum(torch.square(self._actions - self.previous_action), dim=1)
        
#         # print("range  ", self.lidar_range)
#         # print("lidar_scan  ", self.lidar_scan)
#         # print("range - lidar =  ", (self.lidar_range - self.lidar_scan))
#         # print("clamp =  ", (self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range))
#         # print("data ", (self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range).shape)
#         # print("data ", (self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range).mean(dim=2)[2015])
#         reward_safety_static = torch.log((self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=2)
#         # print("reward_safety_static shape", reward_safety_static.shape)
#         # print("reward_safety_static", reward_safety_static[2015])
        
#         rewards = {
#             "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
#             "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
#             "distance_to_goal": distance_to_goal_mapped_rew * self.cfg.distance_to_goal_reward_scale * self.step_dt,
#             # "distance_to_guide": guilding_path_rew * self.cfg.distance_to_guide_reward_scale * self.step_dt,
#             # "heading_tracking": head_tracking_path_rew * self.cfg.heading_tracking_reward_scale * self.step_dt,
#             # "potential_rew": potential_rew * self.cfg.potential_rew_scale * self.step_dt,
#             "action_rate": action_rate * -0.1 * self.step_dt,
#         }
#         reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
#         # Logging
#         for key, value in rewards.items():
#             self._episode_sums[key] += value
#         return reward

#     def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
#         ones = torch.ones_like(self.reset_buf)
#         died = torch.zeros_like(self.reset_buf)
        
#         time_out = self.episode_length_buf >= self.max_episode_length - 1
#         died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.5)
#         # print("died1  shape = ", died.sum().item())
#         # died = torch.where(self.distance_to_guide > 0.30, ones, died)  #0.25

#         print("dead1  shape = ", died.shape)

#         # static_collision = einops.reduce(self.lidar_scan, "n 1 w -> n 1", "max") < (self.lidar_range - (self.lidar_range - 0.3))  # 0.3 collision radius
#         # print(static_collision.squeeze(1).shape)

#         # died = died | static_collision.squeeze(1)

#         # print("heading", self._robot.data.heading_w[2015])
#         # died = torch.where(torch.abs(self.angle_diff) > 1.3, ones, died)
#         # died = torch.where((self.ref_heading - self.robot_heading) 
#         # print("\n shape2 = ", torch.any(torch.abs(self.distance_to_guide) > 0.15).shape)
#         # died = died | (torch.abs(self.distance_to_guide) < 1.0).any(dim=-1, keepdim=True)

#         # print("died2  shape = ", died.sum().item())
#         # print("time_out  shape = ", time_out.sum().item())


#         #TODO Terminate drone when its facing opposite to the target
#         # died = torch.where(torch.abs(self.robot_heading) > 1.5, ones, died)

#         return died, time_out

#     def _reset_idx(self, env_ids: torch.Tensor | None):
#         if env_ids is None or len(env_ids) == self.num_envs:
#             env_ids = self._robot._ALL_INDICES

#         # Logging
#         # final_distance_to_goal = torch.linalg.norm(
#         #     self._desired_goal[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
#         # ).mean()
#         final_distance_to_goal = torch.linalg.norm(
#             self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
#         ).mean()
#         extras = dict()
#         for key in self._episode_sums.keys():
#             episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
#             extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
#             self._episode_sums[key][env_ids] = 0.0
#         self.extras["log"] = dict()
#         self.extras["log"].update(extras)
#         extras = dict()
#         extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
#         extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
#         extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
#         self.extras["log"].update(extras)

#         self._robot.reset(env_ids)
#         super()._reset_idx(env_ids)
#         if len(env_ids) == self.num_envs:
#             # Spread out the resets to avoid spikes in training when many environments reset at a similar time
#             self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

#         self._actions[env_ids] = 0.0
#         self.previous_action[env_ids] = 0.0
        
#         # Sample new commands
#         self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
#         self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
#         self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

#         # self.goal_pos[0] = torch.zeros_like(self.goal_pos[0]).uniform_(-5.0, 5.0)
#         # self.goal_pos[1] = torch.zeros_like(self.goal_pos[1]).uniform_(5.0, 20.0)
#         # self.goal_pos[2] = torch.ones_like(self.goal_pos[2])
#         # self._desired_goal = self.goal_pos.repeat(self.num_envs, 1) + self._terrain.env_origins[:, :]
#         # self.target_path = self.guilding_planner.run(start=(0, 0, 1), goal=self.goal_pos)

#         # Reset robot state
#         joint_pos = self._robot.data.default_joint_pos[env_ids]
#         joint_vel = self._robot.data.default_joint_vel[env_ids]
#         default_root_state = self._robot.data.default_root_state[env_ids]
#         default_root_state[:, :3] += self._terrain.env_origins[env_ids]
#         self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
#         self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
#         self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

#     def _set_debug_vis_impl(self, debug_vis: bool):
#         # create markers if necessary for the first tome
#         if debug_vis:
#             if not hasattr(self, "goal_pos_visualizer"):
#                 marker_cfg = CUBOID_MARKER_CFG.copy()
#                 marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
#                 marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
#                 # -- goal pose
#                 marker_cfg.prim_path = "/Visuals/Command/goal_position"
#                 self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)  

#                 targer_cfg = CUBOID_MARKER_CFG.copy()
#                 targer_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
#                 targer_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
#                 # -- goal pose
#                 targer_cfg.prim_path = "/Visuals/Command/target_position"
#                 self.target_visualizer = VisualizationMarkers(targer_cfg)
#             # set their visibility to true
#             self.target_visualizer.set_visibility(True)

#             # set their visibility to true
#             self.goal_pos_visualizer.set_visibility(True)
#         else:
#             if hasattr(self, "goal_pos_visualizer"):
#                 self.goal_pos_visualizer.set_visibility(False)
#                 self.target_visualizer.set_visibility(False)

#     def _debug_vis_callback(self, event):
#         # update the markers
#         # self.goal_pos_visualizer.visualize(self._desired_goal)
#         self.goal_pos_visualizer.visualize(self._desired_pos_w)
#         # self.target_visualizer.visualize(self.guilding_target[:, 0, :])

# # (0, 4.5, 1)
# # (-3, 8, 1)
# class PotentialFieldPlanner:
#     def __init__(self, map_size=(-10, 10), obstacles=[(-0.5, 4.5, 1), (-3, 8, 1), (2, 10, 1)], obstacle_radius=1.2,  #1.5,
#                  attractive_gain=0.7, repulsive_gain=3000.0, step_size=0.025, max_iters=10000, num_envs = 4096, env_origins = None, device = 'cuda', progress_buf=None):
#                 #  attractive_gain=0.7, repulsive_gain=6000.0, step_size=0.025, max_iters=10000, num_envs = 4096, env_origins = None, device = 'cuda', progress_buf=None):
#         self.device = device
#         self.map_size = map_size
#         self.obstacle_radius = obstacle_radius
#         self.attractive_gain = attractive_gain
#         self.repulsive_gain = repulsive_gain
#         self.step_size = step_size
#         self.max_iters = max_iters
#         self.goal = torch.tensor((0.5, 15.5), dtype=torch.float32)  # Default goal
#         # print("\n\n\n\n\n len obstacles ", len(obstacles))
#         self.obstacles = [torch.tensor(obs, dtype=torch.float32, device=self.device) for obs in obstacles]
        
#         self.future_traj_steps = 4

#         self.draw = _debug_draw.acquire_debug_draw_interface()
#         self.draw.clear_lines()

#         # env information
#         # self.progress_buf = progress_buf
#         self.num_envs = num_envs
#         self.env_origin_pos = env_origins
#         self.central_env_idx = self.env_origin_pos.norm(dim=-1).argmin()  
#         self.origin = torch.tensor([0.0, 0.0, 0.0], device=self.device) 

        

#     def set_goal(self, goal):
#         self.goal = torch.tensor(goal, dtype=torch.float32, device=self.device)

#     # def _compute_potential_gradient(self, pos):
#     #     pos = torch.tensor(pos, dtype=torch.float32)
#     #     att_grad = self.attractive_gain * (pos - self.goal)
#     #     rep_grad = torch.zeros(2, dtype=torch.float32)

#     #     for obs in self.obstacles:
#     #         d = torch.norm(pos - obs)
#     #         if d < self.obstacle_radius:
#     #             rep_grad += self.repulsive_gain * (1.0 / d**2 - 1.0 / self.obstacle_radius**2) * (pos - obs) / d**3

#     #     total_grad = att_grad - rep_grad
#     #     return total_grad

#     def _compute_potential_gradient(self, pos):
#         pos = pos.to(self.device)
#         att_grad = self.attractive_gain * (pos - self.goal)  # 3D Attraction
#         rep_grad = torch.zeros(3, dtype=torch.float32, device=self.device)

#         for obs in self.obstacles:
#             d = torch.norm(pos - obs)
#             if d < self.obstacle_radius:
#                 rep_grad += self.repulsive_gain * (1.0 / d**2 - 1.0 / self.obstacle_radius**2) * (pos - obs) / d**3

#         total_grad = att_grad - rep_grad
#         return total_grad
    
#     def _apply_low_pass_filter(self, path, alpha=0.1):
#         filtered_path = torch.clone(path)
#         for i in range(1, len(path)):
#             filtered_path[i] = alpha * path[i] + (1 - alpha) * filtered_path[i - 1]
#         return filtered_path
    
#     def resample_cycle_torch_linear(self, cycle_tensor: torch.Tensor, num_points: int) -> torch.Tensor:
#         """
#         Pure PyTorch implementation of linear interpolation for trajectory resampling.

#         Args:
#             cycle_tensor: (N, D) tensor, where N is original number of points, D is feature dimension (e.g., 3 for XYZ).
#             num_points: Number of points to resample to.

#         Returns:
#             Resampled tensor of shape (num_points, D).
#         """
#         N, D = cycle_tensor.shape
#         device = cycle_tensor.device
#         dtype = cycle_tensor.dtype

#         # Original and target normalized indices (0 to 1)
#         original_idx = torch.linspace(0, 1, steps=N, device=device, dtype=dtype)
#         new_idx = torch.linspace(0, 1, steps=num_points, device=device, dtype=dtype)

#         # Find indices in original_idx where each new_idx would be inserted
#         idxs = torch.searchsorted(original_idx, new_idx, right=True)
#         idxs = torch.clamp(idxs, 1, N - 1)

#         left = idxs - 1
#         right = idxs

#         left_x = original_idx[left]
#         right_x = original_idx[right]
#         left_y = cycle_tensor[left]
#         right_y = cycle_tensor[right]

#         # Linear interpolation weights
#         weights = (new_idx - left_x) / (right_x - left_x)
#         weights = weights.unsqueeze(1)  # Shape (num_points, 1)

#         interpolated = left_y + weights * (right_y - left_y)
#         return interpolated
    
#     def resample_by_arclength(self, points: torch.Tensor, num_points: int) -> torch.Tensor:
#         """
#         Resample a trajectory to have uniform spatial spacing using arc-length.
        
#         Args:
#             points: (N, D) torch tensor representing the trajectory (e.g., N points in 3D space).
#             num_points: Number of points in the resampled trajectory.

#         Returns:
#             (num_points, D) torch tensor with uniform spatial density.
#         """
#         device = points.device
#         dtype = points.dtype
#         N, D = points.shape

#         # 1. Compute distances between consecutive points
#         deltas = points[1:] - points[:-1]
#         segment_lengths = torch.norm(deltas, dim=1)

#         # 2. Compute cumulative arc-length
#         arc_lengths = torch.cat([torch.zeros(1, device=device, dtype=dtype), torch.cumsum(segment_lengths, dim=0)])
#         total_length = arc_lengths[-1]

#         # 3. Generate new equally spaced arc-length positions
#         target_lengths = torch.linspace(0, total_length, num_points, device=device, dtype=dtype)

#         # 4. Find corresponding segments for interpolation
#         idxs = torch.searchsorted(arc_lengths, target_lengths, right=True)
#         idxs = torch.clamp(idxs, 1, N - 1)
#         left = idxs - 1
#         right = idxs

#         # 5. Interpolate positions
#         left_points = points[left]
#         right_points = points[right]
#         left_l = arc_lengths[left]
#         right_l = arc_lengths[right]

#         t = (target_lengths - left_l) / (right_l - left_l)
#         t = t.unsqueeze(1)  # (num_points, 1)

#         resampled_points = left_points + t * (right_points - left_points)
#         return resampled_points
    
#     def find_path(self, start):
#         path = [torch.tensor(start, dtype=torch.float32, device=self.device)]
#         pos = torch.tensor(start, dtype=torch.float32, device=self.device)

#         for _ in range(self.max_iters):
#             grad = self._compute_potential_gradient(pos)
#             grad_norm = torch.norm(grad)
#             if grad_norm < 1e-3:
#                 break
#             next_pos = pos - self.step_size * grad / grad_norm
            
#             if torch.norm(next_pos - self.goal) < self.step_size:
#                 path.append(self.goal)
#                 break

#             next_pos[0] = torch.clamp(next_pos[0], self.map_size[0]*2, self.map_size[1]*2)  # X
#             next_pos[1] = torch.clamp(next_pos[1], 0, self.map_size[1] * 10)  # Y
#             next_pos[2] = torch.clamp(next_pos[2], 0, 10)  # Z (adjust as needed)
            
#             path.append(next_pos)
#             pos = next_pos

#         path = torch.stack(path)
#         path_filtered = self._apply_low_pass_filter(path).to(device=self.device)
        
#         # print("path_filtered shape = ", path_filtered.shape)

#         resampled_path = self.resample_by_arclength(path_filtered, num_points=round(path_filtered.shape[0] * 1.55))
#         # print("resampled_path shape = ", resampled_path.shape)

#         return resampled_path
#         # return path_filtered

#     def plot(self, start, path):
#         plt.figure(figsize=(8, 8))
#         for obs in self.obstacles:
#             obstacle_circle = plt.Circle(obs[:2].cpu().numpy(), self.obstacle_radius, color='red', alpha=0.5)
#             plt.gca().add_patch(obstacle_circle)
        
#         plt.scatter(start[0], start[1], color='green', s=100, label='Start')
#         plt.scatter(self.goal[0].cpu().numpy(), self.goal[1].cpu().numpy(), color='orange', s=100, label='Goal')
#         plt.plot(path[:, 0].cpu().numpy(), path[:, 1].cpu().numpy(), color='blue', linewidth=2, label='Path')
        
#         plt.xlim(self.map_size[0], self.map_size[1])
#         plt.ylim(0, self.map_size[1] * 2)
#         plt.legend()
#         plt.grid(True)
#         plt.show()
    
#     def compute_shortest_traj(self, input_path, eps_pro , steps: int, env_ids=None, step_size=1):
#         if env_ids is None:
#             env_ids = ...
#         # ----------------------------------------------
#         device = input_path.device  # Ensure compatibility with GPU if needed
#         # Get initial indices from eps_pro[env_ids] and reshape to match batch size
#         start_indices = eps_pro[env_ids].unsqueeze(-1).long()  # Shape [4096, 1]

#         # Compute sliding window indices
#         sliding_indices = start_indices + step_size * torch.arange(steps, device=device).unsqueeze(0)  # Shape [4096, 4]

#         # Clip indices to avoid out-of-bounds errors (ensure they are within [0, 299])
#         sliding_indices = sliding_indices.clamp(0, input_path.size(1) - 1)

#         # Extract trajectory slices
#         self.traj_target = input_path[torch.arange(input_path.size(0)).unsqueeze(1), sliding_indices]  # Shape [4096, 4, 3]

#         # print("traj_target shape = ", self.traj_target.shape)
#         # print("self.env_origin_pos shape = ", self.env_origin_pos.shape)
#         return self.env_origin_pos[:, None, :] + self.traj_target
    
#     def run(self, start, goal=None):
#         if goal is not None:
#             self.set_goal(goal)
#         path = self.find_path(start)
#         if path is not None:
#             print("Path found!")
#             print("path shape = ", path.shape)
#             path_x = path[:, 0]
#             path_y = path[:, 1]
#             path_z = path[:, 2]
#             # path_z = torch.ones_like(path_x)

#             self.path_xyz = torch.stack((path_x, path_y, path_z), dim=1)   # Combine x, y, z into a single tensor   -- > #size [path point,3]    
#             self.duplicated_path_xyz = self.path_xyz.unsqueeze(0).repeat(self.num_envs, 1, 1)   # [num_env,length_bspline,xyz]    
#             self.path_xyz = self.path_xyz.to(self.device) + self.env_origin_pos[self.central_env_idx]
#             point_list_0 = self.path_xyz[:-1].tolist()   # cut the endding point to make a line ex. whose line is 1,2,3,4; point_list_0 = 1,2,3  
#             point_list_1 = self.path_xyz[1:].tolist()    # cut the starting point to make a line ex. whose line is 1,2,3,4; point_list_1 = 2,3,4   then the line is 1-2, 2-3, 3-4

#             debug_plot = True
#             if debug_plot == True:
#                 colors = [(1.0, 1.0, 0.0, 1.0) for _ in range(len(point_list_0))]
#                 sizes = [2 for _ in range(len(point_list_0))]
#                 self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)  # draw the line
            
#             return self.duplicated_path_xyz
#             # while(True):
#             #     pass
#             # self.plot(start, path)
#         else:
#             print("Failed to find a path.")
#             return None
    
#     def _calVector(self, robot_pos, obs_pos):
#         if isinstance(obs_pos, list):
#             obs_pos = torch.stack(obs_pos).to(robot_pos.device)
#         # print("robot_pos ", robot_pos.shape)
#         # print("obs_pos ", obs_pos.shape)
#         vectors = obs_pos - robot_pos[:, None, :]  # Shape (4096, 3, 3)
#         distances = torch.norm(vectors, dim=2)
#         return vectors, distances
    
#     def _calClosestObstacle(self, robot_pos, obs_pos):

#         vectors, distances = self._calVector(robot_pos, obs_pos)

#         # Find the index of the closest obstacle
#         closest_indices = torch.argmin(distances, dim=1)  # Shape (4096,)

#         # Get the minimum distances
#         min_distances = distances[torch.arange(self.num_envs), closest_indices]  # Shape (4096,)

#         # Get the vector to the closest obstacle
#         closest_vectors = vectors[torch.arange(self.num_envs), closest_indices]  # Shape (4096, 3)
#         dx, dy, dz = closest_vectors[:, 0], closest_vectors[:, 1], closest_vectors[:, 2]

#         # Convert to Euler angles
#         yaw = torch.atan2(dy, dx)  # Rotation around Z-axis
#         pitch = torch.atan2(dz, torch.sqrt(dx**2 + dy**2))  # Rotation around Y-axis

#         # Stack yaw and pitch into a (4096, 2) tensor
#         euler_angles = torch.stack((yaw, pitch), dim=1)  # Shape (4096, 2)

#         return min_distances, euler_angles , closest_vectors
        
#     def _calPotential(self, robot_pos, obs_pos, weight_potential):
#         """
#         Compute the potential field from obstacles.

#         Args:
#         - robot_pos: Tensor of shape (4096, 3).
#         - obs_pos: Tensor of shape (4096, 3, 3).
#         - weight_potential: Scalar weight for the potential.

#         Returns:
#         - potential: Tensor of shape (4096), potential field values for each obstacle.
#         """
#         # print("robot_pos ", robot_pos[self.central_env_idx])
#         # print("obs_pos ", obs_pos[self.central_env_idx])
#         vec, euler , dist_vec = self._calClosestObstacle(robot_pos, obs_pos)
#         dist = torch.norm(dist_vec, dim=1)

#         sigma = 1.5  # Standard deviation of Gaussian function
#         gaussian_factor = 1 / (0.1 * torch.sqrt(2 * torch.tensor(torch.pi)))  # Precomputed constant
#         # print("dist shape", dist.shape)
#         potential = weight_potential * gaussian_factor * torch.exp(-dist**2 / (2 * sigma**2))
#         # print("potential shape ", potential.shape)
#         # print("potential ", potential)
#         # print("dist ", dist[self.central_env_idx])
        
#         #for debug potential
#         # if(dist[self.central_env_idx] < 2.0):
#         #     print("dist ", dist[self.central_env_idx])
#         #     print("potential ", potential[self.central_env_idx])

#         # potential = torch.min(potential, torch.tensor(0.0, device=potential.device))
#         return potential

#     def _dot_vector(self, a, b):
#         return torch.sum(a * b, dim=1)
    
#     def _compute_unit_vector(self, vectors):
#         """
#         Compute the unit vector of a given tensor.

#         Args:
#         - vectors: torch.Tensor of shape (..., 3), representing 3D vectors.

#         Returns:
#         - unit_vectors: torch.Tensor of the same shape (..., 3), normalized to unit length.
#         """
#         # Compute vector magnitude
#         norms = torch.norm(vectors, dim=-1, keepdim=True)  # Shape (..., 1)

#         # Avoid division by zero by replacing zeros with a small value
#         norms = torch.where(norms == 0, torch.tensor(1e-8, device=vectors.device), norms)

#         # Compute unit vector
#         unit_vectors = vectors / norms  # Shape (..., 3)
#         return unit_vectors

#     def potentialReward(self, robot_pos, robot_heading_vec, weight_potential=1.0):
#         _ , euler_angles, closet_vector = self._calClosestObstacle(robot_pos, self.obstacles)
        
#         norm_robot_heading = self._compute_unit_vector(robot_heading_vec)
#         norm_closet_obs_vec = self._compute_unit_vector(closet_vector)

#         dot_product = self._dot_vector(norm_robot_heading, norm_closet_obs_vec)

#         # print("euler_angles = ", euler_angles)
#         # print("robot_heading = ", robot_heading)
#         potential = self._calPotential(robot_pos, self.obstacles, weight_potential)
#         # print("potential ", potential.shape)
#         # print("dot_product ", dot_product.shape)

#         reward = potential * dot_product
#         # print("reward ", reward[self.central_env_idx])
#         return reward

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
from isaaclab.utils.math import subtract_frame_transforms

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
    observation_space = 72
    # observation_space = 17 #with 5 beams lidar
    # observation_space = 12
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow
    
    viewer = ViewerCfg(eye=(-19.8, -23.8, 11.5), lookat=(-24.0, -8.5, -1.7), origin_type='env', env_index=2015)

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
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(10.0, 20.0), horizontal_fov_range=(-90.0, 90.0),horizontal_res=3.0),     
        # pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(10.0, 20.0), horizontal_fov_range=(-48.0, 48.0),horizontal_res=3.0),   
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    thrust_to_weight = 5.0
    moment_scale = 0.7

    # reward scales
    lin_vel_reward_scale = -0.5
    ang_vel_reward_scale = -0.05
    distance_to_goal_reward_scale = 15.0
    action_rate_reward_scale = -0.5
    velocity_direction = 15.0
    reward_safety_static = 10.0
    head_tracking = 20.0


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

        self.lidar_scan = ((self._lidar_sensor.data.ray_hits_w - self._lidar_sensor.data.pos_w.unsqueeze(1)).norm(dim=-1).clamp_max(self.lidar_range).reshape(self.num_envs, 1, self.lidar_resolution))
        # print(self.lidar_scan.squeeze(1)[2015])
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                self.lidar_scan.squeeze(1),
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
        # print(self._robot.data.root_lin_vel_b)
        # print(unit_relative_err_pos)
        rew_vel_dir = self._robot.data.root_lin_vel_w * unit_relative_err_pos
        # print("rew vel = ", rew_vel_dir)
        rew_vel_dir = torch.sum(rew_vel_dir, dim=-1)
        # print("reward vel = ", rew_vel_dir)
        # print("reward vel shape = ", rew_vel_dir.shape)

        # lidar safety reward
        reward_safety_static = 1 - torch.tanh((self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=2).squeeze(1)
        
        # heading tracking reward
        self.ref_heading = torch.atan2(relative_err_pos[:, 1], relative_err_pos[:, 0])  # radian
        self.robot_heading = self._robot.data.heading_w
        self.angle_diff = self.ref_heading - self.robot_heading
        head_tracking_path_rew = 1 - torch.tanh(torch.abs(self.angle_diff) / 0.5)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "action_rate": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "rew_velocity_dir": rew_vel_dir * self.cfg.velocity_direction * self.step_dt,
            "reward_safety_static": reward_safety_static * self.cfg.reward_safety_static * self.step_dt,
            "head_tracking": head_tracking_path_rew * self.cfg.head_tracking * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        self.previous_action = self._actions.clone()

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.3, self._robot.data.root_pos_w[:, 2] > 3.0)
        # reach_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1) < 0.25
        # died = died | reach_goal
        # print("dead shape", died.shape)

        # print("lidar dist",einops.reduce(self.lidar_scan, "n 1 w -> n 1", "min"))
        static_collision = einops.reduce(self.lidar_scan, "n 1 w -> n 1", "min") < 0.5  # 0.3 collision radius
        # print("static_collision", static_collision.squeeze(1))
        # num_collisions = static_collision.sum()

        # If you want it as a Python int:
        # num_collisions = static_collision.sum().item()
        # print("Number of static collisions:", num_collisions)
        
        # static_collision = einops.reduce(self.lidar_scan, "n 1 w -> n 1", "max") < (self.lidar_range - (self.lidar_range - 0.3))  # 0.3 collision radius
        # print(self._robot.data.root_lin_vel_b.shape)
        # print("shape norm", torch.norm(self._robot.data.root_lin_vel_b, dim=1).shape)
        # print("norm", torch.norm(self._robot.data.root_lin_vel_b.squeeze(0), dim=1, keepdim=True))

        limit_vel = torch.norm(self._robot.data.root_lin_vel_b, dim=1) > 4.0

        # print(static_collision.squeeze(1).shape)

        died = died | static_collision.squeeze(1) | limit_vel
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
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-15.0, 15.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(1.0, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
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
