# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import time
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
# Import extensions to set up environment tasks
import PerceptionAwareDrone.tasks  # noqa: F401

# DRAGGER #
import numpy as np
import torch
from torchinfo import summary
from tqdm import tqdm
import pandas as pd
import wandb
from datetime import datetime

import dragger_imitation.imitation as imitation
import csv



def play_student_model(student_policy, inputs):
    student_policy.eval()
    with torch.no_grad(): # No gradient is required during validation    
        # inputs = torch.clamp(inputs, min=-10.0, max=10.0)     
        y_pred = student_policy(inputs)
    return y_pred


def create_csv_from_dataset(dataloader, csv_filename="dataset.csv"):
    data = []

    for obs_batch, action_batch in dataloader:
        # Ensure observations and actions are tensors
        obs_batch = obs_batch.cpu().numpy()
        action_batch = action_batch.cpu().numpy()

        # Combine observations and actions
        for obs, action in zip(obs_batch, action_batch):
            row = list(obs) + list(action)  # Concatenate observation and action
            data.append(row)

    # Create a pandas DataFrame
    num_obs = obs_batch.shape[1]  # Number of observation features
    num_actions = action_batch.shape[1]  # Number of action features
    column_names = [f"obs_{i+1}" for i in range(num_obs)] + [f"action_{i+1}" for i in range(num_actions)]

    df = pd.DataFrame(data, columns=column_names)

    # Save to CSV
    df.to_csv(csv_filename, index=False)
    print(f"Dataset saved to {csv_filename}")


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    checkpoint_save = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint_save, path)
    print(f"Checkpoint saved to {path}")


def save_best_checkpoint(model, optimizer, scheduler, epoch, loss, best_loss, path):
    if loss < best_loss:
        save_checkpoint(model, optimizer, scheduler, epoch, loss, path)
        return loss
    return best_loss
# DRAGGER #

def main():
    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.physics_dt

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.

    # dragger imitation
    # Load the checkpoint
    checkpoint_dir = os.path.join("checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    checkpoint = torch.load(checkpoint_path)
    # Load the model state
    # imitation_model = imitation.StudentPolicy()
    # imitation_model.load_state_dict(checkpoint['model_state_dict'])
    # imitation_model.eval()


    student_model = imitation.StudentPolicy()
    student_model = student_model.to(rl_device)
    
    student_model.load_state_dict(checkpoint['model_state_dict'])
    student_model.eval()

    # optimizer = torch.optim.Adam(student_model.parameters(), lr=imitation.CONFIG_IMITATION['lr'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     'min',
    #     factor=imitation.CONFIG_IMITATION['scheduler_factor'],
    #     patience=imitation.CONFIG_IMITATION['scheduler_patience'],
    #     min_lr=imitation.CONFIG_IMITATION['scheduler_min_lr']
    # )
    # loss_fn = imitation.StudentPolicy().loss()
    
    # # Start wandb run
    # wandb.init(
    #     project='training-imitation',
    #     config=imitation.CONFIG_IMITATION,
    # )
    # # Log parameters and gradients
    # wandb.watch(student_model, log='all')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"dataset_{timestamp}.pt"

    log_filename = f"./log_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(log_filename, mode="w", newline='')
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "lin_vel_x", "lin_vel_y", "lin_vel_z",
        "ang_vel_x", "ang_vel_y", "ang_vel_z",
        "unit_des_x", "unit_des_y", "unit_des_z",
        "des_dist_xy", "des_dist_z",
        # Example for lidar: save first 5 lidar inputs only
        *[f"lidar_{i}" for i in range(60)]
    ])


    
    max_teacher_timesteps = 3000
    lidar_resolution = (60)
    lidar_range = 5.0
    beta = 0.0  # Play only student model
    while simulation_app.is_running():
        start_time = time.time()
        
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            lin_vel = obs[:, :3]
            ang_vel = obs[:, 3:6]
            unit_desired_pos = obs[:, 9:12]
            
            desired_dist_2d =  obs[:, 12:13]
            desired_dist_z = obs[:, 13:14]
            # print("desired_dist_2d", desired_dist_2d.shape)
            # print("desired_dist_z", desired_dist_z.shape)

            lidar_scan = (
                (
                    env.env.scene["lidar_sensor"].data.ray_hits_w
                    - env.env.scene["lidar_sensor"].data.pos_w.unsqueeze(1)
                )
                .norm(dim=-1)
                .clamp_max(lidar_range)
                .reshape(env.unwrapped.num_envs, 1, lidar_resolution)
            )
            drone_state = torch.cat((lin_vel, ang_vel, unit_desired_pos, desired_dist_2d, desired_dist_z), dim=1)
            input_model = torch.cat((drone_state, lidar_scan.squeeze(1)), dim=1)  # shape (num_envs, 73)
            # print("input model shape", input_model.shape)
            student_input = input_model

            # csv_writer.writerow(
            #     [
            #         lin_vel[0, 0].item(), lin_vel[0, 1].item(), lin_vel[0, 2].item(),
            #         ang_vel[0, 0].item(), ang_vel[0, 1].item(), ang_vel[0, 2].item(),
            #         unit_desired_pos[0, 0].item(), unit_desired_pos[0, 1].item(), unit_desired_pos[0, 2].item(),
            #         desired_dist_2d[0].item(), desired_dist_z[0].item(),
            #         *lidar_scan[0].squeeze(1).cpu().numpy().tolist()
            #     ]
            # )
            # teacher_actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            student_actions = play_student_model(student_model, student_input).squeeze(1)

            policy_actions = student_actions
        
            # new_data.extend(zip(student_input, teacher_actions))
            # progress_bar.update(step - progress_bar.n)

            obs, reward , dones, _ = env.step(policy_actions)

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # loop_duration = time.time() - start_time
        # print(f"Loop time: {loop_duration:.4f} seconds, Frequency: {1.0 / loop_duration:.2f} Hz")

        
        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            print(f"Sleeping for {sleep_time:.2f} seconds to maintain real-time.")
            time.sleep(sleep_time)
                
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


# Wait for add into Readme
#  python scripts/rl_games/train_imi_random.py --task=Isaac-Agile-Lidar-Direct-v0 --num_envs 256