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


def train_student_policy(student_policy, dataloader, val_loader, optimizer, loss_fn, config, scheduler):
    train_losses = []
    learning_rates = []
    
    for epoch in range(config['epochs']):
        train_loss = []
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        student_policy.train()

        print(f"Training epoch {epoch+1}...")
        print(f"Current LR: {current_lr}")

        for obs_batch, action_batch in dataloader:
            optimizer.zero_grad()
            
            # Replace NaNs in input data and ensure range consistency
            # obs_batch = torch.nan_to_num(obs_batch, nan=0.0)
            # action_batch = torch.nan_to_num(action_batch, nan=0.0)

            # Check for NaNs or Infs and clamp values
            if torch.isnan(obs_batch).any() or torch.isinf(obs_batch).any():
                print("NaN or Inf detected in obs_batch. Skipping this batch.")
                continue
            if torch.isnan(action_batch).any() or torch.isinf(action_batch).any():
                print("NaN or Inf detected in action_batch. Skipping this batch.")
                continue

            # Scale inputs and targets to manageable range
            # obs_batch = torch.clamp(obs_batch, min=-10.0, max=10.0)  # need to normalize the data
            # action_batch = torch.clamp(action_batch, min=-1.0, max=1.0)

            # Forward pass
            predictions = student_policy(obs_batch)
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                print("NaN or Inf detected in predictions. Skipping this batch.")
                continue

            # Calculate loss
            loss = loss_fn(predictions, action_batch)
            if torch.isnan(loss):
                print("NaN detected in loss. Skipping update.")
                continue

            # Backward pass
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(student_policy.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss.append(loss)
        
        
        
        avg_train_loss = torch.stack(train_loss).mean().item()
        train_losses.append(avg_train_loss)

        scheduler.step(avg_train_loss)

        print(f"Epoch {epoch+1} train loss: {avg_train_loss:.4f}")


# Validation
        student_policy.eval()
        val_loss = []
        with torch.no_grad():
            for obs_batch, action_batch in val_loader:
                # obs_batch = torch.clamp(obs_batch, min=-10.0, max=10.0)
                # action_batch = torch.clamp(action_batch, min=-1.0, max=1.0)
                predictions = student_policy(obs_batch)
                loss = loss_fn(predictions, action_batch)
                val_loss.append(loss)
            
            avg_val_loss = torch.stack(val_loss).mean().item()
            val_loss.append(avg_val_loss)
            print(f"Epoch {epoch+1} val loss: {avg_val_loss:.4f}")

        wandb.log({'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'lr': current_lr})

    return train_loss


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

    student_model = imitation.StudentPolicy()
    student_model = student_model.to(rl_device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=imitation.CONFIG_IMITATION['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        factor=imitation.CONFIG_IMITATION['scheduler_factor'],
        patience=imitation.CONFIG_IMITATION['scheduler_patience'],
        min_lr=imitation.CONFIG_IMITATION['scheduler_min_lr']
    )
    loss_fn = imitation.StudentPolicy().loss()
    
    # Start wandb run
    wandb.init(
        project='training-imitation',
        config=imitation.CONFIG_IMITATION,
    )
    # Log parameters and gradients
    wandb.watch(student_model, log='all')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"dataset_{timestamp}.csv"
    D = []
    
    max_teacher_timesteps = 1000
    
    while simulation_app.is_running():
        start_time = time.time()
        for iteration in tqdm(range(imitation.CONFIG_IMITATION['num_iterations']), desc="DAgger Iterations"):
            with tqdm(total=imitation.CONFIG_IMITATION['max_samples'], desc="samples", unit="sample") as progress_bar:
                new_data = []
                teacher_timestep = 0
                while len(new_data) < imitation.CONFIG_IMITATION['max_samples']:  # TODO: change to for loop of num_episodes_progress
                    with torch.inference_mode():
                        obs = agent.obs_to_torch(obs)
                        lin_vel = obs[:, :3]
                        ang_vel = obs[:, 3:6]
                        robot_orientation = obs[:, 9:12]
                        target = obs[:, 12:15]

                        lidar_scan = (env.env.scene["lidar_sensor"].data.ray_hits_w - env.env.scene["lidar_sensor"].data.pos_w.unsqueeze(1)).norm(dim=-1).clamp_max(10).reshape(env.unwrapped.num_envs, 5)
                        
                        drone_state = torch.cat((lin_vel, ang_vel, robot_orientation, target, lidar_scan), dim=1)
                        student_input = drone_state
                        
                        if iteration < 1:  # 2 iterations, use teacher's action
                            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic) # Shape [envs, action_dim], [2,4]
                            teacher_actions = actions
                            # print("action shape ",actions.shape)
                            print("pure teacher action ")
                            
                        else:
                            if teacher_timestep < max_teacher_timesteps: 
                                actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)  # Teacher's action
                                teacher_actions = actions
                                # print("Teacher guild action") 
                            else:  
                                actions = play_student_model(student_model, student_input).squeeze(1)  # Student's action for partial observation
                                teacher_actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)  # Teacher's action / for comparison with student action
                                # print("student action", actions)
                                # print("Teacher action ", teacher_actions)
                        # print("teacher_timestepp", teacher_timestep)
                        teacher_timestep += 1
        
                        if teacher_timestep > 1000: 
                            teacher_timestep = 0

                        new_data.extend(zip(drone_state, teacher_actions))
                        progress_bar.update(len(new_data) - progress_bar.n)

                        obs, _, dones, _ = env.step(actions)

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
                            # time delay for real-time evaluation
                    sleep_time = dt - (time.time() - start_time)
                    if args_cli.real_time and sleep_time > 0:
                        time.sleep(sleep_time)
                
                D.extend(new_data)
                print("size of dataset", len(D))

                dag_dataset = imitation.DAggerDataset(D)
                train_size = int(0.8 * len(dag_dataset))
                val_size = len(dag_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(dag_dataset, [train_size, val_size])

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=imitation.CONFIG_IMITATION['batch_size'], shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=imitation.CONFIG_IMITATION['batch_size'], shuffle=False)

                # create_csv_from_dataset(train_loader, file_name)
                train_student_policy(student_model, train_loader, val_loader, optimizer, loss_fn, imitation.CONFIG_IMITATION, scheduler)
                max_teacher_timesteps = max(0, round(max_teacher_timesteps - (iteration * 20)))   # adaptive teacher timesteps




        
        # # run everything in inference mode
        # with torch.inference_mode():
        #     # convert obs to agent format
        #     obs = agent.obs_to_torch(obs)
        #     # agent stepping
        #     actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
        #     # env stepping
        #     obs, _, dones, _ = env.step(actions)

        #     # perform operations for terminated episodes
        #     if len(dones) > 0:
        #         # reset rnn state for terminated episodes
        #         if agent.is_rnn and agent.states is not None:
        #             for s in agent.states:
        #                 s[:, dones, :] = 0.0
        # if args_cli.video:
        #     timestep += 1
        #     # Exit the play loop after recording one video
        #     if timestep == args_cli.video_length:
        #         break

        # # time delay for real-time evaluation
        # sleep_time = dt - (time.time() - start_time)
        # if args_cli.real_time and sleep_time > 0:
        #     time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
