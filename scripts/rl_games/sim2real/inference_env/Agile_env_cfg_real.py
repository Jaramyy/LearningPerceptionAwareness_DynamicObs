# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from dataclasses import dataclass
from typing import Literal


@dataclass
class AgileDroneEnvCfgReal():
    decimation = 1
    dt = 0.02  # 50 Hz
    cmd_publish_dt = 0.005  # 200 Hz
    max_episode_length_s = 3600
    action_scale = 0.25
    ctrl_delay_step_range = [0, 0]
    default_rfi_lim = 0
    robot = "unitree_h1"

    extend_body_parent_names = ["left_elbow_link", "right_elbow_link", "pelvis"]
    extend_body_names = ["left_hand_link", "right_hand_link", "head_link"]
    extend_body_pos = torch.tensor([[0.3, 0, 0], [0.3, 0, 0], [0, 0, 0.75]])

    tracked_body_names = [
        "left_hand_link",
        "right_hand_link",
        "head_link",
    ]

    # Distillation parameters:
    single_history_dim = 63
    observation_history_length = 25
    num_bodies = 20
    num_joints = 19
    mask_length = calculate_mask_length(
        num_bodies=num_bodies + len(extend_body_parent_names),
        num_joints=num_joints,
    )

    # hardware parameters
    subscriber_freq = 10
    reset_duration = 10.0  # seconds
    reset_step_dt = 0.01  # seconds
    robot_command_mode = "position"  # position or torque
    gravity_value = -9.8  # m/s^2
