"""
DAgger policy distillation: privileged teacher (77D) -> onboard student (16D).

Teacher obs layout (77D, from QuadcopterEnvCfg):
    [0:3]   root_lin_vel_b      -- IMU
    [3:6]   root_ang_vel_b      -- IMU gyro
    [6:9]   projected_gravity_b -- IMU  (dropped for student: redundant with ang_vel for a level drone)
    [9:12]  unit_desired_pos_b  -- GPS/waypoint direction (3D unit vec)
    [12]    desired_dist_2d     -- 2D distance to goal
    [13]    desired_dist_z      -- Z distance to goal
    [14:74] lidar_scan (60)     -- full 360° LiDAR (6°/beam, beam0=-179°, beam30≈0°=forward)
    [74:76] nearest_obs_dir_b   -- PRIVILEGED (requires 360° scan; teacher only)
    [76]    potential            -- PRIVILEGED (Gaussian of nearest obstacle; teacher only)

Student obs (16D) — deployable with partial front-facing LiDAR only:
    [0:3]   root_lin_vel_b      -- lin velocity in FLU body frame
    [3:6]   root_ang_vel_b      -- ang velocity in FLU body frame
    [6:9]   unit_desired_pos_b  -- unit vector to goal in FLU body frame
    [9]     desired_dist_2d     -- 2D distance to goal
    [10]    desired_dist_z      -- Z distance to goal
    [11:16] 5 front-facing beams (teacher indices 28–32, covering ±12° around forward)
            beam_angle = -179 + beam_idx × 6°
            beam 28: -11°, 29: -5°, 30: +1°, 31: +7°, 32: +13°

DAgger algorithm:
    1. Rollout with mixed policy: teacher with prob beta, student with prob 1-beta
    2. Label every state with the TEACHER action (even when student acted)
    3. Aggregate (obs_74D, teacher_action) into replay buffer
    4. Train student on random batches from buffer (MSE loss)
    5. Decay beta and repeat

Run:
    ./isaaclab.sh -p scripts/rl_games/dagger_distill.py \\
        --task Isaac-Agile-Lidar-Vel-PA-v0 \\
        --teacher_checkpoint logs/rl_games/quadcopter_agile_direct/nn/quadcopter_agile_direct.pth \\
        --num_envs 256 \\
        --num_dagger_iters 200 \\
        --rollout_steps 256 \\
        --num_train_steps 500
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DAgger distillation: privileged teacher -> onboard student")
parser.add_argument("--task", type=str, default="Isaac-Agile-Lidar-Vel-PA-v0")
parser.add_argument("--teacher_checkpoint", type=str, default=None,
                    help="Path to teacher .pth. Loads latest from logs/ if omitted.")
parser.add_argument("--student_checkpoint", type=str, default=None,
                    help="Resume student from this .pth (optional).")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_dagger_iters", type=int, default=200,
                    help="Number of DAgger rollout+train iterations.")
parser.add_argument("--rollout_steps", type=int, default=256,
                    help="Steps per env per DAgger iteration during rollout.")
parser.add_argument("--num_train_steps", type=int, default=500,
                    help="Gradient steps per DAgger iteration.")
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--buffer_size", type=int, default=500_000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--beta_init", type=float, default=1.0,
                    help="Initial teacher mixing ratio (1.0 = pure teacher).")
parser.add_argument("--beta_decay", type=float, default=0.97,
                    help="Multiplicative decay applied each DAgger iteration.")
parser.add_argument("--beta_min", type=float, default=0.05,
                    help="Floor for beta so teacher never fully disappears.")
parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128])
parser.add_argument("--log_dir", type=str, default="logs/dagger")
parser.add_argument("--save_every", type=int, default=10,
                    help="Save student checkpoint every N DAgger iters.")
parser.add_argument("--wandb_project", type=str, default=None,
                    help="W&B project name. Omit to disable W&B logging.")
parser.add_argument("--wandb_run_name", type=str, default=None,
                    help="W&B run name (optional, auto-generated if omitted).")
parser.add_argument("--wandb_entity", type=str, default=None,
                    help="W&B entity/team (optional).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""All other imports after Isaac Sim is launched."""

import math
import os
import time
from typing import Any, cast

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import PerceptionAwareDrone.tasks  # noqa: F401

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── Obs layout constants ────────────────────────────────────────────────────
TEACHER_OBS_DIM = 77  # full privileged teacher obs
ACTION_DIM = 4

# Teacher obs slices
LIDAR_START = 14  # first lidar beam index in teacher obs
LIDAR_END = 74  # one past last lidar beam (60 beams total)
NEAREST_OBS_START = 74  # nearest_obs_dir_b [74:76] — privileged, dropped for student
POTENTIAL_IDX = 76  # potential scalar — privileged, dropped for student

# 5 front-facing beams from the 60-beam 360° scan:
#   beam_i angle = -179 + i × 6°  →  beam 30 ≈ 0° (forward)
#   selected range: beams 28–32 cover ±12° around forward
FRONT_BEAM_INDICES = [28, 29, 30, 31, 32]  # ±12° FOV

STUDENT_OBS_DIM = 11 + len(FRONT_BEAM_INDICES)  # = 16  (no gravity)


def extract_student_obs(teacher_obs: torch.Tensor) -> torch.Tensor:
    """Slice 16-dim student obs from the 77-dim teacher obs tensor.

    Drops: projected_gravity_b [6:9], full 360° lidar (60→5 beams),
           nearest_obs_dir_b, potential.
    Keeps: lin_vel [0:6] + goal+dist [9:14] + 5 front-facing beams.
    """
    lin_ang = teacher_obs[:, 0:6]                  # (N, 6)  lin_vel + ang_vel
    goal    = teacher_obs[:, 9:LIDAR_START]         # (N, 5)  goal_dir + dist_2d + dist_z
    lidar   = teacher_obs[:, LIDAR_START:LIDAR_END] # (N, 60)
    front   = lidar[:, FRONT_BEAM_INDICES]          # (N, 5)
    return torch.cat([lin_ang, goal, front], dim=-1)  # (N, 16)


# ── Networks ────────────────────────────────────────────────────────────────

class RunningNormalizer(nn.Module):
    """Online Welford mean/std for normalizing student observations."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0.0))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """Batch Welford update. x shape: (N, dim)."""
        x = x.float()
        batch_n = float(x.shape[0])
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)

        total_n = self.count + batch_n
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_n / total_n)
        new_var = (
            self.var * self.count
            + batch_var * batch_n
            + delta ** 2 * self.count * batch_n / total_n
        ) / total_n

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(torch.tensor(total_n))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) / (self.var + self.eps).sqrt()

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() * (self.var + self.eps).sqrt() + self.mean


class StudentPolicy(nn.Module):
    """MLP mapping 19D student obs -> 4D normalized action in [-1, 1]."""

    def __init__(self, obs_dim: int = STUDENT_OBS_DIM, action_dim: int = ACTION_DIM,
                 hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        linears = [m for m in self.modules() if isinstance(m, nn.Linear)]
        for i, m in enumerate(linears):
            gain = 0.01 if i == len(linears) - 1 else 1.0  # small gain on output layer
            nn.init.orthogonal_(m.weight, gain=gain)  # type: ignore[arg-type]
            nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ── Replay buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Circular buffer storing (student_obs, teacher_action) pairs."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.ptr = 0
        self.size = 0

    def add(self, obs: torch.Tensor, actions: torch.Tensor):
        """Add batch of (obs, actions) to buffer."""
        n = obs.shape[0]
        if n >= self.capacity:
            # Oversized batch: keep only the last capacity elements
            obs = obs[-self.capacity:]
            actions = actions[-self.capacity:]
            n = self.capacity

        end = (self.ptr + n) % self.capacity
        if end > self.ptr:
            self.obs[self.ptr:end] = obs
            self.actions[self.ptr:end] = actions
        else:
            # Wraps around
            first = self.capacity - self.ptr
            self.obs[self.ptr:] = obs[:first]
            self.actions[self.ptr:] = actions[:first]
            self.obs[:end] = obs[first:]
            self.actions[:end] = actions[first:]

        self.ptr = end
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.obs[idx], self.actions[idx]

    def __len__(self) -> int:
        return self.size


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_best_checkpoint(log_root: str, exp_name: str) -> str:
    """
    Search for <exp_name>.pth recursively under log_root.
    Falls back to the most recently modified .pth if not found.
    """
    import glob

    # Prefer the exact name (rl_games saves this as the "best" checkpoint)
    exact = glob.glob(os.path.join(log_root, "**", f"{exp_name}.pth"), recursive=True)
    if exact:
        # Pick the most recently modified one in case there are multiple runs
        return max(exact, key=os.path.getmtime)

    # Fallback: most recently modified .pth in any subdirectory
    any_pth = glob.glob(os.path.join(log_root, "**", "*.pth"), recursive=True)
    if any_pth:
        best = max(any_pth, key=os.path.getmtime)
        print(f"[WARN] Exact checkpoint not found. Using most recent: {best}")
        return best

    raise FileNotFoundError(
        f"No .pth checkpoint found under {log_root}. "
        "Pass --teacher_checkpoint /absolute/path/to/checkpoint.pth"
    )


def load_teacher(
    args_cli: argparse.Namespace,
    agent_cfg: dict[str, Any],
) -> BasePlayer:
    """Load rl_games teacher from checkpoint."""
    if args_cli.teacher_checkpoint is not None:
        resume_path = os.path.abspath(args_cli.teacher_checkpoint)
    else:
        log_root = os.path.abspath(
            os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
        )
        exp_name = agent_cfg["params"]["config"]["name"]
        resume_path = _find_best_checkpoint(log_root, exp_name)
    print(f"[INFO] Teacher checkpoint: {resume_path}")

    teacher_cfg = dict(agent_cfg)
    teacher_cfg["params"]["load_checkpoint"] = True
    teacher_cfg["params"]["load_path"] = resume_path
    teacher_cfg["params"]["config"]["num_actors"] = args_cli.num_envs

    runner = Runner()
    runner.load(teacher_cfg)
    teacher = runner.create_player()
    teacher.restore(resume_path)
    teacher.reset()
    teacher.is_deterministic = True
    return teacher


@torch.no_grad()
def eval_student(
    student: StudentPolicy,
    normalizer: RunningNormalizer,
    env: RlGamesVecEnvWrapper,
    teacher: BasePlayer,
    num_steps: int = 200,
    device: torch.device = torch.device("cuda"),
) -> dict[str, float]:
    """Evaluate student by comparing its actions to teacher actions for num_steps."""
    student.eval()
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    mse_total = 0.0
    for _ in range(num_steps):
        full_obs: torch.Tensor = obs.float()
        student_obs = extract_student_obs(full_obs)     # (N, 19)

        # Teacher labels
        teacher_obs = teacher.obs_to_torch(full_obs)
        teacher_actions: torch.Tensor = teacher.get_action(teacher_obs, is_deterministic=True)  # type: ignore[assignment]

        # Student predictions
        norm_obs = normalizer.normalize(student_obs)
        student_actions = student(norm_obs)

        mse_total += F.mse_loss(student_actions, teacher_actions.float()).item()

        # Step with teacher to keep states diverse
        obs, _, _, _ = env.step(teacher_actions)
        if isinstance(obs, dict):
            obs = obs["obs"]

    student.train()
    return {"eval/action_mse": mse_total / num_steps}


def save_checkpoint(
    path: str,
    student: StudentPolicy,
    normalizer: RunningNormalizer,
    optimizer: torch.optim.Optimizer,
    dagger_iter: int,
    beta: float,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "student": student.state_dict(),
        "normalizer": normalizer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "dagger_iter": dagger_iter,
        "beta": beta,
        "student_obs_dim": STUDENT_OBS_DIM,
        "action_dim": ACTION_DIM,
        "front_beam_indices": FRONT_BEAM_INDICES,
    }, path)
    print(f"[INFO] Saved student checkpoint: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Environment config
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cast(dict[str, Any], load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point"))

    # 2. Isaac Lab env
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env_raw.unwrapped, DirectMARLEnv):
        env_raw = multi_agent_to_single_agent(env_raw)
    env = RlGamesVecEnvWrapper(env_raw, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
    )

    device = torch.device(rl_device)
    print(f"[INFO] Device: {device} | Envs: {args_cli.num_envs}")

    # 3. Teacher
    teacher = load_teacher(args_cli, agent_cfg)

    # 4. Student, normalizer, optimizer, buffer
    student = StudentPolicy(
        obs_dim=STUDENT_OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=args_cli.hidden_dims,
    ).to(device)

    normalizer = RunningNormalizer(STUDENT_OBS_DIM).to(device)

    # Normalizer uses register_buffer (not parameters) — updated via Welford stats, not gradients
    optimizer = torch.optim.Adam(student.parameters(), lr=args_cli.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args_cli.num_dagger_iters,
        eta_min=args_cli.lr * 0.1,
    )

    buffer = ReplayBuffer(
        capacity=args_cli.buffer_size,
        obs_dim=STUDENT_OBS_DIM,
        action_dim=ACTION_DIM,
        device=device,
    )

    # Optionally resume student
    start_iter = 0
    beta = args_cli.beta_init
    if args_cli.student_checkpoint is not None:
        ckpt = torch.load(args_cli.student_checkpoint, map_location=device)
        student.load_state_dict(ckpt["student"])
        normalizer.load_state_dict(ckpt["normalizer"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("dagger_iter", 0) + 1
        beta = ckpt.get("beta", args_cli.beta_init)
        print(f"[INFO] Resumed from iter {start_iter}, beta={beta:.4f}")

    # 5. Initial env reset
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    _ = teacher.get_batch_size(teacher.obs_to_torch(obs.float()), args_cli.num_envs)
    if teacher.is_rnn:
        teacher.init_rnn()

    print("\n[INFO] Starting DAgger distillation")
    print(f"       Teacher obs dim : {TEACHER_OBS_DIM}")
    print(f"       Student obs dim : {STUDENT_OBS_DIM}  (11 base state no gravity + 5 front beams, no potential/nearest_obs)")
    print(f"       Front beams     : indices {FRONT_BEAM_INDICES}  (±12° FOV, beam_angle = -179 + i×6°)")
    print(f"       Hidden dims     : {args_cli.hidden_dims}")
    print(f"       Num envs        : {args_cli.num_envs}")
    print(f"       Buffer capacity : {args_cli.buffer_size:,}")
    print(f"       Beta init/min   : {args_cli.beta_init} / {args_cli.beta_min}")
    print()

    # Log file
    os.makedirs(args_cli.log_dir, exist_ok=True)
    log_path = os.path.join(args_cli.log_dir, "training_log.csv")
    log_header_written = os.path.exists(log_path)
    log_file = open(log_path, "a", buffering=1)
    if not log_header_written:
        log_file.write("iter,beta,buffer_size,train_loss,eval_mse,wall_time\n")

    t_start = time.time()

    # W&B setup
    use_wandb = args_cli.wandb_project is not None and _WANDB_AVAILABLE
    if args_cli.wandb_project is not None and not _WANDB_AVAILABLE:
        print("[WARN] --wandb_project set but wandb is not installed. Run: pip install wandb")
    if use_wandb:
        wandb.init(
            project=args_cli.wandb_project,
            name=args_cli.wandb_run_name,
            entity=args_cli.wandb_entity,
            config=vars(args_cli),
            resume="allow",
        )

    student.train()

    # ── DAgger main loop ─────────────────────────────────────────────────────
    for dagger_iter in range(start_iter, args_cli.num_dagger_iters):
        iter_t0 = time.time()

        # ── Phase 1: Rollout with mixed policy ────────────────────────────
        #
        #   beta   → probability of using TEACHER action for stepping
        #   1-beta → probability of using STUDENT action for stepping
        #   Labels are ALWAYS from the teacher regardless of who acted.
        #
        total_rollout_steps = args_cli.rollout_steps  # per env

        student.eval()
        rollout_obs_list: list[torch.Tensor] = []
        rollout_act_list: list[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(total_rollout_steps):
                full_obs = obs.float()                          # (N, 77)
                student_obs = extract_student_obs(full_obs)     # (N, 19)

                # Teacher action (labels)
                teacher_obs_t = teacher.obs_to_torch(full_obs)
                teacher_actions: torch.Tensor = teacher.get_action(teacher_obs_t, is_deterministic=True)  # type: ignore[assignment]

                # Student action
                normalizer.update(student_obs)
                norm_obs = normalizer.normalize(student_obs)
                student_actions = student(norm_obs)

                # Mix: choose who controls the drone this step
                use_teacher = (
                    torch.rand(args_cli.num_envs, 1, device=device) < beta
                )
                mixed_actions = torch.where(use_teacher, teacher_actions.float(), student_actions)

                # Step env
                obs, _, dones, _ = env.step(mixed_actions)
                if isinstance(obs, dict):
                    obs = obs["obs"]

                # Reset RNN state for done envs
                if teacher.is_rnn and teacher.states is not None and dones.any():
                    for s in teacher.states:
                        s[:, dones, :] = 0.0

                # Aggregate: (student_obs, teacher_label)
                rollout_obs_list.append(student_obs.clone())
                rollout_act_list.append(teacher_actions.float().clone())

        # Add all rollout data to buffer at once
        all_obs = torch.cat(rollout_obs_list, dim=0)     # (N*steps, 19)
        all_acts = torch.cat(rollout_act_list, dim=0)    # (N*steps, 4)
        buffer.add(all_obs, all_acts)

        # ── Phase 2: Train student on buffer ─────────────────────────────
        student.train()
        if len(buffer) < args_cli.batch_size:
            print(f"[iter {dagger_iter:04d}] Buffer too small ({len(buffer)}), skipping train.")
            continue

        train_losses: list[float] = []
        for _ in range(args_cli.num_train_steps):
            obs_b, act_b = buffer.sample(args_cli.batch_size)
            norm_obs_b = normalizer.normalize(obs_b)

            pred_actions = student(norm_obs_b)
            loss = F.mse_loss(pred_actions, act_b)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        mean_train_loss = sum(train_losses) / len(train_losses)

        # ── Decay beta ────────────────────────────────────────────────────
        beta = max(args_cli.beta_min, beta * args_cli.beta_decay)

        # ── Logging ───────────────────────────────────────────────────────
        iter_time = time.time() - iter_t0
        wall_time = time.time() - t_start

        print(
            f"[iter {dagger_iter:04d}] "
            f"beta={beta:.4f} | "
            f"buf={len(buffer):>7,} | "
            f"train_loss={mean_train_loss:.5f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"time={iter_time:.1f}s"
        )

        # Periodic evaluation
        eval_metrics: dict[str, float] = {}
        if dagger_iter % 10 == 0:
            eval_metrics = eval_student(student, normalizer, env, teacher, num_steps=200, device=device)
            print(f"         eval_action_mse={eval_metrics['eval/action_mse']:.5f}")
            # Restore obs after eval
            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]

        log_file.write(
            f"{dagger_iter},{beta:.6f},{len(buffer)},"
            f"{mean_train_loss:.6f},"
            f"{eval_metrics.get('eval/action_mse', float('nan')):.6f},"
            f"{wall_time:.1f}\n"
        )

        if use_wandb:
            log_dict = {
                "train/loss": mean_train_loss,
                "train/beta": beta,
                "train/lr": scheduler.get_last_lr()[0],
                "buffer/size": len(buffer),
                "time/iter_s": iter_time,
            }
            log_dict.update(eval_metrics)  # adds eval/action_mse when available
            wandb.log(log_dict, step=dagger_iter)

        # Save checkpoint
        if dagger_iter % args_cli.save_every == 0 or dagger_iter == args_cli.num_dagger_iters - 1:
            save_checkpoint(
                path=os.path.join(args_cli.log_dir, f"student_iter{dagger_iter:04d}.pth"),
                student=student,
                normalizer=normalizer,
                optimizer=optimizer,
                dagger_iter=dagger_iter,
                beta=beta,
            )
            # Also keep a "latest" pointer
            save_checkpoint(
                path=os.path.join(args_cli.log_dir, "student_latest.pth"),
                student=student,
                normalizer=normalizer,
                optimizer=optimizer,
                dagger_iter=dagger_iter,
                beta=beta,
            )

    log_file.close()
    print(f"\n[INFO] DAgger finished. Checkpoints in: {args_cli.log_dir}/")
    if use_wandb:
        wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
