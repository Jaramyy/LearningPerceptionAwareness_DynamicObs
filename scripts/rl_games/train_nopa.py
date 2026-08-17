"""Non-PA teacher + student training in one script.

Two phases selected with --phase:

  teacher — PPO via rl_games on Isaac-Agile-Lidar-Vel-Controller-v0.
             The non-PA env has the same 75-D obs layout as the PA env but uses
             simpler rewards (no obstacle-facing shaping) and flat terrain.
             Checkpoints: logs/rl_games_nopa/<teacher_run_name>/nn/<name>.pth

  student — DAgger distillation from a trained non-PA teacher into the same
             16-D deployable student used for PA inference.
             Checkpoints: logs/dagger_nopa/student_latest.pth

Teacher obs layout (75-D, quadcopter_env_lidar_vel_cont.py):
    [0:3]   root_lin_vel_b
    [3:6]   root_ang_vel_b
    [6:9]   projected_gravity_b
    [9:12]  unit_desired_pos_b
    [12]    desired_dist_2d
    [13]    desired_dist_z
    [14:74] lidar_scan  (60 beams, 360 deg, 6 deg/beam, max 5 m)
    [74]    potential   (Gaussian of nearest obstacle distance)

Student obs layout (16-D — identical extraction to PA DAgger):
    [0:6]   root_lin_vel_b + root_ang_vel_b  (body frame, FLU)
    [6:9]   unit_desired_pos_b
    [9]     desired_dist_2d
    [10]    desired_dist_z
    [11:16] 5 sector min ranges (-90 to +90 deg), normalised to [0,1]

Run (teacher — first time):
    ./isaaclab.sh -p scripts/rl_games/train_nopa.py \\
        --phase teacher --num_envs 256 --max_epochs 5000

Run (teacher — resume):
    ./isaaclab.sh -p scripts/rl_games/train_nopa.py \\
        --phase teacher --num_envs 256 \\
        --resume_teacher logs/rl_games_nopa/quadcopter_nopa/nn/quadcopter_nopa.pth

Run (student — DAgger from trained non-PA teacher):
    ./isaaclab.sh -p scripts/rl_games/train_nopa.py \\
        --phase student \\
        --teacher_checkpoint logs/rl_games_nopa/quadcopter_nopa/nn/quadcopter_nopa.pth \\
        --num_envs 256 --num_dagger_iters 200

The student checkpoint is drop-in compatible with student_ros2_node_icp.py.
To deploy, pass --checkpoint logs/dagger_nopa/student_latest.pth to the eval scripts.
"""

"""Launch Isaac Sim first (must precede all other imports)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Non-PA teacher (PPO) + student (DAgger) training"
)
parser.add_argument("--phase", choices=["teacher", "student"], required=True,
                    help="'teacher': run PPO RL training.  'student': run DAgger distillation.")

# ── Shared ────────────────────────────────────────────────────────────────────
parser.add_argument("--task", type=str, default="Isaac-Agile-Lidar-Vel-Controller-v0",
                    help="Isaac Lab gym task ID for the non-PA environment.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--disable_fabric", action="store_true", default=False)

# ── Teacher-phase args ────────────────────────────────────────────────────────
parser.add_argument("--max_epochs", type=int, default=5000,
                    help="[teacher] PPO training epochs.")
parser.add_argument("--resume_teacher", type=str, default=None,
                    help="[teacher] Resume PPO training from this .pth checkpoint.")
parser.add_argument("--teacher_log_dir", type=str, default="logs/rl_games_nopa",
                    help="[teacher] Root directory for rl_games checkpoints and logs.")
parser.add_argument("--teacher_run_name", type=str, default="quadcopter_nopa",
                    help="[teacher] Run name (used as checkpoint subdirectory name).")

# ── Student-phase (DAgger) args ───────────────────────────────────────────────
parser.add_argument("--teacher_checkpoint", type=str, default=None,
                    help="[student] Path to teacher .pth. Auto-searched under "
                         "--teacher_log_dir if omitted.")
parser.add_argument("--student_checkpoint", type=str, default=None,
                    help="[student] Resume student DAgger from this .pth (optional).")
parser.add_argument("--num_dagger_iters", type=int, default=200,
                    help="[student] DAgger rollout+train iterations.")
parser.add_argument("--rollout_steps", type=int, default=256,
                    help="[student] Steps per env per DAgger iteration.")
parser.add_argument("--num_train_steps", type=int, default=500,
                    help="[student] Gradient steps per DAgger iteration.")
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--buffer_size", type=int, default=500_000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--beta_init", type=float, default=1.0,
                    help="[student] Initial teacher mixing ratio.")
parser.add_argument("--beta_decay", type=float, default=0.97,
                    help="[student] Multiplicative beta decay per iteration.")
parser.add_argument("--beta_min", type=float, default=0.05,
                    help="[student] Floor for beta.")
parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128],
                    help="[student] MLP hidden layer sizes.")
parser.add_argument("--student_log_dir", type=str, default="logs/dagger_nopa",
                    help="[student] Directory for student checkpoints and CSV log.")
parser.add_argument("--save_every", type=int, default=10,
                    help="[student] Save checkpoint every N DAgger iters.")
parser.add_argument("--wandb_project", type=str, default=None,
                    help="[student] W&B project name. Omit to disable W&B.")
parser.add_argument("--wandb_run_name", type=str, default=None)
parser.add_argument("--wandb_entity", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""All other imports after Isaac Sim is launched."""

import glob
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

# ── Obs layout constants ───────────────────────────────────────────────────────
TEACHER_OBS_DIM = 75
ACTION_DIM      = 4
LIDAR_START     = 14    # first beam index in teacher obs
LIDAR_END       = 74    # one past last beam (60 beams total)
LIDAR_RANGE     = 5.0   # sensor max range in metres

# 5 front sectors (-90 to +90 deg, 36 deg each).
# beam_i angle = -179 + i*6 deg  ->  beam 30 ~= 0 deg (forward)
FRONT_BEAM_SECTORS = [
    (15, 21),   # sector 0: -90 to -54 deg
    (21, 27),   # sector 1: -54 to -18 deg
    (27, 33),   # sector 2: -18 to +18 deg  (forward)
    (33, 39),   # sector 3: +18 to +54 deg
    (39, 45),   # sector 4: +54 to +90 deg
]

STUDENT_OBS_DIM = 11 + len(FRONT_BEAM_SECTORS)   # 16


# ── Obs extraction ─────────────────────────────────────────────────────────────

def extract_student_obs(teacher_obs: torch.Tensor) -> torch.Tensor:
    """75-D teacher obs -> 16-D deployable student obs.

    Drops: projected_gravity_b [6:9], full 360 lidar -> 5 front sector mins,
           potential [74].
    """
    lin_ang = teacher_obs[:, 0:6]
    goal    = teacher_obs[:, 9:LIDAR_START]           # unit_pos_b + dist_2d + dist_z
    lidar   = teacher_obs[:, LIDAR_START:LIDAR_END]   # (N, 60)
    front   = torch.cat([
        lidar[:, s:e].min(dim=1, keepdim=True).values / LIDAR_RANGE
        for s, e in FRONT_BEAM_SECTORS
    ], dim=1)                                           # (N, 5), [0,1]
    return torch.cat([lin_ang, goal, front], dim=-1)   # (N, 16)


# ── Networks (drop-in compatible with student_ros2_node_icp.py) ───────────────

class RunningNormalizer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean",  torch.zeros(dim))
        self.register_buffer("var",   torch.ones(dim))
        self.register_buffer("count", torch.tensor(0.0))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x          = x.float()
        batch_n    = float(x.shape[0])
        batch_mean = x.mean(0)
        batch_var  = x.var(0, unbiased=False)
        total_n    = self.count + batch_n
        delta      = batch_mean - self.mean
        new_mean   = self.mean + delta * (batch_n / total_n)
        new_var    = (
            self.var * self.count
            + batch_var * batch_n
            + delta ** 2 * self.count * batch_n / total_n
        ) / total_n
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(torch.tensor(total_n))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) / (self.var + self.eps).sqrt()


class StudentPolicy(nn.Module):
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
            gain = 0.01 if i == len(linears) - 1 else 1.0
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ── Replay buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device   = device
        self.obs      = torch.zeros(capacity, obs_dim,    device=device)
        self.actions  = torch.zeros(capacity, action_dim, device=device)
        self.ptr      = 0
        self.size     = 0

    def add(self, obs: torch.Tensor, actions: torch.Tensor):
        n = obs.shape[0]
        if n >= self.capacity:
            obs = obs[-self.capacity:]
            actions = actions[-self.capacity:]
            n = self.capacity
        end = (self.ptr + n) % self.capacity
        if end > self.ptr:
            self.obs[self.ptr:end]     = obs
            self.actions[self.ptr:end] = actions
        else:
            first = self.capacity - self.ptr
            self.obs[self.ptr:]     = obs[:first]
            self.obs[:end]          = obs[first:]
            self.actions[self.ptr:] = actions[:first]
            self.actions[:end]      = actions[first:]
        self.ptr  = end
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.obs[idx], self.actions[idx]

    def __len__(self) -> int:
        return self.size


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _build_env(rl_device: str) -> tuple[RlGamesVecEnvWrapper, dict[str, Any]]:
    env_cfg   = parse_env_cfg(
        args_cli.task, device=rl_device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cast(dict[str, Any], load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point"))
    clip_obs     = agent_cfg["params"]["env"].get("clip_observations", math.inf)
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
    return env, agent_cfg


def _find_teacher_ckpt(log_dir: str, run_name: str) -> str:
    exact = glob.glob(os.path.join(log_dir, "**", f"{run_name}.pth"), recursive=True)
    if exact:
        return max(exact, key=os.path.getmtime)
    any_pth = glob.glob(os.path.join(log_dir, "**", "*.pth"), recursive=True)
    if any_pth:
        best = max(any_pth, key=os.path.getmtime)
        print(f"[WARN] Exact checkpoint not found. Using most recent: {best}")
        return best
    raise FileNotFoundError(
        f"No .pth found under {log_dir}. "
        "Pass --teacher_checkpoint /path/to/checkpoint.pth"
    )


def _load_teacher(agent_cfg: dict[str, Any], checkpoint: str, device: str) -> BasePlayer:
    cfg = dict(agent_cfg)
    cfg["params"]["load_checkpoint"]          = True
    cfg["params"]["load_path"]                = checkpoint
    cfg["params"]["config"]["num_actors"]     = args_cli.num_envs
    cfg["params"]["config"]["device"]         = device
    cfg["params"]["config"]["device_name"]    = device
    runner  = Runner()
    runner.load(cfg)
    teacher = runner.create_player()
    teacher.restore(checkpoint)
    teacher.reset()
    teacher.is_deterministic = True
    return teacher


def _save_student(path: str, student: StudentPolicy, normalizer: RunningNormalizer,
                  optimizer: torch.optim.Optimizer, dagger_iter: int, beta: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "student":            student.state_dict(),
        "normalizer":         normalizer.state_dict(),
        "optimizer":          optimizer.state_dict(),
        "dagger_iter":        dagger_iter,
        "beta":               beta,
        "student_obs_dim":    STUDENT_OBS_DIM,
        "action_dim":         ACTION_DIM,
        "front_beam_sectors": FRONT_BEAM_SECTORS,
        "method":             "nopa",
    }, path)
    print(f"[INFO] Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Teacher PPO training
# ══════════════════════════════════════════════════════════════════════════════

def train_teacher():
    rl_device = "cuda:0"
    env, agent_cfg = _build_env(rl_device)

    cfg = agent_cfg["params"]["config"]
    cfg["name"]       = args_cli.teacher_run_name
    cfg["num_actors"] = args_cli.num_envs
    cfg["max_epochs"] = args_cli.max_epochs
    cfg["train_dir"]  = os.path.abspath(args_cli.teacher_log_dir)
    cfg["device"]     = rl_device
    cfg["device_name"] = rl_device

    if args_cli.resume_teacher:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"]       = os.path.abspath(args_cli.resume_teacher)
        print(f"[INFO] Resuming teacher from: {args_cli.resume_teacher}")
    else:
        agent_cfg["params"]["load_checkpoint"] = False
        agent_cfg["params"]["load_path"]       = ""

    ckpt_path = os.path.join(
        args_cli.teacher_log_dir, args_cli.teacher_run_name,
        "nn", f"{args_cli.teacher_run_name}.pth"
    )
    print(f"\n[INFO] Non-PA teacher training (PPO)")
    print(f"       Task       : {args_cli.task}")
    print(f"       Envs       : {args_cli.num_envs}")
    print(f"       Max epochs : {args_cli.max_epochs}")
    print(f"       Log dir    : {os.path.abspath(args_cli.teacher_log_dir)}")
    print(f"       Checkpoint : {os.path.abspath(ckpt_path)}")
    print()

    runner = Runner()
    runner.load(agent_cfg)
    runner.run({"train": True})

    env.close()
    print(f"\n[INFO] Teacher training done. Checkpoint: {ckpt_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Student DAgger distillation
# ══════════════════════════════════════════════════════════════════════════════

def train_student():
    rl_device = "cuda:0"
    env, agent_cfg = _build_env(rl_device)
    device = torch.device(rl_device)

    # Locate teacher checkpoint
    if args_cli.teacher_checkpoint:
        teacher_ckpt = os.path.abspath(args_cli.teacher_checkpoint)
    else:
        teacher_ckpt = _find_teacher_ckpt(args_cli.teacher_log_dir, args_cli.teacher_run_name)
    print(f"[INFO] Teacher checkpoint: {teacher_ckpt}")
    teacher = _load_teacher(agent_cfg, teacher_ckpt, rl_device)

    # Student, normalizer, optimizer, buffer
    student    = StudentPolicy(STUDENT_OBS_DIM, ACTION_DIM, args_cli.hidden_dims).to(device)
    normalizer = RunningNormalizer(STUDENT_OBS_DIM).to(device)
    optimizer  = torch.optim.Adam(student.parameters(), lr=args_cli.lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args_cli.num_dagger_iters, eta_min=args_cli.lr * 0.1,
    )
    buffer = ReplayBuffer(args_cli.buffer_size, STUDENT_OBS_DIM, ACTION_DIM, device)

    start_iter = 0
    beta = args_cli.beta_init
    if args_cli.student_checkpoint:
        ckpt       = torch.load(args_cli.student_checkpoint, map_location=device)
        student.load_state_dict(ckpt["student"])
        normalizer.load_state_dict(ckpt["normalizer"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("dagger_iter", 0) + 1
        beta       = ckpt.get("beta", args_cli.beta_init)
        print(f"[INFO] Resumed student from iter {start_iter}, beta={beta:.4f}")

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    _ = teacher.get_batch_size(teacher.obs_to_torch(obs.float()), args_cli.num_envs)
    if teacher.is_rnn:
        teacher.init_rnn()

    print(f"\n[INFO] Non-PA DAgger distillation")
    print(f"       Task          : {args_cli.task}")
    print(f"       Teacher obs   : {TEACHER_OBS_DIM}-D")
    print(f"       Student obs   : {STUDENT_OBS_DIM}-D")
    print(f"       Hidden dims   : {args_cli.hidden_dims}")
    print(f"       Envs          : {args_cli.num_envs}")
    print(f"       DAgger iters  : {args_cli.num_dagger_iters}")
    print(f"       Buffer cap    : {args_cli.buffer_size:,}")
    print(f"       Beta init/min : {args_cli.beta_init} / {args_cli.beta_min}")
    print(f"       Log dir       : {os.path.abspath(args_cli.student_log_dir)}")
    print()

    os.makedirs(args_cli.student_log_dir, exist_ok=True)
    log_path    = os.path.join(args_cli.student_log_dir, "training_log.csv")
    log_existed = os.path.exists(log_path)
    log_file    = open(log_path, "a", buffering=1)
    if not log_existed:
        log_file.write("iter,beta,buffer_size,train_loss,eval_mse,wall_time\n")

    use_wandb = args_cli.wandb_project is not None and _WANDB_AVAILABLE
    if args_cli.wandb_project and not _WANDB_AVAILABLE:
        print("[WARN] --wandb_project set but wandb not installed (pip install wandb).")
    if use_wandb:
        wandb.init(
            project=args_cli.wandb_project,
            name=args_cli.wandb_run_name or f"nopa_dagger_{int(time.time())}",
            entity=args_cli.wandb_entity,
            config=vars(args_cli),
            resume="allow",
        )

    student.train()
    t_start = time.time()

    for dagger_iter in range(start_iter, args_cli.num_dagger_iters):
        iter_t0 = time.time()

        # ── Rollout: mixed teacher/student policy ─────────────────────────────
        student.eval()
        buf_obs: list[torch.Tensor] = []
        buf_act: list[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(args_cli.rollout_steps):
                full_obs    = obs.float()
                student_obs = extract_student_obs(full_obs)

                t_obs  = teacher.obs_to_torch(full_obs)
                t_act: torch.Tensor = teacher.get_action(t_obs, is_deterministic=True)  # type: ignore[assignment]

                normalizer.update(student_obs)
                s_act = student(normalizer.normalize(student_obs))

                use_teacher   = torch.rand(args_cli.num_envs, 1, device=device) < beta
                mixed         = torch.where(use_teacher, t_act.float(), s_act)

                obs, _, dones, _ = env.step(mixed)
                if isinstance(obs, dict):
                    obs = obs["obs"]

                if teacher.is_rnn and teacher.states is not None and dones.any():
                    for s in teacher.states:
                        s[:, dones, :] = 0.0

                buf_obs.append(student_obs.clone())
                buf_act.append(t_act.float().clone())

        buffer.add(torch.cat(buf_obs, dim=0), torch.cat(buf_act, dim=0))

        # ── Train student on buffer ────────────────────────────────────────────
        student.train()
        if len(buffer) < args_cli.batch_size:
            print(f"[iter {dagger_iter:04d}] buffer too small ({len(buffer)}), skipping train.")
            continue

        losses: list[float] = []
        for _ in range(args_cli.num_train_steps):
            obs_b, act_b = buffer.sample(args_cli.batch_size)
            pred  = student(normalizer.normalize(obs_b))
            loss  = F.mse_loss(pred, act_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        mean_loss = sum(losses) / len(losses)
        beta      = max(args_cli.beta_min, beta * args_cli.beta_decay)

        iter_time = time.time() - iter_t0
        wall_time = time.time() - t_start
        print(
            f"[iter {dagger_iter:04d}] beta={beta:.4f} | buf={len(buffer):>7,} | "
            f"loss={mean_loss:.5f} | lr={scheduler.get_last_lr()[0]:.2e} | "
            f"time={iter_time:.1f}s"
        )

        # ── Periodic eval (action MSE vs teacher) ─────────────────────────────
        eval_mse = float("nan")
        if dagger_iter % 10 == 0:
            student.eval()
            eval_obs = env.reset()
            if isinstance(eval_obs, dict):
                eval_obs = eval_obs["obs"]
            mse_acc = 0.0
            with torch.no_grad():
                for _ in range(200):
                    full_obs = eval_obs.float()
                    s_obs    = extract_student_obs(full_obs)
                    t_obs    = teacher.obs_to_torch(full_obs)
                    t_act_e: torch.Tensor = teacher.get_action(t_obs, is_deterministic=True)  # type: ignore[assignment]
                    s_act_e  = student(normalizer.normalize(s_obs))
                    mse_acc += F.mse_loss(s_act_e, t_act_e.float()).item()
                    eval_obs, _, _, _ = env.step(t_act_e)
                    if isinstance(eval_obs, dict):
                        eval_obs = eval_obs["obs"]
            eval_mse = mse_acc / 200
            print(f"         eval_mse={eval_mse:.5f}")
            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            student.train()

        log_file.write(
            f"{dagger_iter},{beta:.6f},{len(buffer)},"
            f"{mean_loss:.6f},{eval_mse:.6f},{wall_time:.1f}\n"
        )

        if use_wandb:
            log_dict: dict[str, Any] = {
                "train/loss":  mean_loss,
                "train/beta":  beta,
                "train/lr":    scheduler.get_last_lr()[0],
                "buffer/size": len(buffer),
                "time/iter_s": iter_time,
            }
            if not math.isnan(eval_mse):
                log_dict["eval/action_mse"] = eval_mse
            wandb.log(log_dict, step=dagger_iter)

        if dagger_iter % args_cli.save_every == 0 or dagger_iter == args_cli.num_dagger_iters - 1:
            _save_student(
                os.path.join(args_cli.student_log_dir, f"student_iter{dagger_iter:04d}.pth"),
                student, normalizer, optimizer, dagger_iter, beta,
            )
            _save_student(
                os.path.join(args_cli.student_log_dir, "student_latest.pth"),
                student, normalizer, optimizer, dagger_iter, beta,
            )

    log_file.close()
    if use_wandb:
        wandb.finish()
    env.close()
    print(f"\n[INFO] DAgger done. Checkpoints in: {args_cli.student_log_dir}/")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    if args_cli.phase == "teacher":
        train_teacher()
    else:
        train_student()


if __name__ == "__main__":
    main()
    simulation_app.close()
