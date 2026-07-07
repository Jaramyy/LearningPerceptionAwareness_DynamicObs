"""Verify obs consistency while the student policy is actively flying in Isaac Lab.

Unlike verify_obs.py (which uses zero actions), this loads the student checkpoint
and lets the policy control the drone.  At each step it shows:
  - Isaac-native 19D obs  (ground truth from training)
  - PX4-reconstructed 19D obs  (same math as student_ros2_node.py)
  - Per-dimension absolute diff

If all diffs are near zero during flight, the sim2real obs pipeline is correct.

Run:
    ./isaaclab.sh -p scripts/rl_games/verify_obs_flight.py \\
        --task Isaac-Agile-Lidar-Vel-PA-v0 --num_envs 1 --headless \\
        --checkpoint logs/dagger/student_latest.pth \\
        --num_steps 1000 --print_every 25
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Agile-Lidar-Vel-PA-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Student .pth from dagger_distill.py")
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--print_every", type=int, default=25,
                    help="Print per-step table every N steps (0 = summary only)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from typing import Any, cast

import gymnasium as gym
import torch
import torch.nn as nn

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import PerceptionAwareDrone.tasks  # noqa: F401

# ── Obs layout ────────────────────────────────────────────────────────────────
LIDAR_START = 14
LIDAR_END = 74
FRONT_BEAM_INDICES = [28, 29, 30, 31, 32]
STUDENT_OBS_DIM = 11 + len(FRONT_BEAM_INDICES)  # 16  (no gravity)
LIDAR_RANGE = 5.0

OBS_NAMES = [
    "lin_vel_x  (fwd  m/s)",
    "lin_vel_y  (left m/s)",
    "lin_vel_z  (up   m/s)",
    "ang_vel_x  (rad/s)   ",
    "ang_vel_y  (rad/s)   ",
    "ang_vel_z  (rad/s)   ",
    "goal_dir_x (fwd)     ",
    "goal_dir_y (left)    ",
    "goal_dir_z (up)      ",
    "dist_2d    (m)       ",
    "dist_z     (m)       ",
    "beam_28  -11 deg[0-1]",
    "beam_29   -5 deg[0-1]",
    "beam_30   +1 deg[0-1]",
    "beam_31   +7 deg[0-1]",
    "beam_32  +13 deg[0-1]",
]
assert len(OBS_NAMES) == STUDENT_OBS_DIM

HIGH_DIFF = 0.05   # flag threshold for per-step diffs
MAE_FAIL  = 0.05   # threshold for MISMATCH in summary


# ── Student network ───────────────────────────────────────────────────────────

class RunningNormalizer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0.0))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) / (self.var + self.eps).sqrt()


class StudentPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _infer_hidden_dims(sd: dict, action_dim: int) -> list:
    return [
        v.shape[0] for k, v in sd.items()
        if k.endswith(".weight") and "net" in k and v.shape[0] != action_dim
    ]


# ── Rotation helpers (identical to verify_obs.py) ────────────────────────────

def _quat_to_rot(q_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q_wxyz.float().unbind(-1)
    return torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ]).reshape(3, 3)


def _ned_to_flu_mat(R_P: torch.Tensor, v_ned: torch.Tensor) -> torch.Tensor:
    v_frd = R_P.T @ v_ned
    return torch.stack([v_frd[0], v_frd[1], -v_frd[2]])  # only negate Z


def _isaac_to_R_px4(q_isaac: torch.Tensor) -> torch.Tensor:
    device = q_isaac.device
    R_I = _quat_to_rot(q_isaac)
    B = torch.diag(torch.tensor([1., 1., -1.], device=device))
    A = torch.diag(torch.tensor([1., -1., -1.], device=device))
    return B @ R_I @ A


# ── Isaac native obs extractor ────────────────────────────────────────────────

def _extract_isaac_obs(teacher_obs_77: torch.Tensor) -> torch.Tensor:
    lin_ang = teacher_obs_77[0, 0:6]               # lin_vel + ang_vel
    goal    = teacher_obs_77[0, 9:LIDAR_START]      # goal_dir + dist_2d + dist_z (skip gravity [6:9])
    front   = teacher_obs_77[0, LIDAR_START:LIDAR_END][FRONT_BEAM_INDICES]
    return torch.cat([lin_ang, goal, front])        # (16,)


# ── PX4-style obs builder (mirrors student_ros2_node._build_obs) ──────────────

def _build_px4_obs(raw_env, goal_ned: tuple) -> torch.Tensor:
    device = raw_env.device

    pos_neup = raw_env._robot.data.root_pos_w[0].float()
    vel_neup = raw_env._robot.data.root_lin_vel_w[0].float()
    q_isaac  = raw_env._robot.data.root_quat_w[0].float()
    ang_flu  = raw_env._robot.data.root_ang_vel_b[0].float()

    # NEUp -> NED (flip z)
    pos_ned = torch.tensor([pos_neup[0], pos_neup[1], -pos_neup[2]], device=device)
    vel_ned = torch.tensor([vel_neup[0], vel_neup[1], -vel_neup[2]], device=device)
    # Isaac body -> FRD: only negate Z (pitch Y also negated for FRD sign convention)
    ang_frd = torch.tensor([ang_flu[0].item(), -ang_flu[1].item(), ang_flu[2].item()], device=device)

    R_P = _isaac_to_R_px4(q_isaac)

    vel_flu_recon = _ned_to_flu_mat(R_P, vel_ned)
    # FRD -> Isaac body: only negate pitch Y, keep yaw Z same sign
    ang_flu_recon = torch.tensor([ang_frd[0].item(), -ang_frd[1].item(), ang_frd[2].item()], device=device)

    gN, gE, gD = goal_ned
    pN, pE, pD = pos_ned.tolist()
    goal_ned_vec = torch.tensor([gN-pN, gE-pE, gD-pD], device=device)
    goal_flu = _ned_to_flu_mat(R_P, goal_ned_vec)
    goal_flu = goal_flu / (goal_flu.norm() + 1e-6)

    dist_2d = math.sqrt((gN-pN)**2 + (gE-pE)**2)
    dist_z  = pD - gD  # positive = goal above

    lidar_scan = raw_env.lidar_scan  # (N, 1, 60) in meters
    if lidar_scan.dim() == 3:
        beams = lidar_scan[0, 0, FRONT_BEAM_INDICES].float()
    else:
        beams = lidar_scan[0, FRONT_BEAM_INDICES].float()
    beams = beams.clamp(max=LIDAR_RANGE) / LIDAR_RANGE  # normalize to [0,1]

    return torch.cat([
        vel_flu_recon, ang_flu_recon, goal_flu,
        torch.tensor([dist_2d, dist_z], device=device),
        beams,
    ])


# ── Print helpers ─────────────────────────────────────────────────────────────

_C = 14

def _header():
    print(f"\n{'#':<4} {'Obs name':<25} {'Isaac':>{_C}} {'PX4-recon':>{_C}} {'|diff|':>8}  flag")
    print("-" * (4 + 25 + _C + _C + 8 + 8))

def _row(i, name, a, b):
    d = abs(a - b)
    flag = " HIGH ***" if d > HIGH_DIFF else ""
    print(f"{i:<4} {name:<25} {a:>{_C}.5f} {b:>{_C}.5f} {d:>8.5f}{flag}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Load student checkpoint
    ckpt = torch.load(os.path.abspath(args_cli.checkpoint), map_location="cpu", weights_only=False)
    obs_dim     = ckpt.get("student_obs_dim", STUDENT_OBS_DIM)
    action_dim  = ckpt.get("action_dim", 4)
    hidden_dims = _infer_hidden_dims(ckpt["student"], action_dim)
    print(f"[INFO] Checkpoint  : {args_cli.checkpoint}")
    print(f"[INFO] Obs dim     : {obs_dim}  hidden: {hidden_dims}  action: {action_dim}")
    print(f"[INFO] DAgger iter : {ckpt.get('dagger_iter','?')}  beta: {ckpt.get('beta',float('nan')):.4f}")

    # 2. Build env
    env_cfg   = parse_env_cfg(args_cli.task, device=args_cli.device,
                               num_envs=args_cli.num_envs, use_fabric=True)
    agent_cfg = cast(dict[str, Any], load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point"))
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs  = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_act  = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env_raw.unwrapped, DirectMARLEnv):
        env_raw = multi_agent_to_single_agent(env_raw)
    env     = RlGamesVecEnvWrapper(env_raw, rl_device, clip_obs, clip_act)
    raw_env = env_raw.unwrapped
    device  = torch.device(rl_device)

    # 3. Load student onto device
    student    = StudentPolicy(obs_dim, action_dim, hidden_dims).to(device)
    normalizer = RunningNormalizer(obs_dim).to(device)
    student.load_state_dict(ckpt["student"])
    normalizer.load_state_dict(ckpt["normalizer"])
    student.eval()

    # 4. Reset and set goal
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    GOAL_NEUP = torch.tensor([5.0, 0.0, 1.5], device=device)
    raw_env._desired_pos_w[:] = GOAL_NEUP
    GOAL_NED  = (GOAL_NEUP[0].item(), GOAL_NEUP[1].item(), -GOAL_NEUP[2].item())

    print(f"\n[INFO] Goal (Isaac NEUp) : {GOAL_NEUP.tolist()}")
    print(f"[INFO] Goal (PX4 NED)    : {GOAL_NED}")
    print(f"[INFO] Steps to run      : {args_cli.num_steps}")
    print()

    # 5. Accumulate errors
    errors    = torch.zeros(STUDENT_OBS_DIM)
    max_errs  = torch.zeros(STUDENT_OBS_DIM)
    n_steps   = 0
    n_resets  = 0

    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break

            # ── student policy inference ──────────────────────────────────
            teacher_obs  = obs.float()                     # (1, 77)
            student_obs  = _extract_isaac_obs(teacher_obs).unsqueeze(0)  # (1, 19)
            norm_obs     = normalizer.normalize(student_obs)
            actions      = student(norm_obs)               # (1, 4)

            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

            if dones.any():
                raw_env._desired_pos_w[:] = GOAL_NEUP
                n_resets += 1
                continue

            # ── build comparison obs ──────────────────────────────────────
            obs_isaac = _extract_isaac_obs(obs.float()).cpu()
            obs_px4   = _build_px4_obs(raw_env, GOAL_NED).cpu()

            diff = (obs_isaac - obs_px4).abs()
            errors  += diff
            max_errs = torch.max(max_errs, diff)
            n_steps += 1

            # ── print per-step table every N steps ────────────────────────
            if args_cli.print_every > 0 and step % args_cli.print_every == 0:
                pos   = raw_env._robot.data.root_pos_w[0].float()
                gN, gE, gD = GOAL_NED
                dist  = math.sqrt((gN-pos[0].item())**2 + (gE-pos[1].item())**2)
                distz = -pos[2].item() - gD   # NED Down sign
                print(f"\n[step {step:05d}]  pos_NEUp=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})"
                      f"  dist2d={dist:.2f}m  dist_z={distz:.2f}m  resets={n_resets}")
                _header()
                for i, name in enumerate(OBS_NAMES):
                    _row(i, name, obs_isaac[i].item(), obs_px4[i].item())

    env.close()

    if n_steps == 0:
        print("\n[VERIFY] No steps collected (all resets or no steps).")
        return

    mae = errors / n_steps

    print(f"\n\n{'='*80}")
    print(f"  SUMMARY over {n_steps} steps ({n_resets} episode resets)")
    print(f"{'='*80}")
    print(f"{'#':<4} {'Obs name':<25} {'MAE':>12} {'Max|diff|':>12}  Status")
    print("-" * 72)
    all_ok = True
    for i, name in enumerate(OBS_NAMES):
        ok   = mae[i].item() < MAE_FAIL
        flag = "OK" if ok else "MISMATCH <---"
        if not ok:
            all_ok = False
        print(f"{i:<4} {name:<25} {mae[i].item():>12.5f} {max_errs[i].item():>12.5f}  {flag}")

    print("=" * 80)
    if all_ok:
        print("  RESULT: ALL dims match (MAE < 0.05).  Obs reconstruction is CORRECT during flight.")
    else:
        print("  RESULT: Some dims have MAE >= 0.05 — check MISMATCH rows.")
        print("  Common causes:")
        print("    lin_vel / ang_vel   -> wrong FRD<->FLU sign")
        print("    gravity             -> wrong gravity direction in NED")
        print("    goal_dir            -> goal NED z-sign (alt vs Down)")
        print("    dist_z              -> pD - gD vs gD - pD")
        print("    beams               -> not normalized by LIDAR_RANGE")
    print("=" * 80)


if __name__ == "__main__":
    main()
    simulation_app.close()
