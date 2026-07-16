"""Play and evaluate a student policy trained by DAgger distillation.

Student obs layout (16D, matches dagger_distill.py):
    [0:6]   lin_vel + ang_vel in body frame
    [6:11]  goal_dir (3D unit vec) + dist_2d + dist_z
    [11:16] 5 sector min ranges  (5 x 36 deg sectors, -90 to +90)
            each entry = min(all beams in that 36 deg sector)
            sectors: -90->-54, -54->-18, -18->+18, +18->+54, +54->+90
    NOTE: gravity (teacher obs[6:9]) is EXCLUDED from student obs.

Run (visual play, press Ctrl-C to stop):
    ./isaaclab.sh -p scripts/rl_games/dagger_play.py \
        --task Isaac-Agile-Lidar-Vel-PA-v0 \
        --checkpoint logs/dagger/student_latest.pth \
        --num_envs 4

Run (evaluation mode, stops after N episodes):
    ./isaaclab.sh -p scripts/rl_games/dagger_play.py \
        --task Isaac-Agile-Lidar-Vel-PA-v0 \
        --checkpoint logs/dagger/student_latest.pth \
        --num_envs 16 --num_episodes 200 --headless
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play/evaluate a DAgger student policy")
parser.add_argument("--task", type=str, default="Isaac-Agile-Lidar-Vel-PA-v0")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to student .pth saved by dagger_distill.py")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=0,
                    help="Episodes to evaluate then exit. 0 = run forever (visual play).")
parser.add_argument("--goal_radius", type=float, default=1.5,
                    help="Distance threshold for success (m).")
parser.add_argument("--save_csv", type=str, default=None,
                    help="Optional path to save per-episode CSV.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""All other imports after Isaac Sim is launched."""

import math
import os
from typing import Any, cast

import gymnasium as gym
import torch
import torch.nn as nn

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import PerceptionAwareDrone.tasks  # noqa: F401

# ── Obs layout (must match dagger_distill.py) ───────────────────────────────
LIDAR_START = 14
LIDAR_END = 74
FRONT_BEAM_SECTORS = [
    (15, 21),  # sector 0: -90 to -54 deg
    (21, 27),  # sector 1: -54 to -18 deg
    (27, 33),  # sector 2: -18 to +18 deg (forward)
    (33, 39),  # sector 3: +18 to +54 deg
    (39, 45),  # sector 4: +54 to +90 deg
]
STUDENT_OBS_DIM = 11 + len(FRONT_BEAM_SECTORS)  # 16  (no gravity, matches dagger_distill.py)
ACTION_DIM = 4
LIDAR_RANGE = 5.0  # normalize sector values to [0,1] — must match dagger_distill.py


def extract_student_obs(teacher_obs: torch.Tensor) -> torch.Tensor:
    """Build 16D student obs — no gravity, 5 sector min ranges.

    Matches dagger_distill.py extract_student_obs exactly:
      [0:6]   lin_vel + ang_vel
      [6:11]  goal_dir + dist_2d + dist_z  (teacher obs[9:14], skipping gravity [6:9])
      [11:16] 5 front sector min ranges
    """
    lin_ang = teacher_obs[:, 0:6]
    goal = teacher_obs[:, 9:LIDAR_START]
    lidar = teacher_obs[:, LIDAR_START:LIDAR_END]
    front = torch.cat([
        lidar[:, s:e].min(dim=1, keepdim=True).values / LIDAR_RANGE
        for s, e in FRONT_BEAM_SECTORS
    ], dim=1)
    return torch.cat([lin_ang, goal, front], dim=-1)


# ── Network definitions (identical to dagger_distill.py) ────────────────────

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
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _dist_to_goal(raw_env) -> torch.Tensor:
    return (raw_env._desired_pos_w - raw_env._robot.data.root_pos_w).norm(dim=-1)


def _is_collision(raw_env) -> torch.Tensor:
    min_beam = raw_env.lidar_scan.min(dim=2).values.squeeze(1)
    return min_beam < 0.3


def _infer_hidden_dims(student_sd: dict, action_dim: int) -> list[int]:
    return [
        v.shape[0]
        for k, v in student_sd.items()
        if k.endswith(".weight") and "net" in k and v.shape[0] != action_dim
    ]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    eval_mode = args_cli.num_episodes > 0

    # 1. Load checkpoint
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading student checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    obs_dim = ckpt.get("student_obs_dim", STUDENT_OBS_DIM)
    action_dim = ckpt.get("action_dim", ACTION_DIM)
    front_sectors = ckpt.get("front_beam_sectors", FRONT_BEAM_SECTORS)
    hidden_dims = _infer_hidden_dims(ckpt["student"], action_dim)

    print(f"[INFO] Obs dim       : {obs_dim}")
    print(f"[INFO] Hidden dims   : {hidden_dims}")
    print(f"[INFO] Front sectors : {front_sectors}")
    print(f"[INFO] DAgger iter   : {ckpt.get('dagger_iter', '?')}")
    print(f"[INFO] Final beta    : {ckpt.get('beta', float('nan')):.4f}")
    if obs_dim != STUDENT_OBS_DIM:
        print(f"[WARN] Checkpoint obs_dim={obs_dim} != script STUDENT_OBS_DIM={STUDENT_OBS_DIM}. "
              "Using checkpoint value.")
    if front_sectors != FRONT_BEAM_SECTORS:
        print("[WARN] Checkpoint sectors differ from script - obs mismatch!")
        print(f"       checkpoint : {front_sectors}")
        print(f"       script     : {FRONT_BEAM_SECTORS}")

    # 2. Environment
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cast(dict[str, Any], load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point"))
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env_raw.unwrapped, DirectMARLEnv):
        env_raw = multi_agent_to_single_agent(env_raw)
    env = RlGamesVecEnvWrapper(env_raw, rl_device, clip_obs, clip_actions)

    device = torch.device(rl_device)
    raw_env = env.unwrapped
    N = raw_env.num_envs

    # 3. Student + normalizer
    student = StudentPolicy(obs_dim, action_dim, hidden_dims).to(device)
    normalizer = RunningNormalizer(obs_dim).to(device)
    student.load_state_dict(ckpt["student"])
    normalizer.load_state_dict(ckpt["normalizer"])
    student.eval()

    print(f"\n[INFO] Device: {device} | Envs: {N}")
    if eval_mode:
        print(f"[INFO] Evaluation mode: {args_cli.num_episodes} episodes, "
              f"goal_radius={args_cli.goal_radius} m")
    else:
        print("[INFO] Play mode: running until Ctrl-C")
    print()

    # 4. Episode accumulators (used in eval mode)
    ep_steps = torch.zeros(N, device=device)
    ep_ever_reached = torch.zeros(N, dtype=torch.bool, device=device)
    results = {
        "final_dist": [],
        "success": [],
        "collision": [],
        "ep_length": [],
    }
    completed = 0

    # 5. Rollout
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    step = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            full_obs = obs.float()
            student_obs = extract_student_obs(full_obs)
            norm_obs = normalizer.normalize(student_obs)
            actions = student(norm_obs)

            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

            if eval_mode:
                dist_goal = _dist_to_goal(raw_env)
                collision = _is_collision(raw_env)

                ep_steps += 1
                ep_ever_reached |= (dist_goal < args_cli.goal_radius)

                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                for i in done_ids:
                    i = i.item()
                    results["final_dist"].append(dist_goal[i].item())
                    results["success"].append(ep_ever_reached[i].item())
                    results["collision"].append(collision[i].item())
                    results["ep_length"].append(ep_steps[i].item())

                    ep_steps[i] = 0.0
                    ep_ever_reached[i] = False

                    completed += 1
                    if completed % 20 == 0:
                        print(f"  ... {completed}/{args_cli.num_episodes} episodes done")
                    if completed >= args_cli.num_episodes:
                        break

                if completed >= args_cli.num_episodes:
                    break

            else:
                if step % 200 == 0:
                    print(f"[step {step:6d}] running...")

            step += 1

    env.close()

    # 6. Print summary (eval mode only)
    if not eval_mode or not results["final_dist"]:
        return

    def _mean(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    def _pct(lst):
        return 100.0 * sum(lst) / len(lst) if lst else float("nan")

    import statistics
    dists = results["final_dist"]
    print("\n" + "=" * 55)
    print("  STUDENT EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Episodes           : {completed}")
    print(f"  Goal radius        : {args_cli.goal_radius} m")
    print()
    print(f"  Success rate       : {_pct(results['success']):.1f} %")
    print(f"  Collision rate     : {_pct(results['collision']):.1f} %")
    print(f"  Final dist (mean)  : {_mean(dists):.3f} +/- "
          f"{statistics.stdev(dists) if len(dists) > 1 else 0.0:.3f} m")
    print(f"  Episode length     : {_mean(results['ep_length']):.1f} steps")
    print("=" * 55)

    if args_cli.save_csv:
        import csv
        keys = list(results.keys())
        with open(args_cli.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for i in range(len(dists)):
                writer.writerow({k: results[k][i] for k in keys})
        print(f"\n[INFO] Per-episode CSV saved -> {args_cli.save_csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()
