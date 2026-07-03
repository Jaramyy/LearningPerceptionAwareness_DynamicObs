"""Evaluate a DAgger student policy.

Metrics per episode:
  - success          : drone reached within goal_radius at any step (ever-reached)
  - collision        : min LiDAR beam < 0.3 m at episode end
  - final_dist       : 3D distance to goal at episode end
  - ep_length        : steps per episode
  - pa_alignment     : mean cosine(heading, nearest obstacle) on PA-active steps
  - pa_active_ratio  : fraction of steps where Gaussian potential > pa_threshold

Student obs (19D):
  [0:14]  base state  (vel, ang_vel, gravity, goal_dir, dist_2d, dist_z)
  [14:19] 5 front-facing LiDAR beams (teacher beam indices 28-32, ±12° FOV)

Run:
    ./isaaclab.sh -p scripts/rl_games/eval_student.py \\
        --task Isaac-Agile-Lidar-Vel-PA-v0 \\
        --checkpoint logs/dagger/student_latest.pth \\
        --num_envs 16 --num_episodes 200 --headless
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate DAgger student policy")
parser.add_argument("--task",         type=str,   default="Isaac-Agile-Lidar-Vel-PA-v0")
parser.add_argument("--checkpoint",   type=str,   required=True,
                    help="Path to student .pth saved by dagger_distill.py")
parser.add_argument("--num_envs",     type=int,   default=16)
parser.add_argument("--num_episodes", type=int,   default=200)
parser.add_argument("--goal_radius",  type=float, default=1.5,
                    help="Distance threshold for success (m).")
parser.add_argument("--pa_threshold", type=float, default=0.3,
                    help="Gaussian potential threshold for PA-active steps.")
parser.add_argument("--save_csv",     type=str,   default=None,
                    help="Optional path to save per-episode CSV.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""All other imports after Isaac Sim is launched."""

import json
import math
import os
import statistics
from typing import Any, cast

import gymnasium as gym
import torch
import torch.nn as nn

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import PerceptionAwareDrone.tasks  # noqa: F401

# ── Obs layout (must match dagger_distill.py) ───────────────────────────────
LIDAR_START         = 14
LIDAR_END           = 74
FRONT_BEAM_INDICES  = [28, 29, 30, 31, 32]   # ±12° FOV
STUDENT_OBS_DIM     = 14 + len(FRONT_BEAM_INDICES)  # 19
ACTION_DIM          = 4


def extract_student_obs(teacher_obs: torch.Tensor) -> torch.Tensor:
    base  = teacher_obs[:, :LIDAR_START]
    front = teacher_obs[:, LIDAR_START:LIDAR_END][:, FRONT_BEAM_INDICES]
    return torch.cat([base, front], dim=-1)


# ── Network definitions (identical to dagger_distill.py) ────────────────────

class RunningNormalizer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean",  torch.zeros(dim))
        self.register_buffer("var",   torch.ones(dim))
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


# ── Raw-env metrics (mirrors eval_pa.py) ────────────────────────────────────

def _compute_pa_step_metrics(raw_env):
    """Return (potential, cosine_sim) for all envs this step."""
    device = raw_env.device

    ray_hits  = raw_env._lidar_sensor.data.ray_hits_w      # (N, beams, 3)
    drone_pos = raw_env._robot.data.root_pos_w              # (N, 3)
    raw_vecs  = ray_hits - drone_pos.unsqueeze(1)           # (N, beams, 3)
    dists = torch.nan_to_num(
        raw_vecs.norm(dim=-1),
        nan=raw_env.lidar_range, posinf=raw_env.lidar_range,
    )                                                        # (N, beams)

    env_idx      = torch.arange(raw_env.num_envs, device=device)
    closest_idx  = torch.argmin(dists, dim=1)
    nearest_dist = dists[env_idx, closest_idx].clamp_max(raw_env.lidar_range)
    nearest_vec  = raw_vecs[env_idx, closest_idx]           # (N, 3)
    nearest_norm = torch.nan_to_num(nearest_vec, nan=0.0) / (
        nearest_vec.norm(dim=-1, keepdim=True).clamp(min=0.1)
    )

    sigma           = 1.5
    gaussian_factor = 1.0 / (0.1 * torch.sqrt(2.0 * torch.tensor(math.pi, device=device)))
    potential       = 0.25 * gaussian_factor * torch.exp(
        -nearest_dist ** 2 / (2 * sigma ** 2)
    )

    heading_w   = raw_env._robot.data.heading_w             # (N,) yaw in radians
    heading_vec = torch.stack(
        [torch.cos(heading_w), torch.sin(heading_w), torch.zeros_like(heading_w)], dim=-1
    )
    heading_vec = heading_vec / heading_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    cosine_sim = (heading_vec * nearest_norm).sum(dim=-1)   # (N,) in [-1, 1]
    return potential, cosine_sim


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
    # 1. Load checkpoint
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    print(f"[EVAL] Checkpoint    : {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    obs_dim     = ckpt.get("student_obs_dim", STUDENT_OBS_DIM)
    action_dim  = ckpt.get("action_dim",      ACTION_DIM)
    front_beams = ckpt.get("front_beam_indices", FRONT_BEAM_INDICES)
    hidden_dims = _infer_hidden_dims(ckpt["student"], action_dim)

    print(f"[EVAL] Obs dim       : {obs_dim}")
    print(f"[EVAL] Hidden dims   : {hidden_dims}")
    print(f"[EVAL] Front beams   : {front_beams}")
    print(f"[EVAL] DAgger iter   : {ckpt.get('dagger_iter', '?')}")
    print(f"[EVAL] Final beta    : {ckpt.get('beta', float('nan')):.4f}")
    print(f"[EVAL] num_envs      : {args_cli.num_envs}")
    print(f"[EVAL] num_episodes  : {args_cli.num_episodes}")
    print(f"[EVAL] goal_radius   : {args_cli.goal_radius} m")
    print(f"[EVAL] pa_threshold  : {args_cli.pa_threshold}")

    if obs_dim != STUDENT_OBS_DIM:
        print(f"[WARN] Checkpoint obs_dim={obs_dim} != script STUDENT_OBS_DIM={STUDENT_OBS_DIM}. "
              "Using checkpoint value.")

    # 2. Environment
    env_cfg   = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cast(dict[str, Any], load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point"))
    rl_device    = agent_cfg["params"]["config"]["device"]
    clip_obs     = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions",      math.inf)

    env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env_raw.unwrapped, DirectMARLEnv):
        env_raw = multi_agent_to_single_agent(env_raw)
    env = RlGamesVecEnvWrapper(env_raw, rl_device, clip_obs, clip_actions)

    device  = torch.device(rl_device)
    raw_env = env.unwrapped
    N       = raw_env.num_envs

    # 3. Student + normalizer
    student    = StudentPolicy(obs_dim, action_dim, hidden_dims).to(device)
    normalizer = RunningNormalizer(obs_dim).to(device)
    student.load_state_dict(ckpt["student"])
    normalizer.load_state_dict(ckpt["normalizer"])
    student.eval()

    # 4. Per-env accumulators
    ep_steps        = torch.zeros(N, device=device)
    ep_pa_align_sum = torch.zeros(N, device=device)
    ep_pa_steps     = torch.zeros(N, device=device)
    ep_ever_reached = torch.zeros(N, dtype=torch.bool, device=device)

    results: dict[str, list] = {
        "pa_alignment":    [],
        "pa_active_ratio": [],
        "final_dist":      [],
        "success":         [],
        "collision":       [],
        "ep_length":       [],
    }
    completed = 0

    # 5. Rollout
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    print("\n[EVAL] Running rollout …")
    with torch.inference_mode():
        while completed < args_cli.num_episodes and simulation_app.is_running():
            full_obs    = obs.float()
            student_obs = extract_student_obs(full_obs)
            norm_obs    = normalizer.normalize(student_obs)
            actions     = student(norm_obs)

            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

            potential, cosine_sim = _compute_pa_step_metrics(raw_env)
            pa_active  = potential > args_cli.pa_threshold
            dist_goal  = _dist_to_goal(raw_env)
            collision  = _is_collision(raw_env)

            ep_steps        += 1
            ep_pa_align_sum += torch.where(pa_active, cosine_sim, torch.zeros_like(cosine_sim))
            ep_pa_steps     += pa_active.float()
            ep_ever_reached |= (dist_goal < args_cli.goal_radius)

            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            for i in done_ids:
                i = i.item()
                pa_steps = ep_pa_steps[i].item()
                pa_align = (ep_pa_align_sum[i] / pa_steps).item() if pa_steps > 0 else float("nan")
                pa_ratio = (pa_steps / ep_steps[i].item()) if ep_steps[i].item() > 0 else 0.0

                results["pa_alignment"].append(pa_align)
                results["pa_active_ratio"].append(pa_ratio)
                results["final_dist"].append(dist_goal[i].item())
                results["success"].append(ep_ever_reached[i].item())
                results["collision"].append(collision[i].item())
                results["ep_length"].append(ep_steps[i].item())

                ep_steps[i]        = 0.0
                ep_pa_align_sum[i] = 0.0
                ep_pa_steps[i]     = 0.0
                ep_ever_reached[i] = False

                completed += 1
                if completed % 20 == 0:
                    print(f"  … {completed}/{args_cli.num_episodes} episodes done")
                if completed >= args_cli.num_episodes:
                    break

    env.close()

    # 6. Aggregate
    def _mean(lst): return sum(lst) / len(lst) if lst else float("nan")
    def _pct(lst):  return 100.0 * sum(lst) / len(lst) if lst else float("nan")

    pa_valid  = [v for v in results["pa_alignment"] if not math.isnan(v)]
    dists     = results["final_dist"]
    ep_lens   = results["ep_length"]

    summary = {
        "checkpoint":               ckpt_path,
        "dagger_iter":              ckpt.get("dagger_iter", -1),
        "episodes_evaluated":       completed,
        "goal_radius_m":            args_cli.goal_radius,
        "pa_threshold":             args_cli.pa_threshold,
        # navigation
        "success_rate_pct":         _pct(results["success"]),
        "collision_rate_pct":       _pct(results["collision"]),
        "final_dist_mean_m":        _mean(dists),
        "final_dist_std_m":         statistics.stdev(dists)  if len(dists)    > 1 else 0.0,
        "ep_length_mean":           _mean(ep_lens),
        # PA behaviour measured from raw env
        "pa_alignment_mean":        _mean(pa_valid),
        "pa_alignment_std":         statistics.stdev(pa_valid) if len(pa_valid) > 1 else 0.0,
        "pa_active_ratio_mean":     _mean(results["pa_active_ratio"]),
        "episodes_with_no_pa_hit":  completed - len(pa_valid),
    }

    print("\n" + "=" * 60)
    print("  STUDENT EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Checkpoint              : {os.path.basename(ckpt_path)}")
    print(f"  DAgger iter             : {summary['dagger_iter']}")
    print(f"  Episodes evaluated      : {completed}")
    print(f"  Goal radius             : {args_cli.goal_radius} m")
    print()
    print("  ── Navigation ───────────────────────────────────────────")
    print(f"  Success rate            : {summary['success_rate_pct']:.1f} %")
    print(f"  Collision rate          : {summary['collision_rate_pct']:.1f} %")
    print(f"  Final dist (mean)       : {summary['final_dist_mean_m']:.3f} ± "
          f"{summary['final_dist_std_m']:.3f} m")
    print(f"  Episode length (mean)   : {summary['ep_length_mean']:.1f} steps")
    print()
    print("  ── Perception-Aware behaviour (from raw env) ────────────")
    print(f"  PA alignment (mean)     : {summary['pa_alignment_mean']:.4f}  (1.0 = perfect facing)")
    print(f"  PA alignment (std)      : {summary['pa_alignment_std']:.4f}")
    print(f"  PA active ratio (mean)  : {summary['pa_active_ratio_mean']:.3f}  (fraction of steps near obstacle)")
    print(f"  Episodes with no PA hit : {summary['episodes_with_no_pa_hit']}")
    print("=" * 60)

    # 7. Save JSON summary next to checkpoint
    ckpt_dir  = os.path.dirname(ckpt_path)
    json_path = os.path.join(ckpt_dir, "student_eval_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[EVAL] Summary saved -> {json_path}")

    # 8. Optional per-episode CSV
    if args_cli.save_csv:
        import csv
        keys = list(results.keys())
        with open(args_cli.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for i in range(len(dists)):
                writer.writerow({k: results[k][i] for k in keys})
        print(f"[EVAL] Per-episode CSV saved -> {args_cli.save_csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()
