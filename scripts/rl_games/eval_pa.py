"""Evaluate PA (Perception-Aware) performance of a trained rl_games checkpoint.

Metrics collected per episode:
  - pa_alignment     : mean cosine(heading, nearest_obstacle) when PA potential > 0.3  (higher = better facing)
  - pa_active_ratio  : fraction of steps where PA was active (potential > 0.3)
  - final_dist       : 3-D distance to goal at episode end (lower = better)
  - success          : drone reached within GOAL_RADIUS of goal before timeout
  - collision        : episode ended because drone hit an obstacle (min_beam < 0.3 m)
  - ep_length        : steps in the episode

Usage:
  python scripts/rl_games/eval_pa.py --task <task> --num_envs 16 --num_episodes 200
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate PA performance of an rl_games checkpoint.")
parser.add_argument("--num_envs",     type=int,   default=16,    help="Parallel environments.")
parser.add_argument("--num_episodes", type=int,   default=200,   help="Total episodes to evaluate.")
parser.add_argument("--task",         type=str,   default=None,  help="Task name.")
parser.add_argument("--checkpoint",   type=str,   default=None,  help="Path to .pth checkpoint. Uses best by default.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="Load last checkpoint instead of best.")
parser.add_argument("--save_csv",     type=str,   default=None,  help="Optional path to save per-episode CSV.")
parser.add_argument("--goal_radius",  type=float, default=1.5,   help="Distance threshold for success (m).")
parser.add_argument("--pa_threshold", type=float, default=0.3,   help="Potential threshold for PA-active steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import time

import gymnasium as gym
import torch
import wandb

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import PerceptionAwareDrone.tasks  # noqa: F401


GOAL_RADIUS  = None  # set from args after parse
PA_THRESHOLD = None  # set from args after parse


# ── helpers ──────────────────────────────────────────────────────────────────

def _compute_pa_step_metrics(raw_env):
    """Return (potential, cosine_similarity, nearest_dist) for all envs this step."""
    device = raw_env.device

    ray_hits = raw_env._lidar_sensor.data.ray_hits_w          # (N, beams, 3)
    drone_pos = raw_env._robot.data.root_pos_w                  # (N, 3)
    raw_vecs  = ray_hits - drone_pos.unsqueeze(1)              # (N, beams, 3)
    dists     = torch.nan_to_num(
        raw_vecs.norm(dim=-1), nan=raw_env.lidar_range, posinf=raw_env.lidar_range
    )                                                          # (N, beams)

    env_idx      = torch.arange(raw_env.num_envs, device=device)
    closest_idx  = torch.argmin(dists, dim=1)
    nearest_dist = dists[env_idx, closest_idx].clamp_max(raw_env.lidar_range)   # (N,)
    nearest_vec  = raw_vecs[env_idx, closest_idx]                                # (N, 3)
    nearest_norm = torch.nan_to_num(nearest_vec, nan=0.0) / (
        nearest_vec.norm(dim=-1, keepdim=True).clamp(min=0.1)
    )                                                                             # (N, 3)

    # Gaussian potential (same formula as training)
    sigma           = 1.5
    gaussian_factor = 1.0 / (0.1 * torch.sqrt(2.0 * torch.tensor(math.pi, device=device)))
    potential       = 0.25 * gaussian_factor * torch.exp(-nearest_dist ** 2 / (2 * sigma ** 2))  # (N,)

    # Drone heading vector (world frame)
    heading_w        = raw_env._robot.data.heading_w          # (N,)  yaw in radians
    heading_vec      = torch.stack([torch.cos(heading_w), torch.sin(heading_w),
                                    torch.zeros_like(heading_w)], dim=-1)  # (N, 3)
    heading_vec_norm = heading_vec / heading_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    cosine_sim = (heading_vec_norm * nearest_norm).sum(dim=-1)  # (N,)  in [-1, 1]

    return potential, cosine_sim, nearest_dist


def _is_collision(raw_env):
    """True for envs where the current min lidar beam is below the collision threshold."""
    min_beam = raw_env.lidar_scan.min(dim=2).values.squeeze(1)  # (N,)
    return min_beam < 0.3                                        # (N,) bool


def _dist_to_goal(raw_env):
    """3-D Euclidean distance from drone to goal for all envs."""
    return (raw_env._desired_pos_w - raw_env._robot.data.root_pos_w).norm(dim=-1)  # (N,)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global GOAL_RADIUS, PA_THRESHOLD
    GOAL_RADIUS  = args_cli.goal_radius
    PA_THRESHOLD = args_cli.pa_threshold

    from datetime import datetime
    wandb.init(
        project="NAV_PA",
        name=f"eval_{args_cli.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=[args_cli.task, "eval"] if args_cli.task else ["eval"],
        config={
            "task":         args_cli.task,
            "num_envs":     args_cli.num_envs,
            "num_episodes": args_cli.num_episodes,
            "goal_radius":  args_cli.goal_radius,
            "pa_threshold": args_cli.pa_threshold,
            "checkpoint":   args_cli.checkpoint,
        },
    )

    # ── env setup ────────────────────────────────────────────────────────────
    env_cfg   = parse_env_cfg(args_cli.task, device=args_cli.device,
                               num_envs=args_cli.num_envs, use_fabric=True)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    log_root = os.path.abspath(
        os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    )

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        ckpt    = ".*" if args_cli.use_last_checkpoint else \
                  f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root, run_dir, ckpt, other_dirs=["nn"])

    print(f"[PA-EVAL] checkpoint : {resume_path}")
    print(f"[PA-EVAL] num_envs   : {args_cli.num_envs}")
    print(f"[PA-EVAL] episodes   : {args_cli.num_episodes}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    rl_device   = agent_cfg["params"]["config"]["device"]
    clip_obs    = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions= agent_cfg["params"]["env"].get("clip_actions",      math.inf)
    env         = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register("IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                           "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"]       = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    raw_env = env.unwrapped  # Isaac Lab DirectRLEnv
    N       = raw_env.num_envs
    device  = raw_env.device

    # ── per-env episode accumulators ────────────────────────────────────────
    ep_steps        = torch.zeros(N, device=device)
    ep_pa_align_sum = torch.zeros(N, device=device)   # sum of cosine_sim on PA-active steps
    ep_pa_steps     = torch.zeros(N, device=device)   # count of PA-active steps
    ep_ever_reached = torch.zeros(N, dtype=torch.bool, device=device)  # True if drone was within GOAL_RADIUS at any step

    # ── completed episode storage ────────────────────────────────────────────
    results = {
        "pa_alignment":  [],   # mean cosine alignment when PA active (NaN if never active)
        "pa_active_ratio": [], # fraction of steps with PA active
        "final_dist":    [],
        "success":       [],
        "collision":     [],
        "ep_length":     [],
    }
    completed = 0

    # ── rollout ──────────────────────────────────────────────────────────────
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    print("[PA-EVAL] Running rollout …")
    while completed < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            obs_t   = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)

            # ── collect step metrics ─────────────────────────────────────────
            potential, cosine_sim, _ = _compute_pa_step_metrics(raw_env)
            pa_active  = potential > PA_THRESHOLD                 # (N,) bool
            collision  = _is_collision(raw_env)                   # (N,) bool
            dist_goal  = _dist_to_goal(raw_env)                   # (N,)

            ep_steps        += 1
            ep_pa_align_sum += torch.where(pa_active, cosine_sim, torch.zeros_like(cosine_sim))
            ep_pa_steps     += pa_active.float()
            ep_ever_reached |= (dist_goal < GOAL_RADIUS)   # latch True the moment drone enters goal radius

            # ── on episode done, save result ─────────────────────────────────
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            for i in done_ids:
                i = i.item()
                pa_steps  = ep_pa_steps[i].item()
                pa_align  = (ep_pa_align_sum[i] / pa_steps).item() if pa_steps > 0 else float("nan")
                pa_ratio  = (pa_steps / ep_steps[i].item()) if ep_steps[i].item() > 0 else 0.0

                results["pa_alignment"].append(pa_align)
                results["pa_active_ratio"].append(pa_ratio)
                results["final_dist"].append(dist_goal[i].item())
                results["success"].append(ep_ever_reached[i].item())   # True if drone EVER entered goal radius
                results["collision"].append(collision[i].item())
                results["ep_length"].append(ep_steps[i].item())

                # reset accumulators
                ep_steps[i]        = 0.0
                ep_pa_align_sum[i] = 0.0
                ep_pa_steps[i]     = 0.0
                ep_ever_reached[i] = False

                completed += 1
                if completed % 20 == 0:
                    print(f"  … {completed}/{args_cli.num_episodes} episodes done")
                if completed >= args_cli.num_episodes:
                    break

            if agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0

    env.close()

    # ── aggregate results ────────────────────────────────────────────────────
    import statistics, json

    def _mean(lst): return sum(lst) / len(lst) if lst else float("nan")
    def _pct(lst):  return 100.0 * sum(lst) / len(lst) if lst else float("nan")

    pa_valid = [v for v in results["pa_alignment"] if not math.isnan(v)]

    summary = {
        "episodes_evaluated": completed,
        "goal_radius_m":      GOAL_RADIUS,
        "pa_threshold":       PA_THRESHOLD,
        # PA quality
        "pa_alignment_mean":  _mean(pa_valid),
        "pa_alignment_std":   statistics.stdev(pa_valid) if len(pa_valid) > 1 else 0.0,
        "pa_active_ratio_mean": _mean(results["pa_active_ratio"]),
        "episodes_with_no_pa_activation": completed - len(pa_valid),
        # Navigation
        "success_rate_pct":  _pct(results["success"]),
        "collision_rate_pct": _pct(results["collision"]),
        "final_dist_mean_m": _mean(results["final_dist"]),
        "final_dist_std_m":  statistics.stdev(results["final_dist"]) if len(results["final_dist"]) > 1 else 0.0,
        "ep_length_mean":    _mean(results["ep_length"]),
    }

    print("\n" + "=" * 60)
    print("  PA EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Episodes evaluated       : {summary['episodes_evaluated']}")
    print(f"  Goal radius              : {GOAL_RADIUS} m")
    print(f"  PA potential threshold   : {PA_THRESHOLD}")
    print()
    print("  ── Navigation ──────────────────────────────────────────")
    print(f"  Success rate             : {summary['success_rate_pct']:.1f} %")
    print(f"  Collision rate           : {summary['collision_rate_pct']:.1f} %")
    print(f"  Final dist to goal (mean): {summary['final_dist_mean_m']:.3f} ± {summary['final_dist_std_m']:.3f} m")
    print(f"  Episode length (mean)    : {summary['ep_length_mean']:.1f} steps")
    print()
    print("  ── Perception-Aware behaviour ──────────────────────────")
    print(f"  PA alignment (mean)      : {summary['pa_alignment_mean']:.4f}  (1.0 = perfect facing)")
    print(f"  PA alignment (std)       : {summary['pa_alignment_std']:.4f}")
    print(f"  PA active ratio (mean)   : {summary['pa_active_ratio_mean']:.3f}  (fraction of steps near obstacle)")
    print(f"  Episodes with no PA hit  : {summary['episodes_with_no_pa_activation']}")
    print("=" * 60)

    # save JSON summary
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    json_path = os.path.join(log_dir, "pa_eval_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[PA-EVAL] Summary saved → {json_path}")

    # optional CSV per episode
    if args_cli.save_csv:
        import csv
        keys = list(results.keys())
        with open(args_cli.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for i in range(len(results["final_dist"])):
                writer.writerow({k: results[k][i] for k in keys})
        print(f"[PA-EVAL] Per-episode CSV saved → {args_cli.save_csv}")

    # ── W&B logging ──────────────────────────────────────────────────────────
    wandb.log({
        "eval/success_rate_pct":       summary["success_rate_pct"],
        "eval/collision_rate_pct":     summary["collision_rate_pct"],
        "eval/final_dist_mean_m":      summary["final_dist_mean_m"],
        "eval/final_dist_std_m":       summary["final_dist_std_m"],
        "eval/ep_length_mean":         summary["ep_length_mean"],
        "eval/pa_alignment_mean":      summary["pa_alignment_mean"],
        "eval/pa_alignment_std":       summary["pa_alignment_std"],
        "eval/pa_active_ratio_mean":   summary["pa_active_ratio_mean"],
        "eval/episodes_no_pa":         summary["episodes_with_no_pa_activation"],
        "eval/episodes_evaluated":     summary["episodes_evaluated"],
    })

    # per-episode table so you can inspect distributions in W&B
    ep_table = wandb.Table(columns=["episode", "success", "collision", "final_dist_m",
                                     "pa_alignment", "pa_active_ratio", "ep_length"])
    for i in range(len(results["final_dist"])):
        ep_table.add_data(
            i,
            results["success"][i],
            results["collision"][i],
            results["final_dist"][i],
            results["pa_alignment"][i],
            results["pa_active_ratio"][i],
            results["ep_length"][i],
        )
    wandb.log({"eval/episodes": ep_table})
    wandb.finish()
    print("[PA-EVAL] W&B run finished.")


if __name__ == "__main__":
    main()
    simulation_app.close()
