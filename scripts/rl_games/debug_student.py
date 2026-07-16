"""
5-point avoidance debug — runs in Isaac, one test per point.

Tests
-----
  [P1] Sector consistency   : dagger_distill FRONT_BEAM_SECTORS == this script's sectors
  [P2] LiDAR obs range      : sector values in [0, 1] each step
  [P3] Sector-direction map : which sector activates for a known nearest-obstacle angle?
  [P4] Student-teacher yaw  : do they agree on turn direction when obstacle is close?
  [P5] Goal direction obs   : does obs[6:9] point toward goal?

Per-step line also prints:
  potential — teacher obs[76]: gaussian from nearest obstacle.
               Near 0 = obstacle far. > 0.01 = within ~5m.

Run:
    ./isaaclab.sh -p scripts/rl_games/debug_student.py \
        --task Isaac-Agile-Lidar-Vel-PA-v0 \
        --student_checkpoint logs/dagger/student_latest.pth \
        --teacher_checkpoint logs/rl_games/quadcopter_agile_direct/nn/quadcopter_agile_direct.pth \
        --num_envs 16 --steps 1000 --close_threshold 2.0

Output: one row per --print_every steps; PASS/FAIL summary at end.
"""

"""Launch Isaac Sim first."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug student avoidance in Isaac")
parser.add_argument("--task", type=str, default="Isaac-Agile-Lidar-Vel-PA-v0")
parser.add_argument("--student_checkpoint", type=str, required=True)
parser.add_argument("--teacher_checkpoint", type=str, default=None,
                    help="Optional -- enables P4 teacher comparison.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1000,
                    help="How many steps to run before printing summary.")
parser.add_argument("--close_threshold", type=float, default=2.0,
                    help="Distance (m) at which an obstacle is considered close for P3/P4.")
parser.add_argument("--print_every", type=int, default=50,
                    help="Print per-step table every N steps (0 = summary only).")
parser.add_argument("--env_idx", type=int, default=0,
                    help="Which env to print detail for.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""All other imports after Isaac Sim is launched."""

import math
import os
from collections import Counter
from typing import Any, cast

import gymnasium as gym
import torch
import torch.nn as nn

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import PerceptionAwareDrone.tasks  # noqa: F401

# Must match dagger_distill.py exactly
LIDAR_START = 14
LIDAR_END = 74
FRONT_BEAM_SECTORS = [
    (15, 21),  # sector 0: -90 to -54 deg
    (21, 27),  # sector 1: -54 to -18 deg
    (27, 33),  # sector 2: -18 to +18 deg (forward)
    (33, 39),  # sector 3: +18 to +54 deg
    (39, 45),  # sector 4: +54 to +90 deg
]
LIDAR_RANGE = 5.0
STUDENT_OBS_DIM = 11 + len(FRONT_BEAM_SECTORS)  # 16
ACTION_DIM = 4


# ── Network (identical to dagger_distill.py) ─────────────────────────────────

class RunningNormalizer(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0.0))

    @torch.no_grad()
    def normalize(self, x):
        return (x - self.mean) / (self.var.sqrt() + self.eps)


class StudentPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims):
        super().__init__()
        layers, in_dim = [], obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Obs extraction (must match dagger_distill.py) ───────────────────────────

def extract_student_obs(teacher_obs):
    lin_ang = teacher_obs[:, 0:6]
    goal = teacher_obs[:, 9:LIDAR_START]
    lidar = teacher_obs[:, LIDAR_START:LIDAR_END]
    front = torch.cat([
        lidar[:, s:e].min(dim=1, keepdim=True).values / LIDAR_RANGE
        for s, e in FRONT_BEAM_SECTORS
    ], dim=1)
    return torch.cat([lin_ang, goal, front], dim=-1)


def _beam_angle_deg(beam_idx):
    return -179.0 + beam_idx * 6.0


# ── P1: static sector consistency check ──────────────────────────────────────

def p1_check_sector_consistency(ckpt):
    print("\n--- [P1] Sector consistency ---")
    saved = ckpt.get("front_beam_sectors", None)
    if saved is None:
        print("  WARN: checkpoint has no 'front_beam_sectors' key (old format).")
        return True
    ok = (saved == FRONT_BEAM_SECTORS)
    if ok:
        print("  PASS: checkpoint sectors match script sectors.")
    else:
        print("  FAIL: MISMATCH!")
        print(f"    checkpoint : {saved}")
        print(f"    this script: {FRONT_BEAM_SECTORS}")
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load student
    ckpt = torch.load(
        os.path.abspath(args_cli.student_checkpoint),
        map_location="cpu",
        weights_only=False,
    )
    obs_dim = ckpt.get("student_obs_dim", STUDENT_OBS_DIM)
    action_dim = ckpt.get("action_dim", ACTION_DIM)
    hidden_dims = [
        v.shape[0]
        for k, v in ckpt["student"].items()
        if k.endswith(".weight") and "net" in k and v.shape[0] != action_dim
    ]

    print(f"\n[INFO] Student obs_dim={obs_dim}  action_dim={action_dim}  hidden={hidden_dims}")
    if obs_dim != STUDENT_OBS_DIM:
        print(f"  WARN: expected {STUDENT_OBS_DIM}, got {obs_dim}. Shape mismatch likely.")

    p1_ok = p1_check_sector_consistency(ckpt)

    # Environment
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cast(
        dict[str, Any],
        load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point"),
    )
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env_raw.unwrapped, DirectMARLEnv):
        env_raw = multi_agent_to_single_agent(env_raw)
    env = RlGamesVecEnvWrapper(env_raw, rl_device, clip_obs, clip_actions)

    # Register rlgpu so Runner.create_player() can find the env
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
    )

    device = torch.device(rl_device)
    N = env.unwrapped.num_envs
    ei = min(args_cli.env_idx, N - 1)

    # Student + normalizer
    student = StudentPolicy(obs_dim, action_dim, hidden_dims).to(device)
    normalizer = RunningNormalizer(obs_dim).to(device)
    student.load_state_dict(ckpt["student"])
    normalizer.load_state_dict(ckpt["normalizer"])
    student.eval()

    # Teacher (optional, P4)
    teacher = None
    if args_cli.teacher_checkpoint:
        teacher_cfg = dict(agent_cfg)
        teacher_cfg["params"]["load_checkpoint"] = True
        teacher_cfg["params"]["load_path"] = os.path.abspath(args_cli.teacher_checkpoint)
        teacher_cfg["params"]["config"]["num_actors"] = args_cli.num_envs
        runner = Runner()
        runner.load(teacher_cfg)
        teacher = runner.create_player()
        teacher.restore(os.path.abspath(args_cli.teacher_checkpoint))
        teacher.reset()
        teacher.is_deterministic = True
        print("[INFO] Teacher loaded for P4.")
    else:
        print("[INFO] No teacher checkpoint -- P4 skipped.")

    # Counters
    p2_fail = 0
    p3_data = []
    p4_agree = 0
    p4_total = 0
    p5_fail = 0
    max_potential = 0.0
    min_lidar_ever = 1.0
    total_steps = 0

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    print(f"\n[INFO] Running {args_cli.steps} steps on {N} envs (detail env={ei}).\n")
    print(
        f"{'step':>5}  {'sectors':^29}  "
        f"{'near_bm':>7}  {'pot':>5}  "
        f"{'gt_s':>4}  {'act_s':>5}  "
        f"{'s_yaw':>6}"
    )
    print("-" * 80)

    with torch.inference_mode():
        for step in range(args_cli.steps):
            full_obs = obs.float()
            student_obs = extract_student_obs(full_obs)
            norm_obs = normalizer.normalize(student_obs)
            actions = student(norm_obs)

            # LiDAR slices
            lidar60 = full_obs[:, LIDAR_START:LIDAR_END]
            sectors = student_obs[:, 11:16]

            # Teacher obs extras: nearest_obs_dir_b (74:76), potential (76)
            potential = full_obs[ei, 76].item()
            max_potential = max(max_potential, potential)

            # P2: range check
            if sectors.min() < -0.01 or sectors.max() > 1.01:
                p2_fail += 1

            # Nearest beam across all 60 (ground truth)
            min_val, min_idx = lidar60[ei].min(dim=0)
            min_val = min_val.item()
            min_idx = min_idx.item()
            min_lidar_ever = min(min_lidar_ever, min_val)
            min_angle = _beam_angle_deg(min_idx)

            # Which front sector does the nearest beam fall in?
            gt_sector = None
            for si, (s, e) in enumerate(FRONT_BEAM_SECTORS):
                if s <= min_idx < e:
                    gt_sector = si
                    break

            # Which sector has the minimum obs value?
            active_sector = sectors[ei].argmin().item()

            close = min_val < args_cli.close_threshold  # min_val is raw metres
            if gt_sector is not None and close:
                p3_data.append((gt_sector, active_sector))

            # P4: teacher-student yaw sign
            # Call teacher.model directly to avoid get_action()'s unsqueeze
            # (which turns (N,77) → (1,N,77) when has_batch_dimension=False).
            t_yaw = None
            if teacher is not None and close:
                t_input = {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": full_obs.to(device),
                    "rnn_states": None,
                }
                t_res = teacher.model(t_input)
                t_actions_all = t_res["mus"]  # (N, 4), deterministic
                t_yaw = float(t_actions_all[ei, 0])
                s_yaw = float(actions[ei, 0])
                p4_total += 1
                if t_yaw * s_yaw > 0 or (abs(t_yaw) < 0.05 and abs(s_yaw) < 0.05):
                    p4_agree += 1

            # P5: goal direction sanity
            goal_dir_b = student_obs[ei, 6:9]
            dist_2d = student_obs[ei, 9].item()
            if dist_2d > 1.0 and goal_dir_b[0] < -0.9:
                p5_fail += 1

            # Per-step print
            if args_cli.print_every > 0 and step % args_cli.print_every == 0:
                s_vals = " ".join(f"{v:.2f}" for v in sectors[ei].tolist())
                s_act = float(actions[ei, 0])
                mismatch = (
                    " *** P3!" if gt_sector is not None and gt_sector != active_sector else ""
                )
                t_str = f" t={t_yaw:+.2f}" if t_yaw is not None else ""
                print(
                    f"{step:5d}  [{s_vals}]  "
                    f"b{min_idx}({min_angle:+.0f}d)={min_val:.2f}  "
                    f"pot={potential:.3f}  "
                    f"gt={gt_sector!s:>4}  act={active_sector}  "
                    f"yaw={s_act:+.3f}{t_str}{mismatch}"
                )

            obs, _, _, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]
            total_steps += 1

    # Summary
    print("\n" + "=" * 60)
    print("  AVOIDANCE DEBUG SUMMARY")
    print("=" * 60)

    # Environment sanity
    print(f"  Min lidar value seen   : {min_lidar_ever:.3f}  "
          f"({'obstacle detected' if min_lidar_ever < 0.99 else 'NO OBSTACLES IN RANGE -- see below'})")
    print(f"  Max potential seen     : {max_potential:.4f}  "
          f"({'obstacle nearby' if max_potential > 0.01 else 'always far from obstacles'})")
    if min_lidar_ever >= 0.99:
        print()
        print("  !! All lidar beams stayed at max range for all steps.")
        print("     Likely causes:")
        print("     1. Too few steps or envs -- try --num_envs 32 --steps 2000")
        print("     2. Lidar vertical_fov_range=(10,20) looks above obstacle tops")
        print("        at the drone's flying altitude -- check lidar elevation config")
        print("     3. Drone spawn range too narrow, always in obstacle-free zone")
        print()

    # P1
    print(f"  [P1] Sector consistency   : {'PASS' if p1_ok else 'FAIL'}")

    # P2
    p2_status = "PASS" if p2_fail == 0 else f"FAIL ({p2_fail}/{total_steps} steps)"
    print(f"  [P2] LiDAR obs range      : {p2_status}")

    # P3
    if p3_data:
        match = sum(1 for gt, ac in p3_data if gt == ac)
        pct = 100.0 * match / len(p3_data)
        p3_tag = "PASS" if pct >= 80 else "FAIL"
        print(f"  [P3] Sector-direction map : {p3_tag} ({match}/{len(p3_data)} = {pct:.0f}%)")
        if pct < 80:
            print("       gt->active counts:")
            for (gt, ac), cnt in sorted(Counter(p3_data).items()):
                tag = "ok" if gt == ac else "!!"
                print(f"         {tag} gt={gt} -> active={ac}  x{cnt}")
    else:
        print("  [P3] Sector-direction map : SKIP (no close-obstacle steps)")

    # P4
    if p4_total > 0:
        pct = 100.0 * p4_agree / p4_total
        p4_tag = "PASS" if pct >= 70 else "FAIL"
        print(f"  [P4] Student-teacher yaw  : {p4_tag} ({p4_agree}/{p4_total} = {pct:.0f}%)")
    else:
        print("  [P4] Student-teacher yaw  : SKIP (no teacher or no close steps)")

    # P5
    p5_status = "PASS" if p5_fail == 0 else f"FAIL ({p5_fail} steps: goal behind drone)"
    print(f"  [P5] Goal direction obs   : {p5_status}")

    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
