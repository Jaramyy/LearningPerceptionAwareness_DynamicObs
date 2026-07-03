"""Verify that student_ros2_node.py obs reconstruction matches Isaac Lab training obs.

No PX4 or Gazebo required.  Uses Isaac Lab as the ground truth, then synthesises
the equivalent "PX4 sensor" inputs from Isaac's raw simulation state and applies
student_ros2_node.py's reconstruction logic.  Prints a per-dimension comparison
and a summary MAE table.

Frame reference:
  Isaac world    : NEUp  (X=North, Y=East, Z=Up)
  Isaac body     : FLU   (X=forward, Y=left,  Z=up)
  PX4 world      : NED   (X=North,  Y=East,   Z=down)
  PX4 body       : FRD   (X=forward, Y=right, Z=down)

Quaternion conversion (FLU->NEUp) -> (FRD->NED):
  R_P = diag(1,1,-1) @ R_I @ diag(1,-1,-1)
  Both B and A flip z (and B also swaps body-y), leaving det(R_P)=+1.

Run:
    ./isaaclab.sh -p scripts/rl_games/verify_obs.py \\
        --task Isaac-Agile-Lidar-Vel-PA-v0 --num_envs 1 --num_steps 500 --headless
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify Isaac vs PX4-reconstructed obs")
parser.add_argument("--task", type=str, default="Isaac-Agile-Lidar-Vel-PA-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=500,
                    help="Steps to collect before printing summary.")
parser.add_argument("--print_every", type=int, default=50,
                    help="Print per-step table every N steps (0 = summary only).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""All other imports after Isaac Sim is launched."""

import math
from typing import Any, cast

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import PerceptionAwareDrone.tasks  # noqa: F401

# ── Obs layout (must match dagger_distill.py) ────────────────────────────────
LIDAR_START = 14
LIDAR_END = 74
FRONT_BEAM_INDICES = [28, 29, 30, 31, 32]
STUDENT_OBS_DIM = 14 + len(FRONT_BEAM_INDICES)  # 19
LIDAR_RANGE = 5.0

OBS_NAMES = [
    "lin_vel_x (fwd m/s)", "lin_vel_y (left m/s)", "lin_vel_z (up m/s)",
    "ang_vel_x (rad/s)",   "ang_vel_y (rad/s)",    "ang_vel_z (rad/s)",
    "gravity_x",           "gravity_y",             "gravity_z",
    "goal_dir_x (fwd)",    "goal_dir_y (left)",     "goal_dir_z (up)",
    "dist_2d (m)",         "dist_z (m)",
    "beam_28 (-11°, m)", "beam_29 (-5°, m)", "beam_30 (+1°, m)",
    "beam_31 (+7°, m)",  "beam_32 (+13°, m)",
]
assert len(OBS_NAMES) == STUDENT_OBS_DIM


# ── Rotation math helpers ─────────────────────────────────────────────────────

def quat_to_rot_matrix(q_wxyz: torch.Tensor) -> torch.Tensor:
    """(4,) wxyz → (3,3) rotation matrix."""
    w, x, y, z = q_wxyz.float().unbind(-1)
    return torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ]).reshape(3, 3)


def ned_to_flu_via_matrix(R_P: torch.Tensor, v_ned: torch.Tensor) -> torch.Tensor:
    """Rotate NED world vector to FLU body using the FRD->NED rotation matrix.

    R_P.T rotates NED->FRD, then we negate Y and Z to go FRD->FLU.
    """
    v_frd = R_P.T @ v_ned           # NED -> FRD body
    return torch.stack([v_frd[0], -v_frd[1], -v_frd[2]])   # FRD -> FLU (stays on same device)


def isaac_to_R_px4(q_isaac_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert Isaac quaternion (FLU->NEUp) to PX4 rotation matrix (FRD->NED).

    R_P = B @ R_I @ A
    where B = diag(1, 1,-1)  maps NEUp -> NED  (negate world-z)
          A = diag(1,-1,-1)  maps FRD -> FLU   (negate body-y and body-z)
    det(R_P) = det(B)*det(R_I)*det(A) = (-1)*(+1)*(-1) = +1  (valid rotation)
    """
    device = q_isaac_wxyz.device
    R_I = quat_to_rot_matrix(q_isaac_wxyz)
    B = torch.diag(torch.tensor([1., 1., -1.], device=device))
    A = torch.diag(torch.tensor([1., -1., -1.], device=device))
    return B @ R_I @ A


# ── Isaac native obs extractor ────────────────────────────────────────────────

def extract_isaac_student_obs(teacher_obs_77: torch.Tensor) -> torch.Tensor:
    """Same as extract_student_obs in dagger_distill.py (env 0 only)."""
    base = teacher_obs_77[0, :LIDAR_START]                                  # (14,)
    lidar = teacher_obs_77[0, LIDAR_START:LIDAR_END]                        # (60,)
    front = lidar[FRONT_BEAM_INDICES]                                        # (5,)
    return torch.cat([base, front])                                           # (19,)


# ── PX4-style obs builder (mirrors student_ros2_node.py) ─────────────────────

def build_px4_style_obs(raw_env, goal_ned: tuple) -> torch.Tensor:
    """Reconstruct 19D obs from Isaac raw state using PX4 sensor conventions.

    Simulates exactly what student_ros2_node._build_obs() would receive if PX4
    uXRCE-DDS topics reported the same physical state as Isaac Lab.
    """
    device = raw_env.device

    # ── Raw Isaac state (env 0) ───────────────────────────────────────────────
    pos_neup = raw_env._robot.data.root_pos_w[0].float()        # NEUp
    vel_neup = raw_env._robot.data.root_lin_vel_w[0].float()    # NEUp
    q_isaac = raw_env._robot.data.root_quat_w[0].float()        # wxyz FLU->NEUp
    ang_flu = raw_env._robot.data.root_ang_vel_b[0].float()     # FLU body

    # ── Synthetic PX4 NED inputs ──────────────────────────────────────────────
    # NEUp -> NED: flip z
    pos_ned = torch.tensor([pos_neup[0], pos_neup[1], -pos_neup[2]], device=device)
    vel_ned = torch.tensor([vel_neup[0], vel_neup[1], -vel_neup[2]], device=device)

    # FLU -> FRD: negate y and z
    ang_frd = torch.tensor([ang_flu[0].item(), -ang_flu[1].item(), -ang_flu[2].item()], device=device)

    # PX4 rotation matrix (FRD->NED)
    R_P = isaac_to_R_px4(q_isaac)

    # ── Reconstruct obs (same formulas as student_ros2_node._build_obs) ───────

    # [0:3] Linear velocity in FLU body frame
    vel_flu = ned_to_flu_via_matrix(R_P, vel_ned)

    # [3:6] Angular velocity in FLU body frame (FRD->FLU: negate y,z)
    ang_flu_recon = torch.tensor([ang_frd[0].item(), -ang_frd[1].item(), -ang_frd[2].item()], device=device)

    # [6:9] Projected gravity in FLU body frame
    # gravity in NED = unit down = (0, 0, +1)
    grav_ned = torch.tensor([0., 0., 1.], device=device)
    grav_flu = ned_to_flu_via_matrix(R_P, grav_ned)
    grav_flu = grav_flu / (grav_flu.norm() + 1e-6)

    # [9:12] Unit goal direction in FLU body frame
    gN, gE, gD = goal_ned
    pN, pE, pD = pos_ned.tolist()
    dN, dE, dD = gN - pN, gE - pE, gD - pD
    goal_ned_vec = torch.tensor([dN, dE, dD], device=device)
    goal_flu = ned_to_flu_via_matrix(R_P, goal_ned_vec)
    goal_flu = goal_flu / (goal_flu.norm() + 1e-6)

    # [12] 2D horizontal distance
    dist_2d = math.sqrt(dN**2 + dE**2)

    # [13] Vertical distance: positive = goal above drone
    # NED Down: goal above <=> goal_D < pos_D => dist_z = pos_D - goal_D > 0
    dist_z = pD - gD

    # [14:19] Front LiDAR beams (same sensor, no frame change needed)
    lidar_scan = raw_env.lidar_scan  # (N, ?, beams) or (N, beams)
    if lidar_scan.dim() == 3:
        beams = lidar_scan[0, 0, FRONT_BEAM_INDICES].float()
    else:
        beams = lidar_scan[0, FRONT_BEAM_INDICES].float()
    beams = beams.clamp(max=LIDAR_RANGE) / LIDAR_RANGE  # normalize to [0, 1] — matches Isaac obs

    return torch.cat([
        vel_flu,
        ang_flu_recon,
        grav_flu,
        goal_flu,
        torch.tensor([dist_2d, dist_z], device=device),
        beams,
    ])


# ── Print helpers ─────────────────────────────────────────────────────────────

_COL = 22   # column width

def _print_header():
    print(f"\n{'Dim':<4} {'Name':<25} {'Isaac':>{_COL}} {'PX4-recon':>{_COL}} {'|diff|':>10}")
    print("-" * (4 + 25 + _COL + _COL + 10 + 6))


def _print_row(i, name, isaac_v, px4_v):
    diff = abs(isaac_v - px4_v)
    flag = " *** HIGH" if diff > 0.15 else ""
    print(f"{i:<4} {name:<25} {isaac_v:>{_COL}.5f} {px4_v:>{_COL}.5f} {diff:>10.5f}{flag}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Environment setup
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    agent_cfg = cast(dict[str, Any], load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point"))
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env_raw = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env_raw.unwrapped, DirectMARLEnv):
        env_raw = multi_agent_to_single_agent(env_raw)
    env = RlGamesVecEnvWrapper(env_raw, rl_device, clip_obs, clip_actions)

    raw_env = env_raw.unwrapped
    device = torch.device(rl_device)

    # 2. Reset and pin a fixed goal so PX4-style goal_ned is meaningful
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    # Fix goal: 5m North, 0m East, 1.5m Up in Isaac world (NEUp)
    GOAL_NEUP = torch.tensor([5.0, 0.0, 1.5], device=device)
    raw_env._desired_pos_w[:] = GOAL_NEUP
    # Equivalent goal in NED: flip z
    GOAL_NED = (GOAL_NEUP[0].item(), GOAL_NEUP[1].item(), -GOAL_NEUP[2].item())

    print(f"\n[VERIFY] Goal (Isaac NEUp) : {GOAL_NEUP.tolist()}")
    print(f"[VERIFY] Goal (PX4 NED)    : {GOAL_NED}")
    print(f"[VERIFY] Steps             : {args_cli.num_steps}")
    print(f"[VERIFY] Print every       : {args_cli.print_every} steps")

    # 3. Accumulate errors
    errors = torch.zeros(STUDENT_OBS_DIM)
    max_errors = torch.zeros(STUDENT_OBS_DIM)
    n_accumulated = 0

    # 4. Step loop
    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break

            # Zero action → drone will fall, but we still get valid obs comparisons
            actions = torch.zeros(args_cli.num_envs, 4, device=device)
            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

            # Re-pin goal after episode reset
            if dones.any():
                raw_env._desired_pos_w[:] = GOAL_NEUP
                # Re-normalise by skipping comparison this step (state was just reset)
                continue

            teacher_obs = obs.float()   # (1, 77)

            # (A) Isaac native 19D obs
            obs_isaac = extract_isaac_student_obs(teacher_obs).cpu()

            # (B) PX4-style reconstructed 19D obs
            obs_px4 = build_px4_style_obs(raw_env, GOAL_NED).cpu()

            diff = (obs_isaac - obs_px4).abs()
            errors += diff
            max_errors = torch.max(max_errors, diff)
            n_accumulated += 1

            if args_cli.print_every > 0 and step % args_cli.print_every == 0:
                print(f"\n{'='*80}")
                print(f"  Step {step}")
                _print_header()
                for i, name in enumerate(OBS_NAMES):
                    _print_row(i, name, obs_isaac[i].item(), obs_px4[i].item())

    env.close()

    # 5. Summary
    if n_accumulated == 0:
        print("\n[VERIFY] No steps collected.")
        return

    mae = errors / n_accumulated

    print(f"\n\n{'='*80}")
    print(f"  SUMMARY over {n_accumulated} steps")
    print(f"{'='*80}")
    print(f"{'Dim':<4} {'Name':<25} {'MAE':>12} {'Max|diff|':>12}  Status")
    print("-" * 70)
    all_ok = True
    for i, name in enumerate(OBS_NAMES):
        ok = mae[i].item() < 0.05
        if not ok:
            all_ok = False
        flag = "OK" if ok else "MISMATCH <---"
        print(f"{i:<4} {name:<25} {mae[i].item():>12.5f} {max_errors[i].item():>12.5f}  {flag}")

    print("=" * 80)
    if all_ok:
        print("  RESULT: ALL dimensions match (MAE < 0.05).  Obs reconstruction is CORRECT.")
    else:
        print("  RESULT: Some dimensions have MAE >= 0.05.  Check MISMATCH rows above.")
        print("  Common causes:")
        print("   lin_vel / ang_vel mismatch -> wrong FRD<->FLU sign convention")
        print("   gravity mismatch           -> wrong gravity direction (NED down = +1)")
        print("   goal_dir mismatch          -> goal_ned z-sign wrong (alt vs D)")
        print("   dist_z mismatch            -> check pD - gD vs gD - pD")
        print("   beam mismatch              -> LiDAR beam indices or z-offset in sensor")
    print("=" * 80)


if __name__ == "__main__":
    main()
    simulation_app.close()
