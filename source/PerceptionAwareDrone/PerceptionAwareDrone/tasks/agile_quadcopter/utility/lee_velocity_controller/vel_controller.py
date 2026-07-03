# from collections.abc import Sequence
# from functools import partial

# import torch

# class GeometricVelocityController:

#     def __init__(self, num_env, mass, inertia, device):
#         self.num_env = num_env
#         self.mass = mass

#         # self.J = inertia.unsqueeze(0).to(device)    # (1,3,3)
#         # self.J = inertia.unsqueeze(0).to(device)  
#         self.J = inertia.unsqueeze(0).expand(self.num_env, 3, 3).to(device)
#         self.J_expand = inertia.unsqueeze(0).expand(self.num_env, 3, 3).to(device)
#         self.device = device
        

#         # Position gains
#         self.k_p = 8.0 
#         self.k_d = 25.0

#         self.k_d_xy = 22.0
#         self.k_d_z = 80.0
        
#         # FOR 1/200
#         # self.k_p = 16.0 # WORKED at 16.0
#         # self.k_d = 180.0

#         # Attitude gains
#         self.kW = 0.1 #0.3

#         self.kR = 1.25 #4.5
#         self.kv = 35.0  # velocity error gain for velocity-only control

#         # Gravity
#         self.g = torch.tensor([0., 0., -9.81], device=device).unsqueeze(0)


#     # --------------------
#     # Utility functions
#     # --------------------
#     def vee_map(self, M):
#         # Extract [M32 - M23, M13 - M31, M21 - M12]
#         return torch.stack((
#             M[:, 2, 1] - M[:, 1, 2],
#             M[:, 0, 2] - M[:, 2, 0],
#             M[:, 1, 0] - M[:, 0, 1],
#         ), dim=1)

#     def matrix_from_quat(self, q):
#         """ IsaacLab format: q = (w,x,y,z) """
#         w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

#         R = torch.zeros((q.shape[0], 3, 3), device=q.device)

#         R[:, 0, 0] = 1 - 2 * (y * y + z * z)
#         R[:, 0, 1] = 2 * (x * y - z * w)
#         R[:, 0, 2] = 2 * (x * z + y * w)

#         R[:, 1, 0] = 2 * (x * y + z * w)
#         R[:, 1, 1] = 1 - 2 * (x * x + z * z)
#         R[:, 1, 2] = 2 * (y * z - x * w)

#         R[:, 2, 0] = 2 * (x * z - y * w)
#         R[:, 2, 1] = 2 * (y * z + x * w)
#         R[:, 2, 2] = 1 - 2 * (x * x + y * y)

#         return R

#     # ----------------------------------------
#     # MAIN CONTROL STEP
#     # ----------------------------------------
#     def update_velocity_only_edit(
#         self,
#         vel_w, quat_w, omega_b,
#         desired_vel_w,
#         desired_yaw_rate
#     ):
#         """
#         Velocity-only control with yaw-rate tracking.
#         Inputs:
#             pos_w          : current position (1,3) [unused, velocity-only]
#             vel_w          : current linear velocity (1,3)
#             quat_w         : current orientation quaternion (1,4)
#             omega_b        : current body angular velocity (1,3)
#             desired_vel_w  : commanded linear velocity (1,3)
#             desired_yaw_rate : commanded body-frame yaw-rate (rad/s)
#         Outputs:
#             thrust  : scalar thrust along body z-axis
#             torque  : body-frame torque (1,3)
#         """
#         # vel_w = vel_w.view(1,3)
#         # quat_w = quat_w.view(1,4)
#         # omega_b = omega_b.view(1,3)
#         # desired_vel_w = desired_vel_w.view(1,3)
#         R = self.matrix_from_quat(quat_w)
        
#         # ---------------------------
#         # 1) VELOCITY ERROR
#         # ---------------------------
#         ev = vel_w - desired_vel_w

#         # ---------------------------
#         # 2) Thrust magnitude (PD on velocity error + gravity compensation)
#         # ---------------------------
#         eps = 1e-8
#         # ev: (B,3)
#         e3 = torch.tensor([0., 0., 1.], device=self.device).view(1,3)      # (1,3)
#         e3 = e3.expand(ev.shape[0], -1)                                  # (B,3)

#         # A: desired total acceleration/force direction term, (B,3)
#         A = (-self.kv * ev) - (self.mass * self.g * e3 * 1.89) # 1.89 is a tuning factor to get better height tracking, can be removed if you want exact gravity compensation 
#         # A = (self.kv * ev) + (self.mass * self.g * e3)

#         # Re3: world-frame body z-axis (B,3)
#         Re3 = torch.matmul(R, e3.unsqueeze(-1)).squeeze(-1)              # (B,3)

#         # thrust per batch (scalar per item)
#         # f = dot(A, Re3)  (see sign conventions in your controller)
#         f = torch.sum(A * Re3, dim=1)                                    # (B,)

#         # desired body axes
#         A_norm = torch.norm(A, dim=1, keepdim=True).clamp_min(eps)       # (B,1)
#         b3c = A / A_norm                                                # (B,3)

#         b1d = R[:, :, 0]                                                 # (B,3)  (current body x-axis)

#         C = torch.cross(b3c, b1d, dim=1)                                 # (B,3)
#         C_norm = torch.norm(C, dim=1, keepdim=True).clamp_min(eps)       # (B,1)

#         b2c = C / C_norm                                                 # (B,3)
#         b1c = -torch.cross(b3c, C, dim=1) / C_norm                       # (B,3)

#         # Compose desired rotation matrix Rc with columns [b1c, b2c, b3c]
#         Rc = torch.stack((b1c, b2c, b3c), dim=2)                         # (B,3,3)
#         omega_c = torch.zeros_like(omega_b)
#         omega_c[:,2] = desired_yaw_rate  # yaw-rate tracking


#         er = 0.5*self.vee_map(Rc.transpose(1,2) @ R - R.transpose(1,2) @ Rc)
#         eOmega = omega_b - omega_c


#         J_omega_c = torch.bmm(self.J, omega_c.unsqueeze(2)).squeeze(2)

#         M = (-self.kR * er) - (self.kW * eOmega) + torch.cross(omega_b, J_omega_c, dim=1) 

#         # print(f"Velocity error: {ev}, Thrust command: {f}")

#         return f, M


# vel_controller.py — Geometric Velocity Controller
# Redesigned to match real drone flight controller behaviour:
#   - Body-frame velocity error (not world frame)
#   - Per-axis PID with integrator and anti-windup
#   - Max tilt angle constraint (prevents flips)
#   - Yaw decoupled from roll/pitch thrust computation
#   - Thrust clamped to physical motor limits

import torch


class GeometricVelocityController:

    def __init__(self, num_env, mass, inertia, device):
        self.num_env = num_env
        self.mass    = mass
        self.device  = device

        self.J = inertia.unsqueeze(0).expand(num_env, 3, 3).to(device)

        # ── Velocity PID gains (per axis) ────────────────────────────
        # XY (horizontal) and Z (altitude) have very different dynamics
        self.kv_xy = 22.0   # horizontal velocity P gain
        self.kv_z  = 80.0   # vertical velocity P gain
        self.ki_xy = 0.3    # horizontal integral gain (removes wind drift)
        self.ki_z  = 1.5    # vertical integral gain   (removes steady hover error)

        # Anti-windup clamp on integrator (body frame, m/s·s)
        self.int_clamp_xy = 1.5
        self.int_clamp_z  = 2.0

        # ── Attitude gains ────────────────────────────────────────────
        self.kR = 1.25   # attitude error gain
        self.kW = 0.1    # angular rate error gain

        # ── Tilt limit ────────────────────────────────────────────────
        # Hard cap on how far the controller can tilt the drone.
        # 35 deg is typical for a racing/agile drone; lower = more stable.
        MAX_TILT_DEG      = 35.0
        self.max_tilt_cos = torch.cos(
            torch.tensor(MAX_TILT_DEG * torch.pi / 180.0)
        ).to(device)

        # ── Thrust limits ─────────────────────────────────────────────
        # Clamp thrust between 10% hover (prevent motors stopping) and
        # 250% hover (physical motor maximum)
        self.thrust_min = float(mass * 9.81 * 0.1)
        self.thrust_max = float(mass * 9.81 * 2.5)

        # ── Gravity ───────────────────────────────────────────────────
        self.g_vec = torch.tensor([0., 0., -9.81], device=device).view(1, 3)

        # ── Integrator state ──────────────────────────────────────────
        # Reset per environment on episode reset (call reset())
        self.vel_int = torch.zeros(num_env, 3, device=device)

        # ── dt for integrator (set from sim dt × decimation) ──────────
        # Default: 1/200 * 2 = 0.01 s  — override if needed
        self.dt = 1.0 / 200.0 * 2.0

    # ----------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset integrator for given environments (call from _reset_idx)."""
        if env_ids is None:
            self.vel_int.zero_()
        else:
            self.vel_int[env_ids] = 0.0

    # ----------------------------------------------------------------
    @staticmethod
    def _matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
        """Isaac Lab quaternion convention: q = (w, x, y, z)."""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
        R[:, 0, 0] = 1 - 2*(y*y + z*z)
        R[:, 0, 1] = 2*(x*y - z*w)
        R[:, 0, 2] = 2*(x*z + y*w)
        R[:, 1, 0] = 2*(x*y + z*w)
        R[:, 1, 1] = 1 - 2*(x*x + z*z)
        R[:, 1, 2] = 2*(y*z - x*w)
        R[:, 2, 0] = 2*(x*z - y*w)
        R[:, 2, 1] = 2*(y*z + x*w)
        R[:, 2, 2] = 1 - 2*(x*x + y*y)
        return R

    @staticmethod
    def _vee_map(M: torch.Tensor) -> torch.Tensor:
        """SO(3) vee map: skew-symmetric matrix → vector."""
        return torch.stack((
            M[:, 2, 1] - M[:, 1, 2],
            M[:, 0, 2] - M[:, 2, 0],
            M[:, 1, 0] - M[:, 0, 1],
        ), dim=1)

    # ----------------------------------------------------------------
    def update_velocity_only_edit(
        self,
        vel_w:          torch.Tensor,   # (B, 3) world-frame linear velocity
        quat_w:         torch.Tensor,   # (B, 4) world-frame quaternion (w,x,y,z)
        omega_b:        torch.Tensor,   # (B, 3) body-frame angular velocity
        desired_vel_w:  torch.Tensor,   # (B, 3) desired world-frame velocity
        desired_yaw_rate: torch.Tensor, # (B,)   desired yaw rate (rad/s)
    ):
        """
        Real-drone-style geometric velocity controller.

        Returns
        -------
        thrust : (B,)   scalar thrust along body +Z axis
        torque : (B, 3) body-frame torque
        """
        B  = vel_w.shape[0]
        R  = self._matrix_from_quat(quat_w)          # (B,3,3)
        RT = R.transpose(1, 2)                         # (B,3,3)  R^T

        # ── 1. Velocity error in BODY frame ───────────────────────────
        # Real FCs operate in body-horizontal / NED frame.
        # Rotating both velocities into body frame before differencing
        # ensures the integral term doesn't drift due to yaw rotation.
        vel_b         = torch.bmm(RT, vel_w.unsqueeze(-1)).squeeze(-1)          # (B,3)
        desired_vel_b = torch.bmm(RT, desired_vel_w.unsqueeze(-1)).squeeze(-1)  # (B,3)
        ev = vel_b - desired_vel_b                                               # (B,3)

        # ── 2. Per-axis PID with anti-windup integrator ───────────────
        kv = torch.tensor(
            [self.kv_xy, self.kv_xy, self.kv_z], device=self.device
        ).view(1, 3)
        ki = torch.tensor(
            [self.ki_xy, self.ki_xy, self.ki_z], device=self.device
        ).view(1, 3)

        # Integrate velocity error; clamp per-axis to prevent windup
        self.vel_int += ev * self.dt
        self.vel_int[:, :2] = self.vel_int[:, :2].clamp(
            -self.int_clamp_xy, self.int_clamp_xy
        )
        self.vel_int[:, 2]  = self.vel_int[:, 2].clamp(
            -self.int_clamp_z,  self.int_clamp_z
        )

        # Acceleration demand in body frame (PID output)
        acc_demand_b = -(kv * ev) - (ki * self.vel_int)   # (B,3)

        # Rotate acceleration demand back to world frame
        acc_demand_w = torch.bmm(R, acc_demand_b.unsqueeze(-1)).squeeze(-1)  # (B,3)

        # Add gravity compensation in world frame
        # g_vec = [0, 0, -9.81], so subtracting adds upward force
        A = acc_demand_w - self.g_vec.expand(B, -1) * self.mass               # (B,3)

        # ── 3. Desired thrust direction (b3c) with TILT LIMIT ─────────
        A_norm = A.norm(dim=1, keepdim=True).clamp(min=1e-6)
        b3c    = A / A_norm                                                    # (B,3)

        # Clamp tilt: ensure b3c z-component never goes below cos(max_tilt)
        # This is what stops the drone from flipping — the controller
        # physically cannot command more tilt than this threshold.
        b3c_z  = b3c[:, 2].clamp(min=float(self.max_tilt_cos))               # (B,)
        b3c_xy = b3c[:, :2]                                                    # (B,2)
        b3c_xy_norm = b3c_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Rescale XY so the vector stays unit length given the clamped Z
        sin_max         = (1.0 - b3c_z**2).clamp(min=0.0).sqrt()             # (B,)
        scale           = sin_max.unsqueeze(1) / b3c_xy_norm                  # (B,1)
        # Only scale down if the original xy was larger than allowed
        scale           = scale.clamp(max=1.0)
        b3c_xy_clamped  = b3c_xy * scale                                      # (B,2)
        b3c = torch.cat(
            [b3c_xy_clamped, b3c_z.unsqueeze(1)], dim=-1
        )                                                                       # (B,3)
        # Re-normalise after clamping (z was clamped, xy rescaled — sum may ≠ 1)
        b3c = b3c / b3c.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # ── 4. Thrust scalar ──────────────────────────────────────────
        Re3 = R[:, :, 2]                                                       # (B,3) body z in world
        f   = (A * Re3).sum(dim=1)                                             # (B,)
        f   = f.clamp(self.thrust_min, self.thrust_max)                        # motor limits

        # ── 5. DECOUPLED desired attitude ─────────────────────────────
        # Extract current yaw from quaternion (Isaac Lab: w,x,y,z)
        qw, qx, qy, qz = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
        yaw_current = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy**2 + qz**2),
        )                                                                       # (B,)

        # Desired heading = hold current yaw (yaw *rate* is applied in
        # omega_c separately, decoupled from roll/pitch thrust direction)
        cos_yaw = torch.cos(yaw_current)
        sin_yaw = torch.sin(yaw_current)
        b1d = torch.stack(
            [cos_yaw, sin_yaw, torch.zeros_like(cos_yaw)], dim=1
        )                                                                       # (B,3)

        # Desired rotation matrix Rc from [b1d, b2c, b3c]
        C    = torch.cross(b3c, b1d, dim=1)                                    # (B,3)
        C    = C / C.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        b2c  = C
        b1c  = torch.cross(b2c, b3c, dim=1)                                   # (B,3)
        b1c  = b1c / b1c.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        Rc   = torch.stack([b1c, b2c, b3c], dim=2)                            # (B,3,3)

        # ── 6. Attitude error and torque ──────────────────────────────
        # Yaw rate command is applied only through omega_c[:,2];
        # it does not feed back into the attitude error rotation matrix.
        omega_c        = torch.zeros_like(omega_b)
        omega_c[:, 2]  = desired_yaw_rate                                      # (B,)

        eR      = 0.5 * self._vee_map(Rc.transpose(1, 2) @ R - RT @ Rc)       # (B,3)
        eOmega  = omega_b - omega_c                                             # (B,3)

        J_omega_c = torch.bmm(self.J, omega_c.unsqueeze(-1)).squeeze(-1)       # (B,3)

        torque = (
            -self.kR * eR
            - self.kW * eOmega
            + torch.cross(omega_b, J_omega_c, dim=1)
        )                                                                       # (B,3)

        return f, torque