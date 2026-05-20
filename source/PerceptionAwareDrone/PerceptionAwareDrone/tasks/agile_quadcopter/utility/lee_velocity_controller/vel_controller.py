from collections.abc import Sequence
from functools import partial

import torch

class GeometricVelocityController:

    def __init__(self, num_env, mass, inertia, device):
        self.num_env = num_env
        self.mass = mass

        # self.J = inertia.unsqueeze(0).to(device)    # (1,3,3)
        # self.J = inertia.unsqueeze(0).to(device)  
        self.J = inertia.unsqueeze(0).expand(self.num_env, 3, 3).to(device)
        self.J_expand = inertia.unsqueeze(0).expand(self.num_env, 3, 3).to(device)
        self.device = device
        

        # Position gains
        self.k_p = 8.0 
        self.k_d = 25.0

        self.k_d_xy = 22.0
        self.k_d_z = 80.0
        
        # FOR 1/200
        # self.k_p = 16.0 # WORKED at 16.0
        # self.k_d = 180.0

        # Attitude gains
        self.kW = 0.1 #0.3

        self.kR = 1.25 #4.5
        self.kv = 35.0  # velocity error gain for velocity-only control

        # Gravity
        self.g = torch.tensor([0., 0., -9.81], device=device).unsqueeze(0)


    # --------------------
    # Utility functions
    # --------------------
    def vee_map(self, M):
        # Extract [M32 - M23, M13 - M31, M21 - M12]
        return torch.stack((
            M[:, 2, 1] - M[:, 1, 2],
            M[:, 0, 2] - M[:, 2, 0],
            M[:, 1, 0] - M[:, 0, 1],
        ), dim=1)

    def matrix_from_quat(self, q):
        """ IsaacLab format: q = (w,x,y,z) """
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.zeros((q.shape[0], 3, 3), device=q.device)

        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)

        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - x * w)

        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)

        return R

    # ----------------------------------------
    # MAIN CONTROL STEP
    # ----------------------------------------
    def update_velocity_only_edit(
        self,
        vel_w, quat_w, omega_b,
        desired_vel_w,
        desired_yaw_rate
    ):
        """
        Velocity-only control with yaw-rate tracking.
        Inputs:
            pos_w          : current position (1,3) [unused, velocity-only]
            vel_w          : current linear velocity (1,3)
            quat_w         : current orientation quaternion (1,4)
            omega_b        : current body angular velocity (1,3)
            desired_vel_w  : commanded linear velocity (1,3)
            desired_yaw_rate : commanded body-frame yaw-rate (rad/s)
        Outputs:
            thrust  : scalar thrust along body z-axis
            torque  : body-frame torque (1,3)
        """
        # vel_w = vel_w.view(1,3)
        # quat_w = quat_w.view(1,4)
        # omega_b = omega_b.view(1,3)
        # desired_vel_w = desired_vel_w.view(1,3)
        R = self.matrix_from_quat(quat_w)
        
        # ---------------------------
        # 1) VELOCITY ERROR
        # ---------------------------
        ev = vel_w - desired_vel_w

        # ---------------------------
        # 2) Thrust magnitude (PD on velocity error + gravity compensation)
        # ---------------------------
        eps = 1e-8
        # ev: (B,3)
        e3 = torch.tensor([0., 0., 1.], device=self.device).view(1,3)      # (1,3)
        e3 = e3.expand(ev.shape[0], -1)                                  # (B,3)

        # A: desired total acceleration/force direction term, (B,3)
        A = (-self.kv * ev) - (self.mass * self.g * e3 * 1.89) # 1.89 is a tuning factor to get better height tracking, can be removed if you want exact gravity compensation 
        # A = (self.kv * ev) + (self.mass * self.g * e3)

        # Re3: world-frame body z-axis (B,3)
        Re3 = torch.matmul(R, e3.unsqueeze(-1)).squeeze(-1)              # (B,3)

        # thrust per batch (scalar per item)
        # f = dot(A, Re3)  (see sign conventions in your controller)
        f = torch.sum(A * Re3, dim=1)                                    # (B,)

        # desired body axes
        A_norm = torch.norm(A, dim=1, keepdim=True).clamp_min(eps)       # (B,1)
        b3c = A / A_norm                                                # (B,3)

        b1d = R[:, :, 0]                                                 # (B,3)  (current body x-axis)

        C = torch.cross(b3c, b1d, dim=1)                                 # (B,3)
        C_norm = torch.norm(C, dim=1, keepdim=True).clamp_min(eps)       # (B,1)

        b2c = C / C_norm                                                 # (B,3)
        b1c = -torch.cross(b3c, C, dim=1) / C_norm                       # (B,3)

        # Compose desired rotation matrix Rc with columns [b1c, b2c, b3c]
        Rc = torch.stack((b1c, b2c, b3c), dim=2)                         # (B,3,3)
        omega_c = torch.zeros_like(omega_b)
        omega_c[:,2] = desired_yaw_rate  # yaw-rate tracking


        er = 0.5*self.vee_map(Rc.transpose(1,2) @ R - R.transpose(1,2) @ Rc)
        eOmega = omega_b - omega_c


        J_omega_c = torch.bmm(self.J, omega_c.unsqueeze(2)).squeeze(2)

        M = (-self.kR * er) - (self.kW * eOmega) + torch.cross(omega_b, J_omega_c, dim=1) 

        # print(f"Velocity error: {ev}, Thrust command: {f}")

        return f, M