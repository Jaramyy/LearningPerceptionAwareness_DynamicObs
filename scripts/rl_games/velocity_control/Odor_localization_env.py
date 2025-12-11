# (Your original license header omitted for brevity in this paste)
import argparse
import torch
import torch.nn as nn
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip
from PerceptionAwareDrone.tasks.agile_quadcopter.robot.agileDrone import AGILE_CFG

from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat,quat_from_matrix , normalize, quat_rotate, euler_xyz_from_quat, quat_mul,quat_inv

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Twist
from olfaction_msgs.msg import GasSensor
from olfaction_msgs.msg import Anemometer

# -------------------- ROS publisher --------------------
class PIDPlotPublisher(Node):
    def __init__(self, name="vel_pid_pub"):
        super().__init__(name)
        self.pub_error = self.create_publisher(Vector3, "/vel_pid/error", 10)
        self.pub_cmd = self.create_publisher(Vector3, "/vel_pid/command", 10)
        self.pub_vel = self.create_publisher(Vector3, "/drone/velocity", 10)
        self.pub_target = self.create_publisher(Vector3, "/drone/target_velocity", 10)

    def publish_vec(self, pub, vec):
        msg = Vector3()
        msg.x = float(vec[0])
        msg.y = float(vec[1])
        msg.z = float(vec[2])
        pub.publish(msg)

# -------------------- Vel Subscriber --------------------
class VelocityCommandSubscriber(Node):
    def __init__(self, name="vel_cmd_sub"):
        super().__init__(name)
        self.subscription = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.listener_callback,
            10,
        )
        self.subscription  # prevent unused variable warning
        self.cmd_velocity = torch.zeros(3)
        self.cmd_ang_velocity = torch.zeros(3)

    def listener_callback(self, msg):
        self.cmd_velocity[0] = msg.linear.x
        self.cmd_velocity[1] = msg.linear.y
        self.cmd_velocity[2] = msg.linear.z
        self.cmd_ang_velocity[0] = msg.angular.x
        self.cmd_ang_velocity[1] = msg.angular.y
        self.cmd_ang_velocity[2] = msg.angular.z

class gasSubscriber(Node):
    def __init__(self, name="gas_subscriber"):
        super().__init__(name)
        self.subscription = self.create_subscription(
            Vector3,
            "/gas_concentration",
            self.listener_callback,
            10,
        )
        self.subscription  # prevent unused variable warning
        self.gas_concentration = 0.0

    def listener_callback(self, msg):
        self.gas_concentration = msg.z  # Assuming gas concentration is published in the z field




# -------------------- Allocation & Motor classes kept (unchanged) --------------------
# (Use your existing Allocation and Motor classes from your script - unchanged)

# Paste your Allocation and Motor classes here (I assume they're defined below in your file).
# For brevity, I will reuse the names alloc_matrix and motor existing in your file.
class Allocation:
    def __init__(self, num_envs, device="cuda", dtype=torch.float32):
        """
        Initializes the allocation matrix for a quadrotor for multiple environments.

        Parameters:
        - num_envs (int): Number of environments
        - arm_length (float): Distance from the center to the rotor
        - thrust_coeff (float): Rotor thrust constant
        - drag_coeff (float): Rotor torque constant
        - device (str): 'cpu' or 'cuda'
        - dtype (torch.dtype): Desired tensor dtype
        """
        
        arm_length: float = 0.130 #0.035
        """Length of the arms of the drone in meters."""
        
        drag_coef: float = 1.5e-9
        """Drag torque coefficient."""
        
        thrust_coef: float = 5.327e-7 #2.25e-7
        """Thrust coefficient.
        Calculated with 5145 rad/s max angular velociy, thrust to weight: 4, mass: 0.6076 kg and gravity: 9.81 m/s^2.
        thrust_coef = (4 * 0.6076 * 9.81) / (4 * 5145**2) = 2.25e-7."""
        
        self.omega_max: float = 5145.0
        """Maximum angular velocity of the drone motors in rad/s.
        Calculated with 1950KV motor, with 6S LiPo battery with 4.2V per cell.
        1950 * 6 * 4.2 = 49,140 RPM ~= 5145 rad/s."""
        

        sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        A = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [arm_length * sqrt2_inv, -arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [-arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [drag_coef, -drag_coef, drag_coef, -drag_coef],
            ],
            dtype=dtype,
            device=device,
        )
        self._allocation_matrix = A.unsqueeze(0).repeat(num_envs, 1, 1)
        self._thrust_coeff = thrust_coef

    def compute(self, omega):
        """
        Computes the total thrust and body torques given the rotor angular velocities.

        Parameters:
        - omega (torch.Tensor): Tensor of shape (num_envs, 4) representing rotor angular velocities

        Returns:
        - thrust_torque (torch.Tensor): Tensor of shape (num_envs, 4)
        """

        
        thrusts_ref = self._thrust_coeff * omega**2
        thrusts_ref_batched = thrusts_ref.unsqueeze(0)
        # print(f"Thrusts reference: {thrusts_ref.unsqueeze(-1).shape}, Allocation matrix: {self._allocation_matrix.shape}")
        # thrust_torque = torch.bmm(self._allocation_matrix, thrusts_ref_batched.unsqueeze(-1)).squeeze(-1)
        # Ensure thrusts_ref_batched is 3D
        
        if thrusts_ref_batched.ndim == 1:
            # [4] → [1, 4, 1]
            thrusts_ref_batched = thrusts_ref_batched.unsqueeze(0).unsqueeze(-1)
        elif thrusts_ref_batched.ndim == 2:
            # [1, 4] or [B, 4] → [B, 4, 1]
            thrusts_ref_batched = thrusts_ref_batched.unsqueeze(-1)
        elif thrusts_ref_batched.ndim == 3 and thrusts_ref_batched.shape[1] == 1:
            # [B, 1, 4] → [B, 4, 1]
            thrusts_ref_batched = thrusts_ref_batched.transpose(1, 2)

        thrust_torque = torch.bmm(self._allocation_matrix, thrusts_ref_batched).squeeze(-1)

        
        # thrust_torque = torch.matmul(thrusts_ref, self._allocation_matrix.T) 
        return thrust_torque
    
    def get_omega_max(self):
        """
        Returns the maximum angular velocity of the motors.

        Returns:
        - omega_max (float): Maximum angular velocity in rad/s.
        """
        return self.omega_max


class Motor:
    def __init__(self, num_envs, taus, init, max_rate, min_rate, dt, use, device="cpu", dtype=torch.float32):
        """
        Initializes the motor model.

        Parameters:
        - num_envs: Number of envs.
        - taus: (4,) Tensor or list specifying time constants per motor.
        - init: (4,) Tensor or list specifying the initial omega per motor. (rad/s)
        - max_rate: (4,) Tensor or list specifying max rate of change of omega per motor. (rad/s^2)
        - min_rate: (4,) Tensor or list specifying min rate of change of omega per motor. (rad/s^2)
        - dt: Time step for integration.
        - use: Boolean indicating whether to use motor dynamics.
        - device: 'cpu' or 'cuda' for tensor operations.
        - dtype: Data type for tensors.
        """
        self.num_envs = num_envs
        self.num_motors = len(taus)
        self.dt = dt
        self.use = use
        self.init = init
        self.device = device
        self.dtype = dtype

        self.omega = torch.tensor(init, device=device).expand(num_envs, -1).clone()  # (num_envs, num_motors)

        # Convert to tensors and expand for all drones
        self.tau = torch.tensor(taus, device=device).expand(num_envs, -1)  # (num_envs, num_motors)
        self.max_rate = torch.tensor(max_rate, device=device).expand(num_envs, -1)  # (num_envs, num_motors)
        self.min_rate = torch.tensor(min_rate, device=device).expand(num_envs, -1)  # (num_envs, num_motors)

    def compute(self, omega_ref):
        """
        Computes the new omega values based on reference omega and motor dynamics.

        Parameters:
        - omega_ref: (num_envs, num_motors) Tensor of reference omega values.

        Returns:
        - omega: (num_envs, num_motors) Tensor of updated omega values.
        """

        if not self.use:
            self.omega = omega_ref
            return self.omega

        # Compute omega rate using first-order motor dynamics
        omega_rate = (1.0 / self.tau) * (omega_ref - self.omega)  # (num_envs, num_motors)
        omega_rate = omega_rate.clamp(self.min_rate, self.max_rate)

        # Integrate
        self.omega += self.dt * omega_rate
        return self.omega

    def reset(self, env_ids):
        """
        Resets the motor model to initial conditions.
        """
        self.omega[env_ids] = torch.tensor(self.init, device=self.device, dtype=self.dtype).expand(len(env_ids), -1)


class PositionController:
    def __init__(self, k_p=1.2, k_d=0.4, max_vel=2.0, device="cpu"):
        self.k_p = k_p
        self.k_d = k_d
        self.max_vel = max_vel
        self.device = device

    def __call__(self, state_w, desired_pos_w):
        pos = state_w["pos_w"]
        vel = state_w["lin_vel_w"]
        R_wb = state_w["rot_w_b"]

        # Ensure shape is (3,)
        if pos.ndim == 2 and pos.shape[0] == 1:
            pos = pos.squeeze(0)
        if vel.ndim == 2 and vel.shape[0] == 1:
            vel = vel.squeeze(0)

        # Position + velocity errors
        pos_error = desired_pos_w - pos
        vel_error = -vel

        # PD output (world frame)
        v_cmd_w = self.k_p * pos_error + self.k_d * vel_error

        # Fix shape (1,3) → (3,)
        if v_cmd_w.ndim == 2 and v_cmd_w.shape[0] == 1:
            v_cmd_w = v_cmd_w.squeeze(0)

        # Clamp
        speed_xy = torch.norm(v_cmd_w[:2])
        if speed_xy > self.max_vel:
            v_cmd_w[:2] = v_cmd_w[:2] / speed_xy * self.max_vel

        v_cmd_w[2] = torch.clamp(v_cmd_w[2], -self.max_vel, self.max_vel)

        # Convert to body frame (v_b = R_wb @ v_w)
        v_cmd_b = R_wb @ v_cmd_w

        return v_cmd_b

def compute_vee_map(skew_matrix):
    # type: (Tensor) -> Tensor

    # return vee map of skew matrix
    vee_map = torch.stack(
        [-skew_matrix[:, 1, 2], skew_matrix[:, 0, 2], -skew_matrix[:, 0, 1]], dim=1
    )
    return vee_map



class GeometricPositionController:

    def __init__(self, mass, inertia, device):
        self.mass = mass
        self.J = inertia.unsqueeze(0).to(device)    # (1,3,3)
        self.device = device

        # Position gains
        self.k_p = 16.0 # WORKED at 16.0
        self.k_d = 8.5

        # Attitude gains
        self.kR = 6.0 #4.5
        self.kW = 0.5 #0.3

        # Gravity
        self.g = torch.tensor([0., 0., -9.81], device=device).unsqueeze(0)

        self.current_euler = torch.zeros((1,3), device=device)


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

    def quat_rotate(self, q, v):
        """Rotate vector v by quaternion q."""
        R = self.matrix_from_quat(q)
        return torch.bmm(R, v.unsqueeze(2)).squeeze(2)


    # ----------------------------------------
    # MAIN CONTROL STEP
    # ----------------------------------------
    def update(
        self,
        pos_w, vel_w, quat_w, omega_b,
        desired_pos_w, desired_vel_w,
        desired_yaw, desired_yaw_rate
    ):

        # Ensure batch dims
        pos_w = pos_w.view(1, 3)
        vel_w = vel_w.view(1, 3)
        quat_w = quat_w.view(1, 4)
        omega_b = omega_b.view(1, 3)
        desired_pos_w = desired_pos_w.view(1, 3)
        desired_vel_w = desired_vel_w.view(1, 3)

        # ---------------------------
        # 1) POSITION + VELOCITY ERRORS
        # ---------------------------
        pos_err = desired_pos_w - pos_w
        vel_err = desired_vel_w - vel_w

        # ---------------------------
        # 2) DESIRED ACCELERATION
        # ---------------------------
        acc_des = self.k_p * pos_err + self.k_d * vel_err + self.g

        # ---------------------------
        # 3) DESIRED BODY Z (b3c)
        # ---------------------------
        b3_c = acc_des / torch.norm(acc_des, dim=1, keepdim=True)

        # ---------------------------
        # 4) DESIRED BODY X,Y FROM YAW
        # ---------------------------
        cy = torch.cos(desired_yaw)
        sy = torch.sin(desired_yaw)

        b1_des = torch.tensor([[cy, sy, 0.0]], device=self.device)

        b2_c = torch.cross(b3_c, b1_des, dim=1)
        b2_c = b2_c / torch.norm(b2_c, dim=1, keepdim=True)

        b1_c = torch.cross(b2_c, b3_c, dim=1)

        # ---------------------------
        # 5) DESIRED ROTATION MATRIX Rd
        # ---------------------------
        Rd = torch.zeros((1, 3, 3), device=self.device)
        Rd[:, :, 0] = b1_c
        Rd[:, :, 1] = b2_c
        Rd[:, :, 2] = b3_c

        # ---------------------------
        # 6) CURRENT ROTATION MATRIX R
        # ---------------------------
        R = self.matrix_from_quat(quat_w)

        # ---------------------------
        # 7) THRUST COMMAND
        # ---------------------------
        # Project desired force onto current body z axis
        b3 = R[:, :, 2]
        thrust = self.mass * torch.sum(acc_des * b3, dim=1)

        # ---------------------------
        # 8) ROTATION ERROR
        # ---------------------------
        rotation_error = 0.5 * self.vee_map(
            torch.bmm(Rd.transpose(1, 2), R) -
            torch.bmm(R.transpose(1, 2), Rd)
        )

        # ---------------------------
        # 9) DESIRED BODY RATES
        # ---------------------------
        omega_c = torch.zeros((1, 3), device=self.device)
        omega_c[:, 2] = desired_yaw_rate

        RRT = torch.bmm(R.transpose(1, 2), Rd)
        omega_c_body = torch.bmm(RRT, omega_c.unsqueeze(2)).squeeze(2)

        omega_err = omega_b - omega_c_body

        # ---------------------------
        # 10) FEEDFORWARD TERM (Ω × JΩc)
        # ---------------------------
        J_omega_c = torch.bmm(self.J, omega_c.unsqueeze(2)).squeeze(2)
        feedforward = torch.cross(omega_b, J_omega_c, dim=1)

        # ---------------------------
        # 11) TORQUE COMMAND
        # ---------------------------
        torque = -self.kR * rotation_error - self.kW * omega_err + feedforward

        return thrust, torque

class GeometricVelocityController:

    def __init__(self, mass, inertia, device):
        self.mass = mass
        self.J = inertia.unsqueeze(0).to(device)    # (1,3,3)
        self.device = device

        # Position gains
        self.k_p = 8.0 # WORKED at 16.0
        self.k_d = 25.0

        self.k_d_xy = 22.0
        self.k_d_z = 80.0
        
        # FOR 1/200
        # self.k_p = 16.0 # WORKED at 16.0
        # self.k_d = 180.0

        # Attitude gains
        self.kR = 2.5 #4.5
        self.kW = 0.1 #0.3

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

    def quat_rotate(self, q, v):
        """Rotate vector v by quaternion q."""
        R = self.matrix_from_quat(q)
        return torch.bmm(R, v.unsqueeze(2)).squeeze(2)

    # ----------------------------------------
    # MAIN CONTROL STEP
    # ----------------------------------------
    def update(
        self,
        pos_w, vel_w, quat_w, omega_b,
        desired_pos_w, desired_vel_w,
        desired_yaw, desired_yaw_rate
    ):

        # Ensure batch dims
        pos_w = pos_w.view(1, 3)
        vel_w = vel_w.view(1, 3)
        quat_w = quat_w.view(1, 4)
        omega_b = omega_b.view(1, 3)
        desired_pos_w = desired_pos_w.view(1, 3)
        desired_vel_w = desired_vel_w.view(1, 3)

        # ---------------------------
        # 1) POSITION + VELOCITY ERRORS
        # ---------------------------
        pos_err = desired_pos_w - desired_pos_w
        vel_err = desired_vel_w - vel_w

        # ---------------------------
        # 2) DESIRED ACCELERATION
        # ---------------------------
        # acc_des = self.k_d * vel_err + self.g
        acc_des = torch.zeros_like(vel_w)
        acc_des[:, 0] = self.k_d_xy * vel_err[:, 0]
        acc_des[:, 1] = self.k_d_xy * vel_err[:, 1]
        acc_des[:, 2] = self.k_d_z * vel_err[:, 2] + self.g[:, 2]

        # ---------------------------
        # 3) DESIRED BODY Z (b3c)
        # ---------------------------
        b3_c = acc_des / torch.norm(acc_des, dim=1, keepdim=True)

        # ---------------------------
        # 4) DESIRED BODY X,Y FROM YAW
        # ---------------------------
        cy = torch.cos(desired_yaw)
        sy = torch.sin(desired_yaw)

        b1_des = torch.tensor([[cy, sy, 0.0]], device=self.device)

        b2_c = torch.cross(b3_c, b1_des, dim=1)
        b2_c = b2_c / torch.norm(b2_c, dim=1, keepdim=True)

        b1_c = torch.cross(b2_c, b3_c, dim=1)

        # ---------------------------
        # 5) DESIRED ROTATION MATRIX Rd
        # ---------------------------
        Rd = torch.zeros((1, 3, 3), device=self.device)
        Rd[:, :, 0] = b1_c
        Rd[:, :, 1] = b2_c
        Rd[:, :, 2] = b3_c

        # ---------------------------
        # 6) CURRENT ROTATION MATRIX R
        # ---------------------------
        R = self.matrix_from_quat(quat_w)

        # ---------------------------
        # 7) THRUST COMMAND
        # ---------------------------
        # Project desired force onto current body z axis
        b3 = R[:, :, 2]
        thrust = self.mass * torch.sum(acc_des * b3, dim=1)

        # ---------------------------
        # 8) ROTATION ERROR
        # ---------------------------
        rotation_error = 0.5 * self.vee_map(
            torch.bmm(Rd.transpose(1, 2), R) -
            torch.bmm(R.transpose(1, 2), Rd)
        )

        # ---------------------------
        # 9) DESIRED BODY RATES
        # ---------------------------
        omega_c = torch.zeros((1, 3), device=self.device)
        omega_c[:, 2] = desired_yaw_rate

        RRT = torch.bmm(R.transpose(1, 2), Rd)
        omega_c_body = torch.bmm(RRT, omega_c.unsqueeze(2)).squeeze(2)

        omega_err = omega_b - omega_c_body
        print(f"Omega error: {omega_err}")

        # ---------------------------
        # 10) FEEDFORWARD TERM (Ω × JΩc)
        # ---------------------------
        J_omega_c = torch.bmm(self.J, omega_c.unsqueeze(2)).squeeze(2)
        feedforward = torch.cross(omega_b, J_omega_c, dim=1)

        # ---------------------------
        # 11) TORQUE COMMAND
        # ---------------------------
        torque = -self.kR * rotation_error - self.kW * omega_err + feedforward

        return thrust, torque
    
    def update_velocity_only(
        self,
        pos_w, vel_w, quat_w, omega_b,
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
        # Ensure batch dims
        vel_w = vel_w.view(1,3)
        quat_w = quat_w.view(1,4)
        omega_b = omega_b.view(1,3)
        desired_vel_w = desired_vel_w.view(1,3)

        # ---------------------------
        # 1) VELOCITY ERROR
        # ---------------------------
        vel_err = desired_vel_w - vel_w

        # ---------------------------
        # 2) DESIRED ACCELERATION (velocity-only)
        # ---------------------------
        acc_des = torch.zeros_like(vel_w)   #force command 
        acc_des[:,0] = self.k_d_xy * vel_err[:,0]
        acc_des[:,1] = self.k_d_xy * vel_err[:,1]
        acc_des[:,2] = self.k_d_z * vel_err[:,2] + self.g[:,2]  # include gravity

        # ---------------------------
        # 3) BODY Z (thrust direction)
        # ---------------------------
        b3_c = acc_des / (torch.norm(acc_des, dim=1, keepdim=True))

        # ---------------------------
        # 4) DESIRED BODY X/Y (arbitrary, yaw free)
        # ---------------------------
        # current yaw from quaternion
        # w, x, y, z = quat_w[:,0], quat_w[:,1], quat_w[:,2], quat_w[:,3]
        
        euler = euler_xyz_from_quat(quat_w)
        if isinstance(euler, tuple):
            # Each tensor shape: (1,) → stack to (3,) → then batch to (1,3)
            self.current_euler = torch.cat(euler, dim=0).view(1, 3)

        cy = torch.cos(self.current_euler[:, 2])
        sy = torch.sin(self.current_euler[:, 2])

        # cp = torch.cos(self.current_euler[:, 1])
        # sp = torch.sin(self.current_euler[:, 1])

        # cr = torch.cos(self.current_euler[:, 0])
        # sr = torch.sin(self.current_euler[:, 0])

        b1_des = torch.tensor([[cy, sy, 0]], device=self.device)
        # b1_des = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)  # yaw-free reference
        b2_c = torch.cross(b3_c, b1_des, dim=1)
        b2_c /= (torch.norm(b2_c, dim=1, keepdim=True))
        b1_c = torch.cross(b2_c, b3_c, dim=1)

        

        # ---------------------------
        # 5) DESIRED ROTATION MATRIX Rd
        # ---------------------------
        Rd = torch.zeros((1,3,3), device=self.device)
        Rd[:,:,0] = b1_c
        Rd[:,:,1] = b2_c
        Rd[:,:,2] = b3_c

        # ---------------------------
        # 6) CURRENT ROTATION MATRIX R
        # ---------------------------
        R = self.matrix_from_quat(quat_w)

        # ---------------------------
        # 7) THRUST
        # ---------------------------
        b3 = R[:,:,2]  # current body z-axis
        thrust = self.mass * torch.sum(acc_des * b3, dim=1)

        # ---------------------------
        # 8) ROTATION ERROR (roll/pitch only, yaw ignored)
        # ---------------------------
        rotation_error = 0.5 * self.vee_map(
            torch.bmm(Rd.transpose(1,2), R) - torch.bmm(R.transpose(1,2), Rd)
        )
        rotation_error[:,2] = 0.0  # zero yaw error

        # ---------------------------
        # 9) DESIRED BODY RATES
        # ---------------------------
        omega_c = torch.zeros_like(omega_b)
        omega_c[:,2] = desired_yaw_rate  # yaw-rate tracking

        omega_err = omega_b - omega_c

        # ---------------------------
        # 10) FEEDFORWARD
        # ---------------------------
        J_omega_c = torch.bmm(self.J, omega_c.unsqueeze(2)).squeeze(2)
        feedforward = torch.cross(omega_b, J_omega_c, dim=1)

        # ---------------------------
        # 11) TORQUE
        # ---------------------------
        torque = -self.kR * rotation_error - self.kW * omega_err + feedforward

        return thrust, torque
    

# -------------------- Main --------------------
def main():
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(
        dt=1/100,   # 1/200
        device=args_cli.device,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.0, 1.0, 2.0], target=[0.0, 0.0, 1.0])

    # Spawn things into stage
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    robot_cfg = AGILE_CFG.replace(prim_path="/World/AgileDrone")
    robot_cfg.spawn.func("/World/AgileDrone", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)
    sim.reset()

    body_id = robot.find_bodies("base_link")[0]
    robot_mass = robot.root_physx_view.get_masses()[0].sum()
    print(f"[INFO]: Robot mass: {robot_mass:.2f} kg")
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    sim_dt = sim.get_physics_dt()
    print(f"[INFO]: Simulation DT: {sim_dt:.4f} seconds")

    forces = torch.zeros(1, 3, device=sim.device)
    torques = torch.zeros_like(forces)
    lidar = torch.ones(robot.num_instances, 60, device=sim.device) * 4.9

    taus = (0.0001, 0.0001, 0.0001, 0.0001)
    init = (2572.5, 2572.5, 2572.5, 2572.5)
    max_rate = (50000.0, 50000.0, 50000.0, 50000.0)
    min_rate = (-50000.0, -50000.0, -50000.0, -50000.0)
    use_motor_model = False

    # create allocation & motor (your classes)
    alloc_matrix = Allocation(num_envs=1, device=sim.device)
    motor = Motor(num_envs=1, taus=taus, init=init, max_rate=max_rate, min_rate=min_rate, dt=sim_dt, use=use_motor_model, device=sim.device)

    # instantiate the new controller: provide mass and inertia J
    # NOTE: replace this approximate J with your robot's true inertia if available.
    # J_default = torch.diag(torch.tensor([0.01, 0.01, 0.02], device=sim.device))  # (3,3) approximate
    # vac = VelocityAttitudeController(mass=float(robot_mass), J=J_default, g=9.81, k_v=2.0, k_R=4.0, k_omega=0.6, device=sim.device)

    # diagnostic publishers (assumes pid_publisher global)
    global pid_publisher

    # desired_vel_w = torch.zeros(1, 3, device=sim.device)  # for purely velocity control, could be set to zeros to hover
    # desired_pos = torch.tensor([[0.0, 0.0, 2.0]], device=sim.device)
    # desired_yaw = torch.tensor([0.0], device=sim.device)
    # accel = torch.zeros(3, device=sim.device)
    # rotation_matrix_desired = torch.zeros((1, 3, 3), device=sim.device)
    # rotmat_euler_to_body_rates = torch.zeros((3, 3), device=sim.device)
    # desired_body_angvel = torch.zeros_like(robot.data.root_ang_vel_b)

    # desired_vel_w[:, 0] = 1.0  # desired forward velocity in world frame
    # desired_vel_w[:, 1] = 0.0  # desired forward velocity in world frame
    # desired_vel_w[:, 2] = 0.0  # desired forward velocity in world frame
    

    sim_time = 0.0
    count = 0
    
    Ixx = 0.00072172
    Iyy = 0.00088563
    Izz = 0.0012558
    controller = GeometricVelocityController(
        mass=robot_mass,
        inertia=torch.diag(torch.tensor([Ixx, Iyy, Izz], device=sim.device)),
        device=sim.device
    )

    
   
    cmd_pos = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
    cmd_vel = torch.tensor([[2.0, -1.0, 1.0]], device=sim.device)
    cmd_yaw = torch.tensor([0.0], device=sim.device)
    cmd_yaw_rate = torch.tensor([0.0], device=sim.device)

    # main loop
    while simulation_app.is_running():
        start_time = time.time()
        rclpy.spin_once(pid_publisher, timeout_sec=0.0)

        with torch.inference_mode():
            
            # thrust, torque = controller.update(
            #     pos_w=robot.data.root_pos_w,
            #     vel_w=robot.data.root_lin_vel_w,
            #     quat_w=robot.data.root_link_quat_w,   # (1,4)
            #     omega_b=robot.data.root_ang_vel_b,

            #     desired_pos_w=cmd_pos,
            #     desired_vel_w=cmd_vel,
            #     desired_yaw=cmd_yaw,
            #     desired_yaw_rate=cmd_yaw_rate
            # )

            thrust, torque = controller.update_velocity_only(
                pos_w=robot.data.root_pos_w,
                vel_w=robot.data.root_lin_vel_w,
                quat_w=robot.data.root_link_quat_w,   # (1,4)
                omega_b=robot.data.root_ang_vel_b,
                desired_vel_w=cmd_vel,
                desired_yaw_rate=cmd_yaw_rate
            )


            # print(f"Thrust: {thrust}, Torque: {torque}")


            # # apply to sim
            forces[:, :] = 0.0
            torques[:, :] = 0.0

            forces[:, 2] = thrust
            torques[:, :] = torque

            # forces[:, 2] = wrrench_command
            # torques[:, :] = torque.squeeze(0)


            # forces[:, 2] = 8.0 * 1.23 * 9.81
            # torques[:, 0] = 0 * 0.7
            # torques[:, 1] = -2 * 0.7
            # torques[:, 2] = 0 * 0.7
            
            # forces[:, 2] = processed_actions[0, 0]
            # torques[:, :] = processed_actions[0, 1:].unsqueeze(0)

            # publish diagnostics (vel err, commanded force, etc.)
            pid_publisher.publish_vec(pid_publisher.pub_error, ((cmd_vel - robot.data.root_lin_vel_w).squeeze(0)).cpu())
            # pid_publisher.publish_vec(pid_publisher.pub_cmd, (f_w.squeeze(0)).cpu())
            pid_publisher.publish_vec(pid_publisher.pub_vel, robot.data.root_lin_vel_w.squeeze(0).cpu())
            # pid_publisher.publish_vec(pid_publisher.pub_target, v_des_w.squeeze(0).cpu())
            # publish yaw rate 
            pid_publisher.publish_vec(pid_publisher.pub_target, robot.data.root_ang_vel_b.squeeze(0).cpu())

            robot.set_external_force_and_torque(forces, torques, body_ids=body_id)
            robot.write_data_to_sim()
            sim.step()
            sim_time += sim_dt
            robot.update(sim_dt)
            
            # cmd_yaw = torch.tensor([torch.tensor(0.2*sim_time)], device=sim.device)

        # simple timing
        loop_duration = time.time() - start_time
        inference_time = 0.01 # 0.03
        remaining_time = inference_time - loop_duration
        print("loop time: {:.4f}, remaining time to wait: {:.4f}".format(loop_duration, remaining_time))
        if remaining_time > 0.0:
            time.sleep(remaining_time)

# -------------------- Entry point --------------------
if __name__ == "__main__":
    rclpy.init()
    pid_publisher = PIDPlotPublisher()
    
    main()
    
    # close sim app
    simulation_app.close()
    # rclpy.init()
    # pid_publisher = PIDPlotPublisher()
    # try:
    #     main()
    # finally:
    #     try:
    #         pid_publisher.destroy_node()
    #         rclpy.shutdown()
    #     except Exception:
    #         pass
    #     try:
    #         simulation_app.close()
    #     except Exception:
    #         pass
