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
from isaaclab.utils.math import quat_from_euler_xyz,subtract_frame_transforms, matrix_from_quat,quat_from_matrix , normalize, quat_rotate, euler_xyz_from_quat, quat_mul,quat_inv

# ROS2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Twist
from olfaction_msgs.msg import GasSensor
from olfaction_msgs.msg import Anemometer
from std_msgs.msg import Float32MultiArray, Float32

from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import TransformBroadcaster
from nav_msgs.msg import Path   

# Model training imports
from torch.utils.data import TensorDataset
import numpy as np

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
            # "/PioneerP3DX/cmd_vel",
            # "/cmd_vel",
            "/insect_cmd_vel",
            self.listener_callback,
            10,
        )
        
        self.cmd_velocity = torch.zeros(3)
        self.cmd_ang_velocity = torch.zeros(3)

    def listener_callback(self, msg):
        self.cmd_velocity[0] = msg.linear.x
        self.cmd_velocity[1] = msg.linear.y
        self.cmd_velocity[2] = msg.linear.z
        self.cmd_ang_velocity[0] = 0.0
        self.cmd_ang_velocity[1] = 0.0
        self.cmd_ang_velocity[2] = msg.angular.z
    
    def get_cmd_velocity(self):
        return self.cmd_velocity
    
    def get_cmd_ang_velocity(self):
        return self.cmd_ang_velocity
    

class GagenSimSubscriber(Node):
    def __init__(self, name="gas_subscriber"):
        super().__init__(name)

        self.gas_left_subscription_ = self.create_subscription(
            GasSensor,
            '/fake_pid_left/Sensor_reading',
            self.gas_left_callback,
            10)
        
        self.gas_right_subscription_ = self.create_subscription(
            GasSensor,
            '/fake_pid_right/Sensor_reading',
            self.gas_right_callback,
            10)
        
        self.wind_subscription_ = self.create_subscription(
            Anemometer,
            '/fake_anemometer/WindSensor_reading',
            self.wind_callback,
            10)
        
        self.gas_left_raw = GasSensor()
        self.gas_right_raw = GasSensor()
        self.wind_data = Anemometer()


        self.gas_left = torch.zeros(1)
        self.gas_right = torch.zeros(1)
        self.wind_dir = torch.zeros(1)
        self.wind_spd = torch.zeros(1)

    def low_pass_filter(self, current_value, previous_value, alpha=0.001):
        return float(alpha * current_value + (1 - alpha) * previous_value)
    
    def gas_left_callback(self, msg):
        self.gas_left_raw = msg.raw
        self.gas_left = torch.tensor([self.gas_left_raw])
        # print(f"Gas Left Data: {self.gas_left}")
        
        # self.gas_left_lowpass.data = self.low_pass_filter(self.gas_left_data, self.prev_gas_left, alpha=0.05)
        # self.prev_gas_left = self.gas_left_lowpass.data
        # self.get_logger().info(f"Gas Left Data: {self.gas_left_data}")

    def gas_right_callback(self, msg):
        self.gas_right_raw = msg.raw
        self.gas_right = torch.tensor([self.gas_right_raw])
        # self.gas_right_lowpass.data = self.low_pass_filter(self.gas_right_data, self.prev_gas_right, alpha=0.05)
        # self.prev_gas_right = self.gas_right_lowpass.data
        # self.get_logger().info(f"Gas Right Data: {self.gas_right_data}")

    def wind_callback(self, msg):
        wind_dir_rad = msg.wind_direction  # radians
        wind_speed = msg.wind_speed  # m/s
        # self.get_logger().info(f"Wind Direction: {wind_dir_rad}, Wind Speed: {wind_speed}")
        self.wind_dir = torch.tensor([wind_dir_rad])
        self.wind_spd = torch.tensor([wind_speed])

    def get_gas_left(self):
        return self.gas_left
    
    def get_gas_right(self):
        return self.gas_right
    
    def get_wind_direction(self):
        return self.wind_dir
    
    def get_wind_speed(self):
        return self.wind_spd
    
class publishTF(Node):
    def __init__(self, name="tf_publisher"):
        super().__init__(name)
        self.broadcaster = TransformBroadcaster(self)
        self.parent_frame = "world"
        self.child_frame = "drone_base_link"

        # path = Path()
        self.pub_path = self.create_publisher(Path, '/drone_path', 10)



    def publish_transform(self, translation, rotation, parent_frame="world", child_frame="drone_base_link"):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])
        t.transform.rotation.x = float(rotation[1])
        t.transform.rotation.y = float(rotation[2])
        t.transform.rotation.z = float(rotation[3])
        t.transform.rotation.w = float(rotation[0])
        self.broadcaster.sendTransform(t)

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = parent_frame
        
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = parent_frame
        pose.pose.position.x = float(translation[0])
        pose.pose.position.y = float(translation[1])
        pose.pose.position.z = float(translation[2])
        pose.pose.orientation.x = float(rotation[1])
        pose.pose.orientation.y = float(rotation[2])
        pose.pose.orientation.z = float(rotation[3])
        pose.pose.orientation.w = float(rotation[0])
        
        path.poses.append(pose)
        self.pub_path.publish(path)



class insect_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(insect_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class setpose_from_goal_pose_topic(Node):
    def __init__(self, name="setpose_from_goal_pose"):
        super().__init__(name)
        self.subscription = self.create_subscription(
            PoseStamped,
            "/PioneerP3DX/goal_pose",
            self.listener_callback,
            10,
        )
        self.goal_position = torch.zeros(2)
        self.updated = False

    def listener_callback(self, msg):
        # if not self.updated:
            # self.get_logger().info("Received first goal position message.")
            # self.updated = not self.updated
        self.goal_position[0] = msg.pose.position.x
        self.goal_position[1] = msg.pose.position.y
        # self.goal_position[2] = msg.pose.position.z
        print(f"Getting goal position: {self.goal_position}")

    def get_goal_position(self):
        return self.goal_position


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


    # diagnostic publishers (assumes pid_publisher global)
    pid_publisher = PIDPlotPublisher()
    velSub = VelocityCommandSubscriber()
    gasInfoSub = GagenSimSubscriber()
    pubTF = publishTF()
    setPose = setpose_from_goal_pose_topic()

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
    # cmd_yaw = torch.tensor([0.0], device=sim.device)
    cmd_yaw = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
    cmd_yaw_rate = torch.tensor([0.0], device=sim.device)

    altitude_desired = 0.4
    atl_error_integral = 0.0
    kp_altitude = 2.0
    ki_altitude = 1.0
    kd_altitude = 3.0


    episode_buffer = []
    
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, :3] += torch.tensor([7.0, 3.0, altitude_desired], device=sim.device) 
    # rotate to face negative x direction
    default_root_state[:, 3:7] = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([3.14159]))
    robot.write_root_pose_to_sim(default_root_state[:, :7])
    robot.write_root_velocity_to_sim(default_root_state[:, 7:])
    
    lastest_pose = torch.zeros(2)

    gas_source_location = torch.tensor([1.45, 3.0, 0.8])
    done = False
    trial_num = 0
    file_idx = 0
    # main loop
    while simulation_app.is_running():
        start_time = time.time()
        rclpy.spin_once(pid_publisher, timeout_sec=0.0)
        rclpy.spin_once(velSub, timeout_sec=0.0)
        rclpy.spin_once(pubTF, timeout_sec=0.0)
        rclpy.spin_once(gasInfoSub, timeout_sec=0.0)
        rclpy.spin_once(setPose, timeout_sec=0.0)

        # check for goal position changing update
        set_pose = setPose.get_goal_position()
        # print(f"Set Pose from topic: {set_pose}")
        # print(f"Lastest Pose: {lastest_pose}")
        if not torch.equal(set_pose, lastest_pose):
            default_root_state[:, 0] = set_pose[0].to(sim.device)
            default_root_state[:, 1] = set_pose[1].to(sim.device)
            default_root_state[:, 2] = altitude_desired
            robot.write_root_pose_to_sim(default_root_state[:, :7])
            robot.write_root_velocity_to_sim(default_root_state[:, 7:])
            print(f"Updated robot position to: {set_pose}")
            lastest_pose = set_pose.clone()
        # else:
            

        # if lastest_pose.norm().item() > 0.0:
        #     default_root_state[:, :2] = lastest_pose.to(sim.device)
        #     default_root_state[:, 2] = altitude_desired
        #     robot.write_root_pose_to_sim(default_root_state[:, :7])
        #     robot.write_root_velocity_to_sim(default_root_state[:, 7:])
        #     print(f"Updated robot position to: {lastest_pose}")


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
            # print("Getting cmd_vel from subscriber...")
            cmd_vel = velSub.get_cmd_velocity().to(sim.device).unsqueeze(0)
            cmd_yaw = velSub.get_cmd_ang_velocity().to(sim.device).unsqueeze(0)

            #covert global frame cmd_vel to body frame
            current_yaw = euler_xyz_from_quat(robot.data.root_link_quat_w)[2]
            print(f"Current Yaw: {current_yaw.item():.4f} rad")
            cy = torch.cos(-current_yaw)
            sy = torch.sin(-current_yaw)
            # rotation_matrix = torch.tensor([[cy, -sy, 0.0],
                                            # [sy, cy, 0.0],
                                            # [0.0, 0.0, 1.0]], device=sim.device)
            # cmd_vel = torch.matmul(rotation_matrix, cmd_vel.squeeze(0).unsqueeze(1)).unsqueeze(0)
            cmd_vel[0,0] = cy * velSub.get_cmd_velocity()[0].to(sim.device) + sy * velSub.get_cmd_velocity()[1].to(sim.device)
            cmd_vel[0,1] = -sy * velSub.get_cmd_velocity()[0].to(sim.device) + cy * velSub.get_cmd_velocity()[1].to(sim.device)

            # print(f"Commanded Yaw Rate: {cmd_yaw}")
            # print(f"Commanded Velocity: {cmd_vel}")
            
            z_vel_error = altitude_desired - robot.data.root_pos_w[0,2]
            atl_error_integral += z_vel_error * sim_dt
            cmd_vel[0, 2] = (kp_altitude * z_vel_error) #+ (ki_altitude * atl_error_integral) - (kd_altitude * robot.data.root_lin_vel_w[0,2])
            # print(f"Altitude Control Velocity Command: {cmd_vel[0,2]}")

            thrust, torque = controller.update_velocity_only(
                pos_w=robot.data.root_pos_w,
                vel_w=robot.data.root_lin_vel_w,
                quat_w=robot.data.root_link_quat_w,   # (1,4)
                omega_b=robot.data.root_ang_vel_b,
                desired_vel_w=cmd_vel,
                desired_yaw_rate=cmd_yaw[0,2]  # yaw rate around body z-axis
            )

            pubTF.publish_transform(
                translation=robot.data.root_pos_w.squeeze(0).cpu().numpy(),
                rotation=robot.data.root_link_quat_w.squeeze(0).cpu().numpy(),
                parent_frame="map",
                child_frame="PioneerP3DX_base_link"
            )
            

            # gas_left = gasInfoSub.get_gas_left().to(sim.device)
            gas_left = gasInfoSub.get_gas_left()
            gas_right = gasInfoSub.get_gas_right()
            wind_dir = gasInfoSub.get_wind_direction()
            wind_spd = gasInfoSub.get_wind_speed()

            # print(f"Gas Left: {gas_left.item():.4f}, Gas Right: {gas_right.item():.4f}, Wind Dir: {wind_dir.item():.4f}, Wind Spd: {wind_spd.item():.4f}")

            # log_data = torch.cat((gas_left, gas_right, wind_dir, wind_spd), dim=0).unsqueeze(0)
            pos = robot.data.root_pos_w.squeeze(0).cpu().numpy()
            sample_data = np.array([gas_left.item(), gas_right.item(), wind_dir.item(), wind_spd.item(), cmd_vel[0,0].item(), cmd_vel[0,1].item(), cmd_vel[0,2].item(), cmd_yaw[0,2].item(), pos[0], pos[1], pos[2]])
            # print(f"Log Data Shape: {log_data.shape}")
            episode_buffer.append(sample_data)
            
            distance_to_source = torch.norm(robot.data.root_pos_w.squeeze(0) - gas_source_location.to(sim.device)).item()
            print(f"Distance to gas source: {distance_to_source:.4f} m")
            if distance_to_source < 1.0:
                # randomize start position for next trial
                if set_pose.norm().item() == 0.0:
                    print("No set pose received, randomizing start position.")
                    default_root_state = robot.data.default_root_state.clone()
                    default_root_state[:, 0] = torch.tensor([7.0], device=sim.device)
                    default_root_state[:, 1] = torch.tensor([3.0], device=sim.device) + torch.empty(1, device=sim.device).uniform_(-2.0, 2.0)
                    default_root_state[:, 2] = altitude_desired
                else:
                    print("\n\n\n\nUsing set pose for next trial start position.\n\n\n")
                    default_root_state[:, 1] = set_pose[1] + torch.empty(1, device=sim.device).uniform_(-2.0, 2.0)
                robot.write_root_pose_to_sim(default_root_state[:, :7])
                robot.write_root_velocity_to_sim(default_root_state[:, 7:])
                
                trial_num += 1
                print(f"Trial {trial_num} completed. Reaching gas source.")
            
            if trial_num >= 1:
                print("Completed 1 trial.")
                # Save episode data
                episode_array = np.stack(episode_buffer, axis=0)
                filename = f"episode_data_{file_idx:05d}.npy" 
                # np.save(f'episode_data_{count//500}.npy', episode_array)
                np.save(filename, episode_array)
                print(f"Saved episode_data_{file_idx:05d}.npy with shape {episode_array.shape}")
                episode_buffer.clear()
                trial_num = 0
                file_idx += 1
            # count += 1
                


            # colloct data transition for training
            # gas_left = gasInfoSub.gas_left_data.data
            # gas_right = gasInfoSub.gas_right_data.data
            # wind_dir = gasInfoSub.wind_data.wind_direction
            # gasInfoSub.dataLog_gas_left = torch.cat((gasInfoSub.dataLog_gas_left, torch.tensor([gas_left], device=sim.device)), dim=0)
            # gasInfoSub.dataLog_gas_right = torch.cat((gasInfoSub.dataLog_gas_right, torch.tensor([gas_right], device=sim.device)), dim=0)
            # gasInfoSub.dataLog_wind_dir = torch.cat((gasInfoSub.dataLog_wind_dir, torch.tensor([wind_dir], device=sim.device)), dim=0)      

            # #trainning data logging
            # if count % 100 == 0:
            #     torch.save(gasInfoSub.dataLog_gas_left, 'gas_left_data.pt')
            #     torch.save(gasInfoSub.dataLog_gas_right, 'gas_right_data.pt')
            #     torch.save(gasInfoSub.dataLog_wind_dir, 'wind_dir_data.pt')
            #     print("Training data saved.")
            # count += 1






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
            # pid_publisher.publish_vec(pid_publisher.pub_error, ((cmd_vel - robot.data.root_lin_vel_w).squeeze(0)).cpu())
            # pid_publisher.publish_vec(pid_publisher.pub_cmd, (f_w.squeeze(0)).cpu())
            # pid_publisher.publish_vec(pid_publisher.pub_vel, robot.data.root_lin_vel_w.squeeze(0).cpu())
            # pid_publisher.publish_vec(pid_publisher.pub_target, v_des_w.squeeze(0).cpu())
            # publish yaw rate 
            # pid_publisher.publish_vec(pid_publisher.pub_target, robot.data.root_ang_vel_b.squeeze(0).cpu())

            robot.set_external_force_and_torque(forces, torques, body_ids=body_id)
            robot.write_data_to_sim()
            sim.step()
            sim_time += sim_dt
            robot.update(sim_dt)
            
            # cmd_yaw = torch.tensor([torch.tensor(0.2*sim_time)], device=sim.device)

        # trainning model
        # insect_model = insect_MLP(input_size=3, hidden_size=16, output_size=2).to(sim.device)
        # dummy input: gas_left, gas_right, wind_dir
        # dummy_input = torch.tensor([[gas_left, gas_right, wind_dir]], device=sim.device)
        # output = insect_model(dummy_input)
        # print(f"Insect MLP Output: {output}")

        # simple timing
        loop_duration = time.time() - start_time
        inference_time = 0.01 # 0.03
        remaining_time = inference_time - loop_duration
        # print("loop time: {:.4f}, remaining time to wait: {:.4f}".format(loop_duration, remaining_time))
        if remaining_time > 0.0:
            time.sleep(remaining_time)

# -------------------- Entry point --------------------
if __name__ == "__main__":
    rclpy.init()
    
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
