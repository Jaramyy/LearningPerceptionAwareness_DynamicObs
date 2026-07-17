"""Student policy ROS2 node — depth-camera input + MODANC ICP adaptive speed.

Same as student_ros2_node_icp.py but replaces the LiDAR LaserScan input with a
depth camera Image topic.  The depth image is sliced into the same 5 angular
sectors used during Isaac Lab training, then fed to the ICP speed controller and
the student policy exactly as before.

Depth preprocessing pipeline:
  depth Image (16UC1 mm or 32FC1 m)
    → crop centre vertical strip (--depth_vstrip fraction of rows)
    → per-column minimum over the strip
    → assign columns to 5 angular sectors using camera --depth_hfov
    → sector min distance (m), clipped to [DEPTH_MIN_RANGE, LIDAR_RANGE]
    → normalised to [0, 1]  (÷ LIDAR_RANGE=5 m)  — identical to LaserScan path

Sectors (RIGHT → LEFT in FLU body frame, matching Isaac training):
  sec0 RIGHT : -90° to -54°   sec1 R-ctr : -54° to -18°   sec2 fwd : -18° to +18°
  sec3 L-ctr : +18° to +54°   sec4 LEFT  : +54° to +90°
  Sectors outside the camera HFOV return max-range (1.0 normalised).

Typical depth camera params:
  RealSense D435 : --depth_hfov 86  --depth_encoding 16UC1
  RealSense D455 : --depth_hfov 87  --depth_encoding 16UC1
  ZED 2          : --depth_hfov 110 --depth_encoding 32FC1

ICP speed adaptation network (MODANC IROS 2025, Section D.1):
  d_min → S1 → X1 (predictive) / X0=Rm(X1) (reflexive)
  S2 = W0·X0 + W1·X1  →  speed_scale = max(0, 1−S2)
  ΔW1 = μs·(dX0/dt)·X1 − γs·(ks + P(t))·W1²

Run:
  python3 scripts/rl_games/student_ros2_node_depth.py \\
      --checkpoint logs/dagger/student_latest.pth \\
      --goal_north 5.0 --goal_east 0.0 --goal_alt 1.5 \\
      --depth_topic /camera/depth/image_raw \\
      --depth_hfov 87.0 --depth_vstrip 0.3 \\
      --vel_scale 0.5 --d_reflexive 1.0 --d_predictive 4.0 --w1_init 0.5

Observation layout (16D, must match dagger_distill.py):
  [0:3]   root_lin_vel_b      linear velocity in FLU body frame (m/s)
  [3:6]   root_ang_vel_b      angular velocity in FLU body frame (rad/s)
  [6:9]   unit_desired_pos_b  unit vector to goal in FLU body frame
  [9]     desired_dist_2d     horizontal distance to goal (m)
  [10]    desired_dist_z      vertical distance to goal, + = goal above (m)
  [11:16] 5 depth sectors (RIGHT→LEFT, -90° to +90°), normalized [0,1]

Frame conventions:
  PX4 body : FRD  (Forward-Right-Down)
  Isaac/Student body : FLU  (Forward-Left-Up)  [right-handed, +Y=left, +Z=up]
  PX4 world : NED  (North-East-Down)
"""

import argparse
import math
import os
import threading
import time

import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy,
    QoSHistoryPolicy, QoSDurabilityPolicy,
)

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    VehicleAttitude,
    VehicleLocalPosition,
    SensorCombined,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster

# ── Obs constants (must match dagger_distill.py) ─────────────────────────────
STUDENT_OBS_DIM = 16
ACTION_DIM = 4
LIDAR_RANGE = 5.0           # training max range (m)
# 5 sectors matching Isaac body-frame angle order.
# Isaac FLU: negative angles = RIGHT, positive = LEFT.
# Gazebo ROS LaserScan (FLU): same convention — negative angles = RIGHT.
# Sector 0 = RIGHT (most negative), sector 4 = LEFT (most positive).
BEAM_SECTOR_DEG = [(-90.0, -54.0), (-54.0, -18.0), (-18.0, 18.0), (18.0, 54.0), (54.0, 90.0)]

# Scale factors (match QuadcopterEnvCfg)
MAX_VELOCITY = 4.0    # m/s
MAX_YAW_RATE = 3.14   # rad/s

# Depth camera preprocessing
DEPTH_MIN_RANGE = 0.15   # m — discard pixels closer than this (sensor blind spot)


# ── Network definitions (identical to dagger_distill.py) ─────────────────────

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


def _infer_hidden_dims(sd: dict, action_dim: int) -> list[int]:
    return [
        v.shape[0] for k, v in sd.items()
        if k.endswith(".weight") and "net" in k and v.shape[0] != action_dim
    ]


# ── Quaternion math (pure Python, no extra deps) ─────────────────────────────

def _quat_apply(q_wxyz: tuple, v: tuple) -> tuple:
    """Rotate vector v by unit quaternion q: v' = q * v * q^-1."""
    w, qx, qy, qz = q_wxyz
    vx, vy, vz = v
    # Cross product q_vec x v
    cx = qy * vz - qz * vy
    cy = qz * vx - qx * vz
    cz = qx * vy - qy * vx
    # Rodrigues rotation
    return (
        vx + 2 * w * cx + 2 * (qy * cz - qz * cy),
        vy + 2 * w * cy + 2 * (qz * cx - qx * cz),
        vz + 2 * w * cz + 2 * (qx * cy - qy * cx),
    )


def _ned_to_flu(q_frd_ned_wxyz: tuple, v_ned: tuple) -> tuple:
    """Transform vector from NED world frame to FLU body frame (Isaac convention).

    q_frd_ned: rotation quaternion from FRD body -> NED world (PX4 convention).
    Returns vector in FLU body frame (+X=fwd, +Y=left, +Z=up).
    FRD → FLU: negate Y (right→left) and Z (down→up).
    """
    w, x, y, z = q_frd_ned_wxyz
    q_inv = (w, -x, -y, -z)        # conjugate: NED → FRD
    v_frd = _quat_apply(q_inv, v_ned)
    return (v_frd[0], -v_frd[1], -v_frd[2])    # FRD → FLU


def _norm3(v: tuple) -> float:
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def _normalize3(v: tuple, eps: float = 1e-6) -> tuple:
    n = max(_norm3(v), eps)
    return (v[0] / n, v[1] / n, v[2] / n)


def _qmul(a: tuple, b: tuple) -> tuple:
    """Quaternion product a ⊗ b (w,x,y,z convention)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _px4_to_enu_quat(q_frd_ned: tuple) -> tuple:
    """Convert PX4 attitude quaternion (FRD→NED) to ROS TF quaternion (FLU→ENU).

    Result is suitable for geometry_msgs/TransformStamped.rotation with
    parent frame 'map' (ENU: x=East, y=North, z=Up) and
    child frame 'base_link' (FLU: x=Forward, y=Left, z=Up).

    Formula: q_enu_flu = q_enu_ned ⊗ q_frd_ned ⊗ q_frd_flu
      q_enu_ned = (0, 1/√2, 1/√2, 0)  — maps NED axes to ENU axes
      q_frd_flu = (0,  1,   0,   0)   — 180° about x: FRD → FLU body
    """
    s = math.sqrt(0.5)
    q_enu_ned = (0.0, s, s, 0.0)
    q_frd_flu = (0.0, 1.0, 0.0, 0.0)
    return _qmul(_qmul(q_enu_ned, q_frd_ned), q_frd_flu)


# ── Depth image preprocessing ────────────────────────────────────────────────

def _parse_depth_image(msg: Image) -> np.ndarray:
    """Decode a ROS2 depth Image message to a float32 array in **metres**.

    Supported encodings:
      16UC1 — uint16 millimetres  (RealSense D435/D455 default)
      32FC1 — float32 metres      (ZED, simulated depth)
    """
    raw = bytes(msg.data)
    H, W = msg.height, msg.width
    if msg.encoding == "16UC1":
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(H, W).astype(np.float32)
        arr /= 1000.0                    # mm → m
    elif msg.encoding == "32FC1":
        arr = np.frombuffer(raw, dtype=np.float32).reshape(H, W).copy()
    else:
        # Fallback: attempt uint16 mm decode
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(H, W).astype(np.float32) / 1000.0
    return arr


def _depth_to_sectors(
    depth: np.ndarray,
    sectors_deg: list,
    hfov_deg: float,
    vstrip_frac: float,
    max_range: float,
    min_range: float,
) -> list[float]:
    """Convert depth image to per-sector minimum distances (metres).

    Args:
        depth       : H×W float32 array in metres.
        sectors_deg : list of (lo_deg, hi_deg) angular sector boundaries.
        hfov_deg    : horizontal field of view of the camera (degrees).
        vstrip_frac : fraction of image height used as the vertical strip
                      (centred, e.g. 0.3 uses the middle 30% of rows).
        max_range   : clamp ceiling — returned when no valid pixel in sector.
        min_range   : pixels closer than this are treated as invalid.

    Returns: list of float, one min-distance per sector (metres).
    """
    H, W = depth.shape

    # Crop centre vertical strip — captures obstacles at drone altitude
    r0 = int(H * (0.5 - vstrip_frac * 0.5))
    r1 = int(H * (0.5 + vstrip_frac * 0.5))
    strip = depth[r0:r1, :]          # (strip_H, W)

    # Horizontal angle for each column: left-to-right = negative-to-positive (FLU)
    col_angles_deg = np.linspace(-hfov_deg * 0.5, hfov_deg * 0.5, W)

    # Per-column minimum (valid pixels only), then assign to sectors
    col_min = np.full(W, max_range, dtype=np.float32)
    valid_mask = (strip >= min_range) & (strip <= max_range) & np.isfinite(strip)
    # For each column take the minimum of valid rows
    for c in range(W):
        col_vals = strip[:, c][valid_mask[:, c]]
        if col_vals.size > 0:
            col_min[c] = col_vals.min()

    results: list[float] = []
    for lo_deg, hi_deg in sectors_deg:
        sector_mask = (col_angles_deg >= lo_deg) & (col_angles_deg < hi_deg)
        if not sector_mask.any():
            results.append(max_range)   # sector outside camera FOV
        else:
            results.append(float(col_min[sector_mask].min()))
    return results


# ── Sensor state container ────────────────────────────────────────────────────

class _State:
    """Thread-safe cache for the latest sensor readings."""

    def __init__(self):
        self._lock = threading.Lock()
        self.pos_ned = (0.0, 0.0, 0.0)         # North, East, Down (m)
        self.vel_ned = (0.0, 0.0, 0.0)         # vN, vE, vD (m/s)
        self.att_q = (1.0, 0.0, 0.0, 0.0)      # w,x,y,z  FRD->NED
        self.ang_vel_frd = (0.0, 0.0, 0.0)     # rad/s FRD body
        self.depth_msg: Image | None = None
        self.px4_yaw = 0.0                      # NED yaw for FSM logging
        # Freshness flags
        self.has_pos = False
        self.has_att = False
        self.has_ang_vel = False
        self.has_depth = False

    def update_local_pos(self, msg: VehicleLocalPosition):
        with self._lock:
            self.pos_ned = (msg.x, msg.y, msg.z)
            self.vel_ned = (msg.vx, msg.vy, msg.vz)
            self.has_pos = True

    def update_attitude(self, msg: VehicleAttitude):
        with self._lock:
            q = msg.q
            self.att_q = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            w, x, y, z = self.att_q
            # NED yaw (CW from North): standard ZYX Euler for FRD→NED quaternion
            # denominator = 1-2*(y²+z²), NOT 1-2*(x²+y²)
            self.px4_yaw = math.atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
            self.has_att = True

    def update_ang_vel(self, msg: SensorCombined):
        with self._lock:
            g = msg.gyro_rad
            self.ang_vel_frd = (float(g[0]), float(g[1]), float(g[2]))
            self.has_ang_vel = True

    def update_depth(self, msg: Image):
        with self._lock:
            self.depth_msg = msg
            self.has_depth = True

    def snapshot(self):
        with self._lock:
            return (
                self.pos_ned, self.vel_ned, self.att_q,
                self.ang_vel_frd, self.depth_msg, self.px4_yaw,
                self.has_pos and self.has_att and self.has_ang_vel and self.has_depth,
            )


# ── Adaptive Speed Controller (MODANC, IROS 2025) ────────────────────────────

class AdaptiveSpeedController:
    """3-neuron speed adaptation network with ICP online learning.

    Reference: T. Phetpoon et al., IROS 2025, Section D.1.

    Network:  d_min → S1 → {X1 (predictive), X0=Rm(X1) (reflexive)}
              S2 = W0·X0 + W1·X1   →   speed_scale = max(0, 1 − S2)

    ICP update (Eq. 4):
      ΔW1 = μs·(dX0/dt)·X1 − γs·(ks + P(t))·W1²
      P(t) = 0.05·X1  if X1 > 0  else  0          (Eq. 5)
    """

    def __init__(
        self,
        d_reflexive: float  = 1.0,    # m  — reflexive (short-range) threshold
        d_predictive: float = 4.0,    # m  — predictive (long-range) threshold
        W0: float           = 1.0,    # reflexive gain (fixed)
        W1_init: float      = 0.5,    # initial predictive gain (learned)
        mu_s: float         = 0.3,    # ICP learning rate
        gamma_s: float      = 0.007,  # ICP forgetting rate  (paper: 0.0001 → ~2.8h; 0.001 → ~100s)
        k_s: float          = 0.1,    # ICP offset           (paper: 0.01  → ~2.8h; 0.1   → ~100s)
    ):
        self.d_ref   = d_reflexive
        self.d_pred  = d_predictive
        self.W0      = W0
        self.W1      = W1_init
        self.mu_s    = mu_s
        self.gamma_s = gamma_s
        self.k_s     = k_s
        self._X0_prev = 0.0

    # ------------------------------------------------------------------
    def _s1(self, d_min: float) -> float:
        """S1 linear neuron: X1 = (d_pred − d_min) / (d_pred − d_ref).

        X1 = 0 at d_pred (no danger), X1 = 1 at d_ref (boundary),
        X1 > 1 when d < d_ref (inside reflexive zone).
        """
        return (self.d_pred - d_min) / (self.d_pred - self.d_ref)

    @staticmethod
    def _rm(x: float) -> float:
        """Modified ReLU Rm: pass-through if x ≥ 1, else 0 (Eq. 3 / paper Fig. 2b)."""
        return x if x >= 1.0 else 0.0

    # ------------------------------------------------------------------
    def step(self, d_min: float, dt: float) -> tuple[float, float]:
        """Run one ICP step and return (speed_scale, W1).

        d_min  : nearest forward obstacle distance in **metres**.
        dt     : seconds since last call.
        Returns: speed_scale ∈ [0, 1]  (1 = full speed, 0 = stop)
                 W1          (current predictive weight, for logging)
        """
        # S1 output: predictive signal
        X1 = max(0.0, self._s1(d_min))

        # Rm output: reflexive signal (fires only when X1 ≥ 1 → d ≤ d_ref)
        X0 = self._rm(X1)

        # ICP weight update (Eq. 4)
        dX0_dt = (X0 - self._X0_prev) / max(dt, 1e-6)
        P_t    = 0.05 * X1 if X1 > 0.0 else 0.0
        dW1    = (self.mu_s * dX0_dt * X1
                  - self.gamma_s * (self.k_s + P_t) * self.W1 ** 2)
        self.W1 = max(0.0, self.W1 + dW1)   # W1 ≥ 0: never accelerate
        self._X0_prev = X0

        # S2: combine reflexive + predictive (Section D.1)
        S2 = self.W0 * X0 + self.W1 * X1

        # Map to speed scale: S2=0 → full; S2≥1 → stop
        speed_scale = max(0.0, 1.0 - S2)
        return speed_scale, self.W1


# ── Main ROS2 node ────────────────────────────────────────────────────────────

class StudentOffboardNode(Node):

    def __init__(self, student: StudentPolicy, normalizer: RunningNormalizer,
                 goal_ned: tuple, vel_scale: float, args):
        super().__init__("student_offboard")

        self._student = student
        self._normalizer = normalizer
        self._goal_ned = goal_ned          # (goal_N, goal_E, goal_D)
        self._eff_max_vel = MAX_VELOCITY * vel_scale
        self._eff_max_yaw = MAX_YAW_RATE * vel_scale
        self._depth_topic  = args.depth_topic
        self._depth_hfov   = args.depth_hfov
        self._depth_vstrip = args.depth_vstrip
        self._max_vz_down = args.max_vz        # downward NED velocity cap (m/s)
        # Altitude floor in NED D: don't go below (goal_alt - 0.5m)
        # NED D is negative altitude, so floor_D = goal_D + 0.5  (less negative = lower altitude)
        self._alt_floor_D = goal_ned[2] + 0.5  # e.g. goal at -1.5m → floor at -1.0m (1.0m above gnd)
        self._state = _State()

        # Adaptive speed controller (MODANC ICP, IROS 2025)
        self._speed_ctrl = AdaptiveSpeedController(
            d_reflexive  = args.d_reflexive,
            d_predictive = args.d_predictive,
            W1_init      = args.w1_init,
            gamma_s      = args.gamma_s,
            k_s          = args.k_s,
        )
        self._last_ctrl_t = time.monotonic()

        # PX4 arming state
        self._fsm_state = "IDLE"
        self._fsm_cnt = 0
        self._arm_request = False
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arm_state = VehicleStatus.ARMING_STATE_DISARMED
        self.flight_check = False
        self.failsafe = False
        self._takeoff_alt = args.takeoff_alt

        # Velocity setpoints (NED), updated each control cycle
        self._v_ned = (0.0, 0.0, 0.0)
        self._yawspeed = 0.0
        self._fixed_alt = args.fixed_alt   # m above home, used by /goal_pose updates

        # TF broadcaster — publishes 'map' → 'base_link' so RViz can show the drone
        self._tf_broadcaster = TransformBroadcaster(self)

        px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscriptions
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position",
            lambda m: self._state.update_local_pos(m), px4_qos)
        self.create_subscription(
            VehicleAttitude, "/fmu/out/vehicle_attitude",
            lambda m: self._state.update_attitude(m), px4_qos)
        self.create_subscription(
            SensorCombined, "/fmu/out/sensor_combined",
            lambda m: self._state.update_ang_vel(m), px4_qos)
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status",
            self._status_cb, px4_qos)
        self.create_subscription(
            Image, self._depth_topic,
            lambda m: self._state.update_depth(m), sensor_qos)
        self.create_subscription(
            Bool, "/arm_message", self._arm_cb, px4_qos)
        # RViz 2D Nav Goal → update target position in flight
        self.create_subscription(
            PoseStamped, "/goal_pose", self._goal_pose_cb, 10)

        # Publishers
        self._pub_offboard = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", px4_qos)
        self._pub_traj = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", px4_qos)
        self._pub_cmd = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10)
        self._pub_markers = self.create_publisher(
            MarkerArray, "/student/markers", 10)

        # Control timer
        self._step = 0
        self.create_timer(1.0 / 100.0, self._control_cb)

        self.get_logger().info(
            f"StudentOffboard ready | goal_NED={goal_ned} | "
            f"max_vel={self._eff_max_vel:.1f} m/s | "
            f"depth={self._depth_topic} hfov={self._depth_hfov}°"
        )
        self.get_logger().info(
            "Arm: ros2 topic pub --once /arm_message std_msgs/msg/Bool \"data: true\""
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _status_cb(self, msg: VehicleStatus):
        if msg.nav_state != self.nav_state:
            self.get_logger().info(f"NAV_STATE -> {msg.nav_state}")
        if msg.arming_state != self.arm_state:
            self.get_logger().info(f"ARM_STATE -> {msg.arming_state}")
        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state
        self.flight_check = msg.pre_flight_checks_pass
        self.failsafe = msg.failsafe

    def _arm_cb(self, msg: Bool):
        self._arm_request = msg.data
        self.get_logger().info(f"Arm request: {msg.data}")

    def _goal_pose_cb(self, msg: PoseStamped):
        """Receive 2D Nav Goal from RViz and update the flight target.

        RViz publishes in the 'map' frame (ENU: x=East, y=North).
        Altitude is fixed at --fixed_alt metres above home.
        """
        goal_N = msg.pose.position.y   # ENU y = North
        goal_E = msg.pose.position.x   # ENU x = East
        goal_D = -self._fixed_alt      # NED Down = -altitude
        self._goal_ned = (goal_N, goal_E, goal_D)
        self._alt_floor_D = goal_D + 0.5   # keep floor 0.5 m below new goal
        self.get_logger().info(
            f"[goal_pose] new goal → N={goal_N:.1f} E={goal_E:.1f} alt={self._fixed_alt:.1f}m"
        )

    def _publish_markers(self, pos_ned: tuple, v_ned: tuple, goal_ned: tuple):
        """Publish RViz markers in the map (ENU) frame.

        Marker 0 (green arrow): commanded velocity vector from drone position.
        Marker 1 (blue arrow):  direction to goal, scaled to 1 m for clarity.
        Marker 2 (yellow sphere): goal position.
        """
        pN, pE, pD = pos_ned
        vN, vE, vD = v_ned
        gN, gE, gD = goal_ned

        # NED → ENU conversion
        ox = pE;  oy = pN;  oz = -pD   # drone ENU position

        now = self.get_clock().now().to_msg()
        def _arrow(mid: int, r: float, g: float, b: float,
                   sx: float, sy: float, sz: float,
                   ex: float, ey: float, ez: float) -> Marker:
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = "student"
            m.id = mid
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.scale.x = 0.08   # shaft diameter
            m.scale.y = 0.16   # head diameter
            m.scale.z = 0.0
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            start = Point(); start.x = sx; start.y = sy; start.z = sz
            end   = Point(); end.x   = ex; end.y   = ey; end.z   = ez
            m.points = [start, end]
            return m

        def _sphere(mid: int, r: float, g: float, b: float,
                    x: float, y: float, z: float, s: float = 0.3) -> Marker:
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = "student"
            m.id = mid
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = z
            m.pose.orientation.w = 1.0
            m.scale.x = s; m.scale.y = s; m.scale.z = s
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.8
            return m

        # 0: commanded velocity (green), length = actual speed (1 m = 1 m/s)
        vel_arrow = _arrow(
            0, 0.0, 1.0, 0.0,
            ox, oy, oz,
            ox + vE, oy + vN, oz - vD,   # NED → ENU for velocity
        )

        # 1: goal direction (blue), fixed 1.5 m length
        dN = gN - pN; dE = gE - pE; dD = gD - pD
        dist = max(math.sqrt(dN**2 + dE**2 + dD**2), 1e-3)
        arrow_len = 1.5
        goal_arrow = _arrow(
            1, 0.0, 0.4, 1.0,
            ox, oy, oz,
            ox + (dE / dist) * arrow_len,
            oy + (dN / dist) * arrow_len,
            oz + (-dD / dist) * arrow_len,
        )

        # 2: goal sphere (yellow)
        goal_sphere = _sphere(2, 1.0, 1.0, 0.0, gE, gN, -gD)

        markers = MarkerArray()
        markers.markers = [vel_arrow, goal_arrow, goal_sphere]  # type: ignore[assignment]
        self._pub_markers.publish(markers)

    def _publish_tf(self, pos_ned: tuple, q_frd_ned: tuple):
        """Broadcast TF: map (ENU) → base_link (FLU drone body)."""
        pN, pE, pD = pos_ned
        # NED → ENU position
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = pE    # East
        t.transform.translation.y = pN    # North
        t.transform.translation.z = -pD   # Up
        # PX4 FRD/NED → ROS FLU/ENU quaternion
        w, x, y, z = _px4_to_enu_quat(q_frd_ned)
        t.transform.rotation.w = w
        t.transform.rotation.x = x
        t.transform.rotation.y = y
        t.transform.rotation.z = z
        self._tf_broadcaster.sendTransform(t)

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_obs(self, pos, vel_ned, q, ang_vel_frd, depth_msg) -> torch.Tensor | None:
        """Build 16D student obs from sensor readings. Returns (1,16) tensor."""
        if depth_msg is None:
            return None

        # [0:3] Linear velocity in FLU body frame
        vel_flu = _ned_to_flu(q, vel_ned)

        # [3:6] Angular velocity in FLU body frame.
        # FRD→FLU: negate Y (right→left, pitch axis flips) and Z (down→up, yaw axis flips).
        # Roll  (X): axis unchanged, sign unchanged.
        # Pitch (Y): FRD +Y=right, FLU +Y=left → pitch-up has opposite sign → negate Y.
        # Yaw   (Z): FRD +Z=down (CW=+), FLU +Z=up (CCW=+) → yaw-right has opposite sign → negate Z.
        ang_flu = (ang_vel_frd[0], -ang_vel_frd[1], -ang_vel_frd[2])

        # [6:9] Unit vector from drone to goal in FLU body frame
        gN, gE, gD = self._goal_ned
        pN, pE, pD = pos
        dN, dE, dD = gN - pN, gE - pE, gD - pD
        goal_flu = _ned_to_flu(q, (dN, dE, dD))
        unit_goal_flu = _normalize3(goal_flu)

        # [9] 2D horizontal distance (m)
        dist_2d = math.sqrt(dN**2 + dE**2)

        # [10] Vertical distance: positive = goal is above drone
        # In NED: D decreases as altitude increases, so goal above means gD < pD
        dist_z = pD - gD   # positive when goal is above (NED D is negative altitude)

        # [11:16] 5 depth sectors normalised to [0,1] — identical layout to LaserScan path
        depth_arr = _parse_depth_image(depth_msg)
        raw_sectors = _depth_to_sectors(
            depth_arr, BEAM_SECTOR_DEG,
            self._depth_hfov, self._depth_vstrip,
            LIDAR_RANGE, DEPTH_MIN_RANGE,
        )
        beams = [b / LIDAR_RANGE for b in raw_sectors]

        obs_list = (
            list(vel_flu)
            + list(ang_flu)
            + list(unit_goal_flu)
            + [dist_2d, dist_z]
            + beams
        )
        return torch.tensor(obs_list, dtype=torch.float32).unsqueeze(0)  # (1, 16)

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_cb(self):
        pos, vel_ned, q, ang_vel_frd, depth_msg, px4_yaw, ready = self._state.snapshot()

        # Always tick FSM and publish to keep the offboard stream alive
        self._tick_fsm()

        if not ready:
            self._publish_zero()
            return

        obs = self._build_obs(pos, vel_ned, q, ang_vel_frd, depth_msg)
        if obs is None:
            self._publish_zero()
            return

        with torch.inference_mode():
            norm_obs = self._normalizer.normalize(obs)
            actions = self._student(norm_obs)   # (1, 4) in [-1, 1]

        # Student policy outputs BODY-FRAME FLU velocities (not world frame).
        # The student was distilled from body-frame-only obs (lin_vel_b, goal_dir_b,
        # grav_b) so it cannot know absolute world heading — it outputs ego-centric
        # commands that must be rotated to world frame using the current PX4 yaw.
        #
        # action[0] = yaw_rate  (CCW positive in Isaac FLU)
        # action[1] = vx_b      (FLU forward velocity)
        # action[2] = vy_b      (FLU left velocity)
        # action[3] = vz_b      (FLU up velocity)
        yaw_rate_flu = float(actions[0, 0]) * self._eff_max_yaw
        vx_b = float(actions[0, 1]) * self._eff_max_vel
        vy_b = float(actions[0, 2]) * self._eff_max_vel
        vz_b = float(actions[0, 3]) * self._eff_max_vel

        # MODANC adaptive speed control (IROS 2025, Section D.1).
        # Uses forward sectors 1-3 (obs[12:15], ±54°) as d_min input.
        # Lateral vy_b and yaw_rate are NOT scaled — student steers freely.
        now_t = time.monotonic()
        dt_ctrl = now_t - self._last_ctrl_t
        self._last_ctrl_t = now_t
        fwd_min_norm = min(obs[0, 12].item(), obs[0, 13].item(), obs[0, 14].item())
        d_min_m = fwd_min_norm * LIDAR_RANGE          # normalized [0,1] → metres
        speed_scale, _W1 = self._speed_ctrl.step(d_min_m, dt_ctrl)
        vx_b *= speed_scale

        # Isaac body FLU (+X=fwd, +Y=left) → NED world using PX4 yaw (CW from North):
        #   fwd  unit in NE = (cos y,  sin y)
        #   left unit in NE = (sin y, -cos y)   [90° CW from fwd in the NE map plane]
        # Verified: yaw=0°: fwd→N✓ left→W(0,-1)✓
        #           yaw=90°: fwd→E✓ left→N(1,0)✓
        cos_y = math.cos(px4_yaw)
        sin_y = math.sin(px4_yaw)
        vx_world = vx_b * cos_y + vy_b * sin_y   # North component
        vy_world = vx_b * sin_y - vy_b * cos_y   # East component

        # NEUp → NED with clamping
        v_ned_n = max(-self._eff_max_vel, min(self._eff_max_vel, vx_world))
        v_ned_e = max(-self._eff_max_vel, min(self._eff_max_vel, vy_world))
        # Vertical: Down = -Up; cap downward conservatively, allow full upward speed
        v_ned_d_raw = -vz_b
        v_ned_d = max(-self._eff_max_vel, min(self._max_vz_down, v_ned_d_raw))
        # Altitude floor: if below goal_alt-0.5m, refuse further descent
        pD = pos[2]
        if pD > self._alt_floor_D and v_ned_d > 0.0:
            v_ned_d = 0.0
        # Yaw: Isaac FRU +Z=up → positive omega_z = CCW from above = yaw LEFT.
        # PX4 NED +Z=down → positive yawspeed = CW from above = yaw RIGHT.
        # Sign is OPPOSITE: negate to convert.
        yawspeed_ned = max(-self._eff_max_yaw, min(self._eff_max_yaw, -yaw_rate_flu))

        self._v_ned = (v_ned_n, v_ned_e, v_ned_d)
        self._yawspeed = yawspeed_ned

        self._publish_tf(pos, q)
        self._publish_markers(pos, (v_ned_n, v_ned_e, v_ned_d), self._goal_ned)

        if self._fsm_state == "OFFBOARD":
            self._publish_velocity(v_ned_n, v_ned_e, v_ned_d, yawspeed_ned)
        else:
            self._publish_zero()

        if self._step % 150 == 0:
            pN, pE, pD = pos
            gN, gE, gD = self._goal_ned
            dist = math.sqrt((gN - pN)**2 + (gE - pE)**2 + (gD - pD)**2)
            v_ned_n, v_ned_e, v_ned_d = self._v_ned
            goal_b = obs[0, 6:9].tolist()
            raw_act = [float(actions[0, i]) for i in range(4)]
            self.get_logger().info(
                f"[{self._step:6d}] {self._fsm_state:8s} | "
                f"dist={dist:.2f}m | yaw={math.degrees(px4_yaw):+.0f}deg | "
                f"goal_b=({goal_b[0]:+.2f},{goal_b[1]:+.2f},{goal_b[2]:+.2f}) | "
                f"act=[yaw={raw_act[0]:+.2f} vx={raw_act[1]:+.2f} vy={raw_act[2]:+.2f} vz={raw_act[3]:+.2f}] | "
                f"vel_NED=({v_ned_n:+.2f},{v_ned_e:+.2f},{v_ned_d:+.2f}) | "
                f"ICP[d={d_min_m:.1f}m spd={speed_scale:.2f} W1={_W1:.3f}] | "
                f"beams={[f'{b:.2f}' for b in obs[0, 11:].tolist()]}"
            )
        self._step += 1

    # ── Arming FSM (same as sim2sim_px4.py) ──────────────────────────────────

    def _tick_fsm(self):
        match self._fsm_state:
            case "IDLE":
                if self.flight_check and self._arm_request:
                    self._fsm_state = "ARMING"
                    self._fsm_cnt = 0
                    self.get_logger().info("-> ARMING")
            case "ARMING":
                self._send_offboard_mode()
                self._send_zero_traj()
                self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                if self.arm_state == VehicleStatus.ARMING_STATE_ARMED and self._fsm_cnt > 10:
                    self._fsm_state = "TAKEOFF"
                    self._fsm_cnt = 0
                    self.get_logger().info("-> TAKEOFF")
                if not self.flight_check:
                    self._fsm_state = "IDLE"
            case "TAKEOFF":
                self._send_offboard_mode()
                self._send_zero_traj()
                self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
                               param1=1.0, param7=self._takeoff_alt)
                if self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                    self._fsm_state = "LOITER"
                    self._fsm_cnt = 0
                    self.get_logger().info("-> LOITER (waiting for hover)")
                if not self.flight_check:
                    self._fsm_state = "IDLE"
            case "LOITER":
                self._send_offboard_mode()
                self._send_zero_traj()
                self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                if self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER:
                    self._fsm_state = "OFFBOARD"
                    self._fsm_cnt = 0
                    self.get_logger().info("-> OFFBOARD  student policy active")
                if not self.flight_check:
                    self._fsm_state = "IDLE"
            case "OFFBOARD":
                if (not self.flight_check
                        or self.arm_state != VehicleStatus.ARMING_STATE_ARMED
                        or self.failsafe):
                    self._fsm_state = "IDLE"
                    self.get_logger().warn("Offboard lost — returning to IDLE")
                else:
                    self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        if self.arm_state != VehicleStatus.ARMING_STATE_ARMED:
            self._arm_request = False
        self._fsm_cnt += 1

    # ── Publishers ────────────────────────────────────────────────────────────

    def _send_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self._pub_offboard.publish(msg)

    def _publish_zero(self):
        self._send_offboard_mode()
        self._send_zero_traj()

    def _send_zero_traj(self):
        msg = TrajectorySetpoint()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.position[0] = float("nan")
        msg.position[1] = float("nan")
        msg.position[2] = float("nan")
        msg.velocity[0] = 0.0
        msg.velocity[1] = 0.0
        msg.velocity[2] = 0.0
        msg.yaw = float("nan")
        msg.yawspeed = 0.0
        self._pub_traj.publish(msg)

    def _publish_velocity(self, vn: float, ve: float, vd: float, yawspeed: float):
        self._send_offboard_mode()
        ts = self.get_clock().now().nanoseconds // 1000
        msg = TrajectorySetpoint()
        msg.timestamp = ts
        msg.position[0] = float("nan")
        msg.position[1] = float("nan")
        msg.position[2] = float("nan")
        msg.acceleration[0] = float("nan")
        msg.acceleration[1] = float("nan")
        msg.acceleration[2] = float("nan")
        msg.velocity[0] = vn
        msg.velocity[1] = ve
        msg.velocity[2] = vd
        msg.yaw = float("nan")
        msg.yawspeed = yawspeed
        self._pub_traj.publish(msg)

    def _send_cmd(self, command, param1: float = 0.0, param2: float = 0.0,
                  param7: float = 0.0):
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.param7 = param7
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self._pub_cmd.publish(msg)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Student policy ROS2 offboard node")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to student .pth from dagger_distill.py")
    parser.add_argument("--goal_north", type=float, default=5.0,
                        help="Goal North of takeoff point (m)")
    parser.add_argument("--goal_east", type=float, default=0.0,
                        help="Goal East of takeoff point (m)")
    parser.add_argument("--goal_alt", type=float, default=1.5,
                        help="Goal altitude above takeoff point (m)")
    parser.add_argument("--takeoff_alt", type=float, default=1.5,
                        help="PX4 auto-takeoff altitude (m). Set equal to goal_alt.")
    parser.add_argument("--vel_scale", type=float, default=0.5,
                        help="Scale factor on MAX_VELOCITY (0.5 = 2 m/s max).")
    parser.add_argument("--max_vz", type=float, default=0.3,
                        help="Max downward NED velocity the policy can command (m/s). "
                             "Lower = safer near goal altitude. Default 0.3 m/s.")
    parser.add_argument("--depth_topic", type=str, default="/camera/depth/image_raw",
                        help="ROS2 depth Image topic (sensor_msgs/Image).")
    parser.add_argument("--depth_hfov", type=float, default=87.0,
                        help="Depth camera horizontal FOV in degrees. "
                             "RealSense D435/D455: 86-87°. ZED 2: 110°.")
    parser.add_argument("--depth_vstrip", type=float, default=0.3,
                        help="Fraction of image height used as the vertical strip "
                             "(centred). 0.3 = middle 30%% of rows.")
    parser.add_argument("--fixed_alt", type=float, default=1.8,
                        help="Fixed altitude (m above home) used by /goal_pose 2D Nav Goal. "
                             "Ignored when goal is set via --goal_north/east/alt.")
    # ── MODANC adaptive speed control (IROS 2025) ──
    parser.add_argument("--d_reflexive", type=float, default=1.0,
                        help="ICP short-range threshold (m): reflexive speed cut triggers "
                             "when d_min ≤ this value. Paper default: 1.0 m.")
    parser.add_argument("--d_predictive", type=float, default=4.0,
                        help="ICP long-range threshold (m): predictive slow-down begins "
                             "when d_min ≤ this value. Paper default: 4.0 m.")
    parser.add_argument("--w1_init", type=float, default=0.5,
                        help="Initial predictive weight W1 (learned online via ICP). "
                             "0.0 = pure reflexive at start; 1.0 = strong predictive. "
                             "Default 0.5 for balanced immediate behavior.")
    parser.add_argument("--gamma_s", type=float, default=0.001,
                        help="ICP forgetting rate γs. Higher = faster W1 decay when "
                             "no obstacles. Paper value 0.0001 (~2.8h to halve); "
                             "default 0.001 (~100s). Try 0.005 for ~20s, 0.01 for ~10s.")
    parser.add_argument("--k_s", type=float, default=0.1,
                        help="ICP baseline decay offset ks. Scales with gamma_s. "
                             "Paper value 0.01; default 0.1.")
    args = parser.parse_args()

    # Load student checkpoint
    ckpt_path = os.path.abspath(args.checkpoint)
    print(f"[INFO] Checkpoint  : {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    obs_dim = ckpt.get("student_obs_dim", STUDENT_OBS_DIM)
    action_dim = ckpt.get("action_dim", ACTION_DIM)
    hidden_dims = _infer_hidden_dims(ckpt["student"], action_dim)

    print(f"[INFO] Obs dim     : {obs_dim}")
    print(f"[INFO] Hidden dims : {hidden_dims}")
    print(f"[INFO] DAgger iter : {ckpt.get('dagger_iter', '?')}")
    print(f"[INFO] Final beta  : {ckpt.get('beta', float('nan')):.4f}")

    student = StudentPolicy(obs_dim, action_dim, hidden_dims)
    normalizer = RunningNormalizer(obs_dim)
    student.load_state_dict(ckpt["student"])
    normalizer.load_state_dict(ckpt["normalizer"])
    student.eval()

    # Goal in NED: (North, East, Down)
    # Down = negative altitude (NED convention: positive D = below home)
    goal_ned = (args.goal_north, args.goal_east, -args.goal_alt)
    print(f"[INFO] Goal NED    : {goal_ned}  (N={args.goal_north}, E={args.goal_east}, alt={args.goal_alt}m)")
    print(f"[INFO] vel_scale   : {args.vel_scale}  (max {MAX_VELOCITY * args.vel_scale:.1f} m/s)")
    print(f"[INFO] Depth topic  : {args.depth_topic}")
    print(f"[INFO] Depth HFOV   : {args.depth_hfov}°  vstrip={args.depth_vstrip}")
    t_half = 1.0 / (max(args.w1_init, 0.1) * args.gamma_s * args.k_s * 100)
    print(f"[INFO] ICP speed   : d_ref={args.d_reflexive}m  d_pred={args.d_predictive}m  "
          f"W1_init={args.w1_init}  γs={args.gamma_s}  ks={args.k_s}  "
          f"(W1 halves in ~{t_half:.0f}s at rest)")

    rclpy.init()
    node = StudentOffboardNode(student, normalizer, goal_ned, args.vel_scale, args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("[INFO] Node shut down.")


if __name__ == "__main__":
    main()
