"""Live monitor: print 19D student obs built from real PX4 + LiDAR sensors.

No Isaac Lab needed.  Run this alongside student_ros2_node.py (or before
arming) to verify the obs look physically correct before sim2real transfer.

What to check while the drone is on the bench:
  lin_vel_x/y/z  ≈ 0.0  (not moving)
  ang_vel_x/y/z  ≈ 0.0  (not rotating)
  gravity_z      ≈ -1.0 (unit gravity points down in FLU = -Z)
  gravity_x/y    ≈ 0.0  (level)
  goal_dir       = unit vector pointing toward the goal in body FLU
  dist_2d        = horizontal distance to goal (m)
  dist_z         = vertical distance to goal, positive = goal is above (m)
  beams 28-32    = LiDAR ranges at -11,-5,+1,+7,+13 deg (should be ≥5 m in open space)

Run:
  python3 scripts/rl_games/verify_student_obs.py \\
      --goal_north 5.0 --goal_east 0.0 --goal_alt 1.5 \\
      --lidar_topic /scan
"""

import argparse
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy,
    QoSHistoryPolicy, QoSDurabilityPolicy,
)

from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition, SensorCombined
from sensor_msgs.msg import LaserScan

# ── Obs layout ───────────────────────────────────────────────────────────────
STUDENT_OBS_DIM = 19
LIDAR_RANGE = 5.0
BEAM_ANGLES_DEG = [-11.0, -5.0, 1.0, 7.0, 13.0]

OBS_NAMES = [
    "lin_vel_x  fwd  (m/s)",
    "lin_vel_y  left (m/s)",
    "lin_vel_z  up   (m/s)",
    "ang_vel_x  roll (r/s)",
    "ang_vel_y  pitch(r/s)",
    "ang_vel_z  yaw  (r/s)",
    "gravity_x  (FLU, unit)",
    "gravity_y  (FLU, unit)",
    "gravity_z  (FLU, unit)",
    "goal_dir_x fwd  (unit)",
    "goal_dir_y left (unit)",
    "goal_dir_z up   (unit)",
    "dist_2d    horiz(m)   ",
    "dist_z     vert (m)   ",
    "beam_28  -11 deg [0-1]",
    "beam_29   -5 deg [0-1]",
    "beam_30   +1 deg [0-1]",
    "beam_31   +7 deg [0-1]",
    "beam_32  +13 deg [0-1]",
]

# Expected range when drone is level and stationary (for quick sanity check)
# (min, max) — None means skip check
SANITY = [
    (-6.0,  6.0),   # lin_vel_x
    (-6.0,  6.0),   # lin_vel_y
    (-6.0,  6.0),   # lin_vel_z
    (-8.0,  8.0),   # ang_vel_x
    (-8.0,  8.0),   # ang_vel_y
    (-8.0,  8.0),   # ang_vel_z
    (-0.3,  0.3),   # gravity_x  (near 0 when level)
    (-0.3,  0.3),   # gravity_y  (near 0 when level)
    (-1.1, -0.9),   # gravity_z  (near -1 when level)
    (-1.1,  1.1),   # goal_dir_x
    (-1.1,  1.1),   # goal_dir_y
    (-1.1,  1.1),   # goal_dir_z
    (0.0,  50.0),   # dist_2d
    (-20.0, 20.0),  # dist_z
    (0.0, 1.05),   # beams normalized [0, 1]
    (0.0, 1.05),
    (0.0, 1.05),
    (0.0, 1.05),
    (0.0, 1.05),
]


# ── Quaternion / geometry helpers ─────────────────────────────────────────────

def _quat_apply(q_wxyz, v):
    """Rotate vector v by unit quaternion q (Rodrigues formula)."""
    w, qx, qy, qz = q_wxyz
    vx, vy, vz = v
    cx = qy * vz - qz * vy
    cy = qz * vx - qx * vz
    cz = qx * vy - qy * vx
    return (
        vx + 2 * w * cx + 2 * (qy * cz - qz * cy),
        vy + 2 * w * cy + 2 * (qz * cx - qx * cz),
        vz + 2 * w * cz + 2 * (qx * cy - qy * cx),
    )


def _ned_to_flu(q_wxyz, v_ned):
    """Rotate NED world vector to FLU body.
    q_wxyz: PX4 attitude quaternion (FRD body -> NED world).
    """
    w, x, y, z = q_wxyz
    v_frd = _quat_apply((w, -x, -y, -z), v_ned)   # conjugate = NED -> FRD
    return (v_frd[0], -v_frd[1], -v_frd[2])        # FRD -> FLU


def _norm3(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def _normalize3(v, eps=1e-6):
    n = max(_norm3(v), eps)
    return (v[0] / n, v[1] / n, v[2] / n)


def _extract_beams(msg: LaserScan, angles_deg, max_range):
    beams = []
    n = len(msg.ranges)
    for deg in angles_deg:
        idx = round((math.radians(deg) - msg.angle_min) / msg.angle_increment)
        idx = max(0, min(n - 1, idx))
        r = msg.ranges[idx]
        if math.isnan(r) or math.isinf(r) or r <= 0.0:
            r = max_range
        beams.append(min(r, max_range))
    return beams


# ── Sensor state ─────────────────────────────────────────────────────────────

class _State:
    def __init__(self):
        self._lock = threading.Lock()
        self.pos_ned = (0.0, 0.0, 0.0)
        self.vel_ned = (0.0, 0.0, 0.0)
        self.att_q = (1.0, 0.0, 0.0, 0.0)
        self.ang_vel_frd = (0.0, 0.0, 0.0)
        self.lidar: LaserScan | None = None
        self.t_pos = 0.0
        self.t_att = 0.0
        self.t_ang = 0.0
        self.t_lidar = 0.0

    def set_pos(self, msg: VehicleLocalPosition):
        with self._lock:
            self.pos_ned = (msg.x, msg.y, msg.z)
            self.vel_ned = (msg.vx, msg.vy, msg.vz)
            self.t_pos = time.monotonic()

    def set_att(self, msg: VehicleAttitude):
        with self._lock:
            q = msg.q
            self.att_q = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            self.t_att = time.monotonic()

    def set_ang(self, msg: SensorCombined):
        with self._lock:
            g = msg.gyro_rad
            self.ang_vel_frd = (float(g[0]), float(g[1]), float(g[2]))
            self.t_ang = time.monotonic()

    def set_lidar(self, msg: LaserScan):
        with self._lock:
            self.lidar = msg
            self.t_lidar = time.monotonic()

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            age = {
                "pos": now - self.t_pos,
                "att": now - self.t_att,
                "ang": now - self.t_ang,
                "lidar": now - self.t_lidar,
            }
            return (
                self.pos_ned, self.vel_ned, self.att_q,
                self.ang_vel_frd, self.lidar, age,
            )


# ── Monitor node ──────────────────────────────────────────────────────────────

class ObsMonitorNode(Node):

    def __init__(self, goal_ned, lidar_topic, print_hz):
        super().__init__("student_obs_monitor")
        self._goal_ned = goal_ned
        self._state = _State()

        px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        lidar_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(VehicleLocalPosition,
            "/fmu/out/vehicle_local_position", self._state.set_pos, px4_qos)
        self.create_subscription(VehicleAttitude,
            "/fmu/out/vehicle_attitude", self._state.set_att, px4_qos)
        self.create_subscription(SensorCombined,
            "/fmu/out/sensor_combined", self._state.set_ang, px4_qos)
        self.create_subscription(LaserScan,
            lidar_topic, self._state.set_lidar, lidar_qos)

        self._print_timer = self.create_timer(1.0 / print_hz, self._print_cb)
        self._tick = 0

    def _build_obs(self, pos, vel_ned, q, ang_frd, lidar):
        """Build 19D obs — mirrors student_ros2_node._build_obs exactly."""
        # [0:3] Linear velocity FLU body
        vel_flu = _ned_to_flu(q, vel_ned)

        # [3:6] Angular velocity FLU body
        ang_flu = (ang_frd[0], -ang_frd[1], -ang_frd[2])

        # [6:9] Projected gravity in FLU body (NED down = (0,0,+1))
        grav_flu = _ned_to_flu(q, (0.0, 0.0, 1.0))
        grav_flu = _normalize3(grav_flu)

        # [9:12] Unit goal direction in FLU body
        gN, gE, gD = self._goal_ned
        pN, pE, pD = pos
        dN, dE, dD = gN - pN, gE - pE, gD - pD
        goal_flu = _ned_to_flu(q, (dN, dE, dD))
        unit_goal = _normalize3(goal_flu)

        # [12] 2D distance
        dist_2d = math.sqrt(dN**2 + dE**2)

        # [13] Vertical: pD - gD > 0 means goal is above (NED: smaller D = higher)
        dist_z = pD - gD

        # [14:19] Front LiDAR beams normalized to [0,1] — matches Isaac training obs
        if lidar is not None:
            beams = [b / LIDAR_RANGE for b in _extract_beams(lidar, BEAM_ANGLES_DEG, LIDAR_RANGE)]
        else:
            beams = [1.0] * 5

        return (
            list(vel_flu)
            + list(ang_flu)
            + list(grav_flu)
            + list(unit_goal)
            + [dist_2d, dist_z]
            + beams
        )

    def _print_cb(self):
        pos, vel_ned, q, ang_frd, lidar, age = self._state.snapshot()

        stale = {k: v > 1.0 for k, v in age.items()}
        any_stale = any(stale.values())

        obs = self._build_obs(pos, vel_ned, q, ang_frd, lidar)

        # Header every 20 prints
        if self._tick % 20 == 0:
            print()
            print(f"{'#':<4} {'Obs name':<26} {'value':>10}  {'sanity':>6}  "
                  f"{'age(pos/att/ang/lidar)':>26}")
            print("-" * 80)

        print(f"\n[tick {self._tick:04d}]  "
              f"pos_NED=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})  "
              f"stale={''.join(k[0] for k, v in stale.items() if v) or 'none'}")

        for i, (name, val, (lo, hi)) in enumerate(zip(OBS_NAMES, obs, SANITY)):
            ok = lo <= val <= hi
            flag = "OK " if ok else "!!!"
            print(f"  [{i:>2}] {name:<26} {val:>10.4f}  {flag}")

        if any_stale:
            ages_str = "/".join(f"{age[k]:.1f}s" for k in ("pos", "att", "ang", "lidar"))
            print(f"  [WARN] Stale sensor data: {ages_str}")
        if lidar is None:
            print("  [WARN] No LiDAR data received yet")

        self._tick += 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live monitor of 19D student obs from PX4 sensors")
    parser.add_argument("--goal_north", type=float, default=5.0,
                        help="Goal North of takeoff (m)")
    parser.add_argument("--goal_east", type=float, default=0.0,
                        help="Goal East of takeoff (m)")
    parser.add_argument("--goal_alt", type=float, default=1.5,
                        help="Goal altitude above takeoff (m, positive up)")
    parser.add_argument("--lidar_topic", type=str, default="/scan",
                        help="ROS2 LaserScan topic")
    parser.add_argument("--print_hz", type=float, default=1.0,
                        help="Print rate (Hz). Use 0.5 for slow scroll, 2.0 for faster.")
    args = parser.parse_args()

    # Goal in NED: (North, East, Down=-alt)
    goal_ned = (args.goal_north, args.goal_east, -args.goal_alt)

    print(f"[MONITOR] Goal NED  : {goal_ned}")
    print(f"[MONITOR] LiDAR     : {args.lidar_topic}")
    print(f"[MONITOR] Print Hz  : {args.print_hz}")
    print()
    print("Expected values (drone level, stationary, goal ahead):")
    print("  lin_vel_x/y/z  ≈ 0.0")
    print("  ang_vel_x/y/z  ≈ 0.0")
    print("  gravity_z      ≈ -1.0  (gravity down = -Z in FLU)")
    print("  goal_dir_x     ≈ +1.0  (goal is ahead when facing goal)")
    print("  dist_2d        = horizontal distance to goal")
    print("  dist_z         = altitude diff (+ means goal is above)")
    print("  beams 28-32    ≈ 5.0 m (open space ahead)")
    print()

    rclpy.init()
    node = ObsMonitorNode(goal_ned, args.lidar_topic, args.print_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
