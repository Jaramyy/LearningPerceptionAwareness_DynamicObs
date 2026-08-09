"""ROS2 trial monitor for IROS Gazebo simulation experiments.

Monitors a single flight trial driven by student_ros2_node_icp.py and records:
  - success    : reached goal within success_radius before timeout
  - collision  : drone proximity to known obstacle positions (from world JSON)
  - flight_time: seconds from takeoff detect to trial end
  - latency_ms : per-step control latency (scan stamp → setpoint stamp)
  - orb_mean   : mean ORB keypoint count per camera frame
  - orb_std    : std dev of ORB keypoint count

On trial end the node writes one JSON result line to --output_json and exits.

Coordinate frames
-----------------
  PX4 local NED : x=North, y=East, z=Down  (from VehicleLocalPosition)
  Gazebo ENU    : x=East,  y=North, z=Up   (obstacle JSON is in ENU)
  Conversion    : ned_x = enu_y,  ned_y = enu_x,  ned_z = -enu_z

Run (after student node is already running)
-------------------------------------------
  python3 gz_eval_monitor.py \\
      --method pa \\
      --trial_id 0 \\
      --goal_east 12.0  --goal_north 0.0  --goal_alt 1.5 \\
      --obstacle_json ~/worlds/iros_exp_0.json \\
      --output_json results/pa_trial_0.json \\
      --timeout 60
"""

import argparse
import collections
import json
import math
import os
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from px4_msgs.msg import VehicleLocalPosition, TrajectorySetpoint
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Float32MultiArray

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

DRONE_RADIUS = 0.28     # m (drone arm span ≈ 0.55 m → half)
TAKEOFF_ALT_MIN = 0.5   # m — trial starts when drone climbs above this ENU z
CRASH_ALT = 0.25        # m — abort if drone drops below this while airborne
WALL_THICKNESS = 0.40   # m — default wall thickness from world gen


class EvalMonitor(Node):

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('gz_eval_monitor')

        self._method      = args.method
        self._trial_id    = args.trial_id
        self._goal_ned    = (args.goal_north, args.goal_east, -args.goal_alt)  # NED
        self._goal_enu    = (args.goal_east,  args.goal_north, args.goal_alt)  # ENU
        self._success_r   = args.success_radius
        self._timeout     = args.timeout
        self._output      = args.output_json

        self._obstacles   = self._load_obstacles(args.obstacle_json)
        self._orb         = cv2.ORB_create(nfeatures=500) if _CV2_AVAILABLE else None

        # State
        self._pos_ned: Optional[tuple] = None
        self._takeoff_time: Optional[float] = None
        self._airborne    = False
        self._done        = False
        self._result      = 'timeout'
        self._crash_pos_enu: Optional[tuple] = None
        self._nearest_obs_at_crash: Optional[dict] = None

        # Position history (last 30 ticks × 0.1 s = 3 s pre-crash trail)
        self._pos_hist: collections.deque = collections.deque(maxlen=30)

        # Latency tracking
        self._last_scan_stamp: Optional[float] = None   # ROS time (sec float)
        self._latency_samples: list[float] = []

        # ORB tracking
        self._orb_counts: list[int] = []

        # QoS
        px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        be_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self._cb_pos, px4_qos)
        self.create_subscription(
            LaserScan, '/scan',
            self._cb_scan, be_qos)
        self.create_subscription(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint',
            self._cb_setpoint, px4_qos)
        self.create_subscription(
            Image, '/depth_camera/image',
            self._cb_image, be_qos)

        self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f'EvalMonitor: method={self._method} trial={self._trial_id}  '
            f'goal_ned={self._goal_ned}  timeout={self._timeout}s'
        )

    # ── ROS callbacks ──────────────────────────────────────────────────────

    def _cb_pos(self, msg: VehicleLocalPosition) -> None:
        self._pos_ned = (msg.x, msg.y, msg.z)

    def _cb_scan(self, msg: LaserScan) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._last_scan_stamp = t

    def _cb_setpoint(self, msg: TrajectorySetpoint) -> None:
        self._last_setpoint_wall = time.monotonic()

    def _cb_image(self, msg: Image) -> None:
        if not _CV2_AVAILABLE or self._orb is None or not self._airborne:
            return
        if msg.encoding not in ('rgb8', 'bgr8', 'mono8'):
            return
        import numpy as _np
        arr = bytes(msg.data)
        h, w = msg.height, msg.width
        raw = _np.frombuffer(arr, dtype=_np.uint8)
        if msg.encoding == 'mono8':
            img = raw.reshape(h, w)
        else:
            img = cv2.cvtColor(
                raw.reshape(h, w, 3),
                cv2.COLOR_RGB2GRAY if msg.encoding == 'rgb8' else cv2.COLOR_BGR2GRAY,
            )
        kps = self._orb.detect(img, None)
        self._orb_counts.append(len(kps))

    # ── Timer / state machine ───────────────────────────────────────────────

    def _tick(self) -> None:
        if self._done:
            return

        if self._pos_ned is None:
            return

        ned_x, ned_y, ned_z = self._pos_ned
        enu_x = ned_y   # East
        enu_y = ned_x   # North
        enu_z = -ned_z  # altitude in ENU

        wall = time.monotonic()

        # Detect takeoff
        if not self._airborne and enu_z > TAKEOFF_ALT_MIN:
            self._airborne = True
            self._takeoff_time = wall
            self.get_logger().info(f'Trial {self._trial_id}: airborne at alt={enu_z:.2f} m')

        if not self._airborne:
            return

        elapsed = wall - self._takeoff_time  # type: ignore[operator]

        # Record position for pre-crash trail
        self._pos_hist.append((round(enu_x, 3), round(enu_y, 3), round(enu_z, 3)))

        # ── 1. Goal reached ─────────────────────────────────────────────
        goal_n, goal_e, goal_d = self._goal_ned
        dx = ned_x - goal_n
        dy = ned_y - goal_e
        dz = ned_z - goal_d
        dist_to_goal = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist_to_goal < self._success_r:
            self._result = 'success'
            self._finish(elapsed)
            return

        # ── 2. Collision ─────────────────────────────────────────────────
        if self._check_collision(enu_x, enu_y, enu_z):
            self._result = 'collision'
            self._crash_pos_enu = (enu_x, enu_y, enu_z)
            self._nearest_obs_at_crash = self._nearest_obstacle_info(enu_x, enu_y)
            self._finish(elapsed)
            return

        # ── 3. Altitude crash ────────────────────────────────────────────
        if enu_z < CRASH_ALT:
            self._result = 'crash'
            self._crash_pos_enu = (enu_x, enu_y, enu_z)
            self._finish(elapsed)
            return

        # ── 4. Timeout ───────────────────────────────────────────────────
        if elapsed > self._timeout:
            self._result = 'timeout'
            self._finish(elapsed)
            return

    _PILLAR_HEIGHT = 3.0

    def _nearest_obstacle_info(self, enu_x: float, enu_y: float) -> dict:
        """Return info about the closest obstacle to the given ENU position."""
        best = {'dist_m': 999.0, 'type': None, 'x': None, 'y': None}
        for obs in self._obstacles:
            ox, oy = obs['x'], obs['y']
            if obs['type'] == 'pillar':
                dist = math.hypot(enu_x - ox, enu_y - oy) - obs.get('radius', 0.3)
            else:
                wlen = obs.get('length', 2.0)
                wth = obs.get('thickness', WALL_THICKNESS)
                yaw = obs.get('yaw', 0.0)
                dx = enu_x - ox
                dy = enu_y - oy
                lx = dx * math.cos(yaw) + dy * math.sin(yaw)
                ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
                dist = max(0.0, abs(lx) - wlen / 2) + max(0.0, abs(ly) - wth / 2)
            if dist < best['dist_m']:
                best = {'dist_m': round(dist, 3), 'type': obs['type'],
                        'x': round(ox, 2), 'y': round(oy, 2)}
        return best

    def _check_collision(self, enu_x: float, enu_y: float, enu_z: float) -> bool:
        if enu_z > self._PILLAR_HEIGHT:
            return False
        for obs in self._obstacles:
            ox, oy = obs['x'], obs['y']
            if obs['type'] == 'pillar':
                dist = math.hypot(enu_x - ox, enu_y - oy)
                # surface-to-drone-center < 0.25 m → collision
                if dist - obs.get('radius', 0.3) < 0.25:
                    return True
            elif obs['type'] == 'wall':
                wlen = obs.get('length', 2.0)
                wth  = obs.get('thickness', WALL_THICKNESS)
                yaw  = obs.get('yaw', 0.0)
                dx = enu_x - ox
                dy = enu_y - oy
                lx =  dx * math.cos(yaw) + dy * math.sin(yaw)
                ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
                # surface-to-drone-center < 0.25 m on either axis → collision
                half_l = wlen / 2 + 0.25
                half_t = wth  / 2 + 0.25
                if abs(lx) < half_l and abs(ly) < half_t:
                    return True
        return False

    def _finish(self, elapsed: float) -> None:
        self._done = True

        # Latency stats
        lat_arr = self._latency_samples
        lat_mean = float(sum(lat_arr) / len(lat_arr)) * 1e3 if lat_arr else 0.0
        lat_std  = _std(lat_arr) * 1e3                       if lat_arr else 0.0
        lat_worst = max(lat_arr, default=0.0) * 1e3

        # ORB stats
        orb_arr  = self._orb_counts
        orb_mean = float(sum(orb_arr) / len(orb_arr)) if orb_arr else 0.0
        orb_std  = _std(orb_arr)                       if orb_arr else 0.0

        result = {
            'method':      self._method,
            'trial_id':    self._trial_id,
            'result':      self._result,
            'flight_time_s': round(elapsed, 3),
            'latency_mean_ms':  round(lat_mean,  3),
            'latency_std_ms':   round(lat_std,   3),
            'latency_worst_ms': round(lat_worst, 3),
            'latency_samples':  len(lat_arr),
            'orb_mean':   round(orb_mean, 1),
            'orb_std':    round(orb_std,  1),
            'orb_frames': len(orb_arr),
            'cv2_available': _CV2_AVAILABLE,
            # Crash diagnostics (None for success / timeout)
            'crash_pos_enu': (
                [round(v, 3) for v in self._crash_pos_enu]
                if self._crash_pos_enu else None
            ),
            'nearest_obstacle': self._nearest_obs_at_crash,
            'pre_crash_trail_enu': list(self._pos_hist),
        }

        self.get_logger().info(
            f'Trial {self._trial_id} DONE: result={self._result}  '
            f't={elapsed:.1f}s  orb={orb_mean:.0f}±{orb_std:.0f}  '
            f'lat={lat_mean:.1f}ms'
        )

        if self._output:
            os.makedirs(os.path.dirname(os.path.abspath(self._output)), exist_ok=True)
            with open(self._output, 'w') as fh:
                json.dump(result, fh, indent=2)
            self.get_logger().info(f'Results → {self._output}')

        # Signal the spin loop to exit
        raise SystemExit(0)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _load_obstacles(path: Optional[str]) -> list:
        if not path or not os.path.exists(path):
            return []
        with open(path) as fh:
            data = json.load(fh)
        return data.get('obstacles', [])


# ── Per-step latency tracking via wall-clock ──────────────────────────────────
#
# We measure latency differently from end-to-end timestamps.  We record the
# wall-clock time each /scan message arrives and each /trajectory_setpoint
# arrives, then pair them (setpoint N with the most recent scan before it).

class LatencyTracker:
    """Measures wall-clock gap between scan arrival and next setpoint."""
    def __init__(self) -> None:
        self._last_scan_wall: Optional[float] = None
        self.samples: list[float] = []

    def on_scan(self, wall: float) -> None:
        self._last_scan_wall = wall

    def on_setpoint(self, wall: float) -> None:
        if self._last_scan_wall is not None:
            gap = wall - self._last_scan_wall
            if 0 < gap < 0.5:   # filter outliers (>500 ms → missed message)
                self.samples.append(gap)


def _std(data: list) -> float:
    if len(data) < 2:
        return 0.0
    n = len(data)
    mean = sum(data) / n
    return math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1))


# ── Patched EvalMonitor with wall-clock latency ───────────────────────────────

class EvalMonitorV2(EvalMonitor):
    """Adds wall-clock latency + per-stage pipeline latency from /student/latency."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._lat_tracker = LatencyTracker()

        # Per-stage latency samples from /student/latency
        # [scan_age_ms, policy_ms, icp_ms, total_ms]
        self._stage_lat: list[list[float]] = [[], [], [], []]

        be_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Float32MultiArray, '/student/latency',
            self._cb_pipeline_lat, be_qos)

    def _cb_pipeline_lat(self, msg: Float32MultiArray) -> None:
        if not self._airborne or self._done:
            return
        for i, v in enumerate(msg.data[:4]):
            self._stage_lat[i].append(float(v))

    def _cb_scan(self, msg: LaserScan) -> None:
        super()._cb_scan(msg)
        self._lat_tracker.on_scan(time.monotonic())

    def _cb_setpoint(self, msg: TrajectorySetpoint) -> None:
        self._lat_tracker.on_setpoint(time.monotonic())

    def _finish(self, elapsed: float) -> None:
        self._latency_samples = self._lat_tracker.samples

        # Compute per-stage stats
        stage_names = ['scan_age', 'policy', 'icp', 'total']
        stage_stats = {}
        for name, samples in zip(stage_names, self._stage_lat):
            stage_stats[f'lat_{name}_mean_ms']  = round(sum(samples) / len(samples), 3) if samples else 0.0
            stage_stats[f'lat_{name}_std_ms']   = round(_std(samples), 3)               if samples else 0.0
            stage_stats[f'lat_{name}_worst_ms'] = round(max(samples, default=0.0), 3)

        # Build the base result dict via the parent's logic (re-implemented inline
        # so we can inject the extra fields without patching the base class)
        lat_arr = self._latency_samples
        lat_mean  = float(sum(lat_arr) / len(lat_arr)) * 1e3 if lat_arr else 0.0
        lat_std   = _std(lat_arr) * 1e3                       if lat_arr else 0.0
        lat_worst = max(lat_arr, default=0.0) * 1e3

        orb_arr  = self._orb_counts
        orb_mean = float(sum(orb_arr) / len(orb_arr)) if orb_arr else 0.0
        orb_std  = _std(orb_arr)                       if orb_arr else 0.0

        result = {
            'method':      self._method,
            'trial_id':    self._trial_id,
            'result':      self._result,
            'flight_time_s': round(elapsed, 3),
            'latency_mean_ms':  round(lat_mean,  3),
            'latency_std_ms':   round(lat_std,   3),
            'latency_worst_ms': round(lat_worst, 3),
            'latency_samples':  len(lat_arr),
            **stage_stats,
            'orb_mean':   round(orb_mean, 1),
            'orb_std':    round(orb_std,  1),
            'orb_frames': len(orb_arr),
            'cv2_available': _CV2_AVAILABLE,
            'crash_pos_enu': (
                [round(v, 3) for v in self._crash_pos_enu]
                if self._crash_pos_enu else None
            ),
            'nearest_obstacle': self._nearest_obs_at_crash,
            'pre_crash_trail_enu': list(self._pos_hist),
        }

        self._done = True
        self.get_logger().info(
            f'Trial {self._trial_id} DONE: result={self._result}  '
            f't={elapsed:.1f}s  orb={orb_mean:.0f}±{orb_std:.0f}  '
            f'lat={lat_mean:.1f}ms  '
            f'[scan={stage_stats.get("lat_scan_age_mean_ms",0):.1f} '
            f'policy={stage_stats.get("lat_policy_mean_ms",0):.1f} '
            f'icp={stage_stats.get("lat_icp_mean_ms",0):.1f}]ms'
        )

        if self._output:
            os.makedirs(os.path.dirname(os.path.abspath(self._output)), exist_ok=True)
            with open(self._output, 'w') as fh:
                json.dump(result, fh, indent=2)
            self.get_logger().info(f'Results → {self._output}')

        raise SystemExit(0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not _CV2_AVAILABLE:
        print('[WARN] cv2 not found — ORB feature counting disabled. '
              'Install: pip3 install opencv-python')

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--method',         type=str,   default='pa',
                        choices=['pa', 'nopa', 'navrl', 'panther'],
                        help='Method label written to result JSON')
    parser.add_argument('--trial_id',       type=int,   default=0,
                        help='Trial index (used as JSON label)')
    parser.add_argument('--goal_north',     type=float, default=0.0,
                        help='Goal North in PX4 NED (m)')
    parser.add_argument('--goal_east',      type=float, default=12.0,
                        help='Goal East in PX4 NED (m)')
    parser.add_argument('--goal_alt',       type=float, default=1.5,
                        help='Goal altitude (m, positive up)')
    parser.add_argument('--success_radius', type=float, default=1.5,
                        help='Goal reached radius (m, default 1.5)')
    parser.add_argument('--timeout',        type=float, default=60.0,
                        help='Max trial duration in seconds (default 60)')
    parser.add_argument('--obstacle_json',  type=str,   default=None,
                        help='Path to obstacle JSON from gz_world_gen.py')
    parser.add_argument('--output_json',    type=str,   default=None,
                        help='Path to write result JSON (stdout if omitted)')
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = EvalMonitorV2(args)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
