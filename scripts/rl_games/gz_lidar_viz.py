"""Visualize the 5 front LiDAR beams in Gazebo while the student policy runs.

Publishes colored beam lines directly to Gazebo via gz.transport13.
Color codes by range:
  GREEN  — beam >= 3.0 m (clear)
  YELLOW — beam 1.5–3.0 m (caution)
  RED    — beam < 1.5 m (obstacle close)

Requirements:
  - Gazebo world must have the MarkerManager GUI plugin loaded.
    (obstacle_poles.sdf already has it; default worlds may not.)
  - Run alongside student_ros2_node.py (separate terminal).

Usage:
    python3 scripts/rl_games/gz_lidar_viz.py

Optional flags:
    --lidar_topic /scan      (default /scan)
    --hz          20         (publish rate Hz, default 20)
    --beam_scale  0.03       (line width in Gazebo, default 0.03 m)
"""

import sys
import math
import argparse
import threading

# ── gz.transport / gz.msgs (Gazebo Harmonic) ─────────────────────────────────
sys.path.insert(0, '/usr/lib/python3/dist-packages')
import gz.transport13 as gzt
from gz.msgs10 import marker_pb2

# ── ROS2 ─────────────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy,
    QoSHistoryPolicy, QoSDurabilityPolicy,
)
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude
from sensor_msgs.msg import LaserScan

# ── Config ───────────────────────────────────────────────────────────────────
# 5 sectors: (lo_deg, hi_deg, mid_deg) — reversed vs Isaac angles.
# Isaac=FRU (neg=left), Gazebo/ROS=FLU (neg=right): same angle = opposite sides.
# Sector order (0=LEFT … 4=RIGHT) matches Isaac training convention.
BEAM_SECTOR_DEG = [
    (54.0, 90.0, 72.0),    # sector 0: LEFT  in Gazebo FLU
    (18.0, 54.0, 36.0),    # sector 1: l-ctr
    (-18.0, 18.0, 0.0),    # sector 2: forward
    (-54.0, -18.0, -36.0), # sector 3: r-ctr
    (-90.0, -54.0, -72.0), # sector 4: RIGHT in Gazebo FLU
]
LIDAR_RANGE     = 5.0                               # max range (m)
DRONE_RADIUS    = 0.25                              # skip first 0.25 m (body)

DANGER_RANGE    = 1.5   # RED below this
WARN_RANGE      = 3.0   # YELLOW below this (GREEN above)

_COLOR_GREEN  = (0.0, 1.0, 0.0, 0.9)
_COLOR_YELLOW = (1.0, 0.85, 0.0, 0.9)
_COLOR_RED    = (1.0, 0.0, 0.0, 0.9)


def _beam_color(rng: float):
    if rng < DANGER_RANGE:
        return _COLOR_RED
    if rng < WARN_RANGE:
        return _COLOR_YELLOW
    return _COLOR_GREEN


def _extract_beams(msg: LaserScan) -> list:
    """Return list of (min_range, mid_angle_rad) for each sector."""
    n = len(msg.ranges)
    results = []
    for lo_deg, hi_deg, mid_deg in BEAM_SECTOR_DEG:
        lo_idx = max(0, math.ceil((math.radians(lo_deg) - msg.angle_min) / msg.angle_increment))
        hi_idx = min(n - 1, int((math.radians(hi_deg) - msg.angle_min) / msg.angle_increment))
        min_r = LIDAR_RANGE
        for i in range(lo_idx, hi_idx + 1):
            r = msg.ranges[i]
            if not (math.isnan(r) or math.isinf(r) or r <= 0.0):
                min_r = min(min_r, r)
        results.append((min_r, math.radians(mid_deg)))
    return results


def _make_beam_marker(ns: str, marker_id: int,
                      ox, oy, oz,     # origin in Gazebo ENU world (m)
                      dx, dy,         # direction unit vector in ENU XY
                      length: float,
                      color: tuple,
                      line_width: float,
                      lifetime_s: float = 0.15) -> marker_pb2.Marker:
    """Build a Gazebo Marker LINE_STRIP from drone to beam endpoint."""
    m = marker_pb2.Marker()
    m.ns     = ns
    m.id     = marker_id
    m.action = marker_pb2.Marker.ADD_MODIFY
    m.type   = marker_pb2.Marker.LINE_STRIP
    m.lifetime.sec  = int(lifetime_s)
    m.lifetime.nsec = int((lifetime_s % 1.0) * 1e9)

    # Line width via scale (x = width for LINE_STRIP)
    m.scale.x = line_width
    m.scale.y = line_width
    m.scale.z = line_width

    r, g, b, a = color
    for comp in (m.material.ambient, m.material.diffuse, m.material.emissive):
        comp.r, comp.g, comp.b, comp.a = r, g, b, a

    # Start point (just past drone body)
    p0 = m.point.add()
    p0.x = ox + dx * DRONE_RADIUS
    p0.y = oy + dy * DRONE_RADIUS
    p0.z = oz

    # End point
    p1 = m.point.add()
    p1.x = ox + dx * length
    p1.y = oy + dy * length
    p1.z = oz

    return m


def _make_sphere_marker(ns: str, marker_id: int,
                        x, y, z, radius: float,
                        color: tuple,
                        lifetime_s: float = 0.15) -> marker_pb2.Marker:
    """Small sphere at drone position."""
    m = marker_pb2.Marker()
    m.ns     = ns
    m.id     = marker_id
    m.action = marker_pb2.Marker.ADD_MODIFY
    m.type   = marker_pb2.Marker.SPHERE
    m.lifetime.sec  = int(lifetime_s)
    m.lifetime.nsec = int((lifetime_s % 1.0) * 1e9)

    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = z
    m.pose.orientation.w = 1.0

    m.scale.x = radius * 2
    m.scale.y = radius * 2
    m.scale.z = radius * 2

    r, g, b, a = color
    for comp in (m.material.ambient, m.material.diffuse):
        comp.r, comp.g, comp.b, comp.a = r, g, b, a

    return m


# ── Sensor state ─────────────────────────────────────────────────────────────

class _State:
    def __init__(self):
        self._lock     = threading.Lock()
        self.pos_ned   = (0.0, 0.0, 0.0)   # (North, East, Down)
        self.px4_yaw   = 0.0               # CW from North (rad)
        self.lidar     = None

    def set_pos(self, msg: VehicleLocalPosition):
        with self._lock:
            self.pos_ned = (msg.x, msg.y, msg.z)

    def set_att(self, msg: VehicleAttitude):
        q = msg.q
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        with self._lock:
            self.px4_yaw = math.atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )

    def set_lidar(self, msg: LaserScan):
        with self._lock:
            self.lidar = msg

    def snapshot(self):
        with self._lock:
            return self.pos_ned, self.px4_yaw, self.lidar


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class LidarVizNode(Node):

    def __init__(self, lidar_topic: str, hz: float, beam_scale: float):
        super().__init__('gz_lidar_viz')
        self._state      = _State()
        self._beam_scale = beam_scale

        # ── gz.transport publisher ────────────────────────────────────────
        self._gz_node = gzt.Node()
        self._gz_pub  = self._gz_node.advertise('/marker', marker_pb2.Marker)

        # ── QoS profiles ─────────────────────────────────────────────────
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

        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self._state.set_pos, px4_qos,
        )
        self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self._state.set_att, px4_qos,
        )
        self.create_subscription(
            LaserScan,
            lidar_topic,
            self._state.set_lidar, lidar_qos,
        )

        self._timer = self.create_timer(1.0 / hz, self._publish_markers)
        self.get_logger().info(
            f'[gz_lidar_viz] started — listening on {lidar_topic}, '
            f'publishing to /marker at {hz} Hz'
        )

    # ── Marker publishing ─────────────────────────────────────────────────────

    def _publish_markers(self):
        pos_ned, px4_yaw, lidar = self._state.snapshot()
        if lidar is None:
            return

        # ── Drone position in Gazebo ENU ──────────────────────────────────
        # PX4 NED: (North, East, Down)  →  Gazebo ENU: (East, North, Up)
        pN, pE, pD = pos_ned
        gz_x = pE            # East
        gz_y = pN            # North
        gz_z = -pD           # Up

        # ── Drone yaw in Gazebo ENU (CCW from East +X) ───────────────────
        # PX4 yaw: CW from North.  ENU yaw = π/2 - px4_yaw.
        gz_yaw = math.pi / 2.0 - px4_yaw

        # ── Extract 5 front beams ─────────────────────────────────────────
        beams = _extract_beams(lidar)

        # ── Publish drone sphere ─────────────────────────────────────────
        sphere = _make_sphere_marker(
            'lidar_drone', 0,
            gz_x, gz_y, gz_z, 0.15,
            (0.2, 0.6, 1.0, 0.8),
        )
        self._gz_pub.publish(sphere)

        # ── Publish one marker per beam ──────────────────────────────────
        for i, (rng, beam_angle_rad) in enumerate(beams):
            # Beam direction in Gazebo ENU world:
            # gz_yaw is the drone heading in ENU.
            # In the LaserScan frame (sensor +X = drone forward, CCW positive),
            # beam at angle α: direction in ENU = rotate gz_yaw + α about +Z.
            world_angle = gz_yaw + beam_angle_rad
            dx = math.cos(world_angle)
            dy = math.sin(world_angle)

            marker = _make_beam_marker(
                ns        = 'lidar_beams',
                marker_id = i + 1,
                ox=gz_x, oy=gz_y, oz=gz_z,
                dx=dx, dy=dy,
                length=rng,
                color=_beam_color(rng),
                line_width=self._beam_scale,
            )
            self._gz_pub.publish(marker)

        # ── Print ranges to terminal ──────────────────────────────────────
        beam_str = '  '.join(
            f'{math.degrees(mid):+.0f}°={r:.2f}m' for r, mid in beams
        )
        print(f'\r[beams] {beam_str}   ', end='', flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize 5 front LiDAR beams in Gazebo'
    )
    parser.add_argument('--lidar_topic', default='/scan',
                        help='ROS2 LaserScan topic (default /scan)')
    parser.add_argument('--hz', type=float, default=20.0,
                        help='Publish rate in Hz (default 20)')
    parser.add_argument('--beam_scale', type=float, default=0.04,
                        help='Beam line width in Gazebo (m, default 0.04)')
    args = parser.parse_args()

    rclpy.init()
    node = LidarVizNode(args.lidar_topic, args.hz, args.beam_scale)
    print('[gz_lidar_viz] Waiting for sensors...')
    print('  GREEN  >= 3.0 m  |  YELLOW 1.5-3.0 m  |  RED < 1.5 m')
    print()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
