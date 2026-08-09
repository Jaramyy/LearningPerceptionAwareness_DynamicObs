#!/usr/bin/env python3
"""Publish RViz markers for drone position, heading, FoV cone, goal, and flight path.

Converts PX4 NED (VehicleLocalPosition) → ENU map frame for RViz display:
  ned_x (North) → enu_y
  ned_y (East)  → enu_x
  ned_z (Down)  → enu_z = -ned_z
  heading (NED, CW from North) → enu heading via sin/cos

Published on /viz/markers (visualization_msgs/MarkerArray):
  - drone     : blue sphere at current position + altitude text
  - heading   : orange arrow showing current yaw direction (PA metric)
  - fov       : yellow lines showing ±90 deg camera FoV cone
  - goal      : green cylinder at goal ENU position
  - path      : white line strip of visited positions

Usage:
    python3 gz_viz_markers.py --goal_east 12.0 --goal_north 0.0 --goal_alt 1.5
"""

import argparse
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from px4_msgs.msg import VehicleLocalPosition

# Camera FoV half-angle (student uses ±90 deg sectors)
FOV_HALF_DEG = 90.0
# Arrow length for heading and FoV lines (metres)
HEADING_LEN = 3.0
FOV_LEN     = 2.5


class VizMarkers(Node):

    def __init__(self, goal_enu: tuple[float, float, float]) -> None:
        super().__init__('gz_viz_markers')

        self._goal_enu = goal_enu
        self._path_points: list[Point] = []
        self._heading_trail: list[tuple[Point, float]] = []  # (pos, heading_ned)
        self._pos_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._heading_ned: float = 0.0  # radians, NED CW from North

        px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self._cb_pos, px4_qos)
        self._pub = self.create_publisher(MarkerArray, '/viz/markers', 10)

        self.create_timer(0.2, self._publish)
        self.get_logger().info(
            f'VizMarkers ready — goal ENU={goal_enu}  /viz/markers → RViz')

    # ── ROS callback ───────────────────────────────────────────────────────

    def _cb_pos(self, msg: VehicleLocalPosition) -> None:
        enu_x = msg.y    # East  = NED y
        enu_y = msg.x    # North = NED x
        enu_z = -msg.z   # Up    = -NED z
        self._pos_enu = (enu_x, enu_y, enu_z)
        self._heading_ned = float(msg.heading)  # CW from North in NED

        if enu_z > 0.3:
            pt = Point(x=enu_x, y=enu_y, z=enu_z)
            if (not self._path_points
                    or self._dist(pt, self._path_points[-1]) > 0.1):
                self._path_points.append(pt)
                self._heading_trail.append((pt, self._heading_ned))
                if len(self._path_points) > 2000:
                    self._path_points.pop(0)
                    self._heading_trail.pop(0)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _dist(a: Point, b: Point) -> float:
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

    def _heading_enu_vec(self, offset_deg: float = 0.0) -> tuple[float, float]:
        """Unit vector in ENU XY for NED heading + offset_deg."""
        # NED heading 0 = North (+y ENU), π/2 = East (+x ENU), CW positive
        angle = self._heading_ned + math.radians(offset_deg)
        return math.sin(angle), math.cos(angle)  # (enu_x, enu_y)

    def _arrow_marker(self, ns: str, mid: int,
                      dx: float, dy: float, length: float,
                      color: ColorRGBA) -> Marker:
        """ARROW marker from drone position along (dx, dy) unit vector."""
        x, y, z = self._pos_enu
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.08   # shaft diameter
        m.scale.y = 0.18   # head diameter
        m.scale.z = 0.25   # head length
        m.color = color
        m.lifetime.sec = 1
        m.points = [
            Point(x=x, y=y, z=z),
            Point(x=x + dx * length, y=y + dy * length, z=z),
        ]
        return m

    def _line_marker(self, ns: str, mid: int,
                     end_x: float, end_y: float,
                     color: ColorRGBA) -> Marker:
        """LINE_LIST marker from drone position to (end_x, end_y)."""
        x, y, z = self._pos_enu
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.04
        m.color = color
        m.lifetime.sec = 1
        m.points = [Point(x=x, y=y, z=z), Point(x=end_x, y=end_y, z=z)]
        return m

    # ── Publish ────────────────────────────────────────────────────────────

    def _publish(self) -> None:
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()
        x, y, z = self._pos_enu

        # ── Drone sphere ─────────────────────────────────────────────────
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = now
        m.ns = 'drone'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.4
        m.color = ColorRGBA(r=0.2, g=0.5, b=1.0, a=0.9)
        m.lifetime.sec = 1
        markers.markers.append(m)

        mt = Marker()
        mt.header.frame_id = 'map'
        mt.header.stamp = now
        mt.ns = 'drone'
        mt.id = 1
        mt.type = Marker.TEXT_VIEW_FACING
        mt.action = Marker.ADD
        mt.pose.position.x = x
        mt.pose.position.y = y
        mt.pose.position.z = z + 0.6
        mt.pose.orientation.w = 1.0
        mt.scale.z = 0.3
        mt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        mt.text = f'z={z:.1f}m  hdg={math.degrees(self._heading_ned):.0f}°'
        mt.lifetime.sec = 1
        markers.markers.append(mt)

        # ── Heading arrow (PA key metric) ────────────────────────────────
        # Orange arrow showing where the drone camera is pointing.
        # PA policy actively rotates this toward obstacles.
        hdx, hdy = self._heading_enu_vec(0.0)
        markers.markers.append(self._arrow_marker(
            'heading', 0, hdx, hdy, HEADING_LEN,
            ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0),
        ))

        # ── FoV cone boundaries (±90 deg from heading) ───────────────────
        # Yellow lines show the full 180-deg sensor coverage.
        fov_color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.5)
        for offset in (-FOV_HALF_DEG, FOV_HALF_DEG):
            vx, vy = self._heading_enu_vec(offset)
            mid = 1 if offset < 0 else 2
            markers.markers.append(self._line_marker(
                'fov', mid,
                x + vx * FOV_LEN, y + vy * FOV_LEN,
                fov_color,
            ))

        # FoV arc (8 segments connecting the two boundary tips)
        arc_color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.25)
        n_seg = 8
        arc_pts: list[Point] = []
        for i in range(n_seg + 1):
            frac = i / n_seg
            deg = -FOV_HALF_DEG + frac * 2 * FOV_HALF_DEG
            vx, vy = self._heading_enu_vec(deg)
            arc_pts.append(Point(x=x + vx * FOV_LEN, y=y + vy * FOV_LEN, z=z))
        arc = Marker()
        arc.header.frame_id = 'map'
        arc.header.stamp = now
        arc.ns = 'fov'
        arc.id = 0
        arc.type = Marker.LINE_STRIP
        arc.action = Marker.ADD
        arc.pose.orientation.w = 1.0
        arc.scale.x = 0.03
        arc.color = arc_color
        arc.lifetime.sec = 1
        arc.points = arc_pts
        markers.markers.append(arc)

        # ── Goal cylinder ────────────────────────────────────────────────
        gx, gy, gz = self._goal_enu
        mg = Marker()
        mg.header.frame_id = 'map'
        mg.header.stamp = now
        mg.ns = 'goal'
        mg.id = 0
        mg.type = Marker.CYLINDER
        mg.action = Marker.ADD
        mg.pose.position.x = gx
        mg.pose.position.y = gy
        mg.pose.position.z = gz / 2.0
        mg.pose.orientation.w = 1.0
        mg.scale.x = mg.scale.y = 3.0
        mg.scale.z = gz
        mg.color = ColorRGBA(r=0.0, g=1.0, b=0.2, a=0.20)
        markers.markers.append(mg)

        mgt = Marker()
        mgt.header.frame_id = 'map'
        mgt.header.stamp = now
        mgt.ns = 'goal'
        mgt.id = 1
        mgt.type = Marker.TEXT_VIEW_FACING
        mgt.action = Marker.ADD
        mgt.pose.position.x = gx
        mgt.pose.position.y = gy
        mgt.pose.position.z = gz + 0.7
        mgt.pose.orientation.w = 1.0
        mgt.scale.z = 0.5
        mgt.color = ColorRGBA(r=0.0, g=1.0, b=0.2, a=1.0)
        mgt.text = 'GOAL'
        markers.markers.append(mgt)

        # ── Heading footprint trail ──────────────────────────────────────
        # Mini orange arrows at every recorded waypoint, fading from
        # transparent (old) to solid (recent) — shows PA heading behaviour
        # over the full flight.
        if len(self._heading_trail) > 1:
            trail = Marker()
            trail.header.frame_id = 'map'
            trail.header.stamp = now
            trail.ns = 'heading'
            trail.id = 1
            trail.type = Marker.LINE_LIST
            trail.action = Marker.ADD
            trail.pose.orientation.w = 1.0
            trail.scale.x = 0.04
            n = len(self._heading_trail)
            for i, (pt, hdg) in enumerate(self._heading_trail):
                alpha = 0.15 + 0.85 * (i / (n - 1))  # 0.15 → 1.0
                vx = math.sin(hdg) * 0.7
                vy = math.cos(hdg) * 0.7
                c = ColorRGBA(r=1.0, g=0.5, b=0.0, a=alpha)
                trail.points.append(Point(x=pt.x, y=pt.y, z=pt.z))
                trail.points.append(Point(x=pt.x + vx, y=pt.y + vy, z=pt.z))
                trail.colors.append(c)
                trail.colors.append(c)
            markers.markers.append(trail)

        # ── Flight path ──────────────────────────────────────────────────
        if len(self._path_points) > 1:
            mp = Marker()
            mp.header.frame_id = 'map'
            mp.header.stamp = now
            mp.ns = 'path'
            mp.id = 0
            mp.type = Marker.LINE_STRIP
            mp.action = Marker.ADD
            mp.pose.orientation.w = 1.0
            mp.scale.x = 0.05
            mp.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.7)
            mp.points = list(self._path_points)
            markers.markers.append(mp)

        self._pub.publish(markers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--goal_east', type=float, default=12.0)
    parser.add_argument('--goal_north', type=float, default=0.0)
    parser.add_argument('--goal_alt', type=float, default=1.5)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = VizMarkers(goal_enu=(args.goal_east, args.goal_north, args.goal_alt))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
