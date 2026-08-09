"""depth_image -> LaserScan using /depth_camera/depth_image (32FC1, metres).

Avoids all point-cloud parsing issues.  Reads the raw depth pixels directly.

Angle convention for Gazebo FLU frame (X=forward, Y=left, Z=up):
  angle = atan2( (cx - u), fx )   # NOTE: (cx - u), not (u - cx)
  This gives  positive angle = left (+Y)  and  negative = right (-Y).
  Verified against the point-cloud probe:
    left-edge  u=0   -> angle = +43 deg  (+Y, left)  ✓
    centre     u=320 -> angle =   0 deg  (forward)   ✓
    right-edge u=639 -> angle = -43 deg  (-Y, right) ✓

Range from depth pixel d (= X-distance to surface):
  horizontal_range = d / cos(angle)
"""

import math
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, LaserScan


class DepthToScan(Node):
    def __init__(self, scan_height, range_min, range_max,
                 depth_topic, info_topic, scan_topic, num_bins,
                 scan_row_offset=0):
        super().__init__('depth_to_scan')

        self._scan_height = scan_height
        self._scan_row_offset = scan_row_offset  # rows above cy (positive = shift up)
        self._range_min = range_min
        self._range_max = range_max
        self._num_bins = num_bins
        self._dbg = 0

        # Filled once camera_info arrives
        self._fx = None
        self._cx = None
        self._col_angles = None   # shape (W,) – precomputed per column
        self._angle_min = None
        self._angle_max = None
        self._angle_inc = None

        be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._scan_topic = scan_topic
        self._pub = self.create_publisher(LaserScan, scan_topic, 10)
        self.create_subscription(CameraInfo, info_topic, self._cb_info, be)
        self.create_subscription(Image, depth_topic, self._cb_depth, be)

        self.get_logger().info(
            f'depth_to_scan: {depth_topic} -> {scan_topic}  '
            f'scan_height={scan_height} rows  row_offset={scan_row_offset}  '
            f'range=[{range_min},{range_max}]m'
        )

    def _cb_info(self, msg):
        if self._fx is not None:
            return  # already initialised
        self._fx = float(msg.k[0])   # K[0,0] = fx
        self._cx = float(msg.k[2])   # K[0,2] = cx
        self._cy = float(msg.k[5])   # K[1,2] = cy  (optical centre row)
        W = msg.width
        cols = np.arange(W, dtype=np.float64)
        # FLU angle: positive = left (+Y direction)
        self._col_angles = np.arctan2(self._cx - cols, self._fx)
        self._angle_min = float(self._col_angles[-1])   # right edge (negative)
        self._angle_max = float(self._col_angles[0])    # left edge  (positive)
        self._angle_inc = (self._angle_max - self._angle_min) / (self._num_bins - 1)
        self.get_logger().info(
            f'camera_info: fx={self._fx:.1f} cx={self._cx:.1f} cy={self._cy:.1f} W={W}  '
            f'angle=[{math.degrees(self._angle_min):.1f}, '
            f'{math.degrees(self._angle_max):.1f}] deg'
        )

    def _cb_depth(self, msg):
        if (self._col_angles is None or self._angle_min is None
                or self._angle_max is None or self._angle_inc is None
                or self._cy is None):
            return   # camera_info not yet received

        col_angles: np.ndarray = self._col_angles
        angle_min: float = self._angle_min
        angle_max: float = self._angle_max
        angle_inc: float = self._angle_inc
        cy: int = int(self._cy)
        cx_col: int = int(self._cx) if self._cx is not None else 320

        H, W = msg.height, msg.width

        # Parse depth image – handle both common Gazebo encodings
        if msg.encoding == '32FC1':
            img = np.frombuffer(msg.data, dtype=np.float32).reshape(H, W).astype(np.float64)
        elif msg.encoding in ('16UC1', '16SC1'):
            # uint16 millimetres -> metres
            img = np.frombuffer(msg.data, dtype=np.uint16).reshape(H, W).astype(np.float64) / 1000.0
        else:
            self.get_logger().warn(f'Unknown depth encoding: {msg.encoding}  (expected 32FC1 or 16UC1)')
            return

        # Log encoding + optical-centre depth once
        if self._dbg == 0:
            self.get_logger().info(
                f'depth encoding={msg.encoding}  '
                f'centre_pixel({cx_col},{cy})={img[cy, cx_col]:.4f} m'
            )

        # Sample scan_height rows centred on (cy - scan_row_offset).
        # A positive offset shifts the window UPWARD (smaller row indices = looks up),
        # keeping below-horizon rows out of the strip so ground is never detected.
        centre = cy - self._scan_row_offset
        half = self._scan_height // 2
        row_lo = max(0, centre - half)
        row_hi = min(H - 1, centre + half)
        strip = img[row_lo:row_hi + 1, :].astype(np.float64)   # (rows, W)

        # Broadcast column angles to the strip shape
        angles_2d = np.broadcast_to(col_angles, strip.shape)

        # Valid: positive, finite depth
        valid = (strip > 0.0) & np.isfinite(strip)
        d = strip[valid]
        a = angles_2d[valid]

        # Horizontal range = depth / cos(angle)
        r = d / np.cos(a)

        # Apply filters
        mask = (r >= self._range_min) & (r <= self._range_max) \
             & (a >= angle_min) & (a <= angle_max)
        r = r[mask]
        a = a[mask]

        # Bin into LaserScan (keep minimum per bin)
        idx = np.clip(
            ((a - angle_min) / angle_inc + 0.5).astype(int),
            0, self._num_bins - 1,
        )
        out = np.full(self._num_bins, np.inf)
        if idx.size > 0:
            np.minimum.at(out, idx, r)

        # Log every 15 frames
        self._dbg += 1
        if self._dbg % 15 == 0:
            fwd_idx = self._num_bins // 2
            fwd = out[fwd_idx]
            fwd_str = 'inf' if not np.isfinite(fwd) else f'{fwd:.3f}'
            self.get_logger().info(
                f'scan fwd(0deg)={fwd_str}m  '
                f'valid={np.isfinite(out).sum()}/{self._num_bins}  '
                f'rows={row_lo}-{row_hi}'
            )

        scan = LaserScan()
        scan.header = msg.header
        scan.angle_min = angle_min
        scan.angle_max = angle_max
        scan.angle_increment = angle_inc
        scan.time_increment = 0.0
        scan.scan_time = 0.033
        scan.range_min = self._range_min
        scan.range_max = self._range_max
        scan.ranges = out.tolist()
        self._pub.publish(scan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_height', type=int,   default=80,
                        help='Centre image rows to sample (default 80 = ~14 deg vertical)')
    parser.add_argument('--scan_row_offset', type=int, default=0,
                        help='Shift scan window UP by N rows above cy (positive = upward). '
                             'Use ~20 to exclude below-horizon rows and prevent ground detection.')
    parser.add_argument('--range_min',   type=float, default=0.15)
    parser.add_argument('--range_max',   type=float, default=5.0)
    parser.add_argument('--num_bins',    type=int,   default=640)
    parser.add_argument('--depth_topic', type=str,   default='/depth_camera/depth_image')
    parser.add_argument('--info_topic',  type=str,   default='/depth_camera/camera_info')
    parser.add_argument('--scan_topic',  type=str,   default='/scan')
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = DepthToScan(
        scan_height=args.scan_height,
        range_min=args.range_min,
        range_max=args.range_max,
        depth_topic=args.depth_topic,
        info_topic=args.info_topic,
        scan_topic=args.scan_topic,
        num_bins=args.num_bins,
        scan_row_offset=args.scan_row_offset,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
