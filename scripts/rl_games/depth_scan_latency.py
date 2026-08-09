#!/usr/bin/env python3
"""Latency benchmark: depth camera → LaserScan conversion pipeline.

Stages measured
---------------
  bridge_ms  Gazebo sim stamp → Python callback entry
             Valid when Gazebo clock is forwarded to ROS2 (--use_sim_time).
             Without sim time the raw wall-clock difference is printed instead.
  proc_ms    callback entry → LaserScan published  (numpy conversion, always valid)
  inter_ms   wall-clock gap between consecutive depth frames (≈ 1000/camera_fps)

Published topics
----------------
  /scan_depth                sensor_msgs/LaserScan     — converted scan output
  /depth_scan/latency        std_msgs/Float32MultiArray
      data[0]  bridge_ms
      data[1]  proc_ms
      data[2]  inter_ms

Stats printed every --report_every frames: mean ± std, p95, worst.

Usage
-----
  # proc_ms always valid; bridge_ms = wall-clock difference (inaccurate)
  python3 scripts/rl_games/depth_scan_latency.py

  # bridge_ms valid — requires Gazebo clock forwarded via ros_gz_bridge
  python3 scripts/rl_games/depth_scan_latency.py --use_sim_time

  # Save per-frame CSV
  python3 scripts/rl_games/depth_scan_latency.py --csv results/depth_latency.csv

  # Monitor live in rqt_plot
  ros2 run rqt_plot rqt_plot /depth_scan/latency/data[0]:data[1]:data[2]
"""

import argparse
import csv
import math
import statistics
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Float32MultiArray


class DepthScanLatency(Node):

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            'depth_scan_latency',
            parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, args.use_sim_time)
            ],
        )

        self._range_min   = args.range_min
        self._range_max   = args.range_max
        self._scan_height = args.scan_height
        self._num_bins    = args.num_bins
        self._report_n    = args.report_every

        # Camera intrinsics — filled on first CameraInfo message
        self._fx: float | None = None
        self._cx: float | None = None
        self._cy: float | None = None
        self._col_angles: np.ndarray | None = None
        self._angle_min: float | None = None
        self._angle_max: float | None = None
        self._angle_inc: float | None = None

        # Latency sample buffers (all in ms)
        self._bridge_samples: list[float] = []
        self._proc_samples:   list[float] = []
        self._inter_samples:  list[float] = []
        self._prev_recv_t: float | None   = None
        self._frame = 0

        # Optional CSV log
        self._csv_file   = None
        self._csv_writer = None
        if args.csv:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or '.', exist_ok=True)
            self._csv_file   = open(args.csv, 'w', newline='', buffering=4096)
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(['frame', 'bridge_ms', 'proc_ms', 'inter_ms'])

        be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._pub_scan = self.create_publisher(LaserScan,          args.scan_topic,        10)
        self._pub_lat  = self.create_publisher(Float32MultiArray,  '/depth_scan/latency',  10)

        self.create_subscription(CameraInfo, args.info_topic,  self._cb_info,  be)
        self.create_subscription(Image,      args.depth_topic, self._cb_depth, be)

        sim_note = ('valid — sim time active' if args.use_sim_time
                    else 'WALL-CLOCK (inaccurate) — run with --use_sim_time for true bridge latency')
        self.get_logger().info(
            f'depth_scan_latency ready\n'
            f'  {args.depth_topic} → {args.scan_topic}\n'
            f'  bridge_ms : {sim_note}\n'
            f'  report    : every {args.report_every} frames\n'
            f'  csv       : {args.csv or "disabled"}'
        )

    # ── CameraInfo: compute per-column angles once ────────────────────────────

    def _cb_info(self, msg: CameraInfo) -> None:
        if self._fx is not None:
            return
        self._fx = float(msg.k[0])   # K[0,0] = fx
        self._cx = float(msg.k[2])   # K[0,2] = cx  (principal point)
        self._cy = float(msg.k[5])   # K[1,2] = cy
        W = msg.width
        cols = np.arange(W, dtype=np.float64)
        # FLU convention: positive angle = left (+Y), negative = right (−Y)
        self._col_angles = np.arctan2(self._cx - cols, self._fx)
        self._angle_min  = float(self._col_angles[-1])   # right edge
        self._angle_max  = float(self._col_angles[0])    # left edge
        self._angle_inc  = (self._angle_max - self._angle_min) / (self._num_bins - 1)
        self.get_logger().info(
            f'CameraInfo received: fx={self._fx:.1f} cx={self._cx:.1f} cy={self._cy:.1f} W={W}  '
            f'FOV=[{math.degrees(self._angle_min):.1f}°, {math.degrees(self._angle_max):.1f}°]'
        )

    # ── Depth image callback: convert + time ─────────────────────────────────

    def _cb_depth(self, msg: Image) -> None:
        if self._col_angles is None:
            return   # waiting for CameraInfo

        # ── Stage 1: bridge latency ───────────────────────────────────────────
        t_recv  = time.perf_counter()
        now_ros = self.get_clock().now()

        msg_stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now_s       = now_ros.nanoseconds * 1e-9
        bridge_ms   = (now_s - msg_stamp_s) * 1e3   # valid with use_sim_time

        inter_ms = ((t_recv - self._prev_recv_t) * 1e3
                    if self._prev_recv_t is not None else 0.0)
        self._prev_recv_t = t_recv

        # ── Stage 2: depth → LaserScan (numpy conversion) ────────────────────
        H, W = msg.height, msg.width

        if msg.encoding == '32FC1':
            img = np.frombuffer(msg.data, dtype=np.float32).reshape(H, W).astype(np.float64)
        elif msg.encoding in ('16UC1', '16SC1'):
            img = np.frombuffer(msg.data, dtype=np.uint16).reshape(H, W).astype(np.float64) / 1000.0
        else:
            self.get_logger().warn(f'Unsupported encoding: {msg.encoding}', once=True)
            return

        cy     = int(self._cy)  # type: ignore[arg-type]
        half   = self._scan_height // 2
        row_lo = max(0, cy - half)
        row_hi = min(H - 1, cy + half)
        strip  = img[row_lo:row_hi + 1, :]

        angles_2d = np.broadcast_to(self._col_angles, strip.shape)
        valid     = (strip > 0.0) & np.isfinite(strip)
        d = strip[valid]
        a = angles_2d[valid]
        r = d / np.cos(a)   # horizontal range = depth / cos(angle)

        mask = ((r >= self._range_min) & (r <= self._range_max)
                & (a >= self._angle_min) & (a <= self._angle_max))
        r, a = r[mask], a[mask]

        idx = np.clip(
            ((a - self._angle_min) / self._angle_inc + 0.5).astype(int),
            0, self._num_bins - 1,
        )
        out = np.full(self._num_bins, np.inf)
        if idx.size > 0:
            np.minimum.at(out, idx, r)

        scan = LaserScan()
        scan.header          = msg.header
        scan.angle_min       = self._angle_min     # type: ignore[assignment]
        scan.angle_max       = self._angle_max     # type: ignore[assignment]
        scan.angle_increment = self._angle_inc     # type: ignore[assignment]
        scan.time_increment  = 0.0
        scan.scan_time       = 0.033
        scan.range_min       = self._range_min
        scan.range_max       = self._range_max
        scan.ranges          = out.tolist()
        self._pub_scan.publish(scan)

        # ── Stage timing done ─────────────────────────────────────────────────
        t_pub    = time.perf_counter()
        proc_ms  = (t_pub - t_recv) * 1e3   # callback entry → scan published

        # Publish latency breakdown
        lat_msg      = Float32MultiArray()
        lat_msg.data = [float(bridge_ms), float(proc_ms), float(inter_ms)]
        self._pub_lat.publish(lat_msg)

        # Record samples
        self._bridge_samples.append(bridge_ms)
        self._proc_samples.append(proc_ms)
        if inter_ms > 0.0:
            self._inter_samples.append(inter_ms)

        if self._csv_writer:
            self._csv_writer.writerow([
                self._frame,
                round(bridge_ms, 3), round(proc_ms, 3), round(inter_ms, 3),
            ])

        self._frame += 1
        if self._frame % self._report_n == 0:
            self._print_report()

    # ── Periodic stats report ─────────────────────────────────────────────────

    def _print_report(self) -> None:
        n = self._report_n

        def _fmt(data: list[float]) -> str:
            if len(data) < 2:
                return 'n/a'
            mean  = statistics.mean(data)
            std   = statistics.stdev(data)
            p95   = sorted(data)[max(0, int(len(data) * 0.95) - 1)]
            worst = max(data)
            return (f'mean={mean:6.2f}  std={std:5.2f}  '
                    f'p95={p95:6.2f}  worst={worst:6.2f}  ms')

        bridge_note = '' if self.get_parameter('use_sim_time').value else '  ⚠ wall-clock (inaccurate)'
        self.get_logger().info(
            f'\n{"─"*55}\n'
            f'  Depth→Scan latency  (last {n} frames, frame={self._frame})\n'
            f'  bridge : {_fmt(self._bridge_samples[-n:])}{bridge_note}\n'
            f'  proc   : {_fmt(self._proc_samples[-n:])}\n'
            f'  inter  : {_fmt(self._inter_samples[-n:])}\n'
            f'{"─"*55}'
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--depth_topic',  default='/depth_camera/depth_image',
                        help='Depth image topic (32FC1)')
    parser.add_argument('--info_topic',   default='/depth_camera/camera_info',
                        help='CameraInfo topic for intrinsics')
    parser.add_argument('--scan_topic',   default='/scan_depth',
                        help='Output LaserScan topic (default /scan_depth)')
    parser.add_argument('--scan_height',  type=int,   default=80,
                        help='Depth rows to sample (default 80)')
    parser.add_argument('--num_bins',     type=int,   default=640,
                        help='LaserScan bins (default 640)')
    parser.add_argument('--range_min',    type=float, default=0.15)
    parser.add_argument('--range_max',    type=float, default=5.0)
    parser.add_argument('--report_every', type=int,   default=50,
                        help='Print stats every N frames (default 50)')
    parser.add_argument('--use_sim_time', action='store_true',
                        help='Use ROS sim time — makes bridge_ms valid')
    parser.add_argument('--csv',          default='',
                        help='Path to write per-frame CSV (empty = disabled)')
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = DepthScanLatency(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._csv_file:
            node._csv_file.flush()
            node._csv_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
