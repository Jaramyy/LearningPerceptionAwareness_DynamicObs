#!/usr/bin/env python3
"""Publish ORB-keypoint-annotated camera image for RViz visualization.

Subscribes to /depth_camera/image (color RGB from Gazebo bridge),
draws ORB keypoints as green circles with scale, and republishes on
/orb_debug/image as bgr8 so RViz Image display can show it.

Run alongside the eval stack:
    python3 gz_orb_visualizer.py

Then in RViz: Add → By topic → /orb_debug/image → Image
"""

import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

import numpy as np
import cv2

REPORT_EVERY = 30   # print stats every N frames


class OrbVisualizer(Node):

    def __init__(self) -> None:
        super().__init__('orb_visualizer')

        self._orb = cv2.ORB_create(nfeatures=500)

        # Latency sample buffers (ms)
        self._proc_samples:  list[float] = []   # decode + detect + draw + publish
        self._inter_samples: list[float] = []   # gap between frames (~1/fps)
        self._prev_t: float | None = None
        self._frame = 0

        be_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, '/depth_camera/image', self._cb, be_qos)
        self._pub     = self.create_publisher(Image,            '/orb_debug/image',   10)
        self._pub_lat = self.create_publisher(Float32MultiArray, '/orb/latency',      10)
        self.get_logger().info(
            'ORB visualizer ready\n'
            '  → image   : /orb_debug/image\n'
            '  → latency : /orb/latency  [proc_ms, inter_ms, kp_count]'
        )

    def _cb(self, msg: Image) -> None:
        if msg.encoding not in ('rgb8', 'bgr8', 'mono8'):
            self.get_logger().warn(f'Unsupported encoding: {msg.encoding}', once=True)
            return

        # ── Timing start ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        inter_ms = (t0 - self._prev_t) * 1e3 if self._prev_t is not None else 0.0
        self._prev_t = t0

        # ── Decode ────────────────────────────────────────────────────────────
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        h, w = msg.height, msg.width

        if msg.encoding == 'mono8':
            gray  = arr.reshape(h, w)
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif msg.encoding == 'rgb8':
            color = cv2.cvtColor(arr.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        else:
            color = arr.reshape(h, w, 3).copy()

        # ── ORB detect ────────────────────────────────────────────────────────
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        kps  = self._orb.detect(gray, None)

        # ── Draw + annotate ───────────────────────────────────────────────────
        out = cv2.drawKeypoints(
            color, kps, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.putText(out, f'ORB: {len(kps)} kp', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # ── Publish annotated image ───────────────────────────────────────────
        out_msg          = Image()
        out_msg.header   = msg.header
        out_msg.height   = h
        out_msg.width    = w
        out_msg.encoding = 'bgr8'
        out_msg.step     = w * 3
        out_msg.data     = out.tobytes()
        self._pub.publish(out_msg)

        # ── Timing end ────────────────────────────────────────────────────────
        proc_ms = (time.perf_counter() - t0) * 1e3

        lat_msg      = Float32MultiArray()
        lat_msg.data = [float(proc_ms), float(inter_ms), float(len(kps))]
        self._pub_lat.publish(lat_msg)

        self._proc_samples.append(proc_ms)
        if inter_ms > 0.0:
            self._inter_samples.append(inter_ms)

        self._frame += 1
        if self._frame % REPORT_EVERY == 0:
            self._report(len(kps))


def main() -> None:
    rclpy.init()
    node = OrbVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
