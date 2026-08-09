"""Launch gz→ROS2 depth bridge + pointcloud_to_scan + RViz2.

Bridges the Gazebo RealSense D435 depth camera to ROS2 and opens RViz2
showing both the point cloud and the LaserScan on /scan.

Usage (separate terminal while PX4 + Gazebo are running):

    ros2 launch scripts/rl_games/gz_depth_to_scan.launch.py

Optional args:
    range_max:=5.0        max scan range in metres (default 5.0)
    scan_topic:=/scan     output LaserScan topic (default /scan)
    rviz:=true            open RViz2 (default true)

Gazebo topics consumed:
    /depth_camera/depth_image   32FC1 float metres
    /depth_camera/camera_info
    /depth_camera/points        PointCloudPacked → PointCloud2

ROS2 topics produced:
    /depth_camera/points   sensor_msgs/PointCloud2
    /scan                  sensor_msgs/LaserScan  (same frame as point cloud)
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_RVIZ_CFG = os.path.join(_SCRIPT_DIR, 'gz_depth_viz.rviz')

# Frame Gazebo stamps on every depth camera message
GZ_SENSOR_FRAME = 'agi_drone_depth_0/camera_link/realsense_d435'


def generate_launch_description():
    range_max_arg = DeclareLaunchArgument('range_max', default_value='5.0')
    scan_topic_arg = DeclareLaunchArgument('scan_topic', default_value='/scan')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')

    # ── 1. Gazebo → ROS2 bridge ───────────────────────────────────────────
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_depth_bridge',
        arguments=[
            '/depth_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/depth_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
        ],
        output='screen',
    )

    # ── 2. PointCloud2 → LaserScan ───────────────────────────────────────
    # Gazebo rgbd_camera publishes with X=forward in the sensor frame.
    # depthimage_to_laserscan assumes Z=forward (ROS optical) → misaligned.
    # pointcloud_to_scan.py derives the scan directly from 3-D points so scan
    # dots appear exactly on the point cloud in RViz.
    cloud_to_scan = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(_SCRIPT_DIR, 'pointcloud_to_scan.py'),
            '--depth_topic', '/depth_camera/depth_image',
            '--info_topic', '/depth_camera/camera_info',
            '--scan_topic', '/scan',
            '--scan_height', '5',
            '--range_min', '0.15',
        ],
        output='screen',
    )

    # ── 3. Static TF: map → sensor frame (camera-centric view) ───────────
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_to_sensor',
        arguments=['0', '0', '0', '0', '0', '0', 'map', GZ_SENSOR_FRAME],
        output='screen',
    )

    # ── 4. RViz2 ─────────────────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', _RVIZ_CFG],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen',
    )

    return LaunchDescription([
        range_max_arg,
        scan_topic_arg,
        rviz_arg,
        gz_bridge,
        cloud_to_scan,
        static_tf,
        rviz,
    ])
