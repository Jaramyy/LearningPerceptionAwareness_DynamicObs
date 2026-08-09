"""ROS2 launch file for a single Gazebo IROS evaluation trial.

Starts:
  1. gz → ROS2 bridge  (depth image + color image + camera_info + point cloud)
  2. depth_to_scan      (pointcloud_to_scan.py → /scan LaserScan)
  3. student_ros2_node_icp.py  (offboard PA policy + ICP speed control)
  4. gz_eval_monitor.py        (trial monitor: success / collision / latency / ORB)

The monitor exits automatically when the trial ends; the launch system tears down
everything else via on_exit dependency.

Usage
-----
  # Generate worlds first (once):
  python3 scripts/rl_games/gz_world_gen.py --n_worlds 20 \\
      --outdir ~/PX4-Autopilot/Tools/simulation/gz/worlds/

  # Start PX4 SITL with the desired world in a separate terminal:
  cd ~/PX4-Autopilot
  PX4_GZ_WORLD=iros_exp_0 make px4_sitl gz_agi_drone_depth

  # Then launch this file:
  ros2 launch scripts/rl_games/gz_eval.launch.py \\
      method:=pa \\
      checkpoint:=/path/to/student_pa.pth \\
      seed:=0 \\
      trial_id:=0 \\
      goal_east:=12.0  goal_north:=0.0  goal_alt:=1.5 \\
      worlds_dir:=~/PX4-Autopilot/Tools/simulation/gz/worlds \\
      result_dir:=results/pa \\
      timeout:=60

Launch arguments
----------------
  method        pa | nopa | navrl | panther   (label in output JSON)
  checkpoint    absolute path to .pth file
  seed          integer  — selects world file and obstacle JSON
  trial_id      integer  — label for result file
  goal_east     float (m, PX4 NED East)
  goal_north    float (m, PX4 NED North)
  goal_alt      float (m, positive = up)
  worlds_dir    directory containing iros_exp_<seed>.{sdf,json}
  result_dir    directory where result JSON is written
  timeout       float (s)  — per-trial time limit
  vel_scale     float      — max velocity scale for student node (default 0.5)
  d_reflexive   float (m)  — ICP reflexive threshold (default 1.0)
  d_predictive  float (m)  — ICP predictive threshold (default 4.0)
  rviz          true | false  — open RViz2 (default false for headless eval)
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_RVIZ_CFG   = os.path.join(_SCRIPT_DIR, 'gz_eval_viz.rviz')

# Sensor frame stamped on every Gazebo depth camera message
GZ_SENSOR_FRAME = 'agi_drone_depth_0/camera_link/realsense_d435'


def generate_launch_description() -> LaunchDescription:

    # ── Declare launch arguments ───────────────────────────────────────────
    method_arg      = DeclareLaunchArgument('method',       default_value='pa')
    checkpoint_arg  = DeclareLaunchArgument('checkpoint',   default_value='')
    seed_arg        = DeclareLaunchArgument('seed',         default_value='0')
    trial_id_arg    = DeclareLaunchArgument('trial_id',     default_value='0')
    goal_east_arg   = DeclareLaunchArgument('goal_east',    default_value='12.0')
    goal_north_arg  = DeclareLaunchArgument('goal_north',   default_value='0.0')
    goal_alt_arg    = DeclareLaunchArgument('goal_alt',     default_value='2.0')
    worlds_dir_arg  = DeclareLaunchArgument(
        'worlds_dir',
        default_value=os.path.expanduser(
            '~/PX4-Autopilot/Tools/simulation/gz/worlds'))
    result_dir_arg  = DeclareLaunchArgument('result_dir',  default_value='results/pa')
    timeout_arg     = DeclareLaunchArgument('timeout',     default_value='60.0')
    vel_scale_arg   = DeclareLaunchArgument('vel_scale',   default_value='0.375')
    d_reflex_arg    = DeclareLaunchArgument('d_reflexive', default_value='1.0')
    d_pred_arg      = DeclareLaunchArgument('d_predictive', default_value='4.0')
    no_icp_arg      = DeclareLaunchArgument('no_icp',      default_value='false',
                                            description='Disable ICP speed scaling (true/false)')
    rviz_arg        = DeclareLaunchArgument('rviz',        default_value='false')
    trial_log_arg   = DeclareLaunchArgument('trial_log',   default_value='',
                                            description='Path for per-step CSV log (empty = disabled)')

    # ── 1. Gazebo → ROS2 bridge ──────────────────────────────────────────
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_eval_bridge',
        arguments=[
            # GPU LiDAR → LaserScan (main perception input for student policy)
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            # Color image (rgb8) — needed for ORB feature counting
            '/depth_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            # Depth image (32FC1, metres) — kept for diagnostics / depth_viz
            '/depth_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            # Camera calibration
            '/depth_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen',
    )

    # ── 3. Static TF  map → sensor frame ─────────────────────────────────
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_eval',
        arguments=['0', '0', '0', '0', '0', '0', 'map', GZ_SENSOR_FRAME],
        output='screen',
    )

    # ── 4. Student policy + optional ICP speed adaptation ────────────────
    def _make_student_node(context):
        cmd = [
            'python3',
            os.path.join(_SCRIPT_DIR, 'student_ros2_node_icp.py'),
            '--checkpoint',   LaunchConfiguration('checkpoint').perform(context),
            '--goal_north',   LaunchConfiguration('goal_north').perform(context),
            '--goal_east',    LaunchConfiguration('goal_east').perform(context),
            '--goal_alt',     LaunchConfiguration('goal_alt').perform(context),
            '--takeoff_alt',  LaunchConfiguration('goal_alt').perform(context),
            '--vel_scale',    LaunchConfiguration('vel_scale').perform(context),
            '--d_reflexive',  LaunchConfiguration('d_reflexive').perform(context),
            '--d_predictive', LaunchConfiguration('d_predictive').perform(context),
            '--lidar_topic',  '/scan',
            '--trial_log',    LaunchConfiguration('trial_log').perform(context),
        ]
        if LaunchConfiguration('no_icp').perform(context).lower() == 'true':
            cmd.append('--no_icp')
        return [ExecuteProcess(cmd=cmd, output='screen')]

    student_node = OpaqueFunction(function=_make_student_node)

    # ── 5. Trial monitor (exits when done → tears down everything) ────────
    #  obstacle JSON path = worlds_dir/iros_exp_<seed>.json
    #  result   JSON path = result_dir/<method>_trial_<trial_id>.json
    monitor_node = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(_SCRIPT_DIR, 'gz_eval_monitor.py'),
            '--method',        LaunchConfiguration('method'),
            '--trial_id',      LaunchConfiguration('trial_id'),
            '--goal_north',    LaunchConfiguration('goal_north'),
            '--goal_east',     LaunchConfiguration('goal_east'),
            '--goal_alt',      LaunchConfiguration('goal_alt'),
            '--timeout',       LaunchConfiguration('timeout'),
            # obstacle JSON — the run_gz_experiments.sh expands this before launch
            # so we pass it pre-expanded via a dummy; actual path set by shell env
            '--obstacle_json', [LaunchConfiguration('worlds_dir'),
                                '/iros_exp_',
                                LaunchConfiguration('seed'),
                                '.json'],
            '--output_json',   [LaunchConfiguration('result_dir'),
                                '/',
                                LaunchConfiguration('method'),
                                '_trial_',
                                LaunchConfiguration('trial_id'),
                                '.json'],
        ],
        output='screen',
    )

    # ── 6. ORB visualizer (optional) ──────────────────────────────────────
    orb_viz = ExecuteProcess(
        cmd=['python3', os.path.join(_SCRIPT_DIR, 'gz_orb_visualizer.py')],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen',
    )

    # ── 7. Drone / goal / path markers for RViz ───────────────────────────
    viz_markers = ExecuteProcess(
        cmd=[
            'python3', os.path.join(_SCRIPT_DIR, 'gz_viz_markers.py'),
            '--goal_east',  LaunchConfiguration('goal_east'),
            '--goal_north', LaunchConfiguration('goal_north'),
            '--goal_alt',   LaunchConfiguration('goal_alt'),
        ],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen',
    )

    # ── 8. RViz2 ──────────────────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', _RVIZ_CFG],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen',
    )

    return LaunchDescription([
        method_arg, checkpoint_arg, seed_arg, trial_id_arg,
        goal_east_arg, goal_north_arg, goal_alt_arg,
        worlds_dir_arg, result_dir_arg, timeout_arg,
        vel_scale_arg, d_reflex_arg, d_pred_arg, no_icp_arg, rviz_arg, trial_log_arg,
        gz_bridge,
        static_tf,
        student_node,
        monitor_node,
        orb_viz,
        viz_markers,
        rviz,
    ])
