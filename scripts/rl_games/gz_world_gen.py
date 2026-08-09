"""Generate textured Gazebo SDF worlds for IROS perception-aware experiment evaluation.

Each generated world contains:
  - Checkerboard floor  (2 m × 2 m tiles, alternating dark/light) → rich ORB edges
  - Cylindrical pillars with alternating horizontal stripe bands
  - Box-wall obstacles  with alternating horizontal stripe bands
  - Corridor arena  : Gazebo ENU  x ∈ [0, 14],  y ∈ [-5, 5]
  - Drone start     : ENU (0, 0, 0.1)  →  PX4 NED (0, 0) takeoff to 1.5 m
  - Navigation goal : ENU (12, 0, 1.5) →  PX4 NED --goal_north 0 --goal_east 12 --goal_alt 1.5

Alongside each .sdf file a .json obstacle map is saved (used by gz_eval_monitor.py for
collision-proximity detection).

Usage
-----
Single world:
    python3 gz_world_gen.py --seed 0 --output ~/PX4-Autopilot/Tools/simulation/gz/worlds/iros_exp_0.sdf

Multiple worlds (e.g. 10 seeds):
    python3 gz_world_gen.py --n_worlds 10 \\
        --outdir ~/PX4-Autopilot/Tools/simulation/gz/worlds/
"""

import argparse
import json
import math
import os
import random
from typing import Any

# ── Arena / obstacle constants ────────────────────────────────────────────────

ARENA_X_MIN, ARENA_X_MAX = -8.0, 17.0   # Gazebo ENU x (East direction)
ARENA_Y_MIN, ARENA_Y_MAX = -8.0, 8.0    # Gazebo ENU y (North direction)
DRONE_START = (1.0, 1.0)                 # ENU (x, y)
GOAL_ENU    = (12.0, 0.0)               # ENU (x, y)

PILLAR_RADIUS   = 0.30
PILLAR_HEIGHT   = 4.0
WALL_THICKNESS  = 0.40

OBSTACLE_FREE_RADIUS = 1.8   # m around start and goal
MIN_OBSTACLE_SEP     = 0.70  # m minimum gap between obstacle surfaces
MAX_PLACEMENT_TRIES  = 500

# ── Dense-mode placement zone (flight corridor only) ─────────────────────────
# Standard mode scatters obstacles across the full arena [-8,17] × [-8,8].
# Dense mode concentrates them inside the actual flight corridor so every
# obstacle is a potential collision threat for the x=0→12, y≈0 trajectory.
DENSE_N_PILLARS    = 8
DENSE_N_WALLS      = 0       # poles-only by default in dense mode
DENSE_MIN_SEP      = 0.70    # minimum clear gap between pillar surfaces (m)
DENSE_WALL_LEN_MIN = 1.2
DENSE_WALL_LEN_MAX = 2.8
DENSE_PLACE_X      = (1.5, 11.5)   # corridor start+goal clear zones excluded
DENSE_PLACE_Y      = (-4.5,  4.5)
DENSE_PILLAR_RADIUS = 0.25   # pillar radius (m)

# ── SDF boiler-plate ─────────────────────────────────────────────────────────

def _sdf_header(world_name: str) -> str:
    return f"""\
<sdf version='1.10'>
  <world name='{world_name}'>
    <physics type='ode'>
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <plugin name='gz::sim::systems::Physics'       filename='gz-sim-physics-system'/>
    <plugin name='gz::sim::systems::UserCommands'  filename='gz-sim-user-commands-system'/>
    <plugin name='gz::sim::systems::SceneBroadcaster' filename='gz-sim-scene-broadcaster-system'/>
    <plugin name='gz::sim::systems::Contact'       filename='gz-sim-contact-system'/>
    <plugin name='gz::sim::systems::Imu'           filename='gz-sim-imu-system'/>
    <plugin name='gz::sim::systems::AirPressure'   filename='gz-sim-air-pressure-system'/>
    <plugin name='gz::sim::systems::ApplyLinkWrench' filename='gz-sim-apply-link-wrench-system'/>
    <plugin name='gz::sim::systems::NavSat'        filename='gz-sim-navsat-system'/>
    <plugin name='gz::sim::systems::Sensors'       filename='gz-sim-sensors-system'>
      <render_engine>ogre2</render_engine>
    </plugin>
    <gui fullscreen='false'>
      <plugin name='3D View' filename='MinimalScene'>
        <gz-gui>
          <title>3D View</title>
          <property type='bool'   key='showTitleBar'>false</property>
          <property type='string' key='state'>docked</property>
        </gz-gui>
        <engine>ogre2</engine>
        <scene>scene</scene>
        <ambient_light>0.4 0.4 0.4</ambient_light>
        <background_color>0.8 0.8 0.8</background_color>
        <camera_pose>-4 0 5 0 0.45 0</camera_pose>
        <camera_clip><near>0.25</near><far>25000</far></camera_clip>
      </plugin>
      <plugin name='Entity context menu'        filename='EntityContextMenuPlugin'>
        <gz-gui><property key='state' type='string'>floating</property>
                <property key='width' type='double'>5</property>
                <property key='height' type='double'>5</property>
                <property key='showTitleBar' type='bool'>false</property></gz-gui>
      </plugin>
      <plugin name='Scene Manager'              filename='GzSceneManager'>
        <gz-gui><property key='resizable' type='bool'>false</property>
                <property key='width' type='double'>5</property>
                <property key='height' type='double'>5</property>
                <property key='state' type='string'>floating</property>
                <property key='showTitleBar' type='bool'>false</property></gz-gui>
      </plugin>
      <plugin name='Interactive view control'   filename='InteractiveViewControl'>
        <gz-gui><property key='resizable' type='bool'>false</property>
                <property key='width' type='double'>5</property>
                <property key='height' type='double'>5</property>
                <property key='state' type='string'>floating</property>
                <property key='showTitleBar' type='bool'>false</property></gz-gui>
      </plugin>
      <plugin name='World control' filename='WorldControl'>
        <gz-gui>
          <title>World control</title>
          <property type='bool'   key='showTitleBar'>0</property>
          <property type='bool'   key='resizable'>0</property>
          <property type='double' key='height'>72</property>
          <property type='double' key='width'>121</property>
          <property type='double' key='z'>1</property>
          <property type='string' key='state'>floating</property>
          <anchors target='3D View'>
            <line own='left'   target='left'/>
            <line own='bottom' target='bottom'/>
          </anchors>
        </gz-gui>
        <play_pause>1</play_pause>
        <step>1</step>
        <start_paused>0</start_paused>
      </plugin>
      <plugin name='World stats' filename='WorldStats'>
        <gz-gui>
          <title>World stats</title>
          <property type='bool'   key='showTitleBar'>0</property>
          <property type='bool'   key='resizable'>0</property>
          <property type='double' key='height'>110</property>
          <property type='double' key='width'>290</property>
          <property type='double' key='z'>1</property>
          <property type='string' key='state'>floating</property>
          <anchors target='3D View'>
            <line own='right'  target='right'/>
            <line own='bottom' target='bottom'/>
          </anchors>
        </gz-gui>
        <sim_time>1</sim_time>
        <real_time>1</real_time>
        <real_time_factor>1</real_time_factor>
        <iterations>1</iterations>
      </plugin>
      <plugin name='Entity tree' filename='EntityTree'/>
    </gui>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type='adiabatic'/>
    <scene>
      <grid>false</grid>
      <ambient>0.5 0.5 0.5 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.397971057728974</latitude_deg>
      <longitude_deg>8.546163739800146</longitude_deg>
      <elevation>0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>
"""

SDF_FOOTER = """\
  </world>
</sdf>
"""

SDF_DRONE = """\
    <include>
      <uri>file:///home/jaramy/PX4-Autopilot/Tools/simulation/gz/models/agi_drone_depth</uri>
      <name>agi_drone_depth_0</name>
      <pose>0 0 0.1 0 0 0</pose>
    </include>
"""

SDF_LIGHT = """\
    <light name='sunUTC' type='directional'>
      <pose>0 0 500 0 0 0</pose>
      <cast_shadows>true</cast_shadows>
      <intensity>1</intensity>
      <direction>0.001 0.625 -0.78</direction>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.27 0.27 0.27 1</specular>
      <attenuation><range>2000</range><linear>0</linear><constant>1</constant><quadratic>0</quadratic></attenuation>
    </light>
    <light name='fill_light' type='directional'>
      <pose>0 0 500 0 0 0</pose>
      <cast_shadows>false</cast_shadows>
      <intensity>0.4</intensity>
      <direction>-0.5 -0.3 -0.8</direction>
      <diffuse>0.9 0.9 1.0 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation><range>2000</range><linear>0</linear><constant>1</constant><quadratic>0</quadratic></attenuation>
    </light>
"""

# ── SDF fragment builders ─────────────────────────────────────────────────────

def _mat(r: float, g: float, b: float, spec: float = 0.15) -> str:
    s = spec
    return (f'<material>'
            f'<ambient>{r:.3f} {g:.3f} {b:.3f} 1</ambient>'
            f'<diffuse>{r:.3f} {g:.3f} {b:.3f} 1</diffuse>'
            f'<specular>{s:.2f} {s:.2f} {s:.2f} 1</specular>'
            f'</material>')


def _base_ground_sdf() -> str:
    """Large dark-grey collision plane + invisible ground."""
    return """\
    <model name='ground_plane'>
      <static>true</static>
      <link name='link'>
        <collision name='collision'>
          <geometry><plane><normal>0 0 1</normal><size>1 1</size></plane></geometry>
        </collision>
        <visual name='visual'>
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material>
            <ambient>0.22 0.22 0.22 1</ambient>
            <diffuse>0.22 0.22 0.22 1</diffuse>
            <specular>0.05 0.05 0.05 1</specular>
          </material>
        </visual>
      </link>
    </model>
"""


def _checkerboard_tiles_sdf(tile_size: float = 2.0, tile_h: float = 0.03) -> str:
    """Generate 2 m × 2 m tiles over the arena for ORB-detectable floor edges."""
    dark  = (0.20, 0.20, 0.20)
    light = (0.82, 0.82, 0.82)

    x_start = ARENA_X_MIN - tile_size
    y_start = ARENA_Y_MIN - tile_size
    x_end   = ARENA_X_MAX + tile_size
    y_end   = ARENA_Y_MAX + tile_size

    pieces = []
    row, col = 0, 0
    y = y_start
    while y < y_end:
        col = 0
        x = x_start
        while x < x_end:
            is_light = ((row + col) % 2 == 0)
            r, g, b = light if is_light else dark
            cx = x + tile_size / 2
            cy = y + tile_size / 2
            name = f'floor_tile_{row}_{col}'
            pieces.append(
                f'    <model name="{name}">\n'
                f'      <static>true</static>\n'
                f'      <pose>{cx:.2f} {cy:.2f} {tile_h/2:.4f} 0 0 0</pose>\n'
                f'      <link name="link">\n'
                f'        <visual name="vis">\n'
                f'          <geometry><box><size>{tile_size} {tile_size} {tile_h:.3f}</size></box></geometry>\n'
                f'          {_mat(r, g, b, 0.05)}\n'
                f'        </visual>\n'
                f'      </link>\n'
                f'    </model>\n'
            )
            x += tile_size
            col += 1
        y += tile_size
        row += 1
    return ''.join(pieces)


def _goal_marker_sdf(x: float, y: float) -> str:
    """Bright green ring on the floor marking the goal position."""
    return (
        f'    <model name="goal_marker">\n'
        f'      <static>true</static>\n'
        f'      <pose>{x:.2f} {y:.2f} 0.02 0 0 0</pose>\n'
        f'      <link name="link">\n'
        f'        <visual name="ring">\n'
        f'          <geometry><cylinder><radius>0.8</radius><length>0.04</length></cylinder></geometry>\n'
        f'          <material>\n'
        f'            <ambient>0.0 0.9 0.1 1</ambient>\n'
        f'            <diffuse>0.0 0.9 0.1 1</diffuse>\n'
        f'            <specular>0.1 0.5 0.1 1</specular>\n'
        f'          </material>\n'
        f'        </visual>\n'
        f'        <visual name="inner">\n'
        f'          <geometry><cylinder><radius>0.5</radius><length>0.04</length></cylinder></geometry>\n'
        f'          <material>\n'
        f'            <ambient>0.22 0.22 0.22 1</ambient>\n'
        f'            <diffuse>0.22 0.22 0.22 1</diffuse>\n'
        f'          </material>\n'
        f'        </visual>\n'
        f'      </link>\n'
        f'    </model>\n'
    )


def _pillar_sdf(name: str, x: float, y: float,
                radius: float = PILLAR_RADIUS,
                height: float = PILLAR_HEIGHT,
                n_bands: int = 6) -> str:
    """Cylinder with alternating dark / light horizontal bands."""
    cz = height / 2.0
    band_h = height / n_bands

    collision = (
        f'        <collision name="col">\n'
        f'          <geometry><cylinder><radius>{radius}</radius><length>{height}</length></cylinder></geometry>\n'
        f'          <surface><friction><ode/></friction><bounce/><contact/></surface>\n'
        f'        </collision>\n'
    )

    visuals = []
    for i in range(n_bands):
        bz = -height / 2 + (i + 0.5) * band_h
        is_light = (i % 2 == 0)
        r, g, b = (0.93, 0.93, 0.93) if is_light else (0.12, 0.12, 0.12)
        gap = 0.015
        bh = band_h - gap
        # Slightly wider radius so bands are visible at grazing angles
        br = radius + 0.008
        visuals.append(
            f'        <visual name="band_{i}">\n'
            f'          <pose>0 0 {bz:.4f} 0 0 0</pose>\n'
            f'          <geometry><cylinder><radius>{br:.4f}</radius><length>{bh:.4f}</length></cylinder></geometry>\n'
            f'          {_mat(r, g, b, 0.2)}\n'
            f'        </visual>\n'
        )

    body = (
        f'    <model name="{name}">\n'
        f'      <static>true</static>\n'
        f'      <pose>{x:.3f} {y:.3f} {cz:.3f} 0 0 0</pose>\n'
        f'      <link name="link">\n'
        f'{collision}'
        + ''.join(visuals) +
        '        <inertial><mass>1</mass>'
        '<inertia><ixx>1</ixx><ixy>0</ixy><ixz>0</ixz><iyy>1</iyy><iyz>0</iyz><izz>1</izz></inertia>'
        '</inertial>\n'
        '      </link>\n'
        '    </model>\n'
    )
    return body


def _wall_sdf(name: str, x: float, y: float, yaw: float,
              length: float, height: float = PILLAR_HEIGHT,
              thickness: float = WALL_THICKNESS,
              n_bands: int = 5) -> str:
    """Box wall with alternating horizontal stripe bands.

    yaw rotates the wall around Z (ENU frame).  The long axis is X before
    rotation so yaw=0 → East-facing wall, yaw=π/2 → North-facing wall.
    """
    cz = height / 2.0
    band_h = height / n_bands
    eps = 0.012   # visual slightly thicker than collision

    collision = (
        f'        <collision name="col">\n'
        f'          <geometry><box><size>{length:.3f} {thickness:.3f} {height:.3f}</size></box></geometry>\n'
        f'          <surface><friction><ode/></friction><bounce/><contact/></surface>\n'
        f'        </collision>\n'
    )

    visuals = []
    for i in range(n_bands):
        bz = -height / 2 + (i + 0.5) * band_h
        is_light = (i % 2 == 0)
        r, g, b = (0.90, 0.90, 0.90) if is_light else (0.14, 0.14, 0.14)
        gap = 0.015
        bh = band_h - gap
        visuals.append(
            f'        <visual name="band_{i}">\n'
            f'          <pose>0 0 {bz:.4f} 0 0 0</pose>\n'
            f'          <geometry><box>'
            f'<size>{length:.3f} {thickness + eps:.3f} {bh:.4f}</size>'
            f'</box></geometry>\n'
            f'          {_mat(r, g, b, 0.2)}\n'
            f'        </visual>\n'
        )

    body = (
        f'    <model name="{name}">\n'
        f'      <static>true</static>\n'
        f'      <pose>{x:.3f} {y:.3f} {cz:.3f} 0 0 {yaw:.4f}</pose>\n'
        f'      <link name="link">\n'
        f'{collision}'
        + ''.join(visuals) +
        '        <inertial><mass>1</mass>'
        '<inertia><ixx>1</ixx><ixy>0</ixy><ixz>0</ixz><iyy>1</iyy><iyz>0</iyz><izz>1</izz></inertia>'
        '</inertial>\n'
        '      </link>\n'
        '    </model>\n'
    )
    return body


def _arena_boundary_sdf() -> str:
    """Four outer walls enclosing the arena (no stripes — boundary-only)."""
    walls = []
    bh, bth = 4.0, 0.5
    # Arena X: -8 to 17 (centre 4.5), Y: -8 to 8 (centre 0)
    params = [
        # (name,             x      y       yaw           length)
        ('boundary_east',   17.25,  0.0,   math.pi / 2,  16.5),
        ('boundary_west',   -8.25,  0.0,   math.pi / 2,  16.5),
        ('boundary_north',  4.5,    8.25,  0.0,          26.0),
        ('boundary_south',  4.5,   -8.25,  0.0,          26.0),
    ]
    for (nm, x, y, yaw, length) in params:
        cz = bh / 2
        walls.append(
            f'    <model name="{nm}">\n'
            f'      <static>true</static>\n'
            f'      <pose>{x:.2f} {y:.2f} {cz:.2f} 0 0 {yaw:.4f}</pose>\n'
            f'      <link name="link">\n'
            f'        <collision name="col"><geometry><box>'
            f'<size>{length:.2f} {bth:.2f} {bh:.2f}</size>'
            f'</box></geometry></collision>\n'
            f'        <visual name="vis"><geometry><box>'
            f'<size>{length:.2f} {bth:.2f} {bh:.2f}</size>'
            f'</box></geometry>'
            f'{_mat(0.35, 0.35, 0.38, 0.1)}</visual>\n'
            f'        <inertial><mass>1</mass>'
            f'<inertia><ixx>1</ixx><ixy>0</ixy><ixz>0</ixz><iyy>1</iyy><iyz>0</iyz><izz>1</izz></inertia>'
            f'</inertial>\n'
            f'      </link>\n'
            f'    </model>\n'
        )
    return ''.join(walls)


# ── Obstacle placement ────────────────────────────────────────────────────────

def _dist2d(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _generate_obstacles(
    seed: int,
    n_pillars: int = 8,
    n_walls: int = 3,
    min_sep: float = MIN_OBSTACLE_SEP,
    pillar_radius: float = PILLAR_RADIUS,
    wall_len_range: tuple = (3.0, 4.5),
    place_x: tuple = (ARENA_X_MIN + 1.5, ARENA_X_MAX - 2.0),
    place_y: tuple = (ARENA_Y_MIN + 1.0, ARENA_Y_MAX - 1.0),
) -> list[dict[str, Any]]:
    """Return list of obstacle dicts placed without overlapping start/goal zones.

    min_sep is the minimum *clear gap* between obstacle surfaces (not centre-to-centre).
    The clearance passed to _blocked accounts for both the new obstacle's radius and
    the existing obstacle's radius so the gap formula is correct for any radius.
    """
    rng = random.Random(seed)
    obstacles: list[dict[str, Any]] = []

    # Free-zone radius scales with pillar size so start/goal always have clear space
    free_r = max(OBSTACLE_FREE_RADIUS, pillar_radius + 1.5)

    def _blocked(x: float, y: float, new_r: float) -> bool:
        if _dist2d(x, y, *DRONE_START) < free_r:
            return True
        if _dist2d(x, y, *GOAL_ENU) < free_r:
            return True
        for obs in obstacles:
            obs_r = obs.get('radius', PILLAR_RADIUS) if obs['type'] == 'pillar' \
                else obs.get('length', 1.0) / 2
            # minimum centre-to-centre = new_r + obs_r + min_sep (clear gap)
            if _dist2d(x, y, obs['x'], obs['y']) < new_r + obs_r + min_sep:
                return True
        return False

    for i in range(n_pillars):
        for _ in range(MAX_PLACEMENT_TRIES):
            x = rng.uniform(*place_x)
            y = rng.uniform(*place_y)
            if not _blocked(x, y, pillar_radius):
                obstacles.append({'type': 'pillar', 'x': x, 'y': y,
                                   'radius': pillar_radius, 'height': PILLAR_HEIGHT})
                break

    for i in range(n_walls):
        for _ in range(MAX_PLACEMENT_TRIES):
            x    = rng.uniform(*place_x)
            y    = rng.uniform(*place_y)
            yaw  = rng.uniform(0.0, math.pi)
            wlen = rng.uniform(*wall_len_range)
            if not _blocked(x, y, wlen / 2):
                obstacles.append({'type': 'wall', 'x': x, 'y': y,
                                   'yaw': yaw, 'length': wlen,
                                   'height': PILLAR_HEIGHT, 'thickness': WALL_THICKNESS})
                break

    return obstacles


# ── World assembly ────────────────────────────────────────────────────────────

def build_world(seed: int, n_pillars: int = 8, n_walls: int = 3,
                dense: bool = False,
                pillar_radius: float = PILLAR_RADIUS,
                world_name: str = 'iros_exp') -> tuple[str, list[dict]]:
    """Return (sdf_string, obstacle_list)."""
    if dense:
        obstacles = _generate_obstacles(
            seed, n_pillars, n_walls,
            min_sep=DENSE_MIN_SEP,
            pillar_radius=pillar_radius,
            wall_len_range=(DENSE_WALL_LEN_MIN, DENSE_WALL_LEN_MAX),
            place_x=DENSE_PLACE_X,
            place_y=DENSE_PLACE_Y,
        )
    else:
        obstacles = _generate_obstacles(seed, n_pillars, n_walls,
                                        pillar_radius=pillar_radius)

    sdf_parts = [_sdf_header(world_name)]

    sdf_parts.append(_base_ground_sdf())
    sdf_parts.append(_checkerboard_tiles_sdf())
    sdf_parts.append(_arena_boundary_sdf())
    sdf_parts.append(_goal_marker_sdf(*GOAL_ENU))

    for i, obs in enumerate(obstacles):
        if obs['type'] == 'pillar':
            sdf_parts.append(_pillar_sdf(f'pillar_{i}', obs['x'], obs['y'],
                                         radius=obs.get('radius', PILLAR_RADIUS)))
        else:
            sdf_parts.append(_wall_sdf(
                f'wall_{i}', obs['x'], obs['y'], obs['yaw'],
                obs['length'], obs['height'], obs['thickness'],
            ))

    # Drone is NOT included here — PX4's gz_bridge spawns it via the entity
    # creation service.  Including it in the SDF AND having PX4 try to spawn
    # it again causes a service-call conflict and a timeout error.
    sdf_parts.append(SDF_LIGHT)
    sdf_parts.append(SDF_FOOTER)

    return ''.join(sdf_parts), obstacles


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed',       type=int, default=0,
                        help='World seed (default 0)')
    parser.add_argument('--n_pillars',  type=int, default=None,
                        help='Number of cylindrical pillar obstacles '
                             f'(default 8 standard, {DENSE_N_PILLARS} dense)')
    parser.add_argument('--n_walls',    type=int, default=None,
                        help='Number of box-wall obstacles '
                             f'(default 3 standard, {DENSE_N_WALLS} dense)')
    parser.add_argument('--pillar_radius', type=float, default=None,
                        help=f'Pillar radius in metres '
                             f'(default {PILLAR_RADIUS} standard, {DENSE_PILLAR_RADIUS} dense)')
    parser.add_argument('--dense',      action='store_true',
                        help='Dense mode: pack obstacles into the flight corridor '
                             f'x∈{DENSE_PLACE_X} y∈{DENSE_PLACE_Y}, '
                             f'sep={DENSE_MIN_SEP} m, poles only')
    parser.add_argument('--output',     type=str, default=None,
                        help='Output .sdf path (single world). '
                             'JSON saved alongside with same stem.')
    parser.add_argument('--n_worlds',   type=int, default=1,
                        help='Number of worlds to generate (seeds 0..n-1)')
    parser.add_argument('--outdir',     type=str, default='.',
                        help='Output directory when generating multiple worlds')
    parser.add_argument('--prefix',     type=str, default='iros_exp',
                        help='World file name prefix (default iros_exp)')
    args = parser.parse_args()

    # Apply dense-mode defaults when --dense is set and counts are not overridden
    if args.dense:
        n_pillars     = args.n_pillars     if args.n_pillars     is not None else DENSE_N_PILLARS
        n_walls       = args.n_walls       if args.n_walls       is not None else DENSE_N_WALLS
        pillar_radius = args.pillar_radius if args.pillar_radius is not None else DENSE_PILLAR_RADIUS
    else:
        n_pillars     = args.n_pillars     if args.n_pillars     is not None else 8
        n_walls       = args.n_walls       if args.n_walls       is not None else 3
        pillar_radius = args.pillar_radius if args.pillar_radius is not None else PILLAR_RADIUS

    meta = {'dense': args.dense, 'pillar_radius': pillar_radius,
            'n_pillars': n_pillars, 'n_walls': n_walls, 'goal_enu': list(GOAL_ENU)}

    if args.output is not None and args.n_worlds == 1:
        stem_name = os.path.splitext(os.path.basename(args.output))[0]
        sdf, obstacles = build_world(args.seed, n_pillars, n_walls,
                                     dense=args.dense, pillar_radius=pillar_radius,
                                     world_name=stem_name)
        out_path  = args.output
        json_path = os.path.splitext(out_path)[0] + '.json'
        with open(out_path,  'w') as f:
            f.write(sdf)
        with open(json_path, 'w') as f:
            json.dump({**meta, 'seed': args.seed, 'obstacles': obstacles}, f, indent=2)
        print(f'Wrote {out_path}')
        print(f'      {len(obstacles)} obstacles  '
              f'pillars={sum(1 for o in obstacles if o["type"]=="pillar")}  '
              f'walls={sum(1 for o in obstacles if o["type"]=="wall")}  '
              f'radius={pillar_radius}m  dense={args.dense}')
        print(f'Wrote {json_path}')

    else:
        os.makedirs(args.outdir, exist_ok=True)
        for seed in range(args.n_worlds):
            world_name = f'{args.prefix}_{seed}'
            sdf, obstacles = build_world(seed, n_pillars, n_walls,
                                         dense=args.dense, pillar_radius=pillar_radius,
                                         world_name=world_name)
            stem = os.path.join(args.outdir, world_name)
            with open(f'{stem}.sdf', 'w') as f:
                f.write(sdf)
            with open(f'{stem}.json', 'w') as f:
                json.dump({**meta, 'seed': seed, 'obstacles': obstacles}, f, indent=2)
            print(f'Wrote {stem}.sdf + .json  '
                  f'({len(obstacles)} obstacles, r={pillar_radius}m)')


if __name__ == '__main__':
    main()
