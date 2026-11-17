"""
Launch file for Milestone 3.

Starts four nodes: one safety node, master_control_node, one follower, and the lap
counter. Pick which safety node and follower to run, and how many laps to drive, with
launch arguments:

    ros2 launch milestone3 milestone3_launch.py safety:=lidar mode:=gap laps:=3

    safety:=lidar | camera          (default lidar)
    mode:=gap | wall | vision | straight   (default gap)
    speed:=<m/s>                    (default 1.0, only used by mode:=straight)
    laps:=<whole number>            (default 3)

Every node reads its own block from config/params.yaml.
"""

# Group B6

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE = 'milestone3'

# safety:= value -> executable name from setup.py console_scripts
SAFETY_EXECUTABLES = {
    'camera': 'vision_safety_node',
    'lidar': 'safety_node',
}

# mode:= value -> follower executable name from setup.py console_scripts
FOLLOWER_EXECUTABLES = {
    'gap': 'gap_follow_node',
    'wall': 'wall_follow_node',
    'vision': 'vision_follow_node',
    'straight': 'straight_drive_node',  # testing: drives straight at speed:= m/s
}


def launch_nodes(context: LaunchContext, *args, **kwargs) -> list:
    """
    Build the list of nodes to start, based on the safety, mode and laps launch arguments.

    This runs through an OpaqueFunction so the argument values can be read as plain strings.

    Args:
        context (LaunchContext): Launch context, used to read the argument values.
        *args: Unused, required by OpaqueFunction.
        **kwargs: Unused, required by OpaqueFunction.

    Returns:
        list: The four node actions (safety, master_control, follower, lap counter).

    Raises:
        RuntimeError: If safety or mode has a value we don't recognise.
        ValueError: If laps isn't a whole number.
    """
    safety_mode = LaunchConfiguration('safety').perform(context)
    follow_mode = LaunchConfiguration('mode').perform(context)
    laps = LaunchConfiguration('laps').perform(context)
    speed = LaunchConfiguration('speed').perform(context)

    # fail early with a clear message instead of launching the wrong thing
    if safety_mode not in SAFETY_EXECUTABLES:
        raise RuntimeError(f'safety must be one of {list(SAFETY_EXECUTABLES)}, got "{safety_mode}"')

    if follow_mode not in FOLLOWER_EXECUTABLES:
        raise RuntimeError(f'mode must be one of {list(FOLLOWER_EXECUTABLES)}, got "{follow_mode}"')

    # shared params file (same idea as Milestone 2); skipped if it doesn't exist
    params_file = os.path.join(get_package_share_directory(PACKAGE), 'config', 'params.yaml')
    params = [params_file] if os.path.exists(params_file) else []

    follower_exec = FOLLOWER_EXECUTABLES[follow_mode]
    # speed:= only means something to the straight-line test node
    follower_params = params + [{'speed_mps': float(speed)}] if follow_mode == 'straight' else params

    return [
        # safety: only one of the two runs, each with its own params.yaml block
        Node(package=PACKAGE, executable=SAFETY_EXECUTABLES[safety_mode],
             name=SAFETY_EXECUTABLES[safety_mode], parameters=params, output='screen'),
        # arbiter: forwards follower commands unless safety says unsafe
        Node(package=PACKAGE, executable='master_control_node',
             name='master_control_node',
             parameters=params + [{'follow_mode': follow_mode}], output='screen'),
        # the active follower
        Node(package=PACKAGE, executable=follower_exec,
             name=follower_exec, parameters=follower_params, output='screen'),
        # lap counter: runs in every mode, publishes /lap_count and /race_complete.
        # laps:= overrides target_laps, so the lap count can change without editing code
        Node(package=PACKAGE, executable='lap_counter_node',
             name='lap_counter_node', parameters=params + [{'target_laps': int(laps)}], output='screen'),
    ]


def generate_launch_description() -> LaunchDescription:
    """
    Declare the safety, mode and laps launch arguments and hand off to launch_nodes().

    Returns:
        LaunchDescription: What ros2 launch runs.
    """
    return LaunchDescription([
        DeclareLaunchArgument('safety', default_value='lidar',
                              description='lidar or camera safety node'),
        DeclareLaunchArgument('mode', default_value='gap',
                              description='gap, wall, vision, or straight follower'),
        DeclareLaunchArgument('speed', default_value='1.0',
                              description='speed in m/s for mode:=straight'),
        DeclareLaunchArgument('laps', default_value='3',
                              description='number of laps to drive before stopping'),
        OpaqueFunction(function=launch_nodes),
    ])
