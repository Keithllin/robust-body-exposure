"""ROS2 launch for Stretch BedPull: driver, D435i, origin, executor, optional bag."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description() -> LaunchDescription:
    stretch_core = get_package_share_directory("stretch_core")
    robe_share = get_package_share_directory("robe_stretch")
    config = os.path.join(robe_share, "config", "executor.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=config),
            DeclareLaunchArgument("canonical_frame_path", default_value=""),
            DeclareLaunchArgument("manifest_dir", default_value="/tmp/robe_stretch"),
            DeclareLaunchArgument("record_bag", default_value="true"),
            DeclareLaunchArgument("bag_dir", default_value="/tmp/robe_stretch_bag"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(stretch_core, "launch", "stretch_driver.launch.py")
                ),
                launch_arguments={"broadcast_odom_tf": "True"}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        stretch_core, "launch", "d435i_high_resolution.launch.py"
                    )
                )
            ),
            Node(
                package="robe_stretch",
                executable="bed_origin_node",
                name="bed_origin_node",
                output="screen",
                parameters=[LaunchConfiguration("config")],
            ),
            Node(
                package="robe_stretch",
                executable="action_executor",
                name="bed_pull_executor",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "canonical_frame_path": LaunchConfiguration(
                            "canonical_frame_path"
                        ),
                        "manifest_dir": LaunchConfiguration("manifest_dir"),
                    },
                ],
            ),
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "bag",
                    "record",
                    "-o",
                    LaunchConfiguration("bag_dir"),
                    "/stretch/joint_states",
                    "/tf",
                    "/tf_static",
                    "/camera/color/image_raw",
                    "/runstop",
                    "/bed_pull/_action/feedback",
                    "/bed_pull/_action/status",
                    "/robe_stretch/layout_sample",
                    "/robe_stretch/frozen_layout_tf",
                ],
                output="screen",
                condition=IfCondition(LaunchConfiguration("record_bag")),
            ),
        ]
    )
