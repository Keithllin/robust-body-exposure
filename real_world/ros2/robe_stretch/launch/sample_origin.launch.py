"""Driver + D435i + one-shot origin sample. Does not start BedPull.

Override the pan sweep on the command line, e.g.
  pan_start:=-1.05 pan_end:=-2.59 n_stops:=2
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
            DeclareLaunchArgument("output_dir", default_value="/tmp/robe"),
            DeclareLaunchArgument("pan_start", default_value="-1.05"),
            DeclareLaunchArgument("pan_end", default_value="-2.59"),
            DeclareLaunchArgument("n_stops", default_value="2"),
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
                executable="sample_origin",
                name="sample_origin",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "canonical_frame_path": LaunchConfiguration(
                            "canonical_frame_path"
                        ),
                        "output_dir": LaunchConfiguration("output_dir"),
                        "localization.pan_start": ParameterValue(
                            LaunchConfiguration("pan_start"), value_type=float
                        ),
                        "localization.pan_end": ParameterValue(
                            LaunchConfiguration("pan_end"), value_type=float
                        ),
                        "localization.n_stops": ParameterValue(
                            LaunchConfiguration("n_stops"), value_type=int
                        ),
                    },
                ],
            ),
        ]
    )
