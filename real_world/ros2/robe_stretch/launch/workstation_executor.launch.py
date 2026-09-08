"""BedPull executor on the workstation.

Stretch must only run stretch_driver + D435i. Camera pixels stay on the
robot. Launch with no trial paths: session origin from sessions/current,
canonical_bed_frame + manifest_dir from session/active_trial.json written
by run_trial before /bed_pull. Do not pass layout_snapshot_path.
Do not launch execute_pull.launch.py here (it would start the robot driver).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from pathlib import Path


def generate_launch_description() -> LaunchDescription:
    robe_share = get_package_share_directory("robe_stretch")
    config = os.path.join(robe_share, "config", "executor.yaml")
    tune = os.path.join(robe_share, "config", "grasp_tune.yaml")
    code_dir = str(Path.home() / "robe/real_world/code")
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROBE_CODE_DIR", code_dir),
            SetEnvironmentVariable(
                "PYTHONPATH",
                code_dir + os.pathsep + os.environ.get("PYTHONPATH", ""),
            ),
            DeclareLaunchArgument("config", default_value=config),
            DeclareLaunchArgument("tune_config", default_value=tune),
            DeclareLaunchArgument("canonical_frame_path", default_value=""),
            DeclareLaunchArgument("layout_snapshot_path", default_value=""),
            DeclareLaunchArgument("manifest_dir", default_value="/tmp/robe_stretch"),
            Node(
                package="robe_stretch",
                executable="action_executor",
                name="bed_pull_executor",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    LaunchConfiguration("tune_config"),
                    {
                        "canonical_frame_path": LaunchConfiguration(
                            "canonical_frame_path"
                        ),
                        "layout_snapshot_path": LaunchConfiguration(
                            "layout_snapshot_path"
                        ),
                        "manifest_dir": LaunchConfiguration("manifest_dir"),
                    },
                ],
            ),
        ]
    )
