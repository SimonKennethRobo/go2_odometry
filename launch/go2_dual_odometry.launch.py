"""Two InEKF estimators on the same /lowstate, for Go2 + arm whole-body control.

- inekf_odom        IMU + leg kinematics only -> /go2_x5/slam/odom (canonical) and its legacy
                    mirror /odometry/filtered (frame odom_leg, no TF).
                    What the locomotion policy was trained/deployed with (rl_sar).
- inekf_odom_mocap  additionally fuses mocap  -> /go2_x5/slam/odom_mocap (frame odom, owns the
                    odom -> base TF). Drift-free world frame for the whole-body MPC and its
                    targets (ocs2_arm_controller floating base).
Both share config/inekf.yaml; only the mocap switch, frames and outputs differ.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def setup(context):
    args = context.launch_configurations
    share = get_package_share_directory("go2_odometry")
    with open(os.path.join(share, "config", "inekf.yaml")) as f:
        common = yaml.safe_load(f)["inekf_odom"]["ros__parameters"]

    def estimator(name, overrides, remappings=()):
        return Node(
            package="go2_odometry",
            executable="inekf_odom.py",
            name=name,
            output="screen",
            parameters=[{**common, **overrides}],
            remappings=list(remappings),
            # The filter runs 4x4..15x15 numpy algebra at 500 Hz: BLAS worker threads only
            # add spin-wait overhead (several cores per estimator on the Jetson).
            additional_env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, "launch", "go2_state_publisher.launch.py"))
        ),
        estimator("inekf_odom", {"mocap_enabled": False, "publish_tf": False, "odom_frame": "odom_leg"}),
        estimator(
            "inekf_odom_mocap",
            {
                "mocap_enabled": True,
                "publish_tf": True,
                "odom_frame": "odom",
                "mocap_topic": args["mocap_topic"],
                # Not the canonical topic: that one stays the leg-only estimate.
                "output_topic": args["fused_topic"],
                "legacy_output_topic": "",
            },
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("mocap_topic", default_value="/vrpn_mocap/go2/pose"),
            DeclareLaunchArgument("fused_topic", default_value="/go2_x5/slam/odom_mocap"),
            OpaqueFunction(function=setup),
        ]
    )
