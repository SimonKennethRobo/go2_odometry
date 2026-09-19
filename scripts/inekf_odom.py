#!/bin/env python3

from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, qos_profile_sensor_data

from nav_msgs.msg import Odometry
from unitree_go.msg import LowState
import pinocchio as pin

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import PoseStamped, TransformStamped
from rcl_interfaces.msg import ParameterDescriptor as PD
from inekf import RobotState, NoiseParams, InEKF, Kinematics
from unitree_description.path import GO2_DESCRIPTION_URDF_PATH


def skew(vector):
    """Return the matrix representing the cross product with ``vector``."""
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quaternion_to_rotation(quaternion):
    """Convert a ROS (x, y, z, w) quaternion to a rotation matrix."""
    quaternion = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("invalid quaternion")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def rotation_exp(rotation_vector):
    """SO(3) exponential map."""
    theta = np.linalg.norm(rotation_vector)
    omega = skew(rotation_vector)
    if theta < 1.0e-8:
        return np.eye(3) + omega + 0.5 * omega @ omega
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * omega
        + ((1.0 - np.cos(theta)) / theta**2) * omega @ omega
    )


def rotation_log(rotation):
    """SO(3) logarithm map, including rotations close to pi."""
    cos_theta = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if theta < 1.0e-7:
        return 0.5 * vee
    if np.pi - theta < 1.0e-5:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis /= np.linalg.norm(axis)
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * vee


def so3_left_jacobian(rotation_vector):
    """Left Jacobian used by the SE_K(3) exponential map."""
    theta = np.linalg.norm(rotation_vector)
    omega = skew(rotation_vector)
    if theta < 1.0e-8:
        return np.eye(3) + 0.5 * omega + (1.0 / 6.0) * omega @ omega
    return (
        np.eye(3)
        + ((1.0 - np.cos(theta)) / theta**2) * omega
        + ((theta - np.sin(theta)) / theta**3) * omega @ omega
    )


# ==============================================================================
# Main Class
# ==============================================================================
class Inekf(Node):
    def __init__(self):
        super().__init__("inekf")

        # Ros params
        # fmt: off
        self.declare_parameters(
            namespace="",
            parameters=[
                ("base_frame", "base", PD(description="Robot base frame name (for TF)")),
                ("odom_frame", "odom", PD(description="World frame name (for TF)")),
                ("output_topic", "/go2_x5/slam/odom",
                 PD(description="Canonical odometry topic (go2_x5_interfaces kSlamOdometryTopic)")),
                ("legacy_output_topic", "/odometry/filtered",
                 PD(description="Legacy mirror of the same message; empty or equal to output_topic disables it")),
                ("publish_tf", True,
                 PD(description="Broadcast odom_frame -> base_frame; disable for all but one estimator")),
                ("robot_freq", 500.0, PD(description="Frequency at which the robot publish its state")),
                ("use_lowstate_tick", True, PD(description="Use the LowState millisecond tick for propagation dt")),
                ("max_propagation_dt", 0.02, PD(description="Maximum accepted propagation dt before fallback")),
                ("gyroscope_noise", 0.01, PD(description="Inekf covariance value")),
                ("accelerometer_noise", 0.1, PD(description="Inekf covariance value")),
                ("gyroscopeBias_noise", 0.00001, PD(description="Inekf covariance value")),
                ("accelerometerBias_noise", 0.0001, PD(description="Inekf covariance value")),
                ("contact_noise", 0.001, PD(description="Inekf covariance value")),
                ("joint_position_noise", 0.001, PD(description="Noise on joint configuration measurements to project using jacobian")),
                ("contact_velocity_noise", 0.001, PD(description="Noise on contact velocity")),
                ("mocap_enabled", True, PD(description="Fuse mocap PoseStamped measurements")),
                ("mocap_topic", "/vrpn_mocap/go2/pose", PD(description="Mocap PoseStamped input topic")),
                ("mocap_convert_axes", True, PD(description="Apply the VRPN world/body axis conversion")),
                ("mocap_timeout", 0.1, PD(description="Maximum mocap measurement age in seconds")),
                ("mocap_position_std", 0.001, PD(description="Mocap position standard deviation in metres")),
                ("mocap_orientation_std", 0.005, PD(description="Mocap orientation standard deviation in radians")),
                ("mocap_position_gate", 0.5,
                 PD(description="Reject position innovations above this distance; <= 0 disables")),
                ("mocap_orientation_gate", 0.8,
                 PD(description="Reject orientation innovations above this angle; <= 0 disables")),
                ("mocap_initialize", True, PD(description="Use a fresh mocap pose when initializing the filter")),
                ("mocap_window_enabled", True, PD(description="Reject locally inconsistent mocap samples")),
                ("mocap_window_size", 7, PD(description="Number of accepted mocap poses in the local window")),
                ("mocap_prediction_gate", 0.02, PD(description="Local position prediction gate in metres")),
                ("mocap_jump_gate", 0.03, PD(description="Single-sample position jump gate in metres")),
                ("mocap_orientation_prediction_gate", 0.08,
                 PD(description="Local orientation prediction gate in radians")),
                ("mocap_recovery_samples", 3, PD(description="Consistent samples needed after a mocap jump")),
                ("mocap_nis_enabled", True, PD(description="Enable covariance-aware pose innovation gates")),
                ("mocap_position_nis_gate", 16.27, PD(description="3-DoF position NIS threshold")),
                ("mocap_orientation_nis_gate", 16.27, PD(description="3-DoF orientation NIS threshold")),
                ("mocap_position_nis_floor_std", 0.10, PD(description="Position std floor used only for NIS")),
                ("mocap_orientation_nis_floor_std", 0.05,
                 PD(description="Orientation std floor used only for NIS")),
                ("mocap_nis_recovery_samples", 3, PD(description="Rejected NIS samples before forced recovery")),
                ("mocap_velocity_enabled", True, PD(description="Fuse velocity estimated from accepted mocap poses")),
                ("mocap_velocity_min_samples", 5, PD(description="Samples required for mocap velocity estimation")),
                ("mocap_velocity_std", 0.08, PD(description="Mocap linear velocity standard deviation in m/s")),
                ("mocap_velocity_nis_gate", 16.27, PD(description="3-DoF velocity NIS threshold")),
                ("mocap_velocity_nis_floor_std", 0.30, PD(description="Velocity std floor used only for NIS")),
                ("mocap_max_velocity", 5.0, PD(description="Reject mocap velocity estimates above this speed")),
            ],
        )
        # fmt: on

        self.base_frame = self.get_parameter("base_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.publish_tf = self.get_parameter("publish_tf").value
        self.dt = 1.0 / self.get_parameter("robot_freq").value
        self.use_lowstate_tick = self.get_parameter("use_lowstate_tick").value
        self.max_propagation_dt = self.get_parameter("max_propagation_dt").value
        self.last_lowstate_tick = None
        if self.dt <= 0.0 or self.max_propagation_dt <= 0.0:
            raise ValueError("robot_freq and max_propagation_dt must be positive")
        self.pause = True  # By default filter is paused and wait for the first feet contact to start

        self.mocap_enabled = self.get_parameter("mocap_enabled").value
        self.mocap_convert_axes = self.get_parameter("mocap_convert_axes").value
        self.mocap_timeout = self.get_parameter("mocap_timeout").value
        self.mocap_position_std = self.get_parameter("mocap_position_std").value
        self.mocap_orientation_std = self.get_parameter("mocap_orientation_std").value
        self.mocap_position_gate = self.get_parameter("mocap_position_gate").value
        self.mocap_orientation_gate = self.get_parameter("mocap_orientation_gate").value
        self.mocap_initialize = self.get_parameter("mocap_initialize").value
        self.mocap_window_enabled = self.get_parameter("mocap_window_enabled").value
        self.mocap_window_size = self.get_parameter("mocap_window_size").value
        self.mocap_prediction_gate = self.get_parameter("mocap_prediction_gate").value
        self.mocap_jump_gate = self.get_parameter("mocap_jump_gate").value
        self.mocap_orientation_prediction_gate = self.get_parameter(
            "mocap_orientation_prediction_gate"
        ).value
        self.mocap_recovery_samples = self.get_parameter("mocap_recovery_samples").value
        self.mocap_nis_enabled = self.get_parameter("mocap_nis_enabled").value
        self.mocap_position_nis_gate = self.get_parameter("mocap_position_nis_gate").value
        self.mocap_orientation_nis_gate = self.get_parameter("mocap_orientation_nis_gate").value
        self.mocap_position_nis_floor_std = self.get_parameter(
            "mocap_position_nis_floor_std"
        ).value
        self.mocap_orientation_nis_floor_std = self.get_parameter(
            "mocap_orientation_nis_floor_std"
        ).value
        self.mocap_nis_recovery_samples = self.get_parameter("mocap_nis_recovery_samples").value
        self.mocap_velocity_enabled = self.get_parameter("mocap_velocity_enabled").value
        self.mocap_velocity_min_samples = self.get_parameter("mocap_velocity_min_samples").value
        self.mocap_velocity_std = self.get_parameter("mocap_velocity_std").value
        self.mocap_velocity_nis_gate = self.get_parameter("mocap_velocity_nis_gate").value
        self.mocap_velocity_nis_floor_std = self.get_parameter(
            "mocap_velocity_nis_floor_std"
        ).value
        self.mocap_max_velocity = self.get_parameter("mocap_max_velocity").value
        if (
            self.mocap_timeout <= 0.0
            or self.mocap_position_std <= 0.0
            or self.mocap_orientation_std <= 0.0
            or self.mocap_window_size < 3
            or self.mocap_recovery_samples < 2
            or self.mocap_velocity_min_samples < 3
            or self.mocap_velocity_min_samples > self.mocap_window_size
            or self.mocap_nis_recovery_samples < 2
            or self.mocap_velocity_std <= 0.0
            or self.mocap_max_velocity <= 0.0
        ):
            raise ValueError("invalid mocap timeout, window, velocity, or noise parameter")

        # Same calibrated conversions as mocap_state_estimator. The mocap
        # source world is x=left, y=back, z=up and its rigid-body axes need an
        # additional +90 degree yaw correction to become x=front, y=left, z=up.
        self.mocap_world_change = np.diag([-1.0, -1.0, 1.0])
        self.mocap_body_change = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.latest_mocap_pose = None
        self.latest_mocap_velocity = None
        self.latest_mocap_arrival_ns = None
        self.mocap_generation = 0
        self.fused_mocap_generation = 0
        self.mocap_has_fused = False
        self.mocap_history = deque(maxlen=self.mocap_window_size)
        self.mocap_recovery_candidates = deque(maxlen=self.mocap_recovery_samples)
        self.mocap_nis_rejection_streak = 0
        self.mocap_stats = {
            "received": 0,
            "accepted": 0,
            "local_rejected": 0,
            "recoveries": 0,
            "stale": 0,
            "hard_gate_rejected": 0,
            "nis_pose_rejected": 0,
            "nis_pose_recoveries": 0,
            "nis_velocity_rejected": 0,
            "pose_fused": 0,
            "velocity_fused": 0,
        }

        # Load robot model
        # State estimation only needs the kinematic model. loadGo2() also
        # builds visual/collision geometry, which is expensive and can fail on
        # resource-constrained onboard computers.
        robot_model = pin.buildModelFromUrdf(
            GO2_DESCRIPTION_URDF_PATH, pin.JointModelFreeFlyer()
        )
        self.robot = pin.RobotWrapper(robot_model)
        self.foot_frame_name = [prefix + "_foot" for prefix in ["FL", "FR", "RL", "RR"]]
        self.foot_frame_id = [self.robot.model.getFrameId(frame_name) for frame_name in self.foot_frame_name]
        self.imu_frame_id = self.robot.model.getFrameId("imu")
        assert self.imu_frame_id < len(self.robot.model.frames)
        self.base_frame_id = self.robot.model.getFrameId(self.base_frame)
        assert self.base_frame_id < len(self.robot.model.frames)

        # Save rigid transform between imu (filter frame) and base (output frame)
        pin.forwardKinematics(self.robot.model, self.robot.data, pin.neutral(self.robot.model))
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        oMimu = self.robot.data.oMf[self.imu_frame_id]
        oMbase = self.robot.data.oMf[self.base_frame_id]
        self.imuMbase = oMimu.actInv(oMbase)

        # In/Out topics
        self.lowstate_subscription = self.create_subscription(
            LowState, "/lowstate", self.listener_callback, QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        )
        if self.mocap_enabled:
            self.mocap_subscription = self.create_subscription(
                PoseStamped,
                self.get_parameter("mocap_topic").value,
                self.mocap_callback,
                qos_profile_sensor_data,
            )
        output_topic = self.get_parameter("output_topic").value
        legacy_topic = self.get_parameter("legacy_output_topic").value
        self.odom_publisher = self.create_publisher(Odometry, output_topic, 1)
        self.legacy_odom_publisher = (
            self.create_publisher(Odometry, legacy_topic, 1) if legacy_topic and legacy_topic != output_topic else None
        )
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        # Invariant EKF
        gravity = np.array([0, 0, -9.81])

        initial_state = RobotState()
        initial_state.setRotation(np.eye(3))
        initial_state.setVelocity(np.zeros(3))
        initial_state.setPosition(np.zeros(3))
        initial_state.setGyroscopeBias(np.zeros(3))
        initial_state.setAccelerometerBias(np.zeros(3))

        # Initialize state covariance
        noise_params = NoiseParams()
        noise_params.setGyroscopeNoise(self.get_parameter("gyroscope_noise").value)
        noise_params.setAccelerometerNoise(self.get_parameter("accelerometer_noise").value)
        noise_params.setGyroscopeBiasNoise(self.get_parameter("gyroscopeBias_noise").value)
        noise_params.setAccelerometerBiasNoise(self.get_parameter("accelerometerBias_noise").value)
        noise_params.setContactNoise(self.get_parameter("contact_noise").value)

        self.joint_pos_noise = self.get_parameter("joint_position_noise").value
        self.contact_vel_noise = self.get_parameter("contact_velocity_noise").value

        self.filter = InEKF(initial_state, noise_params)
        self.filter.setGravity(gravity)

        if self.mocap_enabled:
            self.get_logger().info(
                "Mocap pose fusion enabled on %s" % self.get_parameter("mocap_topic").value
            )

    def listener_callback(self, msg):
        propagation_dt = self.get_propagation_dt(msg)

        # Format IMU measurements
        imu_state = np.concatenate([msg.imu_state.gyroscope, msg.imu_state.accelerometer])

        # Feet kinematic data
        contact_list, pose_list, normed_covariance_list = self.feet_transformations(msg)

        if self.pause:
            if all(contact_list):
                self.pause = False
                self.initialize_filter(msg)
                self.get_logger().info("All feet in contact with the ground: starting filter.")
            else:
                self.get_logger().info("Waiting for all feet to touch the ground to start filter.", once=True)
                return  # Skip the rest of the filter

        # Propagation step: using IMU
        self.filter.propagate(imu_state, propagation_dt)

        # TODO: use IMU quaternion for extra correction step ?

        # Correction step: using feet kinematics
        contact_pairs = []
        kinematics_list = []
        for i in range(len(self.foot_frame_name)):
            contact_pairs.append((i, contact_list[i]))

            velocity = np.zeros(3)

            kinematics = Kinematics(
                i,
                pose_list[i].translation,
                self.joint_pos_noise * normed_covariance_list[i],
                velocity,
                self.contact_vel_noise * np.eye(3),
            )
            kinematics_list.append(kinematics)

        self.filter.setContacts(contact_pairs)
        self.filter.correctKinematics(kinematics_list)

        # Consume every mocap packet at most once. Running the correction here
        # serializes it with propagation even with a multi-threaded executor.
        self.correct_mocap()

        self.publish_state(self.filter.getState(), msg.imu_state.gyroscope)

    def get_propagation_dt(self, msg):
        """Use Unitree's millisecond tick so dropped callbacks do not lose time."""
        tick = int(msg.tick)
        propagation_dt = self.dt
        if self.use_lowstate_tick and self.last_lowstate_tick is not None:
            tick_delta = (tick - self.last_lowstate_tick) & 0xFFFFFFFF
            measured_dt = tick_delta * 1.0e-3
            if 0.0 < measured_dt <= self.max_propagation_dt:
                propagation_dt = measured_dt
        self.last_lowstate_tick = tick
        return propagation_dt

    def mocap_callback(self, msg):
        self.mocap_stats["received"] += 1
        position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        quaternion = np.array(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w],
            dtype=float,
        )
        if not np.all(np.isfinite(position)):
            self.get_logger().warning("Ignoring mocap pose with a non-finite position")
            return
        try:
            rotation = quaternion_to_rotation(quaternion)
        except ValueError:
            self.get_logger().warning("Ignoring mocap pose with an invalid quaternion")
            return

        if self.mocap_convert_axes:
            position = self.mocap_world_change @ position
            rotation = self.mocap_world_change @ rotation @ self.mocap_body_change.T

        # The mocap rigid-body definition represents the robot base, whereas
        # InEKF estimates the IMU pose. Convert base pose to IMU pose here.
        world_mocap_base = pin.SE3(rotation, position)
        mocap_pose = world_mocap_base.act(self.imuMbase.inverse())
        arrival_ns = self.get_clock().now().nanoseconds
        source_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1.0e-9
        if source_time <= 0.0:
            source_time = arrival_ns * 1.0e-9

        if self.mocap_window_enabled and self.mocap_sample_is_outlier(source_time, mocap_pose):
            self.mocap_stats["local_rejected"] += 1
            self.mocap_recovery_candidates.append((source_time, mocap_pose))
            if not self.mocap_recovery_is_consistent():
                return
            self.mocap_history.clear()
            self.mocap_history.extend(self.mocap_recovery_candidates)
            self.mocap_recovery_candidates.clear()
            self.mocap_stats["recoveries"] += 1
        else:
            self.mocap_recovery_candidates.clear()
            self.mocap_history.append((source_time, mocap_pose))

        self.latest_mocap_pose = mocap_pose
        self.latest_mocap_velocity = self.estimate_mocap_velocity()
        self.latest_mocap_arrival_ns = arrival_ns
        self.mocap_generation += 1
        self.mocap_stats["accepted"] += 1

    def mocap_sample_is_outlier(self, source_time, mocap_pose):
        """Detect a pose inconsistent with both a local prediction and the last sample."""
        if len(self.mocap_history) < 3:
            return False
        last_time, last_pose = self.mocap_history[-1]
        dt = source_time - last_time
        if dt <= 0.0:
            return True
        if dt > self.mocap_timeout:
            self.mocap_history.clear()
            return False

        linear_velocities = []
        angular_velocities = []
        for (time_a, pose_a), (time_b, pose_b) in zip(
            list(self.mocap_history)[:-1], list(self.mocap_history)[1:]
        ):
            sample_dt = time_b - time_a
            if sample_dt <= 0.0 or sample_dt > self.mocap_timeout:
                continue
            linear_velocities.append((pose_b.translation - pose_a.translation) / sample_dt)
            angular_velocities.append(
                rotation_log(pose_b.rotation @ pose_a.rotation.T) / sample_dt
            )
        if not linear_velocities:
            return False

        linear_velocity = np.median(np.asarray(linear_velocities), axis=0)
        angular_velocity = np.median(np.asarray(angular_velocities), axis=0)
        predicted_position = last_pose.translation + linear_velocity * dt
        predicted_rotation = rotation_exp(angular_velocity * dt) @ last_pose.rotation
        prediction_error = np.linalg.norm(mocap_pose.translation - predicted_position)
        position_jump = np.linalg.norm(mocap_pose.translation - last_pose.translation)
        orientation_prediction_error = np.linalg.norm(
            rotation_log(mocap_pose.rotation @ predicted_rotation.T)
        )
        orientation_jump = np.linalg.norm(
            rotation_log(mocap_pose.rotation @ last_pose.rotation.T)
        )
        position_outlier = (
            prediction_error > self.mocap_prediction_gate
            and position_jump > self.mocap_jump_gate
        )
        orientation_outlier = (
            orientation_prediction_error > self.mocap_orientation_prediction_gate
            and orientation_jump > 2.0 * self.mocap_orientation_prediction_gate
        )
        return position_outlier or orientation_outlier

    def mocap_recovery_is_consistent(self):
        if len(self.mocap_recovery_candidates) < self.mocap_recovery_samples:
            return False
        candidates = list(self.mocap_recovery_candidates)
        for (time_a, pose_a), (time_b, pose_b) in zip(candidates[:-1], candidates[1:]):
            dt = time_b - time_a
            position_step = np.linalg.norm(pose_b.translation - pose_a.translation)
            orientation_step = np.linalg.norm(
                rotation_log(pose_b.rotation @ pose_a.rotation.T)
            )
            if (
                dt <= 0.0
                or dt > self.mocap_timeout
                or position_step > self.mocap_jump_gate
                or orientation_step > 2.0 * self.mocap_orientation_prediction_gate
            ):
                return False
        return True

    def estimate_mocap_velocity(self):
        """Estimate current world-frame IMU velocity with a causal quadratic fit."""
        if not self.mocap_velocity_enabled or len(self.mocap_history) < self.mocap_velocity_min_samples:
            return None
        samples = list(self.mocap_history)[-self.mocap_window_size :]
        times = np.asarray([sample[0] for sample in samples])
        times -= times[-1]
        if np.any(np.diff(times) <= 0.0) or -times[0] < 0.02:
            return None
        positions = np.asarray([sample[1].translation for sample in samples])
        design = np.column_stack([np.ones(len(times)), times, times**2])
        coefficients, _, _, _ = np.linalg.lstsq(design, positions, rcond=None)
        velocity = coefficients[1]
        if not np.all(np.isfinite(velocity)) or np.linalg.norm(velocity) > self.mocap_max_velocity:
            return None
        return velocity

    def mocap_is_fresh(self):
        if self.latest_mocap_pose is None:
            return False
        age = (self.get_clock().now().nanoseconds - self.latest_mocap_arrival_ns) * 1.0e-9
        return 0.0 <= age <= self.mocap_timeout

    def correct_mocap(self):
        if (
            not self.mocap_enabled
            or self.latest_mocap_pose is None
            or self.mocap_generation == self.fused_mocap_generation
        ):
            return

        # Mark it consumed even if stale/outlying so the same bad sample is not
        # reconsidered at every IMU tick.
        self.fused_mocap_generation = self.mocap_generation
        if not self.mocap_is_fresh():
            self.mocap_stats["stale"] += 1
            self.get_logger().warning("Ignoring stale mocap pose", throttle_duration_sec=2.0)
            return

        state = self.filter.getState()
        estimated_rotation = np.asarray(state.getRotation())
        estimated_velocity = np.asarray(state.getVelocity()).reshape(3)
        estimated_position = np.asarray(state.getPosition()).reshape(3)
        orientation_residual = rotation_log(
            self.latest_mocap_pose.rotation @ estimated_rotation.T
        )
        innovation_rotation = rotation_exp(orientation_residual)
        innovation_jacobian = so3_left_jacobian(orientation_residual)
        # Log of T_mocap * T_estimate^-1. Expressing the full innovation in
        # the same right-invariant tangent space as InEKF avoids a large-angle
        # position/orientation linearization error at startup.
        position_residual = np.linalg.solve(
            innovation_jacobian,
            self.latest_mocap_pose.translation - innovation_rotation @ estimated_position,
        )

        position_error = np.linalg.norm(
            self.latest_mocap_pose.translation - estimated_position
        )
        orientation_error = np.linalg.norm(orientation_residual)
        position_outlier = self.mocap_position_gate > 0.0 and position_error > self.mocap_position_gate
        orientation_outlier = (
            self.mocap_orientation_gate > 0.0
            and orientation_error > self.mocap_orientation_gate
        )
        if self.mocap_has_fused and (position_outlier or orientation_outlier):
            self.mocap_stats["hard_gate_rejected"] += 1
            self.get_logger().warning(
                "Rejecting mocap innovation: %.3f m, %.3f rad" % (position_error, orientation_error),
                throttle_duration_sec=2.0,
            )
            return

        covariance = np.asarray(state.getP()).copy()
        state_dimension = covariance.shape[0]
        orientation_observation = np.zeros((3, state_dimension))
        orientation_observation[:, :3] = np.eye(3)
        position_observation = np.zeros((3, state_dimension))
        position_observation[:, 6:9] = np.eye(3)

        if self.mocap_nis_enabled and self.mocap_has_fused:
            orientation_nis = self.innovation_nis(
                orientation_residual,
                orientation_observation,
                covariance,
                self.mocap_orientation_std,
                self.mocap_orientation_nis_floor_std,
            )
            position_nis = self.innovation_nis(
                position_residual,
                position_observation,
                covariance,
                self.mocap_position_std,
                self.mocap_position_nis_floor_std,
            )
            if (
                orientation_nis > self.mocap_orientation_nis_gate
                or position_nis > self.mocap_position_nis_gate
            ):
                self.mocap_stats["nis_pose_rejected"] += 1
                self.mocap_nis_rejection_streak += 1
                self.get_logger().warning(
                    "Rejecting mocap pose NIS: position %.2f, orientation %.2f"
                    % (position_nis, orientation_nis),
                    throttle_duration_sec=2.0,
                )
                if self.mocap_nis_rejection_streak < self.mocap_nis_recovery_samples:
                    return
                # Locally consistent mocap samples must eventually win over an
                # over-confident or temporarily divergent filter covariance.
                self.mocap_nis_rejection_streak = 0
                self.mocap_stats["nis_pose_recoveries"] += 1
            else:
                self.mocap_nis_rejection_streak = 0

        observation_blocks = [orientation_observation, position_observation]
        residual_blocks = [orientation_residual, position_residual]
        measurement_variances = [self.mocap_orientation_std**2] * 3
        measurement_variances += [self.mocap_position_std**2] * 3

        velocity_was_fused = False
        if self.mocap_velocity_enabled and self.latest_mocap_velocity is not None:
            velocity_residual = np.linalg.solve(
                innovation_jacobian,
                self.latest_mocap_velocity - innovation_rotation @ estimated_velocity,
            )
            velocity_observation = np.zeros((3, state_dimension))
            velocity_observation[:, 3:6] = np.eye(3)
            velocity_nis = self.innovation_nis(
                velocity_residual,
                velocity_observation,
                covariance,
                self.mocap_velocity_std,
                self.mocap_velocity_nis_floor_std,
            )
            if not self.mocap_nis_enabled or velocity_nis <= self.mocap_velocity_nis_gate:
                observation_blocks.append(velocity_observation)
                residual_blocks.append(velocity_residual)
                measurement_variances += [self.mocap_velocity_std**2] * 3
                velocity_was_fused = True
            else:
                self.mocap_stats["nis_velocity_rejected"] += 1

        observation = np.vstack(observation_blocks)
        residual = np.concatenate(residual_blocks)
        measurement_covariance = np.diag(measurement_variances)
        innovation_covariance = (
            observation @ covariance @ observation.T + measurement_covariance
        )
        try:
            kalman_gain = np.linalg.solve(innovation_covariance, observation @ covariance).T
        except np.linalg.LinAlgError:
            self.get_logger().error("Mocap correction skipped: singular innovation covariance")
            return

        delta = kalman_gain @ residual
        group_dimension = 3 * (state.dimX() - 2)
        group_delta = delta[:group_dimension]
        rotation_delta = group_delta[:3]
        delta_rotation = rotation_exp(rotation_delta)
        left_jacobian = so3_left_jacobian(rotation_delta)

        state_matrix = np.asarray(state.getX()).copy()
        correction = np.eye(state.dimX())
        correction[:3, :3] = delta_rotation
        for column in range(3, state.dimX()):
            vector_index = 3 + 3 * (column - 3)
            correction[:3, column] = left_jacobian @ group_delta[vector_index : vector_index + 3]

        identity = np.eye(state_dimension)
        ikh = identity - kalman_gain @ observation
        corrected_covariance = (
            ikh @ covariance @ ikh.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )
        corrected_covariance = 0.5 * (corrected_covariance + corrected_covariance.T)

        state.setX(correction @ state_matrix)
        state.setTheta(np.asarray(state.getTheta()).reshape(-1) + delta[group_dimension:])
        state.setP(corrected_covariance)
        self.filter.setState(state)
        self.mocap_has_fused = True
        self.mocap_stats["pose_fused"] += 1
        if velocity_was_fused:
            self.mocap_stats["velocity_fused"] += 1

    @staticmethod
    def innovation_nis(residual, observation, covariance, measurement_std, floor_std):
        gate_covariance = observation @ covariance @ observation.T
        gate_covariance += (measurement_std**2 + floor_std**2) * np.eye(3)
        try:
            return float(residual @ np.linalg.solve(gate_covariance, residual))
        except np.linalg.LinAlgError:
            return np.inf

    def log_mocap_statistics(self):
        self.get_logger().info(
            "Mocap statistics: "
            + ", ".join("%s=%d" % item for item in self.mocap_stats.items())
        )

    def get_qvf_pinocchio(state_msg):
        def unitree_to_urdf_vec(vec):
            # fmt: off
            return  [vec[3],  vec[4],  vec[5],
                     vec[0],  vec[1],  vec[2],
                     vec[9],  vec[10], vec[11],
                     vec[6],  vec[7],  vec[8],]
            # fmt: on

        # Get sensor measurement
        q_unitree = [j.q for j in state_msg.motor_state[:12]]
        v_unitree = [j.dq for j in state_msg.motor_state[:12]]
        f_unitree = state_msg.foot_force

        # Rearrange joints according to urdf
        q_pin = np.array([0] * 6 + [1] + unitree_to_urdf_vec(q_unitree))
        v_pin = np.array([0] * 6 + unitree_to_urdf_vec(v_unitree))
        f_pin = [f_unitree[i] for i in [1, 0, 3, 2]]

        return q_pin, v_pin, f_pin

    def initialize_filter(self, state_msg):
        # Unitree configuration
        q, v, _ = Inekf.get_qvf_pinocchio(state_msg)

        # Use robot IMU guess to initialize the filter
        q[3] = state_msg.imu_state.quaternion[1]
        q[4] = state_msg.imu_state.quaternion[2]
        q[5] = state_msg.imu_state.quaternion[3]
        q[6] = state_msg.imu_state.quaternion[0]

        q[3:7] /= np.linalg.norm(q[3:7])  # Normalize quaternion

        # Compute FK
        pin.forwardKinematics(self.robot.model, self.robot.data, q, v)
        pin.updateFramePlacements(self.robot.model, self.robot.data)

        # Correct initial rotation
        oMbase = self.robot.data.oMf[self.base_frame_id]
        rpy = pin.rpy.matrixToRpy(oMbase.rotation)
        rpy[2] = 0  # Set yaw to 0 for robot to always face x axis at start
        oMbase.rotation = pin.rpy.rpyToMatrix(rpy)

        # Compute average foot height
        z_avg = 0
        for i in range(4):
            oMfoot = self.robot.data.oMf[self.foot_frame_id[i]]
            z_avg += oMfoot.translation[2]
        z_avg /= 4.0

        # Correct base position
        oMbase.translation[:2] = np.zeros(2)  # centered in XY
        oMbase.translation[2] -= z_avg - 0.025  # Add foot thickness of 2.5 cm

        # Convert base pose to IMU (since filter state is in IMU frame)
        oMimu = oMbase.act(self.imuMbase.inverse())

        # Prefer the absolute mocap pose at startup when available; otherwise
        # retain the original foot/IMU based initialization.
        if self.mocap_enabled and self.mocap_initialize and self.mocap_is_fresh():
            oMimu = self.latest_mocap_pose
            self.fused_mocap_generation = self.mocap_generation
            self.mocap_has_fused = True

        # Set filter initial state
        state = self.filter.getState()
        state.setRotation(oMimu.rotation)
        state.setPosition(oMimu.translation)
        self.filter.setState(state)

    def feet_transformations(self, state_msg):
        def feet_contacts(feet_forces):
            return [bool(f >= 20) for f in feet_forces]

        # Get configuration
        q_pin, v_pin, f_pin = Inekf.get_qvf_pinocchio(state_msg)

        # Compute positions and velocities
        pin.forwardKinematics(self.robot.model, self.robot.data, q_pin, v_pin)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        pin.computeJointJacobians(self.robot.model, self.robot.data)

        # Compute foot kinematics adn jacobian
        oMimu = self.robot.data.oMf[self.imu_frame_id]
        contact_list = feet_contacts(f_pin)
        pose_list = []
        normed_covariance_list = []
        for i in range(4):
            oMfoot = self.robot.data.oMf[self.foot_frame_id[i]]
            imuMfoot = oMimu.actInv(oMfoot)
            pose_list.append(imuMfoot)

            Jc = pin.getFrameJacobian(self.robot.model, self.robot.data, self.foot_frame_id[i], pin.LOCAL)[:3, 6:]
            normed_cov_pose = Jc @ Jc.transpose()
            normed_covariance_list.append(normed_cov_pose)

        return contact_list, pose_list, normed_covariance_list

    def publish_state(self, filter_state, twist_angular_vel):
        # Get filter state
        timestamp = self.get_clock().now().to_msg()

        # Get filter state (imu frame)
        oMimu = pin.SE3(filter_state.getRotation(), filter_state.getPosition())
        v_linear_imu_world = filter_state.getX()[0:3, 3].reshape(-1)
        v_linear_imu_local = oMimu.inverse().rotation @ v_linear_imu_world
        v_imu_local = pin.Motion(linear=v_linear_imu_local, angular=twist_angular_vel)

        # Transform to base frame
        base_pose = oMimu.act(self.imuMbase)
        base_velocity = self.imuMbase.actInv(v_imu_local)

        # Convert to quaternion
        base_quaternion = pin.Quaternion(base_pose.rotation)
        base_quaternion.normalize()

        # TF2 messages
        transform_msg = TransformStamped()
        transform_msg.header.stamp = timestamp
        transform_msg.child_frame_id = self.base_frame
        transform_msg.header.frame_id = self.odom_frame

        transform_msg.transform.translation.x = float(base_pose.translation[0])
        transform_msg.transform.translation.y = float(base_pose.translation[1])
        transform_msg.transform.translation.z = float(base_pose.translation[2])

        transform_msg.transform.rotation.x = base_quaternion.x
        transform_msg.transform.rotation.y = base_quaternion.y
        transform_msg.transform.rotation.z = base_quaternion.z
        transform_msg.transform.rotation.w = base_quaternion.w

        if self.tf_broadcaster is not None:
            self.tf_broadcaster.sendTransform(transform_msg)

        # Odometry topic
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp
        odom_msg.child_frame_id = self.base_frame
        odom_msg.header.frame_id = self.odom_frame

        odom_msg.pose.pose.position.x = float(base_pose.translation[0])
        odom_msg.pose.pose.position.y = float(base_pose.translation[1])
        odom_msg.pose.pose.position.z = float(base_pose.translation[2])

        odom_msg.pose.pose.orientation.x = base_quaternion.x
        odom_msg.pose.pose.orientation.y = base_quaternion.y
        odom_msg.pose.pose.orientation.z = base_quaternion.z
        odom_msg.pose.pose.orientation.w = base_quaternion.w

        odom_msg.twist.twist.linear.x = float(base_velocity.linear[0])
        odom_msg.twist.twist.linear.y = float(base_velocity.linear[1])
        odom_msg.twist.twist.linear.z = float(base_velocity.linear[2])

        odom_msg.twist.twist.angular.x = float(base_velocity.angular[0])
        odom_msg.twist.twist.angular.y = float(base_velocity.angular[1])
        odom_msg.twist.twist.angular.z = float(base_velocity.angular[2])

        self.odom_publisher.publish(odom_msg)
        if self.legacy_odom_publisher:
            self.legacy_odom_publisher.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)

    inekf_node = Inekf()
    try:
        rclpy.spin(inekf_node)
    except KeyboardInterrupt:
        pass
    finally:
        inekf_node.log_mocap_statistics()
        inekf_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
