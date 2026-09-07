#!/usr/bin/env python3
"""Plot mocap-fused InEKF output against mocap ground truth from a ROS 2 bag."""

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


AXES = ("x", "y", "z")
C_WORLD = np.diag([-1.0, -1.0, 1.0])
C_BODY = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=120.0)
    parser.add_argument("--derivative-window", type=int, default=41)
    parser.add_argument("--settling-time", type=float, default=1.0)
    parser.add_argument("--odom-topic", default="/odometry/filtered")
    parser.add_argument("--gt-pose-topic", default="/mocap_state_estimator/pose")
    parser.add_argument(
        "--gt-odom-topic",
        help="use an Odometry topic as the reference and align frames at startup",
    )
    parser.add_argument("--gt-twist-topic", default="/mocap_state_estimator/twist")
    parser.add_argument("--gt-accel-topic", default="/mocap_state_estimator/accel")
    parser.add_argument(
        "--derive-gt-from-pose",
        action="store_true",
        help="derive GT twist/accel from the raw VRPN pose instead of reading estimator topics",
    )
    parser.add_argument(
        "--alignment-window",
        type=float,
        default=1.0,
        help="seconds used to align estimate and reference odometry frames",
    )
    return parser.parse_args()


def database_paths(bag):
    if bag.is_file():
        return [bag]
    metadata = bag / "metadata.yaml"
    if metadata.exists():
        info = yaml.safe_load(metadata.read_text())["rosbag2_bagfile_information"]
        return [bag / name for name in info["relative_file_paths"]]
    return sorted(bag.glob("*.db3"))


def read_topics(bag, topics):
    records = {topic: [] for topic in topics}
    for database in database_paths(bag):
        connection = sqlite3.connect(str(database))
        topic_rows = connection.execute("SELECT id, name, type FROM topics").fetchall()
        topic_info = {topic_id: (name, type_name) for topic_id, name, type_name in topic_rows}
        selected = [topic_id for topic_id, (name, _) in topic_info.items() if name in topics]
        if selected:
            marks = ",".join("?" for _ in selected)
            query = (
                "SELECT topic_id, timestamp, data FROM messages "
                f"WHERE topic_id IN ({marks}) ORDER BY timestamp"
            )
            type_cache = {}
            for topic_id, timestamp, data in connection.execute(query, selected):
                name, type_name = topic_info[topic_id]
                message_type = type_cache.setdefault(type_name, get_message(type_name))
                records[name].append((timestamp * 1.0e-9, deserialize_message(data, message_type)))
        connection.close()
    missing = [topic for topic, values in records.items() if not values]
    if missing:
        raise RuntimeError("Missing topics: " + ", ".join(missing))
    return records


def normalized_quaternion(message):
    quaternion = np.array(
        [message.x, message.y, message.z, message.w], dtype=float
    )
    return quaternion / np.linalg.norm(quaternion)


def pose_records(records, odometry=False):
    times, positions, quaternions = [], [], []
    for stamp, message in records:
        pose = message.pose.pose if odometry else message.pose
        times.append(stamp)
        positions.append([pose.position.x, pose.position.y, pose.position.z])
        quaternions.append(normalized_quaternion(pose.orientation))
    return np.asarray(times), np.asarray(positions), np.asarray(quaternions)


def odom_twist_records(records):
    times, linear, angular = [], [], []
    for stamp, message in records:
        twist = message.twist.twist
        times.append(stamp)
        linear.append([twist.linear.x, twist.linear.y, twist.linear.z])
        angular.append([twist.angular.x, twist.angular.y, twist.angular.z])
    return np.asarray(times), np.asarray(linear), np.asarray(angular)


def stamped_vector_records(records, field):
    times, linear, angular = [], [], []
    for stamp, message in records:
        value = getattr(message, field)
        times.append(stamp)
        linear.append([value.linear.x, value.linear.y, value.linear.z])
        angular.append([value.angular.x, value.angular.y, value.angular.z])
    return np.asarray(times), np.asarray(linear), np.asarray(angular)


def unique_samples(times, *arrays):
    keep = np.r_[True, np.diff(times) > 1.0e-7]
    return (times[keep], *(array[keep] for array in arrays))


def interpolate_vectors(times, values, grid):
    return np.column_stack(
        [np.interp(grid, times, values[:, axis]) for axis in range(3)]
    )


def interpolate_rotations(times, quaternions, grid):
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    return Slerp(times, Rotation.from_quat(quaternions))(grid)


def rotation_matrices(rotation):
    """Support both old SciPy (as_dcm) and current SciPy (as_matrix)."""
    return rotation.as_matrix() if hasattr(rotation, "as_matrix") else rotation.as_dcm()


def metrics(reference, estimate, valid):
    error = estimate[valid] - reference[valid]
    axis_metrics = {}
    for axis, name in enumerate(AXES):
        axis_metrics[name] = {
            "rmse": float(np.sqrt(np.mean(error[:, axis] ** 2))),
            "mae": float(np.mean(np.abs(error[:, axis]))),
            "max_abs": float(np.max(np.abs(error[:, axis]))),
        }
    axis_metrics["vector_rmse"] = float(
        np.sqrt(np.mean(np.sum(error**2, axis=1)))
    )
    return axis_metrics


def plot_components(path, time, reference, estimate, title, unit, reference_label="GT (mocap)"):
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
    for axis, name in enumerate(AXES):
        axes[axis].plot(time, reference[:, axis], label=reference_label, linewidth=1.4)
        axes[axis].plot(time, estimate[:, axis], label="InEKF + mocap", linewidth=1.0)
        axes[axis].set_ylabel(f"{name} [{unit}]")
        axes[axis].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trajectory(path, gt_position, estimate_position, reference_label="GT (mocap)"):
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.plot(gt_position[:, 0], gt_position[:, 1], label=reference_label, linewidth=1.5)
    axis.plot(estimate_position[:, 0], estimate_position[:, 1], label="InEKF + mocap", linewidth=1.0)
    axis.scatter(gt_position[0, 0], gt_position[0, 1], marker="o", label="start")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.set_title("XY trajectory")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if args.rate <= 0.0:
        raise ValueError("--rate must be positive")
    if args.derivative_window % 2 == 0 or args.derivative_window < 5:
        raise ValueError("--derivative-window must be odd and >= 5")
    if args.alignment_window <= 0.0:
        raise ValueError("--alignment-window must be positive")
    using_gt_odom = args.gt_odom_topic is not None
    derive_gt_from_pose = args.derive_gt_from_pose or using_gt_odom
    gt_pose_topic = args.gt_odom_topic if using_gt_odom else args.gt_pose_topic
    reference_label = "Reference odometry" if using_gt_odom else "GT (mocap)"
    topics = [args.odom_topic, gt_pose_topic]
    if not derive_gt_from_pose:
        topics.extend([args.gt_twist_topic, args.gt_accel_topic])
    records = read_topics(args.bag, topics)

    odom_t, odom_position, odom_quaternion = pose_records(
        records[args.odom_topic], odometry=True
    )
    _, odom_linear_body, odom_angular_body = odom_twist_records(
        records[args.odom_topic]
    )
    odom_t, odom_position, odom_quaternion, odom_linear_body, odom_angular_body = unique_samples(
        odom_t, odom_position, odom_quaternion, odom_linear_body, odom_angular_body
    )
    gt_pose_t, gt_position, gt_quaternion = pose_records(
        records[gt_pose_topic], odometry=using_gt_odom
    )
    gt_pose_t, gt_position, gt_quaternion = unique_samples(
        gt_pose_t, gt_position, gt_quaternion
    )
    if using_gt_odom:
        gt_twist_t = gt_pose_t
        gt_accel_t = gt_pose_t
    elif args.derive_gt_from_pose:
        gt_position = np.einsum("ij,nj->ni", C_WORLD, gt_position)
        source_matrices = rotation_matrices(Rotation.from_quat(gt_quaternion))
        target_matrices = C_WORLD[None, :, :] @ source_matrices @ C_BODY.T[None, :, :]
        gt_quaternion = Rotation.from_dcm(target_matrices).as_quat()
        gt_twist_t = gt_pose_t
        gt_accel_t = gt_pose_t
    elif not using_gt_odom:
        gt_twist_t, gt_linear_velocity, gt_angular_velocity = stamped_vector_records(
            records[args.gt_twist_topic], "twist"
        )
        gt_accel_t, gt_linear_accel, gt_angular_accel = stamped_vector_records(
            records[args.gt_accel_topic], "accel"
        )

    start = max(odom_t[0], gt_pose_t[0], gt_twist_t[0], gt_accel_t[0])
    stop = min(odom_t[-1], gt_pose_t[-1], gt_twist_t[-1], gt_accel_t[-1])
    sample_count = int(np.floor((stop - start) * args.rate)) + 1
    grid = start + np.arange(sample_count) / args.rate
    display_time = grid - grid[0]

    estimate_position = interpolate_vectors(odom_t, odom_position, grid)
    estimate_rotation = interpolate_rotations(odom_t, odom_quaternion.copy(), grid)
    estimate_linear_body = interpolate_vectors(odom_t, odom_linear_body, grid)
    estimate_angular_body = interpolate_vectors(odom_t, odom_angular_body, grid)

    reference_position = interpolate_vectors(gt_pose_t, gt_position, grid)
    reference_rotation = interpolate_rotations(gt_pose_t, gt_quaternion.copy(), grid)

    alignment = None
    if using_gt_odom:
        alignment_samples = display_time <= args.alignment_window
        relative_rotation_matrices = rotation_matrices(
            reference_rotation[alignment_samples]
            * estimate_rotation[alignment_samples].inv()
        )
        mean_matrix = np.mean(relative_rotation_matrices, axis=0)
        left, _, right = np.linalg.svd(mean_matrix)
        alignment_matrix = left @ right
        if np.linalg.det(alignment_matrix) < 0.0:
            left[:, -1] *= -1.0
            alignment_matrix = left @ right
        alignment_rotation = Rotation.from_dcm(alignment_matrix)
        alignment_translation = np.mean(
            reference_position[alignment_samples]
            - alignment_rotation.apply(estimate_position[alignment_samples]),
            axis=0,
        )
        estimate_position = (
            alignment_rotation.apply(estimate_position) + alignment_translation
        )
        estimate_rotation = alignment_rotation * estimate_rotation
        alignment = {
            "method": "mean pose over startup window",
            "window_s": args.alignment_window,
            "rotation_vector_rad": alignment_rotation.as_rotvec().tolist(),
            "translation_m": alignment_translation.tolist(),
        }

    estimate_linear_velocity = estimate_rotation.apply(estimate_linear_body)
    estimate_angular_velocity = estimate_rotation.apply(estimate_angular_body)

    dt = 1.0 / args.rate
    if derive_gt_from_pose:
        reference_linear_velocity = savgol_filter(
            reference_position,
            args.derivative_window,
            3,
            deriv=1,
            delta=dt,
            axis=0,
        )
        reference_linear_accel = savgol_filter(
            reference_position,
            args.derivative_window,
            3,
            deriv=2,
            delta=dt,
            axis=0,
        )
        angular_intervals = (
            reference_rotation[1:] * reference_rotation[:-1].inv()
        ).as_rotvec() / dt
        angular_time = 0.5 * (grid[1:] + grid[:-1])
        reference_angular_velocity = interpolate_vectors(
            angular_time, angular_intervals, grid
        )
        reference_angular_velocity = savgol_filter(
            reference_angular_velocity,
            args.derivative_window,
            3,
            axis=0,
        )
        reference_angular_accel = savgol_filter(
            reference_angular_velocity,
            args.derivative_window,
            3,
            deriv=1,
            delta=dt,
            axis=0,
        )
    else:
        reference_linear_velocity = interpolate_vectors(
            gt_twist_t, gt_linear_velocity, grid
        )
        reference_angular_velocity = interpolate_vectors(
            gt_twist_t, gt_angular_velocity, grid
        )
        reference_linear_accel = interpolate_vectors(
            gt_accel_t, gt_linear_accel, grid
        )
        reference_angular_accel = interpolate_vectors(
            gt_accel_t, gt_angular_accel, grid
        )
    estimate_linear_accel = savgol_filter(
        estimate_linear_velocity,
        args.derivative_window,
        3,
        deriv=1,
        delta=dt,
        axis=0,
    )
    estimate_angular_accel = savgol_filter(
        estimate_angular_velocity,
        args.derivative_window,
        3,
        deriv=1,
        delta=dt,
        axis=0,
    )

    reference_rpy = np.unwrap(reference_rotation.as_euler("xyz"), axis=0)
    estimate_rpy = np.unwrap(estimate_rotation.as_euler("xyz"), axis=0)
    orientation_error = np.linalg.norm(
        (reference_rotation.inv() * estimate_rotation).as_rotvec(), axis=1
    )
    valid = display_time >= args.settling_time

    report = {
        "reference_topic": gt_pose_topic,
        "reference_is_ground_truth": not using_gt_odom,
        "alignment": alignment,
        "duration_s": float(display_time[-1]),
        "samples": int(len(grid)),
        "rate_hz": args.rate,
        "settling_time_excluded_s": args.settling_time,
        "position_m": metrics(reference_position, estimate_position, valid),
        "orientation_angle_rad": {
            "rmse": float(np.sqrt(np.mean(orientation_error[valid] ** 2))),
            "mae": float(np.mean(np.abs(orientation_error[valid]))),
            "max_abs": float(np.max(np.abs(orientation_error[valid]))),
        },
        "linear_velocity_world_mps": metrics(
            reference_linear_velocity, estimate_linear_velocity, valid
        ),
        "angular_velocity_world_radps": metrics(
            reference_angular_velocity, estimate_angular_velocity, valid
        ),
        "linear_acceleration_world_mps2": metrics(
            reference_linear_accel, estimate_linear_accel, valid
        ),
        "angular_acceleration_world_radps2": metrics(
            reference_angular_accel, estimate_angular_accel, valid
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_components(
        args.output_dir / "pose_position.png",
        display_time,
        reference_position,
        estimate_position,
        f"Position: fused estimate vs {reference_label}",
        "m",
        reference_label,
    )
    plot_components(
        args.output_dir / "pose_orientation_rpy.png",
        display_time,
        reference_rpy,
        estimate_rpy,
        f"Orientation: fused estimate vs {reference_label}",
        "rad",
        reference_label,
    )
    plot_trajectory(
        args.output_dir / "trajectory_xy.png",
        reference_position,
        estimate_position,
        reference_label,
    )
    plot_components(
        args.output_dir / "twist_linear_world.png",
        display_time,
        reference_linear_velocity,
        estimate_linear_velocity,
        "World-frame linear velocity",
        "m/s",
        reference_label,
    )
    plot_components(
        args.output_dir / "twist_angular_world.png",
        display_time,
        reference_angular_velocity,
        estimate_angular_velocity,
        "World-frame angular velocity",
        "rad/s",
        reference_label,
    )
    plot_components(
        args.output_dir / "accel_linear_world.png",
        display_time,
        reference_linear_accel,
        estimate_linear_accel,
        "World-frame linear acceleration",
        "m/s^2",
        reference_label,
    )
    plot_components(
        args.output_dir / "accel_angular_world.png",
        display_time,
        reference_angular_accel,
        estimate_angular_accel,
        "World-frame angular acceleration",
        "rad/s^2",
        reference_label,
    )

    columns = ["time_s"]
    arrays = {
        "gt_position": reference_position,
        "fused_position": estimate_position,
        "gt_rpy": reference_rpy,
        "fused_rpy": estimate_rpy,
        "gt_linear_velocity_world": reference_linear_velocity,
        "fused_linear_velocity_world": estimate_linear_velocity,
        "gt_angular_velocity_world": reference_angular_velocity,
        "fused_angular_velocity_world": estimate_angular_velocity,
        "gt_linear_acceleration_world": reference_linear_accel,
        "fused_linear_acceleration_world": estimate_linear_accel,
        "gt_angular_acceleration_world": reference_angular_accel,
        "fused_angular_acceleration_world": estimate_angular_accel,
    }
    for prefix in arrays:
        columns.extend(f"{prefix}_{axis}" for axis in AXES)
    with (args.output_dir / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for index, time in enumerate(display_time):
            row = [time]
            for array in arrays.values():
                row.extend(array[index])
            writer.writerow(row)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
