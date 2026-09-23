// C++ port of scripts/inekf_odom.py.
//
// The Python node spent most of its CPU inside the rclpy executor rather than
// in the filter: a bare rclpy subscriber costs ~940 us of CPU per message, so
// simply waking up at the 500 Hz /lowstate rate burned ~47% of a Jetson core
// before any estimation ran. Two estimators therefore saturated two cores.
// This node keeps the algorithm identical and removes that per-callback cost.
//
// scripts/inekf_odom.py is kept as the reference implementation; the two must
// stay behaviourally equivalent.

#include <pinocchio/fwd.hpp> // Must precede any boost/ROS header.

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/math/rpy.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "inekf/InEKF.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "unitree_go/msg/low_state.hpp"

namespace
{

/// Return the matrix representing the cross product with `vector`.
Eigen::Matrix3d skew(const Eigen::Vector3d & vector)
{
  Eigen::Matrix3d result;
  // clang-format off
  result <<          0.0, -vector.z(),  vector.y(),
              vector.z(),         0.0, -vector.x(),
             -vector.y(),  vector.x(),         0.0;
  // clang-format on
  return result;
}

/// SO(3) exponential map.
Eigen::Matrix3d rotation_exp(const Eigen::Vector3d & rotation_vector)
{
  const double theta = rotation_vector.norm();
  const Eigen::Matrix3d omega = skew(rotation_vector);
  if (theta < 1.0e-8)
  {
    return Eigen::Matrix3d::Identity() + omega + 0.5 * omega * omega;
  }
  return Eigen::Matrix3d::Identity() + (std::sin(theta) / theta) * omega +
         ((1.0 - std::cos(theta)) / (theta * theta)) * omega * omega;
}

/// SO(3) logarithm map, including rotations close to pi.
Eigen::Vector3d rotation_log(const Eigen::Matrix3d & rotation)
{
  const double cos_theta = std::clamp((rotation.trace() - 1.0) * 0.5, -1.0, 1.0);
  const double theta = std::acos(cos_theta);
  const Eigen::Vector3d vee(rotation(2, 1) - rotation(1, 2), rotation(0, 2) - rotation(2, 0),
                            rotation(1, 0) - rotation(0, 1));
  if (theta < 1.0e-7)
  {
    return 0.5 * vee;
  }
  if (M_PI - theta < 1.0e-5)
  {
    // Near pi the vee form degenerates, so recover the axis from the
    // eigenvector associated with the unit eigenvalue.
    Eigen::EigenSolver<Eigen::Matrix3d> solver(rotation);
    Eigen::Index best = 0;
    double best_distance = std::numeric_limits<double>::infinity();
    for (Eigen::Index i = 0; i < solver.eigenvalues().size(); ++i)
    {
      const double distance = std::abs(solver.eigenvalues()(i) - std::complex<double>(1.0, 0.0));
      if (distance < best_distance)
      {
        best_distance = distance;
        best = i;
      }
    }
    Eigen::Vector3d axis = solver.eigenvectors().col(best).real();
    axis.normalize();
    return theta * axis;
  }
  return theta / (2.0 * std::sin(theta)) * vee;
}

/// Left Jacobian used by the SE_K(3) exponential map.
Eigen::Matrix3d so3_left_jacobian(const Eigen::Vector3d & rotation_vector)
{
  const double theta = rotation_vector.norm();
  const Eigen::Matrix3d omega = skew(rotation_vector);
  if (theta < 1.0e-8)
  {
    return Eigen::Matrix3d::Identity() + 0.5 * omega + (1.0 / 6.0) * omega * omega;
  }
  return Eigen::Matrix3d::Identity() + ((1.0 - std::cos(theta)) / (theta * theta)) * omega +
         ((theta - std::sin(theta)) / (theta * theta * theta)) * omega * omega;
}

/// Convert a ROS (x, y, z, w) quaternion to a rotation matrix.
bool quaternion_to_rotation(const Eigen::Vector4d & quaternion, Eigen::Matrix3d & rotation)
{
  const double norm = quaternion.norm();
  if (!std::isfinite(norm) || norm < 1.0e-12)
  {
    return false;
  }
  const Eigen::Vector4d unit = quaternion / norm;
  rotation = Eigen::Quaterniond(unit(3), unit(0), unit(1), unit(2)).toRotationMatrix();
  return true;
}

/// Component-wise median, matching numpy.median(values, axis=0).
Eigen::Vector3d componentwise_median(const std::vector<Eigen::Vector3d> & values)
{
  Eigen::Vector3d result = Eigen::Vector3d::Zero();
  const size_t count = values.size();
  std::vector<double> component(count);
  for (int axis = 0; axis < 3; ++axis)
  {
    for (size_t i = 0; i < count; ++i)
    {
      component[i] = values[i][axis];
    }
    std::sort(component.begin(), component.end());
    result[axis] = (count % 2 == 1) ? component[count / 2]
                                    : 0.5 * (component[count / 2 - 1] + component[count / 2]);
  }
  return result;
}

/// Solve `system * x = rhs`, reporting failure instead of returning garbage.
bool solve_linear(const Eigen::MatrixXd & system, const Eigen::MatrixXd & rhs,
                  Eigen::MatrixXd & solution)
{
  solution = system.partialPivLu().solve(rhs);
  return solution.allFinite();
}

struct MocapSample
{
  double time;
  pinocchio::SE3 pose;
};

/// An accepted mocap sample awaiting fusion at the next propagation step.
struct PendingMocap
{
  pinocchio::SE3 pose;
  std::optional<Eigen::Vector3d> velocity;
  int64_t arrival_ns;
};

} // namespace

class InekfOdomNode : public rclcpp::Node
{
public:
  InekfOdomNode() : rclcpp::Node("inekf")
  {
    declareParameters();

    base_frame_ = get_parameter("base_frame").as_string();
    odom_frame_ = get_parameter("odom_frame").as_string();
    world_frame_ = get_parameter("world_frame").as_string();
    publish_tf_ = get_parameter("publish_tf").as_bool();
    dt_ = 1.0 / get_parameter("robot_freq").as_double();
    use_lowstate_tick_ = get_parameter("use_lowstate_tick").as_bool();
    max_propagation_dt_ = get_parameter("max_propagation_dt").as_double();
    if (dt_ <= 0.0 || max_propagation_dt_ <= 0.0)
    {
      throw std::runtime_error("robot_freq and max_propagation_dt must be positive");
    }

    mocap_enabled_ = get_parameter("mocap_enabled").as_bool();
    mocap_convert_axes_ = get_parameter("mocap_convert_axes").as_bool();
    mocap_timeout_ = get_parameter("mocap_timeout").as_double();
    mocap_position_std_ = get_parameter("mocap_position_std").as_double();
    mocap_orientation_std_ = get_parameter("mocap_orientation_std").as_double();
    mocap_position_gate_ = get_parameter("mocap_position_gate").as_double();
    mocap_orientation_gate_ = get_parameter("mocap_orientation_gate").as_double();
    mocap_initialize_ = get_parameter("mocap_initialize").as_bool();
    mocap_window_enabled_ = get_parameter("mocap_window_enabled").as_bool();
    mocap_window_size_ = static_cast<size_t>(get_parameter("mocap_window_size").as_int());
    mocap_prediction_gate_ = get_parameter("mocap_prediction_gate").as_double();
    mocap_jump_gate_ = get_parameter("mocap_jump_gate").as_double();
    mocap_orientation_prediction_gate_ =
      get_parameter("mocap_orientation_prediction_gate").as_double();
    mocap_recovery_samples_ = static_cast<size_t>(get_parameter("mocap_recovery_samples").as_int());
    mocap_nis_enabled_ = get_parameter("mocap_nis_enabled").as_bool();
    mocap_position_nis_gate_ = get_parameter("mocap_position_nis_gate").as_double();
    mocap_orientation_nis_gate_ = get_parameter("mocap_orientation_nis_gate").as_double();
    mocap_position_nis_floor_std_ = get_parameter("mocap_position_nis_floor_std").as_double();
    mocap_orientation_nis_floor_std_ = get_parameter("mocap_orientation_nis_floor_std").as_double();
    mocap_nis_recovery_samples_ =
      static_cast<size_t>(get_parameter("mocap_nis_recovery_samples").as_int());
    mocap_velocity_enabled_ = get_parameter("mocap_velocity_enabled").as_bool();
    mocap_velocity_min_samples_ =
      static_cast<size_t>(get_parameter("mocap_velocity_min_samples").as_int());
    mocap_velocity_std_ = get_parameter("mocap_velocity_std").as_double();
    mocap_velocity_nis_gate_ = get_parameter("mocap_velocity_nis_gate").as_double();
    mocap_velocity_nis_floor_std_ = get_parameter("mocap_velocity_nis_floor_std").as_double();
    mocap_max_velocity_ = get_parameter("mocap_max_velocity").as_double();
    if (mocap_timeout_ <= 0.0 || mocap_position_std_ <= 0.0 || mocap_orientation_std_ <= 0.0 ||
        mocap_window_size_ < 3 || mocap_recovery_samples_ < 2 || mocap_velocity_min_samples_ < 3 ||
        mocap_velocity_min_samples_ > mocap_window_size_ || mocap_nis_recovery_samples_ < 2 ||
        mocap_velocity_std_ <= 0.0 || mocap_max_velocity_ <= 0.0)
    {
      throw std::runtime_error("invalid mocap timeout, window, velocity, or noise parameter");
    }

    // Same calibrated conversions as mocap_state_estimator. The mocap source
    // world is x=left, y=back, z=up and its rigid-body axes need an additional
    // +90 degree yaw correction to become x=front, y=left, z=up.
    mocap_world_change_ = Eigen::Vector3d(-1.0, -1.0, 1.0).asDiagonal();
    // clang-format off
    mocap_body_change_ << 0.0, -1.0, 0.0,
                          1.0,  0.0, 0.0,
                          0.0,  0.0, 1.0;
    // clang-format on

    loadRobotModel();

    // In/Out topics
    lowstate_subscription_ = create_subscription<unitree_go::msg::LowState>(
      "/lowstate", rclcpp::QoS(rclcpp::KeepLast(10)),
      [this](unitree_go::msg::LowState::ConstSharedPtr msg) { lowstateCallback(*msg); });
    if (mocap_enabled_)
    {
      mocap_subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        get_parameter("mocap_topic").as_string(), rclcpp::SensorDataQoS(),
        [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) { mocapCallback(*msg); });
    }
    const std::string output_topic = get_parameter("output_topic").as_string();
    const std::string mocap_output_topic = get_parameter("mocap_output_topic").as_string();
    const std::string legacy_topic = get_parameter("legacy_output_topic").as_string();
    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>(output_topic, 1);
    if (!mocap_output_topic.empty() && mocap_output_topic != output_topic)
    {
      mocap_odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>(mocap_output_topic, 1);
    }
    if (!legacy_topic.empty() && legacy_topic != output_topic)
    {
      legacy_odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>(legacy_topic, 1);
    }
    if (publish_tf_)
    {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    // Invariant EKF
    inekf::RobotState initial_state;
    initial_state.setRotation(Eigen::Matrix3d::Identity());
    initial_state.setVelocity(Eigen::Vector3d::Zero());
    initial_state.setPosition(Eigen::Vector3d::Zero());
    initial_state.setGyroscopeBias(Eigen::Vector3d::Zero());
    initial_state.setAccelerometerBias(Eigen::Vector3d::Zero());

    // Initialize state covariance
    inekf::NoiseParams noise_params;
    noise_params.setGyroscopeNoise(get_parameter("gyroscope_noise").as_double());
    noise_params.setAccelerometerNoise(get_parameter("accelerometer_noise").as_double());
    noise_params.setGyroscopeBiasNoise(get_parameter("gyroscopeBias_noise").as_double());
    noise_params.setAccelerometerBiasNoise(get_parameter("accelerometerBias_noise").as_double());
    noise_params.setContactNoise(get_parameter("contact_noise").as_double());

    joint_pos_noise_ = get_parameter("joint_position_noise").as_double();
    contact_vel_noise_ = get_parameter("contact_velocity_noise").as_double();

    filter_ = inekf::InEKF(initial_state, noise_params);
    filter_.setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));

    if (mocap_enabled_)
    {
      RCLCPP_INFO(get_logger(), "Mocap pose fusion enabled on %s",
                  get_parameter("mocap_topic").as_string().c_str());
    }
  }

  void logMocapStatistics() const
  {
    RCLCPP_INFO(get_logger(),
                "Mocap statistics: received=%ld, accepted=%ld, local_rejected=%ld, recoveries=%ld, "
                "stale=%ld, hard_gate_rejected=%ld, nis_pose_rejected=%ld, nis_pose_recoveries=%ld, "
                "nis_velocity_rejected=%ld, pose_fused=%ld, velocity_fused=%ld",
                stats_.received, stats_.accepted, stats_.local_rejected, stats_.recoveries,
                stats_.stale, stats_.hard_gate_rejected, stats_.nis_pose_rejected,
                stats_.nis_pose_recoveries, stats_.nis_velocity_rejected, stats_.pose_fused,
                stats_.velocity_fused);
  }

private:
  struct MocapStats
  {
    int64_t received = 0;
    int64_t accepted = 0;
    int64_t local_rejected = 0;
    int64_t recoveries = 0;
    int64_t stale = 0;
    int64_t hard_gate_rejected = 0;
    int64_t nis_pose_rejected = 0;
    int64_t nis_pose_recoveries = 0;
    int64_t nis_velocity_rejected = 0;
    int64_t pose_fused = 0;
    int64_t velocity_fused = 0;
  };

  void declareParameters()
  {
    declare_parameter("base_frame", "base");           // Robot base frame name (for TF)
    declare_parameter("odom_frame", "odom");           // Local odometry frame name (for TF)
    declare_parameter("world_frame", "world");         // Absolute mocap/world frame name
    // Canonical odometry topic (go2_x5_interfaces kSlamOdometryTopic)
    declare_parameter("output_topic", "/go2_x5/slam/odom");
    // World-frame odometry mirror; falls back to pure odometry when mocap is disabled
    declare_parameter("mocap_output_topic", "/go2_x5/slam/odom_mocap");
    // Legacy mirror of the same message; empty or equal to output_topic disables it
    declare_parameter("legacy_output_topic", "/odometry/filtered");
    // Broadcast odom_frame -> base_frame; disable for all but one estimator
    declare_parameter("publish_tf", true);
    // Path to the robot URDF; empty resolves the installed unitree_description model
    declare_parameter("urdf_path", "");
    declare_parameter("robot_freq", 500.0);        // Frequency at which the robot publish its state
    declare_parameter("use_lowstate_tick", true);  // Use the LowState millisecond tick for dt
    declare_parameter("max_propagation_dt", 0.02); // Maximum accepted propagation dt
    declare_parameter("gyroscope_noise", 0.01);
    declare_parameter("accelerometer_noise", 0.1);
    declare_parameter("gyroscopeBias_noise", 0.00001);
    declare_parameter("accelerometerBias_noise", 0.0001);
    declare_parameter("contact_noise", 0.001);
    declare_parameter("joint_position_noise", 0.001);
    declare_parameter("contact_velocity_noise", 0.001);
    declare_parameter("mocap_enabled", true);
    declare_parameter("mocap_topic", "/vrpn_mocap/go2/pose");
    declare_parameter("mocap_convert_axes", true);
    declare_parameter("mocap_timeout", 0.1);
    declare_parameter("mocap_position_std", 0.001);
    declare_parameter("mocap_orientation_std", 0.005);
    declare_parameter("mocap_position_gate", 0.5);
    declare_parameter("mocap_orientation_gate", 0.8);
    declare_parameter("mocap_initialize", true);
    declare_parameter("mocap_window_enabled", true);
    declare_parameter("mocap_window_size", 7);
    declare_parameter("mocap_prediction_gate", 0.02);
    declare_parameter("mocap_jump_gate", 0.03);
    declare_parameter("mocap_orientation_prediction_gate", 0.08);
    declare_parameter("mocap_recovery_samples", 3);
    declare_parameter("mocap_nis_enabled", true);
    declare_parameter("mocap_position_nis_gate", 16.27);
    declare_parameter("mocap_orientation_nis_gate", 16.27);
    declare_parameter("mocap_position_nis_floor_std", 0.10);
    declare_parameter("mocap_orientation_nis_floor_std", 0.05);
    declare_parameter("mocap_nis_recovery_samples", 3);
    declare_parameter("mocap_velocity_enabled", true);
    declare_parameter("mocap_velocity_min_samples", 5);
    declare_parameter("mocap_velocity_std", 0.08);
    declare_parameter("mocap_velocity_nis_gate", 16.27);
    declare_parameter("mocap_velocity_nis_floor_std", 0.30);
    declare_parameter("mocap_max_velocity", 5.0);
  }

  void loadRobotModel()
  {
    // State estimation only needs the kinematic model, so the visual and
    // collision geometry is never parsed.
    std::string urdf_path = get_parameter("urdf_path").as_string();
    if (urdf_path.empty())
    {
      urdf_path = ament_index_cpp::get_package_share_directory("unitree_description") +
                  "/model/go2/go2.urdf";
    }
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model_);
    data_ = pinocchio::Data(model_);

    const std::array<std::string, 4> prefixes{"FL", "FR", "RL", "RR"};
    for (size_t i = 0; i < 4; ++i)
    {
      const std::string frame_name = prefixes[i] + "_foot";
      if (!model_.existFrame(frame_name))
      {
        throw std::runtime_error("missing URDF frame: " + frame_name);
      }
      foot_frame_id_[i] = model_.getFrameId(frame_name);
    }
    if (!model_.existFrame("imu") || !model_.existFrame(base_frame_))
    {
      throw std::runtime_error("missing URDF frame: imu or " + base_frame_);
    }
    imu_frame_id_ = model_.getFrameId("imu");
    base_frame_id_ = model_.getFrameId(base_frame_);

    // Save rigid transform between imu (filter frame) and base (output frame)
    pinocchio::forwardKinematics(model_, data_, pinocchio::neutral(model_));
    pinocchio::updateFramePlacements(model_, data_);
    imuMbase_ = data_.oMf[imu_frame_id_].actInv(data_.oMf[base_frame_id_]);

    joint_jacobian_.resize(6, model_.nv);
  }

  // ---------------------------------------------------------------------------
  // /lowstate pipeline
  // ---------------------------------------------------------------------------
  void lowstateCallback(const unitree_go::msg::LowState & msg)
  {
    const double propagation_dt = getPropagationDt(msg);

    // Format IMU measurements
    Eigen::Matrix<double, 6, 1> imu_state;
    for (int i = 0; i < 3; ++i)
    {
      imu_state(i) = static_cast<double>(msg.imu_state.gyroscope[i]);
      imu_state(3 + i) = static_cast<double>(msg.imu_state.accelerometer[i]);
    }

    // Feet kinematic data
    feetTransformations(msg);

    if (pause_)
    {
      if (std::all_of(contact_list_.begin(), contact_list_.end(), [](bool c) { return c; }))
      {
        pause_ = false;
        initializeFilter(msg);
        RCLCPP_INFO(get_logger(), "All feet in contact with the ground: starting filter.");
      }
      else
      {
        RCLCPP_INFO_ONCE(get_logger(), "Waiting for all feet to touch the ground to start filter.");
        return; // Skip the rest of the filter
      }
    }

    // Propagation step: using IMU
    filter_.propagate(imu_state, propagation_dt);

    // TODO: use IMU quaternion for extra correction step ?

    // Correction step: using feet kinematics
    std::vector<std::pair<int, bool>> contact_pairs;
    contact_pairs.reserve(4);
    inekf::vectorKinematics kinematics_list;
    kinematics_list.reserve(4);
    for (int i = 0; i < 4; ++i)
    {
      contact_pairs.emplace_back(i, contact_list_[static_cast<size_t>(i)]);
      kinematics_list.emplace_back(i, pose_list_[static_cast<size_t>(i)].translation(),
                                   joint_pos_noise_ * normed_covariance_list_[static_cast<size_t>(i)],
                                   Eigen::Vector3d::Zero(),
                                   contact_vel_noise_ * Eigen::Matrix3d::Identity());
    }

    filter_.setContacts(contact_pairs);
    filter_.correctKinematics(kinematics_list);

    // Consume every mocap packet at most once. Running the correction here
    // serializes it with propagation: both callbacks run on the same executor.
    correctMocap();

    publishState(filter_.getState(), msg.imu_state.gyroscope);
  }

  /// Use Unitree's millisecond tick so dropped callbacks do not lose time.
  double getPropagationDt(const unitree_go::msg::LowState & msg)
  {
    const uint32_t tick = msg.tick;
    double propagation_dt = dt_;
    if (use_lowstate_tick_ && last_lowstate_tick_.has_value())
    {
      const uint32_t tick_delta = tick - last_lowstate_tick_.value();
      const double measured_dt = static_cast<double>(tick_delta) * 1.0e-3;
      if (measured_dt > 0.0 && measured_dt <= max_propagation_dt_)
      {
        propagation_dt = measured_dt;
      }
    }
    last_lowstate_tick_ = tick;
    return propagation_dt;
  }

  /// Rearrange the Unitree joint order into the URDF order.
  static void unitreeToUrdfVec(const double * source, double * destination)
  {
    // clang-format off
    static constexpr std::array<size_t, 12> kOrder{3, 4, 5,
                                                   0, 1, 2,
                                                   9, 10, 11,
                                                   6, 7, 8};
    // clang-format on
    for (size_t i = 0; i < 12; ++i)
    {
      destination[i] = source[kOrder[i]];
    }
  }

  void getQvfPinocchio(const unitree_go::msg::LowState & msg, Eigen::VectorXd & q_pin,
                       Eigen::VectorXd & v_pin, std::array<double, 4> & f_pin) const
  {
    std::array<double, 12> q_unitree;
    std::array<double, 12> v_unitree;
    for (size_t i = 0; i < 12; ++i)
    {
      q_unitree[i] = static_cast<double>(msg.motor_state[i].q);
      v_unitree[i] = static_cast<double>(msg.motor_state[i].dq);
    }

    q_pin.setZero(model_.nq);
    q_pin(6) = 1.0; // Free-flyer unit quaternion
    v_pin.setZero(model_.nv);
    unitreeToUrdfVec(q_unitree.data(), q_pin.data() + 7);
    unitreeToUrdfVec(v_unitree.data(), v_pin.data() + 6);

    static constexpr std::array<size_t, 4> kFootOrder{1, 0, 3, 2};
    for (size_t i = 0; i < 4; ++i)
    {
      f_pin[i] = static_cast<double>(msg.foot_force[kFootOrder[i]]);
    }
  }

  void feetTransformations(const unitree_go::msg::LowState & msg)
  {
    // Get configuration
    std::array<double, 4> f_pin;
    getQvfPinocchio(msg, q_pin_, v_pin_, f_pin);

    // Compute positions and velocities
    pinocchio::forwardKinematics(model_, data_, q_pin_, v_pin_);
    pinocchio::updateFramePlacements(model_, data_);
    pinocchio::computeJointJacobians(model_, data_);

    // Compute foot kinematics and jacobian
    const pinocchio::SE3 & oMimu = data_.oMf[imu_frame_id_];
    for (size_t i = 0; i < 4; ++i)
    {
      contact_list_[i] = f_pin[i] >= 20.0;
      pose_list_[i] = oMimu.actInv(data_.oMf[foot_frame_id_[i]]);

      joint_jacobian_.setZero();
      pinocchio::getFrameJacobian(model_, data_, foot_frame_id_[i], pinocchio::LOCAL,
                                  joint_jacobian_);
      // Actuated columns only: the free-flyer block is not a measurement.
      const Eigen::Matrix<double, 3, 12> Jc = joint_jacobian_.block<3, 12>(0, 6);
      normed_covariance_list_[i] = Jc * Jc.transpose();
    }
  }

  void initializeFilter(const unitree_go::msg::LowState & msg)
  {
    // Unitree configuration
    Eigen::VectorXd q = Eigen::VectorXd::Zero(model_.nq);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model_.nv);
    std::array<double, 4> f_pin;
    getQvfPinocchio(msg, q, v, f_pin);

    // Use robot IMU guess to initialize the filter
    q(3) = static_cast<double>(msg.imu_state.quaternion[1]);
    q(4) = static_cast<double>(msg.imu_state.quaternion[2]);
    q(5) = static_cast<double>(msg.imu_state.quaternion[3]);
    q(6) = static_cast<double>(msg.imu_state.quaternion[0]);
    q.segment<4>(3).normalize(); // Normalize quaternion

    // Compute FK
    pinocchio::forwardKinematics(model_, data_, q, v);
    pinocchio::updateFramePlacements(model_, data_);

    // Correct initial rotation
    pinocchio::SE3 oMbase = data_.oMf[base_frame_id_];
    Eigen::Vector3d rpy = pinocchio::rpy::matrixToRpy(oMbase.rotation());
    rpy(2) = 0.0; // Set yaw to 0 for robot to always face x axis at start
    oMbase.rotation(pinocchio::rpy::rpyToMatrix(rpy));

    // Compute average foot height
    double z_avg = 0.0;
    for (size_t i = 0; i < 4; ++i)
    {
      z_avg += data_.oMf[foot_frame_id_[i]].translation().z();
    }
    z_avg /= 4.0;

    // Correct base position
    Eigen::Vector3d base_translation = oMbase.translation();
    base_translation.x() = 0.0; // centered in XY
    base_translation.y() = 0.0;
    base_translation.z() -= z_avg - 0.025; // Add foot thickness of 2.5 cm
    oMbase.translation(base_translation);

    // Convert base pose to IMU (since filter state is in IMU frame)
    pinocchio::SE3 oMimu = oMbase.act(imuMbase_.inverse());

    // Prefer the absolute mocap pose at startup when available; otherwise
    // retain the original foot/IMU based initialization.
    if (mocap_enabled_ && mocap_initialize_ && mocapIsFresh())
    {
      oMimu = latest_mocap_pose_.value();
      pending_mocap_.clear();
      mocap_has_fused_ = true;
    }

    // Set filter initial state
    inekf::RobotState state = filter_.getState();
    state.setRotation(oMimu.rotation());
    state.setPosition(oMimu.translation());
    filter_.setState(state);
  }

  // ---------------------------------------------------------------------------
  // Mocap pipeline
  // ---------------------------------------------------------------------------
  void mocapCallback(const geometry_msgs::msg::PoseStamped & msg)
  {
    stats_.received += 1;
    Eigen::Vector3d position(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);
    const Eigen::Vector4d quaternion(msg.pose.orientation.x, msg.pose.orientation.y,
                                     msg.pose.orientation.z, msg.pose.orientation.w);
    if (!position.allFinite())
    {
      RCLCPP_WARN(get_logger(), "Ignoring mocap pose with a non-finite position");
      return;
    }
    Eigen::Matrix3d rotation;
    if (!quaternion_to_rotation(quaternion, rotation))
    {
      RCLCPP_WARN(get_logger(), "Ignoring mocap pose with an invalid quaternion");
      return;
    }

    if (mocap_convert_axes_)
    {
      position = mocap_world_change_ * position;
      rotation = mocap_world_change_ * rotation * mocap_body_change_.transpose();
    }

    // The mocap rigid-body definition represents the robot base, whereas InEKF
    // estimates the IMU pose. Convert base pose to IMU pose here.
    const pinocchio::SE3 world_mocap_base(rotation, position);
    const pinocchio::SE3 mocap_pose = world_mocap_base.act(imuMbase_.inverse());
    const int64_t arrival_ns = get_clock()->now().nanoseconds();
    double source_time =
      static_cast<double>(msg.header.stamp.sec) + static_cast<double>(msg.header.stamp.nanosec) * 1.0e-9;
    if (source_time <= 0.0)
    {
      source_time = static_cast<double>(arrival_ns) * 1.0e-9;
    }

    if (mocap_window_enabled_ && mocapSampleIsOutlier(source_time, mocap_pose))
    {
      stats_.local_rejected += 1;
      pushBounded(mocap_recovery_candidates_, {source_time, mocap_pose}, mocap_recovery_samples_);
      if (!mocapRecoveryIsConsistent())
      {
        return;
      }
      mocap_history_.clear();
      for (const MocapSample & candidate : mocap_recovery_candidates_)
      {
        pushBounded(mocap_history_, candidate, mocap_window_size_);
      }
      mocap_recovery_candidates_.clear();
      stats_.recoveries += 1;
    }
    else
    {
      mocap_recovery_candidates_.clear();
      pushBounded(mocap_history_, {source_time, mocap_pose}, mocap_window_size_);
    }

    latest_mocap_pose_ = mocap_pose;
    latest_mocap_arrival_ns_ = arrival_ns;
    pushBounded(pending_mocap_, PendingMocap{mocap_pose, estimateMocapVelocity(), arrival_ns},
                kMaxPendingMocap);
    stats_.accepted += 1;
  }

  /// Detect a pose inconsistent with both a local prediction and the last
  /// sample. Drops the window when the stream itself has gone stale.
  bool mocapSampleIsOutlier(double source_time, const pinocchio::SE3 & mocap_pose)
  {
    if (mocap_history_.size() < 3)
    {
      return false;
    }
    const MocapSample & last = mocap_history_.back();
    const double dt = source_time - last.time;
    if (dt <= 0.0)
    {
      return true;
    }
    if (dt > mocap_timeout_)
    {
      mocap_history_.clear();
      return false;
    }

    std::vector<Eigen::Vector3d> linear_velocities;
    std::vector<Eigen::Vector3d> angular_velocities;
    for (size_t i = 0; i + 1 < mocap_history_.size(); ++i)
    {
      const MocapSample & a = mocap_history_[i];
      const MocapSample & b = mocap_history_[i + 1];
      const double sample_dt = b.time - a.time;
      if (sample_dt <= 0.0 || sample_dt > mocap_timeout_)
      {
        continue;
      }
      linear_velocities.push_back((b.pose.translation() - a.pose.translation()) / sample_dt);
      angular_velocities.push_back(
        rotation_log(b.pose.rotation() * a.pose.rotation().transpose()) / sample_dt);
    }
    if (linear_velocities.empty())
    {
      return false;
    }

    const Eigen::Vector3d linear_velocity = componentwise_median(linear_velocities);
    const Eigen::Vector3d angular_velocity = componentwise_median(angular_velocities);
    const Eigen::Vector3d predicted_position = last.pose.translation() + linear_velocity * dt;
    const Eigen::Matrix3d predicted_rotation =
      rotation_exp(angular_velocity * dt) * last.pose.rotation();
    const double prediction_error = (mocap_pose.translation() - predicted_position).norm();
    const double position_jump = (mocap_pose.translation() - last.pose.translation()).norm();
    const double orientation_prediction_error =
      rotation_log(mocap_pose.rotation() * predicted_rotation.transpose()).norm();
    const double orientation_jump =
      rotation_log(mocap_pose.rotation() * last.pose.rotation().transpose()).norm();
    const bool position_outlier =
      prediction_error > mocap_prediction_gate_ && position_jump > mocap_jump_gate_;
    const bool orientation_outlier =
      orientation_prediction_error > mocap_orientation_prediction_gate_ &&
      orientation_jump > 2.0 * mocap_orientation_prediction_gate_;
    return position_outlier || orientation_outlier;
  }

  bool mocapRecoveryIsConsistent() const
  {
    if (mocap_recovery_candidates_.size() < mocap_recovery_samples_)
    {
      return false;
    }
    for (size_t i = 0; i + 1 < mocap_recovery_candidates_.size(); ++i)
    {
      const MocapSample & a = mocap_recovery_candidates_[i];
      const MocapSample & b = mocap_recovery_candidates_[i + 1];
      const double dt = b.time - a.time;
      const double position_step = (b.pose.translation() - a.pose.translation()).norm();
      const double orientation_step =
        rotation_log(b.pose.rotation() * a.pose.rotation().transpose()).norm();
      if (dt <= 0.0 || dt > mocap_timeout_ || position_step > mocap_jump_gate_ ||
          orientation_step > 2.0 * mocap_orientation_prediction_gate_)
      {
        return false;
      }
    }
    return true;
  }

  /// Estimate current world-frame IMU velocity with a causal quadratic fit.
  std::optional<Eigen::Vector3d> estimateMocapVelocity() const
  {
    if (!mocap_velocity_enabled_ || mocap_history_.size() < mocap_velocity_min_samples_)
    {
      return std::nullopt;
    }
    const size_t count = mocap_history_.size();
    Eigen::VectorXd times(count);
    Eigen::MatrixXd positions(count, 3);
    for (size_t i = 0; i < count; ++i)
    {
      times(static_cast<Eigen::Index>(i)) = mocap_history_[i].time;
      positions.row(static_cast<Eigen::Index>(i)) = mocap_history_[i].pose.translation().transpose();
    }
    const double last_time = times(static_cast<Eigen::Index>(count) - 1);
    times.array() -= last_time;
    for (Eigen::Index i = 1; i < times.size(); ++i)
    {
      if (times(i) - times(i - 1) <= 0.0)
      {
        return std::nullopt;
      }
    }
    if (-times(0) < 0.02)
    {
      return std::nullopt;
    }

    Eigen::MatrixXd design(count, 3);
    design.col(0).setOnes();
    design.col(1) = times;
    design.col(2) = times.array().square();
    const Eigen::MatrixXd coefficients =
      design.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(positions);
    const Eigen::Vector3d velocity = coefficients.row(1).transpose();
    if (!velocity.allFinite() || velocity.norm() > mocap_max_velocity_)
    {
      return std::nullopt;
    }
    return velocity;
  }

  bool arrivalIsFresh(int64_t arrival_ns) const
  {
    const double age = static_cast<double>(get_clock()->now().nanoseconds() - arrival_ns) * 1.0e-9;
    return age >= 0.0 && age <= mocap_timeout_;
  }

  bool mocapIsFresh() const
  {
    return latest_mocap_pose_.has_value() && arrivalIsFresh(latest_mocap_arrival_ns_);
  }

  void correctMocap()
  {
    if (!mocap_enabled_ || pending_mocap_.empty())
    {
      return;
    }

    // Pop it even if stale/outlying so the same bad sample is not reconsidered
    // at every IMU tick. /lowstate runs far faster than the mocap, so the
    // queue drains immediately and no accepted sample is skipped.
    const PendingMocap pending = pending_mocap_.front();
    pending_mocap_.pop_front();
    if (!arrivalIsFresh(pending.arrival_ns))
    {
      stats_.stale += 1;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Ignoring stale mocap pose");
      return;
    }

    inekf::RobotState state = filter_.getState();
    const Eigen::Matrix3d estimated_rotation = state.getRotation();
    const Eigen::Vector3d estimated_velocity = state.getVelocity();
    const Eigen::Vector3d estimated_position = state.getPosition();
    const pinocchio::SE3 & mocap_pose = pending.pose;

    const Eigen::Vector3d orientation_residual =
      rotation_log(mocap_pose.rotation() * estimated_rotation.transpose());
    const Eigen::Matrix3d innovation_rotation = rotation_exp(orientation_residual);
    const Eigen::Matrix3d innovation_jacobian = so3_left_jacobian(orientation_residual);
    // Log of T_mocap * T_estimate^-1. Expressing the full innovation in the
    // same right-invariant tangent space as InEKF avoids a large-angle
    // position/orientation linearization error at startup.
    const Eigen::Vector3d position_residual = innovation_jacobian.partialPivLu().solve(
      Eigen::Vector3d(mocap_pose.translation() - innovation_rotation * estimated_position));

    const double position_error = (mocap_pose.translation() - estimated_position).norm();
    const double orientation_error = orientation_residual.norm();
    const bool position_outlier = mocap_position_gate_ > 0.0 && position_error > mocap_position_gate_;
    const bool orientation_outlier =
      mocap_orientation_gate_ > 0.0 && orientation_error > mocap_orientation_gate_;
    if (mocap_has_fused_ && (position_outlier || orientation_outlier))
    {
      stats_.hard_gate_rejected += 1;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Rejecting mocap innovation: %.3f m, %.3f rad", position_error,
                           orientation_error);
      return;
    }

    const Eigen::MatrixXd covariance = state.getP();
    const Eigen::Index state_dimension = covariance.rows();
    Eigen::MatrixXd orientation_observation =
      Eigen::MatrixXd::Zero(3, state_dimension);
    orientation_observation.leftCols<3>().setIdentity();
    Eigen::MatrixXd position_observation = Eigen::MatrixXd::Zero(3, state_dimension);
    position_observation.block<3, 3>(0, 6).setIdentity();

    if (mocap_nis_enabled_ && mocap_has_fused_)
    {
      const double orientation_nis =
        innovationNis(orientation_residual, orientation_observation, covariance,
                      mocap_orientation_std_, mocap_orientation_nis_floor_std_);
      const double position_nis =
        innovationNis(position_residual, position_observation, covariance, mocap_position_std_,
                      mocap_position_nis_floor_std_);
      if (orientation_nis > mocap_orientation_nis_gate_ || position_nis > mocap_position_nis_gate_)
      {
        stats_.nis_pose_rejected += 1;
        mocap_nis_rejection_streak_ += 1;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Rejecting mocap pose NIS: position %.2f, orientation %.2f",
                             position_nis, orientation_nis);
        if (mocap_nis_rejection_streak_ < mocap_nis_recovery_samples_)
        {
          return;
        }
        // Locally consistent mocap samples must eventually win over an
        // over-confident or temporarily divergent filter covariance.
        mocap_nis_rejection_streak_ = 0;
        stats_.nis_pose_recoveries += 1;
      }
      else
      {
        mocap_nis_rejection_streak_ = 0;
      }
    }

    // Stack the accepted observations: orientation, position and optionally
    // the mocap-derived world velocity.
    Eigen::Index rows = 6;
    bool fuse_velocity = false;
    Eigen::Vector3d velocity_residual = Eigen::Vector3d::Zero();
    Eigen::MatrixXd velocity_observation;
    if (mocap_velocity_enabled_ && pending.velocity.has_value())
    {
      velocity_residual = innovation_jacobian.partialPivLu().solve(Eigen::Vector3d(
        pending.velocity.value() - innovation_rotation * estimated_velocity));
      velocity_observation = Eigen::MatrixXd::Zero(3, state_dimension);
      velocity_observation.block<3, 3>(0, 3).setIdentity();
      const double velocity_nis =
        innovationNis(velocity_residual, velocity_observation, covariance, mocap_velocity_std_,
                      mocap_velocity_nis_floor_std_);
      if (!mocap_nis_enabled_ || velocity_nis <= mocap_velocity_nis_gate_)
      {
        rows = 9;
        fuse_velocity = true;
      }
      else
      {
        stats_.nis_velocity_rejected += 1;
      }
    }

    Eigen::MatrixXd observation(rows, state_dimension);
    Eigen::VectorXd residual(rows);
    Eigen::VectorXd measurement_variances(rows);
    observation.topRows<3>() = orientation_observation;
    observation.middleRows<3>(3) = position_observation;
    residual.head<3>() = orientation_residual;
    residual.segment<3>(3) = position_residual;
    measurement_variances.head<3>().setConstant(mocap_orientation_std_ * mocap_orientation_std_);
    measurement_variances.segment<3>(3).setConstant(mocap_position_std_ * mocap_position_std_);
    if (fuse_velocity)
    {
      observation.middleRows<3>(6) = velocity_observation;
      residual.segment<3>(6) = velocity_residual;
      measurement_variances.segment<3>(6).setConstant(mocap_velocity_std_ * mocap_velocity_std_);
    }

    const Eigen::MatrixXd measurement_covariance = measurement_variances.asDiagonal();
    const Eigen::MatrixXd observed_covariance = observation * covariance;
    const Eigen::MatrixXd innovation_covariance =
      observed_covariance * observation.transpose() + measurement_covariance;
    Eigen::MatrixXd solved;
    if (!solve_linear(innovation_covariance, observed_covariance, solved))
    {
      RCLCPP_ERROR(get_logger(), "Mocap correction skipped: singular innovation covariance");
      return;
    }
    const Eigen::MatrixXd kalman_gain = solved.transpose();

    const Eigen::VectorXd delta = kalman_gain * residual;
    const Eigen::Index group_dimension = 3 * (state.dimX() - 2);
    const Eigen::VectorXd group_delta = delta.head(group_dimension);
    const Eigen::Vector3d rotation_delta = group_delta.head<3>();
    const Eigen::Matrix3d delta_rotation = rotation_exp(rotation_delta);
    const Eigen::Matrix3d left_jacobian = so3_left_jacobian(rotation_delta);

    const Eigen::MatrixXd state_matrix = state.getX();
    Eigen::MatrixXd correction = Eigen::MatrixXd::Identity(state.dimX(), state.dimX());
    correction.topLeftCorner<3, 3>() = delta_rotation;
    for (Eigen::Index column = 3; column < state.dimX(); ++column)
    {
      const Eigen::Index vector_index = 3 + 3 * (column - 3);
      correction.block<3, 1>(0, column) = left_jacobian * group_delta.segment<3>(vector_index);
    }

    const Eigen::MatrixXd ikh =
      Eigen::MatrixXd::Identity(state_dimension, state_dimension) - kalman_gain * observation;
    Eigen::MatrixXd corrected_covariance =
      ikh * covariance * ikh.transpose() +
      kalman_gain * measurement_covariance * kalman_gain.transpose();
    corrected_covariance = 0.5 * (corrected_covariance + corrected_covariance.transpose()).eval();

    state.setX(correction * state_matrix);
    state.setTheta(Eigen::VectorXd(state.getTheta() + delta.tail(state_dimension - group_dimension)));
    state.setP(corrected_covariance);
    filter_.setState(state);
    mocap_has_fused_ = true;
    stats_.pose_fused += 1;
    if (fuse_velocity)
    {
      stats_.velocity_fused += 1;
    }
  }

  static double innovationNis(const Eigen::Vector3d & residual, const Eigen::MatrixXd & observation,
                              const Eigen::MatrixXd & covariance, double measurement_std,
                              double floor_std)
  {
    Eigen::MatrixXd gate_covariance = observation * covariance * observation.transpose();
    gate_covariance += (measurement_std * measurement_std + floor_std * floor_std) *
                       Eigen::Matrix3d::Identity();
    Eigen::MatrixXd solved;
    if (!solve_linear(gate_covariance, residual, solved))
    {
      return std::numeric_limits<double>::infinity();
    }
    return residual.dot(solved.col(0));
  }

  // ---------------------------------------------------------------------------
  // Output
  // ---------------------------------------------------------------------------
  void publishState(const inekf::RobotState & filter_state, const std::array<float, 3> & twist_angular_vel)
  {
    // Get filter state
    const rclcpp::Time timestamp = get_clock()->now();

    // Get filter state (imu frame)
    const pinocchio::SE3 oMimu(Eigen::Matrix3d(filter_state.getRotation()),
                               Eigen::Vector3d(filter_state.getPosition()));
    const Eigen::Vector3d v_linear_imu_world = filter_state.getX().block<3, 1>(0, 3);
    const Eigen::Vector3d v_linear_imu_local = oMimu.rotation().transpose() * v_linear_imu_world;
    const pinocchio::Motion v_imu_local(
      v_linear_imu_local, Eigen::Vector3d(static_cast<double>(twist_angular_vel[0]),
                                          static_cast<double>(twist_angular_vel[1]),
                                          static_cast<double>(twist_angular_vel[2])));

    // Transform to base frame
    const pinocchio::SE3 base_pose = oMimu.act(imuMbase_);
    const pinocchio::Motion base_velocity = imuMbase_.actInv(v_imu_local);

    // The filter state is expressed in `odom`.  Keep odom -> base from the
    // InEKF estimate and derive world -> odom from the latest accepted mocap
    // pose.  When mocap is disabled, world -> odom is identity so the
    // odom_mocap topic remains a live pure-odometry mirror.
    if (!mocap_enabled_)
    {
      world_odom_pose_ = pinocchio::SE3::Identity();
    }
    else if (mocapIsFresh())
    {
      const pinocchio::SE3 mocap_base_pose = latest_mocap_pose_.value().act(imuMbase_);
      world_odom_pose_ = mocap_base_pose.act(base_pose.inverse());
    }
    const pinocchio::SE3 world_base_pose = world_odom_pose_.act(base_pose);

    // Convert to quaternion
    Eigen::Quaterniond base_quaternion(base_pose.rotation());
    base_quaternion.normalize();
    Eigen::Quaterniond world_base_quaternion(world_base_pose.rotation());
    world_base_quaternion.normalize();

    if (tf_broadcaster_)
    {
      // TF2 messages
      transform_msg_.header.stamp = timestamp;
      transform_msg_.child_frame_id = base_frame_;
      transform_msg_.header.frame_id = odom_frame_;

      transform_msg_.transform.translation.x = base_pose.translation().x();
      transform_msg_.transform.translation.y = base_pose.translation().y();
      transform_msg_.transform.translation.z = base_pose.translation().z();

      transform_msg_.transform.rotation.x = base_quaternion.x();
      transform_msg_.transform.rotation.y = base_quaternion.y();
      transform_msg_.transform.rotation.z = base_quaternion.z();
      transform_msg_.transform.rotation.w = base_quaternion.w();

      tf_broadcaster_->sendTransform(transform_msg_);

      if (world_frame_ != odom_frame_)
      {
        world_odom_transform_msg_.header.stamp = timestamp;
        world_odom_transform_msg_.header.frame_id = world_frame_;
        world_odom_transform_msg_.child_frame_id = odom_frame_;
        world_odom_transform_msg_.transform.translation.x = world_odom_pose_.translation().x();
        world_odom_transform_msg_.transform.translation.y = world_odom_pose_.translation().y();
        world_odom_transform_msg_.transform.translation.z = world_odom_pose_.translation().z();
        Eigen::Quaterniond world_odom_quaternion(world_odom_pose_.rotation());
        world_odom_quaternion.normalize();
        world_odom_transform_msg_.transform.rotation.x = world_odom_quaternion.x();
        world_odom_transform_msg_.transform.rotation.y = world_odom_quaternion.y();
        world_odom_transform_msg_.transform.rotation.z = world_odom_quaternion.z();
        world_odom_transform_msg_.transform.rotation.w = world_odom_quaternion.w();
        tf_broadcaster_->sendTransform(world_odom_transform_msg_);
      }
    }

    // Odometry topic
    odom_msg_.header.stamp = timestamp;
    odom_msg_.child_frame_id = base_frame_;
    odom_msg_.header.frame_id = odom_frame_;

    odom_msg_.pose.pose.position.x = base_pose.translation().x();
    odom_msg_.pose.pose.position.y = base_pose.translation().y();
    odom_msg_.pose.pose.position.z = base_pose.translation().z();

    odom_msg_.pose.pose.orientation.x = base_quaternion.x();
    odom_msg_.pose.pose.orientation.y = base_quaternion.y();
    odom_msg_.pose.pose.orientation.z = base_quaternion.z();
    odom_msg_.pose.pose.orientation.w = base_quaternion.w();

    odom_msg_.twist.twist.linear.x = base_velocity.linear().x();
    odom_msg_.twist.twist.linear.y = base_velocity.linear().y();
    odom_msg_.twist.twist.linear.z = base_velocity.linear().z();

    odom_msg_.twist.twist.angular.x = base_velocity.angular().x();
    odom_msg_.twist.twist.angular.y = base_velocity.angular().y();
    odom_msg_.twist.twist.angular.z = base_velocity.angular().z();

    odom_publisher_->publish(odom_msg_);
    if (legacy_odom_publisher_)
    {
      legacy_odom_publisher_->publish(odom_msg_);
    }
    if (mocap_odom_publisher_)
    {
      mocap_odom_msg_.header.stamp = timestamp;
      mocap_odom_msg_.header.frame_id = world_frame_;
      mocap_odom_msg_.child_frame_id = base_frame_;
      mocap_odom_msg_.pose.pose.position.x = world_base_pose.translation().x();
      mocap_odom_msg_.pose.pose.position.y = world_base_pose.translation().y();
      mocap_odom_msg_.pose.pose.position.z = world_base_pose.translation().z();
      mocap_odom_msg_.pose.pose.orientation.x = world_base_quaternion.x();
      mocap_odom_msg_.pose.pose.orientation.y = world_base_quaternion.y();
      mocap_odom_msg_.pose.pose.orientation.z = world_base_quaternion.z();
      mocap_odom_msg_.pose.pose.orientation.w = world_base_quaternion.w();
      mocap_odom_msg_.twist.twist.linear.x = base_velocity.linear().x();
      mocap_odom_msg_.twist.twist.linear.y = base_velocity.linear().y();
      mocap_odom_msg_.twist.twist.linear.z = base_velocity.linear().z();
      mocap_odom_msg_.twist.twist.angular.x = base_velocity.angular().x();
      mocap_odom_msg_.twist.twist.angular.y = base_velocity.angular().y();
      mocap_odom_msg_.twist.twist.angular.z = base_velocity.angular().z();
      mocap_odom_publisher_->publish(mocap_odom_msg_);
    }
  }

  template <typename T>
  static void pushBounded(std::deque<T> & buffer, const T & value, size_t max_size)
  {
    buffer.push_back(value);
    while (buffer.size() > max_size)
    {
      buffer.pop_front();
    }
  }

  // Parameters
  std::string base_frame_;
  std::string odom_frame_;
  std::string world_frame_;
  bool publish_tf_ = true;
  double dt_ = 0.002;
  bool use_lowstate_tick_ = true;
  double max_propagation_dt_ = 0.02;
  double joint_pos_noise_ = 0.001;
  double contact_vel_noise_ = 0.001;
  bool mocap_enabled_ = true;
  bool mocap_convert_axes_ = true;
  double mocap_timeout_ = 0.1;
  double mocap_position_std_ = 0.001;
  double mocap_orientation_std_ = 0.005;
  double mocap_position_gate_ = 0.5;
  double mocap_orientation_gate_ = 0.8;
  bool mocap_initialize_ = true;
  bool mocap_window_enabled_ = true;
  size_t mocap_window_size_ = 7;
  double mocap_prediction_gate_ = 0.02;
  double mocap_jump_gate_ = 0.03;
  double mocap_orientation_prediction_gate_ = 0.08;
  size_t mocap_recovery_samples_ = 3;
  bool mocap_nis_enabled_ = true;
  double mocap_position_nis_gate_ = 16.27;
  double mocap_orientation_nis_gate_ = 16.27;
  double mocap_position_nis_floor_std_ = 0.10;
  double mocap_orientation_nis_floor_std_ = 0.05;
  size_t mocap_nis_recovery_samples_ = 3;
  bool mocap_velocity_enabled_ = true;
  size_t mocap_velocity_min_samples_ = 5;
  double mocap_velocity_std_ = 0.08;
  double mocap_velocity_nis_gate_ = 16.27;
  double mocap_velocity_nis_floor_std_ = 0.30;
  double mocap_max_velocity_ = 5.0;

  // Robot model
  pinocchio::Model model_;
  pinocchio::Data data_;
  std::array<pinocchio::FrameIndex, 4> foot_frame_id_{};
  pinocchio::FrameIndex imu_frame_id_ = 0;
  pinocchio::FrameIndex base_frame_id_ = 0;
  pinocchio::SE3 imuMbase_ = pinocchio::SE3::Identity();

  // Filter
  inekf::InEKF filter_;
  bool pause_ = true; // By default filter is paused and waits for the first feet contact
  std::optional<uint32_t> last_lowstate_tick_;

  // Mocap state
  Eigen::Matrix3d mocap_world_change_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d mocap_body_change_ = Eigen::Matrix3d::Identity();
  std::optional<pinocchio::SE3> latest_mocap_pose_;
  pinocchio::SE3 world_odom_pose_ = pinocchio::SE3::Identity();
  int64_t latest_mocap_arrival_ns_ = 0;
  static constexpr size_t kMaxPendingMocap = 32;
  std::deque<PendingMocap> pending_mocap_;
  bool mocap_has_fused_ = false;
  std::deque<MocapSample> mocap_history_;
  std::deque<MocapSample> mocap_recovery_candidates_;
  size_t mocap_nis_rejection_streak_ = 0;
  MocapStats stats_;

  // Preallocated work buffers: the /lowstate path runs at 500 Hz.
  Eigen::VectorXd q_pin_;
  Eigen::VectorXd v_pin_;
  Eigen::Matrix<double, 6, Eigen::Dynamic> joint_jacobian_;
  std::array<bool, 4> contact_list_{};
  std::array<pinocchio::SE3, 4> pose_list_;
  std::array<Eigen::Matrix3d, 4> normed_covariance_list_;
  nav_msgs::msg::Odometry odom_msg_;
  nav_msgs::msg::Odometry mocap_odom_msg_;
  geometry_msgs::msg::TransformStamped transform_msg_;
  geometry_msgs::msg::TransformStamped world_odom_transform_msg_;

  // ROS interfaces
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mocap_subscription_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr mocap_odom_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr legacy_odom_publisher_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<InekfOdomNode>();
  rclcpp::spin(node);
  node->logMocapStatistics();
  rclcpp::shutdown();
  return 0;
}
