cmake_minimum_required(VERSION 3.8)
project(go2_odometry)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(unitree_go REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pinocchio REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(inekf REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(unitree_description REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(ament_index_cpp REQUIRED)

unset(PYTHON_SOABI)

add_executable(state_converter_node src/go2_state_converter_node.cpp)
ament_target_dependencies(state_converter_node rclcpp unitree_go sensor_msgs)

# C++ port of scripts/inekf_odom.py. The Python node paid ~47% of a Jetson core
# in rclpy wake-ups alone at the 500 Hz /lowstate rate.
add_executable(inekf_odom_node src/inekf_odom_node.cpp)
ament_target_dependencies(inekf_odom_node
  rclcpp
  unitree_go
  nav_msgs
  geometry_msgs
  tf2_ros
  ament_index_cpp
)
target_link_libraries(inekf_odom_node inekf::inekf pinocchio::pinocchio)

install(TARGETS
  state_converter_node
  inekf_odom_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python executables
file(GLOB SCRIPTS_LIST scripts/*.py)

install(PROGRAMS
  ${SCRIPTS_LIST}
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files.
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
