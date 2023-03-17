cmake_minimum_required(VERSION 2.8.3)
project(moveit_planner)
## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_runtime
  message_generation
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  moveit_ros_move_group
  geometry_msgs
  moveit_msgs
  tf
  trajectory_msgs
)

add_service_files(
  FILES
  Inv.srv
  GetTF.srv
  GetPose.srv
  MovePoint.srv
  MoveQuat.srv
  MovePose.srv
  MoveJoint.srv
  MoveCart.srv
  MoveAway.srv
  SetVelocity.srv
  AddCollision.srv
  SetConstraints.srv
  MoveNamedState.srv
  SetJointWithTime.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  moveit_msgs
  trajectory_msgs
)

catkin_package(
  INCLUDE_DIRS include  ## Added this so pkg can be included in other pkgs
  CATKIN_DEPENDS
    roscpp
    moveit_core
    geometry_msgs
    moveit_msgs
    message_runtime
    tf
)

include_directories(include ${catkin_INCLUDE_DIRS})
link_libraries(${catkin_LIBRARIES})

add_executable(main_node src/main.cpp src/moveit_planner.cpp)
add_dependencies(main_node ${moveit_planner_EXPORTED_TARGETS})

add_executable(pose_node src/pose_node)
add_dependencies(pose_node ${moveit_planner_EXPORTED_TARGETS})
