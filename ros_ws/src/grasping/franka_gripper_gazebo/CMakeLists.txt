cmake_minimum_required(VERSION 2.8.3)
add_compile_options(-std=c++11)  ## Added this since it throws errors while compiling on Ubuntu 16.04 --Abhinav
project(franka_gripper_gazebo)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_generation
  message_runtime
  std_msgs
)

add_message_files(
  FILES
  DelayedMotion.msg)

add_service_files(
  FILES
  GripMsg.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS} include
)
link_libraries(
  ${catkin_LIBRARIES}
)

add_executable(gripper_node src/franka_gripper_node.cpp src/franka_gripper.cpp)
add_dependencies(gripper_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
