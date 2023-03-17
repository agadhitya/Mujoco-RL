cmake_minimum_required(VERSION 2.8.3)
project(gripper_controls)
add_compile_options(-std=c++11)

 find_package(catkin REQUIRED COMPONENTS
   roscpp
   rospy
   std_msgs
   message_generation
   rosbag
 )

 add_message_files(DIRECTORY msg
   FILES
   Friction.msg
   JointEffort.msg
)

 add_service_files(
   FILES
   PositionCommand.srv
   Holdcommand.srv
   SetControlType.srv
   SetControlValue.srv
   SetFriction.srv
)


 generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS message_runtime roscpp rospy std_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(high_lvl_ctrl src/high_controller.cpp include/gripper_controls/Hand.cpp)
target_link_libraries(high_lvl_ctrl ${catkin_LIBRARIES})
add_dependencies(high_lvl_ctrl ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(low_lvl_ctrl src/low_controller.cpp)
target_link_libraries(low_lvl_ctrl ${catkin_LIBRARIES})
add_dependencies(low_lvl_ctrl ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(sequential_command_node src/sequential_commands.cpp)
target_link_libraries(sequential_command_node ${catkin_LIBRARIES})
add_dependencies(sequential_command_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(torque_pub_node src/torque_publisher.cpp)
add_dependencies(torque_pub_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(torque_pub_node ${catkin_LIBRARIES})
