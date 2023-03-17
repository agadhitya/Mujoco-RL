cmake_minimum_required(VERSION 2.8.3)
project(arm_controls)
add_compile_options(-std=c++11)

 find_package(catkin REQUIRED COMPONENTS
   roscpp
   rospy
   std_msgs
   message_generation
   rosbag
   moveit_msgs
   moveit_planner
 )

 add_service_files(
   FILES
   PoseChange.srv
   MoveStraight.srv
)

 generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime roscpp rospy std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(arm_ctrl src/arm_high_lvl_ctrl.cpp)
target_link_libraries(arm_ctrl ${catkin_LIBRARIES})
add_dependencies(arm_ctrl ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
