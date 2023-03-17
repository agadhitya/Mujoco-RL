cmake_minimum_required(VERSION 2.8.3)
project(piercing_experiments)

 add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  moveit_core
  moveit_ros_perception
  moveit_ros_planning_interface
  moveit_visual_tools
  moveit_msgs
  pluginlib
  geometry_msgs
  rosbag
  piercing_experiments_msgs
)

find_package(Boost REQUIRED)

catkin_package( 
INCLUDE_DIRS include 
CATKIN_DEPENDS roscpp
)
include_directories(include ${catkin_INCLUDE_DIRS})
 
 add_executable(main_pierce_node src/main_pierce.cpp)
 add_executable(move_node src/move.cpp)
 add_executable(recorder_node src/record.cpp)

 add_dependencies(move_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
 add_dependencies(main_pierce_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
 add_dependencies(recorder_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

 target_link_libraries(main_pierce_node ${catkin_LIBRARIES})
 target_link_libraries(move_node ${catkin_LIBRARIES} ${Boost_LIBRARIES})
 target_link_libraries(recorder_node ${catkin_LIBRARIES})
