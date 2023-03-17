cmake_minimum_required(VERSION 2.8.3)
project(conveyor_picking)
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  message_runtime
  roscpp
  sensor_msgs
  std_msgs
  ros_pcl_manip
  pose_estimator
  moveit_planner
  trajectory_msgs
  franka_gripper_gazebo)

find_package(Eigen3 REQUIRED)

add_service_files(
  FILES
  FindMug.srv)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  geometry_msgs
  std_msgs)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS})

add_executable(find_mug_node src/findMug.cpp)
target_link_libraries(find_mug_node ${catkin_LIBRARIES})
add_dependencies(find_mug_node ${${PROJECT_NAME}_EXPORTED_TARGETS})

add_executable(main_picking_node src/main.cpp)
target_link_libraries(main_picking_node ${catkin_LIBRARIES})
add_dependencies(main_picking_node ${${PROJECT_NAME}_EXPORTED_TARGETS})

add_executable(anticipation_node src/anticipationMain.cpp)
target_link_libraries(anticipation_node ${catkin_LIBRARIES})
add_dependencies(anticipation_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
