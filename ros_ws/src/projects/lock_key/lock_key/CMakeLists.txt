cmake_minimum_required(VERSION 2.8.3)
project(lock_key)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pluginlib
  geometry_msgs
  pcl_ros
  pcl_conversions
  sensor_msgs
  lock_key_msgs
  moveit_planner
  moveit_ros_planning_interface
  moveit_msgs
  moveit_ros_planning
  moveit_ros_move_group
  tf
  franka_gripper
  genmsg
  actionlib_msgs
  message_generation
  )

  find_package(PCL REQUIRED)# COMPONENTS
  #common
  #io
#)

find_package(Boost REQUIRED)

## Generate services in the 'srv' folder
 add_service_files(
   FILES
   imgCapture.srv
   findKey.srv
   getAveWrench.srv
   getWrench.srv
 )

 ## Generate actions in the 'action' folder
 add_action_files(
   DIRECTORY
   action
   FILES
   Jiggle.action
   MoveAbs.action
   SpiralInsert.action
 )

## Generate added messages, actions, and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   sensor_msgs
   geometry_msgs
   actionlib_msgs
 )

catkin_package(DEPENDS moveit_planner)

include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

add_executable(arm_move_node src/arm_move.cpp)
add_executable(img_capture_node src/img_capture.cpp)
add_executable(control_node src/controller.cpp)
add_executable(find_key_node src/find_key.cpp)
add_executable(collision_obj_node src/collision_obj.cpp)
add_executable(key_grasp_node src/key_grasp.cpp)
add_executable(arm_constraints_node src/arm_constraints.cpp)
add_executable(test_move_node src/test_move.cpp)
add_executable(setup_node src/setup.cpp)
add_executable(spiral_insert_node src/spiral_insert.cpp)
add_executable(get_wrench_node src/wrench_server.cpp)
add_executable(actual_control_node src/actual_controller.cpp)
add_executable(jiggle_node src/jiggle.cpp)
add_executable(move_abs_node src/move_absolute.cpp)
add_executable(detect_plane_node src/detect_plane.cpp)
add_executable(spiral_node src/spiral.cpp)
add_executable(rotate_key_node src/rotate_key.cpp)
add_executable(find_planes_node src/find_planes.cpp)

add_dependencies(arm_move_node moveit_planner_gencpp)
add_dependencies(img_capture_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(control_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(find_key_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(collision_obj_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(key_grasp_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(arm_constraints_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(test_move_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(setup_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(spiral_insert_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(get_wrench_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(actual_control_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(jiggle_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(move_abs_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(detect_plane_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(spiral_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(rotate_key_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(find_planes_node ${${PROJECT_NAME}_EXPORTED_TARGETS})

target_link_libraries(arm_move_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(img_capture_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(control_node ${catkin_LIBRARIES})
target_link_libraries(find_key_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(collision_obj_node ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(key_grasp_node ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(arm_constraints_node ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(test_move_node ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(setup_node ${catkin_LIBRARIES})
target_link_libraries(spiral_insert_node ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(get_wrench_node ${catkin_LIBRARIES})
target_link_libraries(actual_control_node ${catkin_LIBRARIES})
target_link_libraries(jiggle_node ${catkin_LIBRARIES})
target_link_libraries(move_abs_node ${catkin_LIBRARIES})
target_link_libraries(detect_plane_node ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(spiral_node ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(rotate_key_node ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(find_planes_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
