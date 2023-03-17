cmake_minimum_required(VERSION 2.8.3)
project(owi_control)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  Eigen3
  rosbag
  cv_bridge
  image_transport
)

find_package(Boost REQUIRED COMPONENTS system)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
 DEPENDENCIES
 std_msgs
 geometry_msgs
 sensor_msgs
)

catkin_package()
include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

add_executable(owi_controller_node src/owi_controller.cpp)
add_dependencies(owi_controller_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(owi_controller_node ${catkin_LIBRARIES} ${EIGEN3_LIBRARIES})

add_executable(owi_record_node src/bag.cpp)
add_dependencies(owi_record_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(owi_record_node ${catkin_LIBRARIES})

add_executable(owi_keyboard_node src/keyboard.cpp)
add_dependencies(owi_keyboard_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(owi_keyboard_node ${catkin_LIBRARIES})

add_executable(owi_trajectory_node src/traj_plot.cpp)
add_dependencies(owi_trajectory_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(owi_trajectory_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(owi_sine_node src/sin_velocity.cpp)
add_dependencies(owi_sine_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(owi_sine_node ${catkin_LIBRARIES})

add_executable(owi_freq_node src/freq_check.cpp)
add_dependencies(owi_freq_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(owi_freq_node ${catkin_LIBRARIES})

add_executable(owi_freq_record_node src/record_data.cpp)
add_dependencies(owi_freq_record_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(owi_freq_record_node ${catkin_LIBRARIES})

add_executable(owi_const_vel_node src/const_vel.cpp)
add_dependencies(owi_const_vel_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(owi_const_vel_node ${catkin_LIBRARIES})

add_executable(owi_positioning_node src/position.cpp)
add_dependencies(owi_positioning_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(owi_positioning_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(owi_sys_id_node src/system_identification.cpp)
add_dependencies(owi_sys_id_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(owi_sys_id_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})