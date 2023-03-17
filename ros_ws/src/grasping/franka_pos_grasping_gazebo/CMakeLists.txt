cmake_minimum_required(VERSION 2.8.3)
project(franka_pos_grasping_gazebo)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED
  roscpp
  message_generation
  message_runtime
  std_msgs
  )

 add_service_files(
   FILES
   GripPos.srv
)

 generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
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

add_executable(pos_grasper_node src/pos_grasper.cpp)
add_dependencies(pos_grasper_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pos_grasper_node ${catkin_LIBRARIES})