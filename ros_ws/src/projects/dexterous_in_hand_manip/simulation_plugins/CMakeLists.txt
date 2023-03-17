cmake_minimum_required(VERSION 2.8.3)
project(simulation_plugins)

 find_package(catkin REQUIRED COMPONENTS
   roscpp
   rospy
   std_msgs
   message_generation
   gripper_controls
   simulation_ui
 )

find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

add_library(friction_plugin SHARED src/friction_plugin.cc)
target_link_libraries(friction_plugin ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES})
add_dependencies(friction_plugin ${catkin_EXPORTED_TARGETS})

add_library(param_plugin SHARED src/surf_param_plugin.cc)
target_link_libraries(param_plugin ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES})
add_dependencies(param_plugin ${catkin_EXPORTED_TARGETS})

add_library(unified_plugin SHARED src/unified_plugin.cc)
target_link_libraries(unified_plugin ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES})
add_dependencies(unified_plugin ${catkin_EXPORTED_TARGETS})

 generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime roscpp rospy std_msgs gripper_controls simulation_ui
)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
)

include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
