cmake_minimum_required(VERSION 2.8.3)
project(owi_robot)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
roscpp
rospy
std_msgs
geometry_msgs
message_generation
lusb
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)

add_service_files(
   FILES
   writecmd.srv
   generatecmd.srv
#   pwm.srv
)

generate_messages(
   DEPENDENCIES
   std_msgs
   geometry_msgs
)

catkin_package()
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(cmd_robot_node src/cmd_robot_srv.cpp)
add_dependencies(cmd_robot_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(cmd_robot_node ${catkin_LIBRARIES} usb-1.0)

add_executable(cmd_gen_node src/cmd_gen_srv.cpp)
add_dependencies(cmd_gen_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(cmd_gen_node ${catkin_LIBRARIES})

add_executable(owi_control_node src/main.cpp)
add_dependencies(owi_control_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(owi_control_node ${catkin_LIBRARIES})

add_executable(pwm_node src/pwm_srv.cpp)
add_dependencies(pwm_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pwm_node ${catkin_LIBRARIES})