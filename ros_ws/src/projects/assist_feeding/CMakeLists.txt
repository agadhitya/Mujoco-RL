cmake_minimum_required(VERSION 2.8.3)
project(assist_feeding)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  dynamixel_sdk
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
 add_service_files(
   FILES
   gripper_open.srv
   gripper_close.srv
   finger_orientation.srv
 )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
 )

catkin_package(CATKIN_DEPENDS message_runtime)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(hand_node src/grippercontrol.cpp)
add_executable(grasp_node src/grasp.cpp src/classes/hand.cpp)

add_dependencies(hand_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(grasp_node ${${PROJECT_NAME}_EXPORTED_TARGETS})

target_link_libraries(hand_node ${catkin_LIBRARIES})
target_link_libraries(grasp_node ${catkin_LIBRARIES})
