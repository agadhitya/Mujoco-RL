cmake_minimum_required(VERSION 2.8.3)
project(friction_finger_gripper)

## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  dynamixel_motors
  common_msgs_gl
  roscpp
  rospy
  message_generation
  rosbag
)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
 add_service_files(
   FILES
   PositionCommand.srv
   Holdcommand.srv
   ActionCommand.srv)
#   Service2.srv
# )

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


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES friction_finger_gripper
  CATKIN_DEPENDS dynamixel_motors roscpp
#  DEPENDS system_lib
)

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/finger.cpp
#   src/controller_client.cpp
# )

add_executable(controller_2 src/controller_2.cpp)
add_executable(controller_new src/controller_new.cpp)
add_executable(Test src/Test.cpp)
add_executable(record_node src/record.cpp)

add_dependencies(controller_2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(controller_new ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(Test ${${PROJECT_NAME}_EXPORTED_TARGETS} 
add_dependencies(record_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
${catkin_EXPORTED_TARGETS})
## Specify libraries to link a library or executable target against
 #target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )
 #target_link_libraries(client_node
  # ${catkin_LIBRARIES}
 #)
 ##target_link_libraries(controller_node
  # ${catkin_LIBRARIES}
 #)
 ##target_link_libraries(controller_client_node
  ## ${catkin_LIBRARIES}
 #)
 ##target_link_libraries(high_level_controller_node
   #${catkin_LIBRARIES}
 #)

  target_link_libraries(record_node ${catkin_LIBRARIES})
  target_link_libraries(controller_2 ${catkin_LIBRARIES})
  target_link_libraries(controller_new
   ${catkin_LIBRARIES}
 )
  target_link_libraries(Test
   ${catkin_LIBRARIES}
 )

