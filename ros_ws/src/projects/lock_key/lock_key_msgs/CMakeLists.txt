cmake_minimum_required(VERSION 2.8.3)
project(lock_key_msgs)

 add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  actionlib_msgs
  geometry_msgs
  sensor_msgs
)

find_package(Boost REQUIRED)

## Generate messages in the 'msg' folder
 add_message_files(
   FILES
   PlaneCoefficients.msg
#   Message2.msg
 )

## Generate services in the 'srv' folder
 add_service_files(
   FILES
   GetLockKeyPoses.srv
   FindPlanes.srv
 )

## Generate actions in the 'action' folder
 add_action_files(
   DIRECTORY
   action
   FILES
   DetectPlane.action
   Spiral.action
   RotateKey.action
 )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
   actionlib_msgs
   geometry_msgs
   sensor_msgs
 )

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES lock_key_msgs
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# add_executable(${PROJECT_NAME}_node src/lock_key_msgs_node.cpp)

# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )
