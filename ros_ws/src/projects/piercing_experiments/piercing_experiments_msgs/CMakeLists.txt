cmake_minimum_required(VERSION 2.8.3)
project(piercing_experiments_msgs)

 add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
roscpp
rospy
std_msgs
geometry_msgs
message_generation
)

## Generate services in the 'srv' folder
 add_service_files(
   FILES
   PrePose.srv
   CartesianPath.srv
 )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
   geometry_msgs
 )

catkin_package()
include_directories(include ${catkin_INCLUDE_DIRS})