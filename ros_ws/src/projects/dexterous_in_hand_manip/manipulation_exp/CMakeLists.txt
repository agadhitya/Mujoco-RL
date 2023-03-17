cmake_minimum_required(VERSION 2.8.3)
project(manipulation_exp)
add_compile_options(-std=c++11)

 find_package(catkin REQUIRED COMPONENTS
   roscpp
   rospy
   std_msgs
   message_generation
   rosbag
   gripper_controls
   arm_controls
 )

add_service_files(
    FILES
    Sequence.srv
 )


generate_messages(
   DEPENDENCIES
   std_msgs
 )

 catkin_package(
   INCLUDE_DIRS include
   CATKIN_DEPENDS message_runtime roscpp rospy std_msgs
 )

 include_directories(
   include
   ${catkin_INCLUDE_DIRS}
 )

add_executable(manip_test src/manipulation_test.cpp)
target_link_libraries(manip_test ${catkin_LIBRARIES})
add_dependencies(manip_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(manip_seq src/sequence_server.cpp include/manipulation_exp/ManipulationSequence.cpp)
target_link_libraries(manip_seq ${catkin_LIBRARIES})
add_dependencies(manip_seq ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
