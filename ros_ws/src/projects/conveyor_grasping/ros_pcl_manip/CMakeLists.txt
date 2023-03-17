cmake_minimum_required(VERSION 2.8.3)
project(ros_pcl_manip)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  message_generation
  message_runtime)
find_package(PCL REQUIRED COMPONENTS
  common
  filters
  sample_consensus
  segmentation
  search
  kdtree
  features
  recognition
  console
  io
  visualization)

add_service_files(
  FILES
  Downsample.srv
  SegmentPlane.srv
  ESeg.srv
  CorrGroup.srv
  LoadFile.srv
  ToFile.srv
  PassFilter.srv)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs)

catkin_package(
  INCLUDE_DIRS include)

include_directories(include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

add_executable(ros_pcl_manip_node src/ros_pcl_manip_node)
target_link_libraries(ros_pcl_manip_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(ros_pcl_manip_node ${${PROJECT_NAME}_EXPORTED_TARGETS})

add_executable(test_vision_node src/test_vision_node)
target_link_libraries(test_vision_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(test_vision_node ${${PROJECT_NAME}_EXPORTED_TARGETS})

add_executable(to_pcd_node src/toPCD.cpp)
target_link_libraries(to_pcd_node ${PCL_LIBRARIES})

add_executable(correspondence_node src/correspondence.cpp)
target_link_libraries(correspondence_node ${PCL_LIBRARIES})
