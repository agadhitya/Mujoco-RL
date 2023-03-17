cmake_minimum_required(VERSION 2.8.3)
project(pcl_processor)
## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)
 
find_package(PCL 1.3 REQUIRED COMPONENTS
  common
  io
)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_conversions
  pcl_ros
  sensor_msgs
)

catkin_package()

include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

add_executable(pcl_processor_node src/main.cpp)
target_link_libraries(pcl_processor_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
