cmake_minimum_required(VERSION 2.8.3)
project(pcl_recorder)
## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)
 
find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_generation
  message_runtime
  sensor_msgs
)
find_package(octomap REQUIRED)

add_service_files(
  FILES
  GetPointCloud.srv
  BeginPointCloudRange.srv
  EndPointCloudRange.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS}
)

add_executable(pcl_recorder_node src/main.cpp)
target_link_libraries(pcl_recorder_node ${catkin_LIBRARIES} ${OCTOMAP_LIBRARIES})
add_dependencies(pcl_recorder_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
