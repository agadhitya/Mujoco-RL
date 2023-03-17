cmake_minimum_required(VERSION 2.8.3)
project(rrbot_pcl)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pluginlib
  PCL REQUIRED
  pcl_ros
  pcl_conversions
  tf
  message_generation
  sensor_msgs
  cv_bridge
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)


## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
add_service_files(
  FILES
  cloudMsg.srv
  processCloud.srv
  concaveHull.srv
  fourierParam.srv
  projectCloud.srv
  skelMsg.srv
)

## Generate added messages, actions, and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  sensor_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)

include_directories( include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

catkin_install_python(PROGRAMS scripts/skeletonization.py 
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

## Declare a C++ executable
add_executable(get_cloud_node src/get_cloud.cpp)
add_executable(rrbot_pcl_node src/main.cpp)
add_executable(process_cloud_node src/process_cloud.cpp)
add_executable(concave_hull_node src/concave_hull.cpp)
add_executable(fourier_param_node src/fourier_param.cpp)
add_executable(project_cloud_node src/project_cloud.cpp)

## Add cmake target dependencies of the executable
add_dependencies(get_cloud_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(rrbot_pcl_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(process_cloud_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(concave_hull_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(fourier_param_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(project_cloud_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(get_cloud_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(rrbot_pcl_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(process_cloud_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(concave_hull_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(fourier_param_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(project_cloud_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})