cmake_minimum_required(VERSION 3.0.2)
project(custom_gazebo_hwi)

find_package(catkin REQUIRED COMPONENTS
	gazebo_dev
  roscpp
	gazebo_ros
	gazebo_ros_control
  std_msgs
  control_toolbox
  controller_manager
  hardware_interface
  transmission_interface
  pluginlib
  joint_limits_interface
  urdf
  angles
)



catkin_package(
  CATKIN_DEPENDS
    roscpp
    std_msgs
		gazebo_ros
		gazebo_ros_control
    controller_manager
    control_toolbox
    pluginlib
    hardware_interface
    transmission_interface
    joint_limits_interface
    urdf
    angles
  INCLUDE_DIRS include
  LIBRARIES my_gazebo_ros_control my_robot_hw_sim
)

link_directories(
  ${catkin_LIBRARY_DIRS}
)

include_directories(include
  ${Boost_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)



## Libraries

add_library(my_robot_hw_sim src/my_robot_hw_sim.cpp)
target_link_libraries(my_robot_hw_sim ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})


add_library(my_gazebo_ros_control src/my_gazebo_ros_control_plugin.cpp)
target_link_libraries(my_gazebo_ros_control my_robot_hw_sim ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

## Install
install(TARGETS my_gazebo_ros_control my_robot_hw_sim
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(FILES ${PROJECT_NAME}_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

