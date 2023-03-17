cmake_minimum_required(VERSION 3.0.2)
project(simulation_ui)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	rospy
	std_msgs
	message_generation
	gripper_controls
)

add_message_files(DIRECTORY msg
	FILES
	SimParam.msg
	SimFriction.msg
)

 add_service_files(
   FILES
   SetParam.srv
)

generate_messages(
 DEPENDENCIES
 std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS message_runtime roscpp rospy std_msgs
)

include_directories(${catkin_INCLUDE_DIRS} ${QT_INCLUDE_DIRS} include forms)

link_libraries(${catkin_LIBRARIES})


QT5_WRAP_UI(wUI forms/mainWindow.ui)
QT5_WRAP_CPP(wMOC include/mainWindow.hpp)

add_executable(sim_ui src/main.cpp src/mainWindow.cpp ${wUI} ${wMOC})
qt5_use_modules(sim_ui Widgets Core)
target_link_libraries(sim_ui ${Qt5Core_LIBRARIES} ${Qt5Widgets_LIBRARIES})
add_dependencies(sim_ui ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(param_setter src/param_setter.cpp)
target_link_libraries(param_setter ${catkin_LIBRARIES})
add_dependencies(param_setter ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
