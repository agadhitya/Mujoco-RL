cmake_minimum_required(VERSION 3.1.0)
## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)
 
project(ref_benchmark)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	moveit_planner
	sensor_msgs
	pose_estimator
)

catkin_package(
	CATKIN_DEPENDS roscpp moveit_planner randomizer pcl_recorder pose_estimator
)

include_directories(${catkin_INCLUDE_DIRS} ${QT_INCLUDE_DIRS} include forms)

link_libraries(${catkin_LIBRARIES})

QT5_WRAP_UI(wUI forms/mainWindow.ui)
QT5_WRAP_CPP(wMOC include/mainWindow.hpp)

add_executable(ref_benchmark_node src/mainWindow.cpp src/main.cpp ${wUI} ${wMOC})
qt5_use_modules(ref_benchmark_node Widgets Core)
target_link_libraries(ref_benchmark_node ${Qt5Core_LIBRARIES} ${Qt5Widgets_LIBRARIES})
add_dependencies(ref_benchmark_node moveit_planner_gencpp)
