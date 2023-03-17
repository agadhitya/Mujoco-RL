cmake_minimum_required(VERSION 2.8.3)
project(randomizer)
## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)
 
find_package(catkin REQUIRED COMPONENTS
  roscpp
  gazebo_msgs
  geometry_msgs
  message_generation
)

add_service_files(
  FILES
  Rand.srv
)

generate_messages()

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(${catkin_INCLUDE_DIRS})

add_executable(randomizer_node src/main.cpp)
add_dependencies(randomizer_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(randomizer_node ${catkin_LIBRARIES})
