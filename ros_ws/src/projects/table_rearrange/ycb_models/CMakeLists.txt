cmake_minimum_required(VERSION 2.8.3)
add_compile_options(-std=c++11)
project(ycb_models)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  gazebo_msgs
  roslib
)

catkin_package()
include_directories(${catkin_INCLUDE_DIRS})

add_executable(object_scene_spawner src/spawner.cpp)
target_link_libraries(object_scene_spawner ${catkin_LIBRARIES})

add_executable(object_description_uploader src/descUploader.cpp)
target_link_libraries(object_description_uploader ${catkin_LIBRARIES})
