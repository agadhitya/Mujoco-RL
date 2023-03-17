cmake_minimum_required(VERSION 3.0.2)
project(active_vision)

## Compile as C++14, supported in ROS Neotic
add_compile_options(-std=c++14)
find_package(PCL 1.10 REQUIRED)
find_package(OpenCV 4.2 REQUIRED)
message("*****OpenCV version: ${OpenCV_VERSION}"*****)
message("*****PCL version: ${PCL_VERSION}"*****)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  pcl_conversions
  pcl_ros
  tf
  pluginlib
  geometry_msgs
  moveit_planner
  moveit_ros_planning_interface
  moveit_msgs
  moveit_ros_planning
  moveit_ros_move_group
  franka_gripper
  genmsg
  actionlib_msgs
  cv_bridge
  image_transport
  message_generation
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
add_service_files(
  FILES
  heuristicPolicySRV.srv
  trainedPolicySRV.srv
  controlSRV.srv
  restartObjSRV.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
  actionlib_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
   INCLUDE_DIRS include
   LIBRARIES active_vision
   CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs #cv_bridge image_transport
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# add_definitions(${PCL_DEFINITIONS})
# link_directories(${PCL_LIBRARY_DIRS})

## Declare a C++ library
add_library(${PROJECT_NAME}
   src/environment.cpp
   src/toolDataHandling.cpp
   src/toolStateVector.cpp
   src/toolVisualization.cpp
   src/toolViewPointCalc.cpp
)

# target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/active_vision_node.cpp)
#add_executable(testingModel src/testingModel.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_active_vision.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

add_executable(environmentTesting src/environmentTesting.cpp)
add_dependencies(environmentTesting active_vision ${catkin_EXPORTED_TARGETS})
target_link_libraries(environmentTesting active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(dataCollector src/dataCollector.cpp)
add_dependencies(dataCollector active_vision ${catkin_EXPORTED_TARGETS})
target_link_libraries(dataCollector active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(visDataRec src/visDataRec.cpp)
add_dependencies(visDataRec active_vision ${catkin_EXPORTED_TARGETS})
target_link_libraries(visDataRec active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(visDataRecV2 src/visDataRecV2.cpp)
add_dependencies(visDataRecV2 active_vision ${catkin_EXPORTED_TARGETS})
target_link_libraries(visDataRecV2 active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(visAllPaths src/visAllPaths.cpp)
add_dependencies(visAllPaths active_vision ${catkin_EXPORTED_TARGETS})
target_link_libraries(visAllPaths active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES})

# add_executable(visViewsphere src/visViewsphere.cpp)
# add_dependencies(visViewsphere active_vision ${catkin_EXPORTED_TARGETS})
# target_link_libraries(visViewsphere active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(visDataCollectionPaths src/visDataCollectionPaths.cpp)
add_dependencies(visDataCollectionPaths active_vision ${catkin_EXPORTED_TARGETS})
target_link_libraries(visDataCollectionPaths active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES})

# add_executable(findStablePos src/findStablePos.cpp)
# add_dependencies(findStablePos active_vision ${catkin_EXPORTED_TARGETS})
# target_link_libraries(findStablePos active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(genStateVec src/genStateVec.cpp)
add_dependencies(genStateVec active_vision ${catkin_EXPORTED_TARGETS})
target_link_libraries(genStateVec active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(policyTester src/policyTester.cpp)
add_dependencies(policyTester active_vision ${catkin_EXPORTED_TARGETS})
target_link_libraries(policyTester active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(kinectService src/kinectService.cpp)
add_dependencies(kinectService active_vision ${catkin_EXPORTED_TARGETS})
target_link_libraries(kinectService active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(heuristicPolicyService src/heuristicPolicyService.cpp)
add_dependencies(heuristicPolicyService active_vision ${catkin_EXPORTED_TARGETS})
target_link_libraries(heuristicPolicyService active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(BFSHeuristicPolicyService src/BFSHeuristicPolicyService.cpp)
add_dependencies(BFSHeuristicPolicyService active_vision ${catkin_EXPORTED_TARGETS})
target_link_libraries(BFSHeuristicPolicyService active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(3DheuristicPolicyService src/3DheuristicPolicyService.cpp)
add_dependencies(3DheuristicPolicyService active_vision ${catkin_EXPORTED_TARGETS})
target_link_libraries(3DheuristicPolicyService active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(probabilisticPaper src/probabilisticPaper.cpp)
add_dependencies(probabilisticPaper active_vision ${catkin_EXPORTED_TARGETS})
target_link_libraries(probabilisticPaper active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES})

# add_executable(bfs src/bfs.cpp)
# add_dependencies(bfs active_vision ${catkin_EXPORTED_TARGETS})
# target_link_libraries(bfs active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES})

# add_executable(rgbd_to_rgb src/rgbd_to_rgb.cpp)
# add_dependencies(rgbd_to_rgb active_vision ${catkin_EXPORTED_TARGETS})
# target_link_libraries(rgbd_to_rgb active_vision ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
