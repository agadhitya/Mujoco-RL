cmake_minimum_required(VERSION 3.0.2)
project(manipulation_planning)

## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  gripper_controls
  moveit_msgs
  moveit_planner
  arm_controls
)

add_service_files(
   FILES
   HandAct.srv
   ArmAct.srv
   PlanExe.srv
   ArmAdjust.srv
   ArmPose.srv
)

generate_messages(
 DEPENDENCIES
 std_msgs
 geometry_msgs
)

catkin_package(
 CATKIN_DEPENDS message_runtime roscpp rospy std_msgs geometry_msgs
)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS scripts/plan_executer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/extended_plan_executer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(manip_acts src/action_server.cpp include/manipulation_planning/ManipulationActions.cpp)
target_link_libraries(manip_acts ${catkin_LIBRARIES})
add_dependencies(manip_acts ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(arm_adjust src/arm_adjust.cpp)
target_link_libraries(arm_adjust ${catkin_LIBRARIES})
add_dependencies(arm_adjust ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(franka_acts src/franka_action_server.cpp)
target_link_libraries(franka_acts ${catkin_LIBRARIES})
add_dependencies(franka_acts ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
