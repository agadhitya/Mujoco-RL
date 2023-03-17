cmake_minimum_required(VERSION 2.8.3)
project(common_msgs_gl)

find_package(catkin REQUIRED
message_generation
geometry_msgs)

add_message_files(
  FILES
  PointArray.msg
  PointArray3D.msg
  Motor_position.msg
)

add_service_files(
  FILES
  SetPose.srv
  SetPosition.srv
  SetOrientation.srv
  SendInt.srv
  SendIntArray.srv
  SendBool.srv
  SendDouble.srv
  SendDoubleArray.srv
  SendDoubleArrayReceiveInt.srv
  SendDoubleArrayReceiveIntDouble.srv
  SendDoubleArrayReceiveDouble.srv
  SendDoubleArrayReceiveDoubleArray.srv
  SendString.srv
  SendPointArray.srv
  GetString.srv
  GetInt.srv
  GetDoubleArray.srv
  GetDouble.srv
  GetBool.srv
  GetBoolArray.srv
  Visual_servo_goal.srv
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
  geometry_msgs  # Or other packages containing msgs
  std_msgs
)

catkin_package(
)
