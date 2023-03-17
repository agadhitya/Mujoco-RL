#include "gripper_controls/Hand.h"
#include <ros/ros.h>
#include <gazebo_msgs/GetJointProperties.h>
#include <string>
#include "gripper_controls/SetControlValue.h"
#include <controller_manager_msgs/SwitchController.h>
#include "gripper_controls/SetControlType.h"
#include <gripper_controls/SetFriction.h>
#include <iostream>
#include <sensor_msgs/JointState.h>

// method for reading joint position motor1: right finger
float Hand::read_position(int motor_num){
  ros::NodeHandle n;
  ros::service::waitForService("gazebo/get_joint_properties");
  ros::ServiceClient client_read_pos = n.serviceClient<gazebo_msgs::GetJointProperties>("gazebo/get_joint_properties");
  gazebo_msgs::GetJointProperties srv;
  std::string joint = "J" + std::to_string(motor_num);
  srv.request.joint_name = joint;
  if (client_read_pos.call(srv)){
    // ROS_INFO("Read_position_success");
    return srv.response.position[0];
  }
  else{
    ROS_INFO("Failure");
    return -1;
  }
}


//
// float Hand::read_effort(int motor_num){
//   ros::NodeHandle n;
//   sensor_msgs::JointStateConstPtr msg;
//   msg = ros::topic::waitForMessage<sensor_msgs::JointState>("/gripper/joint_states",n);
//   float effort_val{0};
//   effort_val = msg->effort[motor_num];
//   return effort_val;
// }

// method for sending position reference to low level controller motor1: right finger
bool Hand::command_position(int motor_num, float position){
  ros::NodeHandle n;
  ros::service::waitForService("pos_change");
  ros::ServiceClient client_position = n.serviceClient<gripper_controls::SetControlValue>("pos_change");
  gripper_controls::SetControlValue srv;
  srv.request.finger = motor_num;
  srv.request.value = position;
  if (client_position.call(srv)){
    // ROS_INFO("Command_position[%d]_Success=[%f]", motor_num, position);
    return 1;
  }
  else{
    ROS_INFO("Failure");
    return -1;
  }
}

// method for sending torque reference to low level controller motor1:right finger
bool Hand::command_torque(int motor_num, float torque){
  ros::NodeHandle n;
  ros::service::waitForService("effort_change");
  ros::ServiceClient client_torque = n.serviceClient<gripper_controls::SetControlValue>("effort_change");
  gripper_controls::SetControlValue srv;
  srv.request.finger = motor_num;
  srv.request.value = torque;
  if (client_torque.call(srv)){
    // ROS_INFO("Command_torque_Success");
    return 1;
  }
  else{
    ROS_INFO("Failure");
    return -1;
  }
}

// method for setting actuator modes
// informs low level controller on controller type and calls controller manager switch service for the actual switch
bool Hand::set_actuator_modes(int size, int mode[]){
  ros::NodeHandle n;
  ros::service::waitForService("vf_hand/controller_manager/switch_controller");
  ros::ServiceClient client_operating_mode = n.serviceClient<controller_manager_msgs::SwitchController>("vf_hand/controller_manager/switch_controller");
  // 3 -> position_controller 0 -> effort_controller
  std::string left{"l"};
  std::string right{"r"};
  std::string position{"_finger_position"};
  std::string effort{"_finger_effort"};
  controller_manager_msgs::SwitchController srv_l;
  if (mode[0]==3){
    srv_l.request.start_controllers.push_back(left+position);
    srv_l.request.stop_controllers.push_back(left+effort);
  }
  else{
    srv_l.request.start_controllers.push_back(left+effort);
    srv_l.request.stop_controllers.push_back(left+position);
  }
  srv_l.request.strictness = 2;
  client_operating_mode.call(srv_l);
  controller_manager_msgs::SwitchController srv_r;
  if (mode[1]==3){
    srv_r.request.start_controllers.push_back(right+position);
    srv_r.request.stop_controllers.push_back(right+effort);
  }
  else{
    srv_r.request.start_controllers.push_back(right+effort);
    srv_r.request.stop_controllers.push_back(right+position);
  }
  srv_r.request.strictness = 2;
  client_operating_mode.call(srv_r);
  ROS_INFO("Set_operating_mode_success");
  ros::service::waitForService("type_change");
  ros::ServiceClient client_control_mode = n.serviceClient<gripper_controls::SetControlType>("type_change");
  gripper_controls::SetControlType srv;
  srv.request.left = mode[0];
  srv.request.right = mode[1];
  client_control_mode.call(srv);
  return 1;
}

// sets left finger friction reference for the low level controller
bool Hand::set_friction_left(bool friction_surface){
  ros::NodeHandle n;
  ros::service::waitForService("set_friction");
  ros::ServiceClient client_friction = n.serviceClient<gripper_controls::SetFriction>("set_friction");
  gripper_controls::SetFriction srv;
  srv.request.finger = 0;
  srv.request.high_friction = friction_surface;
  if (client_friction.call(srv)){
    ROS_INFO("Friction_surface_set_Success");
    return 1;
  }
  else{
    ROS_INFO("Failure");
    return 0;
  }
}

// sets right finger friction reference for the low level controller
bool Hand::set_friction_right(bool friction_surface){
  ros::NodeHandle n;
  ros::service::waitForService("set_friction");
  ros::ServiceClient client_friction = n.serviceClient<gripper_controls::SetFriction>("set_friction");
  gripper_controls::SetFriction srv;
  srv.request.finger = 1;
  srv.request.high_friction = friction_surface;
  if (client_friction.call(srv)){
    ROS_INFO("Friction_surface_set_Success");
    return 1;
  }
  else{
    ROS_INFO("Failure");
    return 0;
  }
}

// Hand object constructor
Hand::Hand(std::string topic){
  std::cout << "Hand object constructed" << std::endl;
  finger_state = 0;
  left_effort = 0;
  right_effort = 0;
  sub_ = n_.subscribe(topic,1000,&Hand::stateCallback, this);
}

// callback for getting joint efforts
void Hand::stateCallback(const sensor_msgs::JointState& msg){
  left_effort = msg.effort[1];
  right_effort = msg.effort[0];
}


bool Hand::slide_left_down(gripper_controls::PositionCommand::Request &req, gripper_controls::PositionCommand::Response &res){
  ros::NodeHandle n;
  // predetermined max effort is always the same (a negative value, towards closing direction)
  // and loaded from parameter server
  float torque_val{0};
  n.getParam("/high_level/finger_torque", torque_val);
  bool set_modes, set_friction_l, set_friction_r, send_torque, send_pos;
  // Set modes - Left -> Position(3), Right -> Effort(0)
  int modes[] = {3,0};
  set_modes = set_actuator_modes(2,modes);
  // if (finger_state != 1){
  //   // Set Friction Surfaces - Left -> Low, Right -> High
  //   set_friction_l = set_friction_left(false);
  //   set_friction_r = set_friction_right(true);
  //   finger_state = 1;
  // }
  set_friction_l = set_friction_left(false);
  set_friction_r = set_friction_right(true);
  // ramped torque command
  float initial_torque;
  // position controller can end with a positive effort (towards opening direction)
  // this statement is to prevent it
  if (right_effort>0)
    initial_torque = 0;
  else
    initial_torque = right_effort;
  ROS_INFO("effort reading %f",initial_torque);
  for (float i = initial_torque; i>=torque_val; i-=0.01){
    send_torque = command_torque(1, i);
  }
  // ramped position command
  if (send_torque){
    float initial_position;
    // read the position of the other finger and multiply it with -1 to get the initial position
    initial_position = -1*read_position(1);
    for (float i = initial_position; i <= req.data; i = i + 0.01){
      send_pos = command_position(0,i);
    }
    if (send_pos)
      return 1;
    else{
      ROS_ERROR("Sending Position Values Failed");
      return 0;
    }
  }
  else{
    ROS_ERROR("Sending Torque Values Failed");
    return 0;
  }
}

bool Hand::slide_left_up(gripper_controls::PositionCommand::Request &req, gripper_controls::PositionCommand::Response &res){
  ros::NodeHandle n;
  float torque_val{0};
  n.getParam("/high_level/finger_torque", torque_val);
  bool set_modes, set_friction_l, set_friction_r, send_torque, send_pos;
  // Set modes - Left -> Torque(0), Right -> Position(3)
  int modes[] = {0,3};
  set_modes = set_actuator_modes(2,modes);
  // if (finger_state != 1){
  //   // Set Friction Surfaces - Left -> Low, Right -> High
  //   set_friction_l = set_friction_left(false);
  //   set_friction_r = set_friction_right(true);
  //   finger_state = 1;
  // }
  set_friction_l = set_friction_left(false);
  set_friction_r = set_friction_right(true);
  // ramped torque command
  float initial_torque;
  if (left_effort>0)
    initial_torque = 0;
  else
    initial_torque = left_effort;
  ROS_INFO("effort reading %f",initial_torque);
  for (float i = initial_torque; i>=torque_val; i-=0.01){
    send_torque = command_torque(0, i);
  }
  // ramped position command
  if (send_torque){
    float initial_position;
    initial_position = -1*read_position(2);
    for (float i = initial_position; i <= req.data; i = i + 0.01){
      send_pos = command_position(1,i);
    }
    if (send_pos)
      return 1;
    else{
      ROS_ERROR("Sending Position Values Failed");
      return 0;
    }
  }
  else{
    ROS_ERROR("Sending Torque Values Failed");
    return 0;
  }
}

bool Hand::slide_right_down(gripper_controls::PositionCommand::Request &req, gripper_controls::PositionCommand::Response &res){
  ros::NodeHandle n;
  float torque_val{0};
  n.getParam("/high_level/finger_torque", torque_val);
  bool set_modes, set_friction_l, set_friction_r, send_torque, send_pos;
  // Set modes - Left -> Effort(0), Right -> Position(3)
  int modes[] = {0,3};
  set_modes = set_actuator_modes(2,modes);
  // if (finger_state != 2){
  //   // Set Friction Surfaces - Left -> High, Right -> Low
  //   set_friction_l = set_friction_left(true);
  //   set_friction_r = set_friction_right(false);
  //   finger_state = 2;
  // }
  set_friction_l = set_friction_left(true);
  set_friction_r = set_friction_right(false);
  // ramped torque command
  float initial_torque;
  if (left_effort>0)
    initial_torque = 0;
  else
    initial_torque = left_effort;
  ROS_INFO("effort reading %f",initial_torque);
  for (float i = initial_torque; i>=torque_val; i-=0.01){
    send_torque = command_torque(0, i);
  }
  // ramped position command
  if (send_torque){
    float initial_position;
    initial_position = -1*read_position(2);
    for (float i = initial_position; i <= req.data; i = i + 0.01){
      send_pos = command_position(1,i);
    }
    if (send_pos)
      return 1;
    else{
      ROS_ERROR("Sending Position Values Failed");
      return 0;
    }
  }
  else{
    ROS_ERROR("Sending Torque Values Failed");
    return 0;
  }
}

bool Hand::slide_right_up(gripper_controls::PositionCommand::Request &req, gripper_controls::PositionCommand::Response &res){
  ros::NodeHandle n;
  float torque_val{0};
  n.getParam("/high_level/finger_torque", torque_val);
  bool set_modes, set_friction_l, set_friction_r, send_torque, send_pos;
  // Set modes - Left -> Position(3), Right -> Effort(0)
  int modes[] = {3,0};
  set_modes = set_actuator_modes(2,modes);
  // if (finger_state != 2){
  //   // Set Friction Surfaces - Left -> High, Right -> Low
  //   set_friction_l = set_friction_left(true);
  //   set_friction_r = set_friction_right(false);
  //   finger_state = 2;
  // }
  set_friction_l = set_friction_left(true);
  set_friction_r = set_friction_right(false);
  // ramped torque command
  float initial_torque;
  // position controller can end with a positive effort (towards opening direction)
  // this statement is to prevent it
  if (right_effort>0)
    initial_torque = 0;
  else
    initial_torque = right_effort;
  ROS_INFO("effort reading %f",initial_torque);
  for (float i = initial_torque; i>=torque_val; i-=0.01){
    send_torque = command_torque(1, i);
  }
  // ramped position command
  if (send_torque){
    float initial_position;
    initial_position = -1*read_position(1);
    for (float i = initial_position; i <= req.data; i = i + 0.01){
      send_pos = command_position(0,i);
    }
    if (send_pos)
      return 1;
    else{
      ROS_ERROR("Sending Position Values Failed");
      return 0;
    }
  }
  else{
    ROS_ERROR("Sending Torque Values Failed");
    return 0;
  }
  return 1;
}

bool Hand::rotate_anticlockwise(gripper_controls::PositionCommand::Request &req, gripper_controls::PositionCommand::Response &res){
  ros::NodeHandle n;
  float torque_val{0};
  n.getParam("/high_level/finger_torque", torque_val);
  bool set_modes, set_friction_l, set_friction_r, send_torque, send_pos;
  // Set modes - Left -> Effort(0), Right -> Position(3)
  int modes[] = {0,3};
  set_modes = set_actuator_modes(2,modes);
  // if (finger_state != 3){
  //   // Set Friction Surfaces - Left -> High, Right -> High
  //   set_friction_l = set_friction_left(true);
  //   set_friction_r = set_friction_right(true);
  //   finger_state = 3;
  // }
  set_friction_l = set_friction_left(true);
  set_friction_r = set_friction_right(true);
  // ramped torque command
  float initial_torque;
  // initial_torque = read_effort(1); // Left finger
  initial_torque = left_effort;
  ROS_INFO("init torq %f",initial_torque);
  for (float i = initial_torque; i>=torque_val; i-=0.01){
    send_torque = command_torque(0, i);
  }
  // ramped position command
  if (send_torque){
    float initial_position;
    initial_position = -1*read_position(2);
    for (float i = initial_position; i <= req.data; i = i + 0.01){
      send_pos = command_position(1,i);
    }
    if (send_pos)
      return 1;
    else{
      ROS_ERROR("Sending Position Values Failed");
      return 0;
    }
  }
  else{
    ROS_ERROR("Sending Torque Values Failed");
    return 0;
  }
}

bool Hand::rotate_clockwise(gripper_controls::PositionCommand::Request &req, gripper_controls::PositionCommand::Response &res){
  ros::NodeHandle n;
  float torque_val{0};
  n.getParam("/high_level/finger_torque", torque_val);
  bool set_modes, set_friction_l, set_friction_r, send_torque, send_pos;
  // Set modes - Left -> Position(3), Right -> Effort(0)
  int modes[] = {3,0};
  set_modes = set_actuator_modes(2,modes);
  // if (finger_state != 3){
  //   // Set Friction Surfaces - Left -> High, Right -> High
  //   set_friction_l = set_friction_left(true);
  //   set_friction_r = set_friction_right(true);
  //   finger_state = 3;
  // }
  set_friction_l = set_friction_left(true);
  set_friction_r = set_friction_right(true);
  // ramped torque command
  float initial_torque;
  // initial_torque = read_effort(0); // Right finger
  initial_torque = right_effort;
  for (float i = initial_torque; i>=torque_val; i-=0.01){
    send_torque = command_torque(1, i);
  }
  // ramped position command
  if (send_torque){
    float initial_position;
    initial_position = -1*read_position(1);
    for (float i = initial_position; i <= req.data; i = i + 0.01){
      send_pos = command_position(0,i);
    }
    if (send_pos)
      return 1;
    else{
      ROS_ERROR("Sending Position Values Failed");
      return 0;
    }
  }
  else{
    ROS_ERROR("Sending Torque Values Failed");
    return 0;
  }
}

bool Hand::hold_object(gripper_controls::Holdcommand::Request &req, gripper_controls::Holdcommand::Response &res){
  bool set_modes, set_friction_l, set_friction_r, send_pos1, send_pos2;
  // Set modes - Left -> Position(3), Right -> Position(3)
  int modes1[] = {3,3};
  set_modes = set_actuator_modes(2,modes1);
  // if (finger_state != 3){
  //   set_friction_l = set_friction_left(true);
  //   set_friction_r = set_friction_right(true);
  //   finger_state = 3;
  // }
  set_friction_l = set_friction_left(true);
  set_friction_r = set_friction_right(true);
  send_pos1 = command_position(0, req.left);
  send_pos2 = command_position(1, req.right);

  ros::Duration(1).sleep();

  if (send_pos1 && send_pos2)
    return true;
  else
    return false;
}
