#include "manipulation_planning/ManipulationActions.h"
#include <arm_controls/PoseChange.h>
#include <arm_controls/MoveStraight.h>
#include <gripper_controls/SetFriction.h>
#include <gripper_controls/PositionCommand.h>
#include "ros/ros.h"
#include <manipulation_planning/HandAct.h>
#include <manipulation_planning/ArmAct.h>

ManipulationActions::ManipulationActions(){
  ros::Duration d{2};
  dur = d;
  std::cout << "ManipulationActions object constructed" << std::endl;
}

bool ManipulationActions::move_up(float val){
  ros::service::waitForService("move_up");
  ros::ServiceClient client = n_.serviceClient<arm_controls::MoveStraight>("move_up");
  arm_controls::MoveStraight srv;
  srv.request.val = val;
  client.call(srv);
  return 1;
}

bool ManipulationActions::move_down(float val){
  ros::service::waitForService("move_down");
  ros::ServiceClient client = n_.serviceClient<arm_controls::MoveStraight>("move_down");
  arm_controls::MoveStraight srv;
  srv.request.val = val;
  client.call(srv);
  return 1;
}

bool ManipulationActions::decrease_friction(){
  ros::service::waitForService("set_friction");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::SetFriction>("set_friction");
  gripper_controls::SetFriction srv_r;
  srv_r.request.finger = 1;
  srv_r.request.high_friction = false;
  client.call(srv_r);
  gripper_controls::SetFriction srv_l;
  srv_l.request.finger = 0;
  srv_l.request.high_friction = false;
  client.call(srv_l);
  return 1;
}

bool ManipulationActions::increase_friction(){
  ros::service::waitForService("set_friction");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::SetFriction>("set_friction");
  gripper_controls::SetFriction srv_r;
  srv_r.request.finger = 1;
  srv_r.request.high_friction = true;
  client.call(srv_r);
  gripper_controls::SetFriction srv_l;
  srv_l.request.finger = 0;
  srv_l.request.high_friction = true;
  client.call(srv_l);
  return 1;
}

bool ManipulationActions::slide_left_finger_down(float val){
  ros::service::waitForService("Slide_Left_Finger_Down");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::PositionCommand>("Slide_Left_Finger_Down");
  gripper_controls::PositionCommand srv;
  srv.request.data = val;
  client.call(srv);
  return 1;
}

bool ManipulationActions::slide_left_finger_up(float val){
  ros::service::waitForService("Slide_Left_Finger_Up");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::PositionCommand>("Slide_Left_Finger_Up");
  gripper_controls::PositionCommand srv;
  srv.request.data = val;
  client.call(srv);
  return 1;
}

bool ManipulationActions::slide_right_finger_down(float val){
  ros::service::waitForService("Slide_Right_Finger_Down");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::PositionCommand>("Slide_Right_Finger_Down");
  gripper_controls::PositionCommand srv;
  srv.request.data = val;
  client.call(srv);
  return 1;
}

bool ManipulationActions::slide_right_finger_up(float val){
  ros::service::waitForService("Slide_Right_Finger_Up");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::PositionCommand>("Slide_Right_Finger_Up");
  gripper_controls::PositionCommand srv;
  srv.request.data = val;
  client.call(srv);
  return 1;
}

bool ManipulationActions::rotate_clockwise(float val){
  ros::service::waitForService("Rotate_clockwise");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::PositionCommand>("Rotate_clockwise");
  gripper_controls::PositionCommand srv;
  srv.request.data = val;
  client.call(srv);
  return 1;
}



bool ManipulationActions::rotate_anticlockwise(float val){
  ros::service::waitForService("Rotate_anticlockwise");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::PositionCommand>("Rotate_anticlockwise");
  gripper_controls::PositionCommand srv;
  srv.request.data = val;
  client.call(srv);
  return 1;
}

bool ManipulationActions::slide_left_down(manipulation_planning::HandAct::Request &req, manipulation_planning::HandAct::Response &res){
  bool slide;
  slide = slide_left_finger_down(req.val);
  dur.sleep();
  return 1;
}

bool ManipulationActions::slide_left_up(manipulation_planning::HandAct::Request &req, manipulation_planning::HandAct::Response &res){
  bool slide;
  slide = slide_left_finger_up(req.val);
  dur.sleep();
  return 1;
}

bool ManipulationActions::slide_right_down(manipulation_planning::HandAct::Request &req, manipulation_planning::HandAct::Response &res){
  bool slide;
  slide = slide_right_finger_down(req.val);
  dur.sleep();
  return 1;
}

bool ManipulationActions::slide_right_up(manipulation_planning::HandAct::Request &req, manipulation_planning::HandAct::Response &res){
  bool slide;
  slide = slide_right_finger_up(req.val);
  dur.sleep();
  return 1;
}

bool ManipulationActions::rotate_cw(manipulation_planning::HandAct::Request &req, manipulation_planning::HandAct::Response &res){
  bool slide;
  slide = rotate_clockwise(req.val);
  dur.sleep();
  return 1;
}

bool ManipulationActions::rotate_acw(manipulation_planning::HandAct::Request &req, manipulation_planning::HandAct::Response &res){
  bool slide;
  slide = rotate_anticlockwise(req.val);
  dur.sleep();
  return 1;
}

bool ManipulationActions::move_contact_up(manipulation_planning::ArmAct::Request &req, manipulation_planning::ArmAct::Response &res){
  // bool high_friction1, up, low_friction, high_friction2;
  // high_friction1 = increase_friction();
  // up = move_up();
  // low_friction = decrease_friction();
  // dur.sleep();
  // high_friction2 = increase_friction();
  bool low_friction, up, high_friction;
  low_friction = decrease_friction();
  // dur.sleep();
  up = move_up(req.val);
  dur.sleep();
  high_friction = increase_friction();
  return 1;
}

bool ManipulationActions::move_contact_down(manipulation_planning::ArmAct::Request &req, manipulation_planning::ArmAct::Response &res){
  bool low_friction, down, high_friction;
  low_friction = decrease_friction();
  down = move_down(req.val);
  high_friction = increase_friction();
  return 1;
}
