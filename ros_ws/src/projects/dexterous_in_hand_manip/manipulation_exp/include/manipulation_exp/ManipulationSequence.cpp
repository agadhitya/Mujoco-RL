#include "manipulation_exp/ManipulationSequence.h"
#include <std_srvs/Empty.h>
#include <gripper_controls/Holdcommand.h>
#include <arm_controls/PoseChange.h>
#include <arm_controls/MoveStraight.h>
#include <gripper_controls/SetFriction.h>
#include <gripper_controls/PositionCommand.h>
#include "ros/ros.h"
#include <manipulation_exp/Sequence.h>

ManipulationSequence::ManipulationSequence(){
  ros::Duration d{3};
  dur = d;
  std::cout << "ManipulationSequence object constructed" << std::endl;
}

bool ManipulationSequence::reset_world(){
  ros::service::waitForService("/gazebo/reset_world");
  ros::ServiceClient client = n_.serviceClient<std_srvs::Empty>("/gazebo/reset_world");
  std_srvs::Empty srv;
  client.call(srv);
  return 1;
}

bool ManipulationSequence::release_object(){
  ros::service::waitForService("Hold_object");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::Holdcommand>("Hold_object");
  gripper_controls::Holdcommand srv;
  srv.request.left = 0.5;
  srv.request.right = 0.5;
  client.call(srv);
  return 1;
}

bool ManipulationSequence::go_to_rest(){
  ros::service::waitForService("rest_pose");
  ros::ServiceClient client = n_.serviceClient<arm_controls::PoseChange>("rest_pose");
  arm_controls::PoseChange srv;
  srv.request.execute = true;
  client.call(srv);
  return 1;
}

bool ManipulationSequence::initialize_arm(){
  ros::service::waitForService("initialize_arm_pose");
  ros::ServiceClient client = n_.serviceClient<arm_controls::PoseChange>("initialize_arm_pose");
  arm_controls::PoseChange srv;
  srv.request.execute = true;
  client.call(srv);
  return 1;
}

bool ManipulationSequence::approach_object(){
  ros::service::waitForService("grasp_pose");
  ros::ServiceClient client = n_.serviceClient<arm_controls::PoseChange>("grasp_pose");
  arm_controls::PoseChange srv;
  srv.request.execute = true;
  client.call(srv);
  return 1;
}

bool ManipulationSequence::grasp_object(){
  ros::service::waitForService("Hold_object");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::Holdcommand>("Hold_object");
  gripper_controls::Holdcommand srv;
  srv.request.left = 0.01;
  srv.request.right = 0.01;
  client.call(srv);
  return 1;
}

bool ManipulationSequence::move_up(float val){
  ros::service::waitForService("move_up");
  ros::ServiceClient client = n_.serviceClient<arm_controls::MoveStraight>("move_up");
  arm_controls::MoveStraight srv;
  srv.request.val = val;
  client.call(srv);
  return 1;
}

bool ManipulationSequence::move_down(float val){
  ros::service::waitForService("move_down");
  ros::ServiceClient client = n_.serviceClient<arm_controls::MoveStraight>("move_down");
  arm_controls::MoveStraight srv;
  srv.request.val = val;
  client.call(srv);
  return 1;
}

bool ManipulationSequence::decrease_friction(){
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

bool ManipulationSequence::increase_friction(){
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

bool ManipulationSequence::slide_left_down(){
  ros::service::waitForService("Slide_Left_Finger_Down");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::PositionCommand>("Slide_Left_Finger_Down");
  gripper_controls::PositionCommand srv;
  srv.request.data = 0.5;
  client.call(srv);
  return 1;
}

bool ManipulationSequence::slide_left_up(){
  ros::service::waitForService("Slide_Left_Finger_Up");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::PositionCommand>("Slide_Left_Finger_Up");
  gripper_controls::PositionCommand srv;
  srv.request.data = 0.0;
  client.call(srv);
  return 1;
}

bool ManipulationSequence::slide_right_down(){
  ros::service::waitForService("Slide_Right_Finger_Down");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::PositionCommand>("Slide_Right_Finger_Down");
  gripper_controls::PositionCommand srv;
  srv.request.data = 0.5;
  client.call(srv);
  return 1;
}

bool ManipulationSequence::slide_right_up(){
  ros::service::waitForService("Slide_Right_Finger_Up");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::PositionCommand>("Slide_Right_Finger_Up");
  gripper_controls::PositionCommand srv;
  srv.request.data = 0.0;
  client.call(srv);
  return 1;
}

bool ManipulationSequence::rotate_clockwise(){
  ros::service::waitForService("Rotate_clockwise");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::PositionCommand>("Rotate_clockwise");
  gripper_controls::PositionCommand srv;
  srv.request.data = 0.5;
  client.call(srv);
  return 1;
}



bool ManipulationSequence::rotate_anticlockwise(){
  ros::service::waitForService("Rotate_anticlockwise");
  ros::ServiceClient client = n_.serviceClient<gripper_controls::PositionCommand>("Rotate_anticlockwise");
  gripper_controls::PositionCommand srv;
  srv.request.data = 0.5;
  client.call(srv);
  return 1;
}

bool ManipulationSequence::reset_sequence(manipulation_exp::Sequence::Request &req, manipulation_exp::Sequence::Response &res){
  bool release, rest, reset;
  release = release_object();
  rest = go_to_rest();
  reset = reset_world();
  return 1;
}

bool ManipulationSequence::prepare_grasp(manipulation_exp::Sequence::Request &req, manipulation_exp::Sequence::Response &res){
  bool initialize, approach;
  initialize = initialize_arm();
  approach = approach_object();
  return 1;
}

bool ManipulationSequence::object_grasp(manipulation_exp::Sequence::Request &req, manipulation_exp::Sequence::Response &res){
  bool grasp;
  grasp = grasp_object();
  return 1;
}

bool ManipulationSequence::gravity_exploit(manipulation_exp::Sequence::Request &req, manipulation_exp::Sequence::Response &res){
  bool up, low_friction, high_friction;
  up = move_up(0.01);
  low_friction = decrease_friction();
  high_friction = increase_friction();
  return 1;
}

bool ManipulationSequence::prehensile_pushing(manipulation_exp::Sequence::Request &req, manipulation_exp::Sequence::Response &res){
  bool low_friction, down, high_friction, up;
  low_friction = decrease_friction();
  down = move_down(0.01);
  high_friction = increase_friction();
  // up = move_up();
  return 1;
}

bool ManipulationSequence::sliding_test(manipulation_exp::Sequence::Request &req, manipulation_exp::Sequence::Response &res){
  bool sld, slu, srd, sru;
  sld = slide_left_down();
  dur.sleep();
  slu = slide_left_up();
  dur.sleep();
  srd = slide_right_down();
  dur.sleep();
  sru = slide_right_up();
  return 1;
}

bool ManipulationSequence::rotating_test(manipulation_exp::Sequence::Request &req, manipulation_exp::Sequence::Response &res){
  bool rc, rac;
  rc = rotate_clockwise();
  dur.sleep();
  rac = rotate_anticlockwise();
  return 1;
}
