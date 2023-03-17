#include <friction_finger_gripper/controller_client.hpp>
#include <friction_finger_gripper/PositionCommand.h>
#include <friction_finger_gripper/Holdcommand.h>
#include <friction_finger_gripper/ActionCommand.h>


bool slide_right_finger_up(friction_finger_gripper::ActionCommand::Request &req,
                    friction_finger_gripper::ActionCommand::Response &res)
{
  ros::NodeHandle n1;
  res.success = slide_right_up(n1,req.position, req.switch_surface);
  return true;
}

bool slide_right_finger_down(friction_finger_gripper::ActionCommand::Request &req,
                    friction_finger_gripper::ActionCommand::Response &res)
{
  ros::NodeHandle n1;
  res.success = slide_right_down(n1,req.position, req.switch_surface);
  return true;
}

bool slide_left_finger_up(friction_finger_gripper::ActionCommand::Request &req,
                    friction_finger_gripper::ActionCommand::Response &res)
{
  ros::NodeHandle n1;
  res.success = slide_left_up(n1,req.position, req.switch_surface);
  return true;
}

bool slide_left_finger_down(friction_finger_gripper::ActionCommand::Request &req,
                    friction_finger_gripper::ActionCommand::Response &res)
{
  ros::NodeHandle n1;
  res.success = slide_left_down(n1,req.position, req.switch_surface);
  return true;
}

bool rotate_clock(friction_finger_gripper::ActionCommand::Request &req,
                    friction_finger_gripper::ActionCommand::Response &res)
{
  ros::NodeHandle n1;
  res.success = rotate_clockwise(n1,req.position, req.switch_surface);
  return true;
}

bool rotate_anticlock(friction_finger_gripper::ActionCommand::Request &req,
                    friction_finger_gripper::ActionCommand::Response &res)
{
  ros::NodeHandle n1;
  res.success = rotate_anticlockwise(n1,req.position, req.switch_surface);
  return true;
}

bool home_position(friction_finger_gripper::PositionCommand::Request &req,
                    friction_finger_gripper::PositionCommand::Response &res)
{
  ros::NodeHandle n1;
  res.success = Home_position(n1);
  return true;
}

//Need to be fixed
bool hold_object1(friction_finger_gripper::Holdcommand::Request &req,
                  friction_finger_gripper::Holdcommand::Response &res)
{
  bool success;
  ros::NodeHandle n1;
  success = hold_object(n1,req.left,req.right);
  return true;
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "High_level_actions_server");
  ros::NodeHandle n;

  ros::ServiceServer service_1 = n.advertiseService("Slide_Right_Finger_Up", slide_right_finger_up);
  
  ros::ServiceServer service_2 = n.advertiseService("Slide_Right_Finger_Down", slide_right_finger_down);
  
  ros::ServiceServer service_3 = n.advertiseService("Slide_Left_Finger_Up", slide_left_finger_up);
  
  ros::ServiceServer service_4 = n.advertiseService("Slide_Left_Finger_Down", slide_left_finger_down);
  
  ros::ServiceServer service_5 = n.advertiseService("Rotate_clockwise", rotate_clock);
  
  ros::ServiceServer service_6 = n.advertiseService("Rotate_anticlockwise", rotate_anticlock);
  
   ros::ServiceServer service_7 = n.advertiseService("Home_position", home_position);
  
   ros::ServiceServer service_8 = n.advertiseService("Hold_object", hold_object1);
  ros::spin();

  return 0;
}
