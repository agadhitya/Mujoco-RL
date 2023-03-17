#include "gripper_controls/Holdcommand.h"
#include "gripper_controls/PositionCommand.h"
#include <ros/ros.h>

int main(int argc, char **argv){
  ros::init(argc,argv,"Sequential_command_node");
  ros::NodeHandle n;
  // Wait for services and get client handles:
  ros::service::waitForService("Hold_object");
  ros::ServiceClient client_hold_obj = n.serviceClient<gripper_controls::Holdcommand>("Hold_object");
  ros::service::waitForService("Rotate_clockwise");
  ros::ServiceClient client_rot_clock = n.serviceClient<gripper_controls::PositionCommand>("Rotate_clockwise");
  ros::service::waitForService("Rotate_anticlockwise");
  ros::ServiceClient client_rot_anti = n.serviceClient<gripper_controls::PositionCommand>("Rotate_anticlockwise");
  ros::service::waitForService("Slide_Right_Finger_Up");
  ros::ServiceClient client_srfu = n.serviceClient<gripper_controls::PositionCommand>("Slide_Right_Finger_Up");
  ros::service::waitForService("Slide_Right_Finger_Down");
  ros::ServiceClient client_srfd = n.serviceClient<gripper_controls::PositionCommand>("Slide_Right_Finger_Down");
  ros::service::waitForService("Slide_Left_Finger_Up");
  ros::ServiceClient client_slfu = n.serviceClient<gripper_controls::PositionCommand>("Slide_Left_Finger_Up");
  ros::service::waitForService("Slide_Left_Finger_Down");
  ros::ServiceClient client_slfd = n.serviceClient<gripper_controls::PositionCommand>("Slide_Left_Finger_Down");

  ros::Duration d(3);
  // 1. Hold object
  gripper_controls::Holdcommand srv_hold;
  srv_hold.request.left = 0.0;
  srv_hold.request.right = 0.0;
  if (client_hold_obj.call(srv_hold)){
    ROS_INFO("Command hold object success");
  }
  else{
    ROS_INFO("Failure holding object");
  }
  d.sleep();

  // 2. Slide left finger down 0.5
  gripper_controls::PositionCommand srv_slfd;
  srv_slfd.request.data = 0.5;
  if (client_slfd.call(srv_slfd)){
    ROS_INFO("Command slide left finger down success");
  }
  else{
    ROS_INFO("Failure slfd");
  }
  d.sleep();

  // 3. Slide left finger up 0
  gripper_controls::PositionCommand srv_slfu;
  srv_slfu.request.data = 0.0;
  if (client_slfu.call(srv_slfu)){
    ROS_INFO("Command slide left finger up success");
  }
  else{
    ROS_INFO("Failure slfu");
  }
  d.sleep();

  // 4. Slide right finger down 0.5
  gripper_controls::PositionCommand srv_srfd;
  srv_srfd.request.data = 0.5;
  if (client_srfd.call(srv_srfd)){
    ROS_INFO("Command slide right finger down success");
  }
  else{
    ROS_INFO("Failure srfd");
  }
  d.sleep();

  // 5. Slide right finger up 0
  gripper_controls::PositionCommand srv_srfu;
  srv_srfu.request.data = 0.0;
  if (client_srfu.call(srv_srfu)){
    ROS_INFO("Command slide right finger up success");
  }
  else{
    ROS_INFO("Failure srfu");
  }
  d.sleep();

  // // 6. Rotate clockwise
  // gripper_controls::PositionCommand srv_rot_clock;
  // srv_rot_clock.request.data = 0.5;
  // if (client_rot_clock.call(srv_rot_clock)){
  //   ROS_INFO("Command rotate clockwise success");
  // }
  // else{
  //   ROS_INFO("Failure rotate clockwise");
  // }
  //
  // // 7. Rotate anticlockwise
  // gripper_controls::PositionCommand srv_rot_anti;
  // srv_rot_anti.request.data = 0.0;
  // if (client_rot_anti.call(srv_rot_anti)){
  //   ROS_INFO("Command rotate anticlockwise success");
  // }
  // else{
  //   ROS_INFO("Failure rotate anticlockwise");
  // }




  return 0;
}
