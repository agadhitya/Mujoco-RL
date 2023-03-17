#include "gripper_controls/Holdcommand.h"
#include "gripper_controls/PositionCommand.h"
#include <ros/ros.h>
#include "arm_controls/PoseChange.h"
#include "arm_controls/MoveStraight.h"
#include <std_srvs/Empty.h>

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
  ros::service::waitForService("initialize_arm_pose");
  ros::ServiceClient client_init_pose = n.serviceClient<arm_controls::PoseChange>("initialize_arm_pose");
  ros::service::waitForService("grasp_pose");
  ros::ServiceClient client_grasp_pose = n.serviceClient<arm_controls::PoseChange>("grasp_pose");
  ros::service::waitForService("rest_pose");
  ros::ServiceClient client_rest_pose = n.serviceClient<arm_controls::PoseChange>("rest_pose");
  ros::service::waitForService("move_up");
  ros::ServiceClient client_move_up = n.serviceClient<arm_controls::MoveStraight>("move_up");
  ros::service::waitForService("move_down");
  ros::ServiceClient client_move_down = n.serviceClient<arm_controls::MoveStraight>("move_down");
  ros::service::waitForService("/gazebo/reset_world");
  ros::ServiceClient client_world_reset = n.serviceClient<std_srvs::Empty>("/gazebo/reset_world");



  ros::Rate loop_rate(1);
  // 1. Open fingers
  gripper_controls::Holdcommand srv_hold;
  srv_hold.request.left = 0.5;
  srv_hold.request.right = 0.5;
  if (client_hold_obj.call(srv_hold)){
    ROS_INFO("Command hold object success");
  }
  else{
    ROS_INFO("Failure holding object");
  }
  // 2. Move to rest pose:
  arm_controls::PoseChange srv_pose;
  srv_pose.request.execute = true;
  client_rest_pose.call(srv_pose);
  // 3. Reset world:
  std_srvs::Empty srv_reset;
  client_world_reset.call(srv_reset);
  // 4. Move to init pose:
  client_init_pose.call(srv_pose);
  // 5. Move to rest pose:
  client_grasp_pose.call(srv_pose);

  // 1. Hold object
  srv_hold.request.left = 0.01;
  srv_hold.request.right = 0.01;
  if (client_hold_obj.call(srv_hold)){
    ROS_INFO("Command hold object success");
  }
  else{
    ROS_INFO("Failure holding object");
  }
  loop_rate.sleep();

  //
  // // 2. Slide left finger down 0.5
  // gripper_controls::PositionCommand srv_slfd;
  // srv_slfd.request.data = 0.3;
  // if (client_slfd.call(srv_slfd)){
  //   ROS_INFO("Command slide left finger down success");
  // }
  // else{
  //   ROS_INFO("Failure slfd");
  // }
  // loop_rate.sleep();
  //
  // // 3. Slide left finger up 0
  // gripper_controls::PositionCommand srv_slfu;
  // srv_slfu.request.data = 0.0;
  // if (client_slfu.call(srv_slfu)){
  //   ROS_INFO("Command slide left finger up success");
  // }
  // else{
  //   ROS_INFO("Failure slfu");
  // }
  // loop_rate.sleep();
  //
  // // 4. Slide right finger down 0.5
  // gripper_controls::PositionCommand srv_srfd;
  // srv_srfd.request.data = 0.3;
  // if (client_srfd.call(srv_srfd)){
  //   ROS_INFO("Command slide right finger down success");
  // }
  // else{
  //   ROS_INFO("Failure srfd");
  // }
  // loop_rate.sleep();
  //
  // // 5. Slide right finger up 0
  // gripper_controls::PositionCommand srv_srfu;
  // srv_srfu.request.data = 0.0;
  // if (client_srfu.call(srv_srfu)){
  //   ROS_INFO("Command slide right finger up success");
  // }
  // else{
  //   ROS_INFO("Failure srfu");
  // }
  // loop_rate.sleep();

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
