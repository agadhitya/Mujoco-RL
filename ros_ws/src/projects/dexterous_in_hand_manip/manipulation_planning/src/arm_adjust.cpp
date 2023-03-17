#include "manipulation_planning/ArmAdjust.h"
#include <ros/ros.h>
#include <gazebo_msgs/GetModelState.h>
#include <geometry_msgs/Pose.h>
#include <arm_controls/MoveStraight.h>

bool adjust(manipulation_planning::ArmAdjust::Request &req, manipulation_planning::ArmAdjust::Response &res){
  ros::NodeHandle n_;
  ros::service::waitForService("gazebo/get_model_state");
  ros::ServiceClient client_pose = n_.serviceClient<gazebo_msgs::GetModelState>("gazebo/get_model_state");
  gazebo_msgs::GetModelState srv;
  srv.request.model_name = "object";
  srv.request.relative_entity_name = "panda_link7";
  client_pose.call(srv);
  geometry_msgs::Pose current_pose{srv.response.pose};
  float x_current{current_pose.position.x};
  float x_des{0};
  x_des = 0.033 - ((8.5 - req.z)*(0.033 - (-0.027)))/(8.5-2.5);
  if (x_current > x_des){
    ros::service::waitForService("move_down");
    ros::ServiceClient client_down = n_.serviceClient<arm_controls::MoveStraight>("move_down");
    arm_controls::MoveStraight srv_down;
    srv_down.request.val = x_current - x_des;
    client_down.call(srv_down);
  }
  else{
    ros::service::waitForService("move_up");
    ros::ServiceClient client_up = n_.serviceClient<arm_controls::MoveStraight>("move_up");
    arm_controls::MoveStraight srv_up;
    srv_up.request.val = x_des - x_current;
    client_up.call(srv_up);
  }
  float z_current{current_pose.position.z};
  float z_des{0};
  z_des = 0.28 - ((9.5 - req.d)*(0.28 - (0.22)))/(9.5-3.0);
  if (z_current > z_des){
    ros::service::waitForService("move_forward");
    ros::ServiceClient client_forward = n_.serviceClient<arm_controls::MoveStraight>("move_forward");
    arm_controls::MoveStraight srv_forward;
    srv_forward.request.val = z_current - z_des;
    client_forward.call(srv_forward);
  }
  else{
    ros::service::waitForService("move_back");
    ros::ServiceClient client_back = n_.serviceClient<arm_controls::MoveStraight>("move_back");
    arm_controls::MoveStraight srv_back;
    srv_back.request.val = z_des - z_current;
    client_back.call(srv_back);
  }
}

int main(int argc, char **argv){
  ros::init(argc, argv, "arm_adjust_server");
  ros::NodeHandle n;
  ros::ServiceServer arm_adj = n.advertiseService("adjust_arm", adjust);
  ros::spin();
  return 0;
}
