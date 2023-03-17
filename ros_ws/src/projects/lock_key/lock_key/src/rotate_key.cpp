#include "ros/ros.h"
#include "moveit_planner/GetTF.h"
#include "moveit_planner/MoveCart.h"
#include <lock_key_msgs/RotateKeyAction.h> // Note: "Action" is appended
#include <actionlib/server/simple_action_server.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <iostream>
#include <cmath>
using namespace std;

class RotateKey{
public:
  RotateKey(string name) : 
    as(nh, name, false),
    action_name(name)
  { //register goal and feedback callbacks
    ROS_INFO("Initializing RotateKey Action Server.");
    as.registerGoalCallback(boost::bind(&RotateKey::goalCB, this));
    as.registerPreemptCallback(boost::bind(&RotateKey::preemptCB, this));
    //Get parameters
    //nh.getParam("spiral/delta_max",delta_max);
    //Wait for services
    ros::service::waitForService("get_transform",timeout);
    ros::service::waitForService("cartesian_move",timeout);
    getTFClient = nh.serviceClient<moveit_planner::GetTF>("get_transform");
    moveCartClient = nh.serviceClient<moveit_planner::MoveCart>("cartesian_move");

    //Get current TF from world to EE
    curTF.request.from="map"; //map
    curTF.request.to="panda_link8"; //end_effector_link

    ROS_INFO("RotateKey Server starting now.");
    as.start();
  }

  ~RotateKey(void){
  }

  void rotateRel(double roll, double pitch, double yaw){
    //Get current TF from world to EE
    getTFClient.call(curTF);
    p=curTF.response.pose;
    p_rot=curTF.response.pose;

    //Convert RPY to quat
    tf2::Quaternion q_orig, q_change, q_new;
    tf2::convert(p_rot.orientation , q_orig);
    q_change.setRPY(roll, pitch, yaw);
    q_new=q_change*q_orig;
    q_new.normalize();
    tf2::convert(q_new, p_rot.orientation);

    //Execute
    cart.request.val.clear();
    cart.request.val.push_back(p_rot);    
    cart.request.val.push_back(p);
    cart.request.execute = true;
    moveCartClient.call(cart);
  }

  void goalCB(){
    auto goal=as.acceptNewGoal();

    // make sure that the action hasn't been canceled
    if (!as.isActive())
      return;

    ROS_INFO("Beginning rotation.");
    //Perform main task
    rotateRel(goal->d_roll,goal->d_pitch,goal->d_yaw);

    //Set result
    as.setSucceeded();
  }

  void preemptCB(){
    ROS_INFO("%s: Preempted", action_name.c_str());
    // set the action state to preempted
    as.setPreempted();
  }

private:
  ros::NodeHandle nh;
  actionlib::SimpleActionServer<lock_key_msgs::RotateKeyAction> as;
  lock_key_msgs::RotateKeyFeedback feedback;
  lock_key_msgs::RotateKeyResult result;
  string action_name;
  ros::ServiceClient getTFClient;
  ros::ServiceClient moveCartClient;
  moveit_planner::GetTF curTF;
  moveit_planner::MoveCart cart;
  geometry_msgs::Pose p;
  geometry_msgs::Pose p_rot;
  int32_t timeout = 1000;
  double current_delta{0.0};
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "rotate_key_node");
  RotateKey rotate_key(ros::this_node::getName());
  ros::spin();

  return 0;
}