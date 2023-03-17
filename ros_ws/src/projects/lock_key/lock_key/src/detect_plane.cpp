#include "ros/ros.h"
//#include <geometry_msgs/PointStamped.h>
#include "lock_key/getWrench.h"
#include "lock_key/getAveWrench.h"
#include "moveit_planner/GetTF.h"
#include "moveit_planner/MoveCart.h"
#include <lock_key_msgs/DetectPlaneAction.h> // Note: "Action" is appended
#include <actionlib/server/simple_action_server.h>
#include <tf/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

class DetectPlane{
public:
  DetectPlane(string name) : 
    as(nh, name, false),
    action_name(name)
  { //register goal and feedback callbacks
    ROS_INFO("Initializing DetectPlane Action Server.");
    as.registerGoalCallback(boost::bind(&DetectPlane::goalCB, this));
    as.registerPreemptCallback(boost::bind(&DetectPlane::preemptCB, this));
    //Get parameters
    nh.getParam("spiral/delta_max",delta_max);
    nh.getParam("spiral/delta_step",delta_step);
    //Wait for services
    ros::service::waitForService("getAveWrench",timeout);
    ros::service::waitForService("getWrench",timeout);
    ros::service::waitForService("get_transform",timeout);
    ros::service::waitForService("cartesian_move",timeout);
    getWrenchclient = nh.serviceClient<lock_key::getWrench>("getWrench");
    getAveWrenchclient = nh.serviceClient<lock_key::getAveWrench>("getAveWrench");
    getTFClient = nh.serviceClient<moveit_planner::GetTF>("get_transform");
    moveCartClient = nh.serviceClient<moveit_planner::MoveCart>("cartesian_move");

    //Get current TF from world to EE
    curTF.request.from="map"; //map
    curTF.request.to="panda_link8"; //end_effector_link

    setGoalOrientation();

    ROS_INFO("DetectPlane Server starting now.");
    as.start();
  }

  ~DetectPlane(void){
  }

  void calculateFTBias(){
      // Retrieve current FT readings
      getAveWrenchclient.call(aveWrench);   
      // Store in global variables
      bias.Fx=aveWrench.response.fx;
      bias.Fy=aveWrench.response.fy;
      bias.Fz=aveWrench.response.fz;
      bias.Tx=aveWrench.response.tx;
      bias.Ty=aveWrench.response.ty;
      bias.Tz=aveWrench.response.tz;
      bias.set=1;

      //Set as ROS params
      nh.setParam("/FT_bias/Fx", bias.Fx);
      nh.setParam("/FT_bias/Fy", bias.Fy);
      nh.setParam("/FT_bias/Fz", bias.Fz);
      nh.setParam("/FT_bias/Tx", bias.Tx);
      nh.setParam("/FT_bias/Ty", bias.Ty);
      nh.setParam("/FT_bias/Tz", bias.Tz);

      ROS_INFO("Successfully retrieved FT sensor biases.");
  }

  bool maxDownForce(double force){
    getWrenchclient.call(currentWrench);

    forces={currentWrench.response.fx-bias.Fx,
            currentWrench.response.fy-bias.Fy,
            currentWrench.response.fz-bias.Fz};
    ROS_INFO("Stop if (Fz: %f > %f)",forces[2], force);

  return forces[2]<force;
  }

  void setGoalOrientation(){
      double roll, pitch, yaw;
      //Retrieve orientation parameters
      nh.getParam("padlock_goal/roll",roll);
      nh.getParam("padlock_goal/pitch",pitch);
      nh.getParam("padlock_goal/yaw",yaw);  
      //Convert RPY orientations to Quaternion
      tf2::Quaternion q_padlock_goal;
      q_padlock_goal.setRPY(roll, pitch, yaw);
      q_padlock_goal.normalize();
      //Update private variable
      tf2::convert(q_padlock_goal, padlock_goal_or);
      //Set orientation of goal position
      p.orientation=padlock_goal_or;
  }

  void moveRel(double x, double y, double z){
    getTFClient.call(curTF);
    p.position=curTF.response.pose.position;
    // if (use_current_or){
    //   p.orientation=curTF.response.pose.orientation;
    // }

    //Define PointStamped in panda_link8 (using xyz args)
    // hand_point.header.frame_id = "panda_link8";
    // hand_point.point.x = x;
    // hand_point.point.y = y;
    // hand_point.point.z = z;
    // //Transform to map frame.
    // listener.waitForTransform("/map", "/panda_link8",
    //                           ros::Time(0), ros::Duration(1.0));
    // listener.transformPoint("map",hand_point,hand_point_tf);
    //Use that as your positions
    // p.position=hand_point_tf.point;
    // ROS_INFO("Current: %.6f, %.6f, %.6f.",p.position.x,p.position.y,p.position.z);

    // //Update position with relative changes
    p.position.x += x;
    p.position.y += y;
    p.position.z += z;

    // ROS_INFO("Target: %.6f, %.6f, %.6f.",p.position.x,p.position.y,p.position.z);

    cart.request.val.clear();
    cart.request.val.push_back(p);
    //cart.request.time = 0;
    cart.request.execute = true;
    moveCartClient.call(cart);
  }

  void goalCB(){
    //Reset variables
    current_delta=0.0;
    // Update params in case they changed
    nh.getParam("spiral/delta_max",delta_max);
    nh.getParam("spiral/delta_step",delta_step);

    auto goal=as.acceptNewGoal();

    if (goal->recalculate_bias){ // | !bias.set
      ROS_INFO("Calculating FT sensor bias.");
      calculateFTBias();
    }

    // if (goal->use_current_orientation == 0){ // | !bias.set
    //   ROS_INFO("Setting goal orientation for padlock.");
    //   setGoalOrientation();
    // }
    // make sure that the action hasn't been canceled
    if (!as.isActive())
      return;

    ROS_INFO("Beginning descent to plane.");
    //Perform main task
    while (current_delta<=delta_max && maxDownForce(goal->F_max)){
        moveRel(0.0,0.0,-delta_step);
        // Update delta. TODO: Use actual position rather than cmd
        current_delta+=delta_step;
        // Publish feedback
        feedback.current_delta=current_delta;
        as.publishFeedback(feedback);
        ROS_INFO("Current Delta: %.6f",current_delta);
    }

    //Set result (whether a plane was detected)
    if (current_delta>=delta_max){
        result.reached_plane=0;
        ROS_INFO("Exceeded max insertion plane detection distance.");
    } else {
      result.reached_plane=1;
      ROS_INFO("Arrived at insertion plane");
    }

    as.setSucceeded(result);
  }

  void preemptCB(){
    ROS_INFO("%s: Preempted", action_name.c_str());
    // set the action state to preempted
    as.setPreempted();
  }

private:
  ros::NodeHandle nh;
  actionlib::SimpleActionServer<lock_key_msgs::DetectPlaneAction> as;
  lock_key_msgs::DetectPlaneFeedback feedback;
  lock_key_msgs::DetectPlaneResult result;
  string action_name;
  ros::ServiceClient getWrenchclient;
  ros::ServiceClient getAveWrenchclient;
  ros::ServiceClient getTFClient;
  ros::ServiceClient moveCartClient;
  lock_key::getAveWrench aveWrench;
  lock_key::getWrench currentWrench;
  moveit_planner::GetTF curTF;
  moveit_planner::MoveCart cart;
  geometry_msgs::Quaternion padlock_goal_or;
  geometry_msgs::Pose p;
  geometry_msgs::PointStamped hand_point;
  geometry_msgs::PointStamped hand_point_tf;
  tf::TransformListener listener;
  int32_t timeout = 1000;
  double current_delta{0.0};
  double delta_max{0.0};
  double delta_step{0.0};
  vector<double> forces;
  struct FT{
    double Fx{0.0};
    double Fy{0.0};
    double Fz{0.0};
    double Tx{0.0};
    double Ty{0.0};
    double Tz{0.0};
    bool set{0};
  };
  FT bias;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "plane_detector_node");
  DetectPlane detect_plane(ros::this_node::getName());
  ros::spin();

  return 0;
}