#include "ros/ros.h"
#include "lock_key/getWrench.h"
#include "lock_key/getAveWrench.h"
#include "moveit_planner/GetTF.h"
#include "moveit_planner/MoveCart.h"
#include <lock_key_msgs/SpiralAction.h> // Note: "Action" is appended
#include <actionlib/server/simple_action_server.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "std_msgs/Float64.h"
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

class Spiral{
public:
  Spiral(string name) : 
    as(nh, name, false),
    action_name(name)
  { //register goal and feedback callbacks
    ROS_INFO("Initializing Spiral Action Server.");
    as.registerGoalCallback(boost::bind(&Spiral::goalCB, this));
    as.registerPreemptCallback(boost::bind(&Spiral::preemptCB, this));
    //Get parameters...
    nh.getParam("FT_bias/Fx", bias.Fx);
    nh.getParam("FT_bias/Fy", bias.Fy);
    nh.getParam("FT_bias/Fz", bias.Fz);
    nh.getParam("FT_bias/Tx", bias.Tx);
    nh.getParam("FT_bias/Ty", bias.Ty);
    nh.getParam("FT_bias/Tz", bias.Tz);
    bias.set=1;
    //Wait for services
    ros::service::waitForService("getAveWrench",timeout);
    ros::service::waitForService("getWrench",timeout);
    ros::service::waitForService("get_transform",timeout);
    ros::service::waitForService("cartesian_move",timeout);
    getWrenchclient = nh.serviceClient<lock_key::getWrench>("getWrench");
    getTFClient = nh.serviceClient<moveit_planner::GetTF>("get_transform");
    moveCartClient = nh.serviceClient<moveit_planner::MoveCart>("cartesian_move");
    //Define publisher
    fz_pub=nh.advertise<std_msgs::Float64>("fz_bias_removed", 1000);
    //Get current TF from world to EE
    curTF.request.from="map"; //map
    curTF.request.to="panda_link8"; //end_effector_link

    setGoalOrientation();

    ROS_INFO("Spiral Server starting now.");
    as.start();
  }

  ~Spiral(void){
  }

  bool maxSpiralForces(double Fd){
      getWrenchclient.call(currentWrench);

      forces={currentWrench.response.fx-bias.Fx,
              currentWrench.response.fy-bias.Fy,
              currentWrench.response.fz-bias.Fz};
      torques={currentWrench.response.tx-bias.Tx,
               currentWrench.response.ty-bias.Ty,
               currentWrench.response.tz-bias.Tz};

      fz_msg.data=forces[2];
      fz_pub.publish(fz_msg);
     
      ROS_INFO("Spiral. Stop if Fz: %.4f < %.4f", forces[2], minSpiralForce);
      ROS_INFO("Spiral. Stop if Tx: %.4f > %.4f", abs(torques[0]), Tx_limit);
      ROS_INFO("Spiral. Stop if Ty: %.4f > %.4f", abs(torques[1]), Ty_limit);

      return ((forces[2]>minSpiralForce) &&
              (sqrt(pow(torques[0],2))<Tx_limit) &&
              (sqrt(pow(torques[1],2))<Ty_limit)); //If true, keep spiralling
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

  void moveAbsSpiral(double x_in, double y_in, double z_in, bool useCurrentZ){
      //Use current z-position for upcoming command
      if (useCurrentZ){
          getTFClient.call(curTF);
          p.position.z=curTF.response.pose.position.z;
      } 
      //Or use an absolute z-position from args
      else{
          p.position.z = z_in;
      }

      //Set remaining position info
      p.position.x = x_in;
      p.position.y = y_in;

      //Execute
      cart.request.val.clear();
      cart.request.val.push_back(p);
      cart.request.execute = true;
      moveCartClient.call(cart);
  }

  void goalCB(){
    // Update params in case they changed
    nn=1.0; x=0.0; y=0.0;
    ROS_INFO("Retrieving spiral parameters");
    nh.getParam("spiral/Tx",Tx_limit);
    nh.getParam("spiral/Ty",Ty_limit);
    nh.getParam("spiral/a",spiral_a);
    nh.getParam("spiral/b",spiral_b);
    nh.getParam("spiral/nmax",spiral_nmax);
    nh.getParam("spiral/rot",spiral_rot);
    nh.getParam("spiral/min_spiral_force",minSpiralForce);
    nh.getParam("spiral/Fd",Fd_max);

    as.acceptNewGoal();

    // make sure that the action hasn't been canceled
    if (!as.isActive())
      return;

    ROS_INFO("Starting Spiral Insertion Motion");
    //Perform main task
    getTFClient.call(curTF);

    while (nn<spiral_nmax && maxSpiralForces(Fd_max)){
        //Calculate new xy
        phi = sqrt(nn/spiral_nmax)*(spiral_rot*2.0*M_PI);
        spiral_rad=(spiral_a-spiral_b*phi);
        ROS_INFO("nn: %.4f, phi: %.4f", nn, phi);
        x+=spiral_rad*cos(phi);
        y+=spiral_rad*sin(phi);
        ROS_INFO("Spiral X: %.4f, Y: %.4f", x, y);
        //Send relative movement command
        moveAbsSpiral(x+curTF.response.pose.position.x,
                      y+curTF.response.pose.position.y,
                      curTF.response.pose.position.z,
                      0);
        // Publish feedback
        feedback.dx=x; feedback.dy=y;
        as.publishFeedback(feedback);
        //Update parameters
        nn+=1.0;
    }

    //Set result (whether a hole was found)
    if (nn>=spiral_nmax){
        result.found_hole=0;
        ROS_INFO("Exceeded max spiral iterations.");
    } else {
      result.found_hole=1;
      ROS_INFO("Found hole");
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
  actionlib::SimpleActionServer<lock_key_msgs::SpiralAction> as;
  lock_key_msgs::SpiralFeedback feedback;
  lock_key_msgs::SpiralResult result;
  string action_name;
  ros::ServiceClient getWrenchclient;
  ros::ServiceClient getTFClient;
  ros::ServiceClient moveCartClient;
  ros::Publisher fz_pub;
  lock_key::getAveWrench aveWrench;
  lock_key::getWrench currentWrench;
  moveit_planner::GetTF curTF;
  moveit_planner::MoveCart cart;
  geometry_msgs::Quaternion padlock_goal_or;
  geometry_msgs::Pose p;
  std_msgs::Float64 fz_msg;
  vector<double> forces;
  vector<double> torques;
  double Fd_max;
  double minSpiralForce;
  double Tx_limit, Ty_limit;
  double spiral_a, spiral_b, spiral_nmax;
  double spiral_rad, spiral_rot, phi;
  double nn{1.0};
  double x, y;
  int32_t timeout = 1000;
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
  ros::init(argc, argv, "spiral_node");
  Spiral spiral(ros::this_node::getName());
  ros::spin();

  return 0;
}