#include "ros/ros.h"
#include <geometry_msgs/WrenchStamped.h>
#include "lock_key/getWrench.h"
#include "lock_key/getAveWrench.h"
#include <vector>
using namespace std;

ros::NodeHandle* n_ptr;
vector<double> forces (3);
vector<double> torques (3);

void wrenchSubCallback(const geometry_msgs::WrenchStamped::ConstPtr& msg){
    //ROS_INFO("Received wrist wrench");
    forces[0]=msg->wrench.force.x;
    forces[1]=msg->wrench.force.y;
    forces[2]=msg->wrench.force.z;
    torques[0]=msg->wrench.torque.x;
    torques[1]=msg->wrench.torque.y;
    torques[2]=msg->wrench.torque.z;
}

bool wrenchServiceCallback(lock_key::getWrench::Request &req, 
                           lock_key::getWrench::Response &res){
  ROS_INFO("Getting wrist wrench.");
  res.fx = forces[0];
  res.fy = forces[1];
  res.fz = forces[2];
  res.tx = torques[0];
  res.ty = torques[1];
  res.tz = torques[2];
  //ROS_INFO("Fx: %.2f, Fy: %.2f, Fz: %.2f", res.fx, res.fy, res.fz);
  //ROS_INFO("Tx: %.2f, Ty: %.2f, Tz: %.2f", res.tx, res.ty, res.tz);
  return true;
}

bool wrenchAveServiceCallback(lock_key::getAveWrench::Request &req, 
                              lock_key::getAveWrench::Response &res){
  int ii_max;
  double ft_sleep;
  n_ptr->getParam("spiral/ft_samples",ii_max);
  n_ptr->getParam("spiral/ft_sleep",ft_sleep);

  ROS_INFO("Getting average wrist wrench over %i samples.", ii_max);
  // Set variables to zero
  res.fx = 0.0;
  res.fy = 0.0;
  res.fz = 0.0;
  res.tx = 0.0;
  res.ty = 0.0;
  res.tz = 0.0;
  // Get ten readings for each and accumulate results
  for (int ii=0;ii<ii_max;ii++){
    res.fx += forces[0];
    res.fy += forces[1];
    res.fz += forces[2];
    res.tx += torques[0];
    res.ty += torques[1];
    res.tz += torques[2];
    ROS_INFO("Collected sample %i/%i.",ii+1,ii_max);
    ROS_INFO("Fx: %.2f, Fy: %.2f, Fz: %.2f", forces[0], forces[1], forces[2]);
    ROS_INFO("Tx: %.2f, Ty: %.2f, Tz: %.2f", torques[0], torques[1], torques[2]);
    ros::Duration(ft_sleep).sleep();
  }
  // Divide by number of samples to get average. Multiply by 1.0 to get double
  res.fx /= (1.0*ii_max);
  res.fy /= (1.0*ii_max);
  res.fz /= (1.0*ii_max);
  res.tx /= (1.0*ii_max);
  res.ty /= (1.0*ii_max);
  res.tz /= (1.0*ii_max);

  ROS_INFO("Fx_ave: %.2f, Fy_ave: %.2f, Fz_ave: %.2f", res.fx, res.fy, res.fz);
  ROS_INFO("Tx_ave: %.2f, Ty_ave: %.2f, Tz_ave: %.2f", res.tx, res.ty, res.tz);
  return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "wrench_server");
    ros::NodeHandle n;
    n_ptr=&n;

    // Use AsyncSpinner so that wrench values will update
    // during service execution.
    ros::AsyncSpinner spinner(3);
    spinner.start();

    //ros::Subscriber wrench_sub = n.subscribe("panda_joint7_wrench", 1000, wrenchSubCallback); //Topic for gazebo simulation
    ros::Subscriber wrench_sub = n.subscribe("franka_state_controller/F_ext", 1000, wrenchSubCallback); // Topic for real panda
    ros::ServiceServer wrench_server = n.advertiseService("getWrench", wrenchServiceCallback);
    ros::ServiceServer wrench_ave_server = n.advertiseService("getAveWrench", wrenchAveServiceCallback);

    ros::waitForShutdown();
}