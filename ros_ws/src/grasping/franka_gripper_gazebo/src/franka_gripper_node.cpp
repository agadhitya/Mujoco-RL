#include <ros/ros.h>

#include <string>

#include "franka_gripper.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "franka_gazebo_gripper_node");
  ros::NodeHandle nh;

  std::string finger1_topic = "/panda/panda_finger1_controller/command";
  std::string finger2_topic = "/panda/panda_finger2_controller/command";

  // TODO: Replace with switch case
  if(argc == 1)
    ROS_INFO("No arguments supplied, creating with defaults");
  else if(argc == 2) {
    ROS_ERROR("Not enough arguments, please either do not supply any arguments or 2 topic names");
    return 1;
  }
  else if(argc == 3) {
    ROS_INFO("Creating with supplied arguments");
    finger1_topic = argv[1];
    finger2_topic = argv[2];
  }
  else {
    ROS_ERROR("Too many arguments supplied");
    return 1;
  }

  FrankaGripper franka_gripper(nh, finger1_topic, finger2_topic);

  return 0;
}
