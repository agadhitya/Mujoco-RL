#include "ros/ros.h"

#include "obtain_pose_gazebo.hpp"

#include <vector>
#include <string>

int main(int argc, char** argv) {
  ros::init(argc, argv, "obtain_pose_gazebo_node");
  ros::NodeHandle nh{};

  ObtainPoseGazebo p{nh};

  return 0;
}
