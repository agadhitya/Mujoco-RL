#include "ros/ros.h"

#include <iostream>
#include <string>

// Main class
#include "moveit_planner.hpp"

using moveit_planner::MoveitPlanner;

int main(int argc, char** argv) {
  ros::init(argc, argv, "main_node");
  ros::NodeHandle nh{};
  std::string group_name;
  std::string base_frame{"/world"};

  if(argc == 1) {
    ROS_INFO("No arm argument given, defaulting to \"arm\"");
    group_name = "arm";
  }
  else if(argc == 2) {
    ROS_INFO_STREAM("Starting moveit_planner with group name " << argv[1]);
    group_name = argv[1];
  }
  else if(argc == 3) {
    ROS_INFO_STREAM("Setting base link as " << argv[2]);
    group_name = argv[1];
    base_frame = argv[2];
  }

  MoveitPlanner planner(nh, group_name);

  return 0;
}
