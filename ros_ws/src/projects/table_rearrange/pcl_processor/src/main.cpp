#include "ros/ros.h"

#include "pcl_processor.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "pcl_processor_node");
  ros::NodeHandle nh{};

  PCLProcessor p{nh};

  return 0;
}
