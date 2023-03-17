// ROS Core Deps
#include "ros/ros.h"
#include "ros/package.h"

// External Deps
#include "geometry_msgs/Pose.h"

// C++ Deps
#include <vector>
#include <string>

// Custom Deps
#include "helpfulUtils.cpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "ycb_desc_uploader_node");
  ros::NodeHandle nh;

  // TODO: Replace duplicates with bool mask, avoids constant copying
  std::vector<std::string> objectNames; // List of object names to load onto server
  std::vector<std::string> spawnableObjectNames; // List of objects with a valid URDF found
  if(!nh.getParam("/ycb_models/names", objectNames)) {
    ROS_WARN("Could not retrieve any names, exiting node");
    return 0;
  }

  std::string fileContents;	// String containing urdf
  for(int i = 0; i < objectNames.size(); ++i) {
    bool found{false};
    fileContents = loadLocal(objectNames[i], found);
    if(!found) {
      ROS_ERROR_STREAM("Could not find/read file for object model: " << objectNames[i]);
      continue;
    }

    // Add to parameter server and string vectors
    ros::param::set("/ycb_models/urdfs/" + objectNames[i], fileContents);
  }

  return 0;
}
