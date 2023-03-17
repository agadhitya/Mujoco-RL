/* 
 * YCB Object spawner node
 */

// ROS Core Deps
#include "ros/ros.h"
#include "ros/package.h"

// External Deps
#include "geometry_msgs/Pose.h"
#include "gazebo_msgs/SpawnModel.h"

// C++ Deps
#include <vector>
#include <string>

// Custom Deps
#include "helpfulUtils.cpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "ycb_spawner_node");
  ros::NodeHandle nh;

  // TODO: Replace duplicates with bool mask, avoids constant copying
  std::vector<std::string> objectNames; // List of object names to spawn
  std::vector<std::string> validObjectNames; // List of VALID (Position exists) objects
  std::vector<std::string> spawnableObjectNames; // List of objects with a valid URDF found
  std::vector<geometry_msgs::Pose> objectPoses; // List of corresponding object poses
  std::vector<geometry_msgs::Pose> spawnableObjectPoses; // Poses of objects with valid URDF
  if(!nh.getParam("/ycb_models/names", objectNames)) {
    ROS_WARN("Could not retrieve any names, exiting node");
    return 0;
  }

  // Obtain details for each node
  std::string baseString;	// Base param string
  for(int i = 0; i < objectNames.size(); ++i) {
    geometry_msgs::Pose tempPose;
    baseString = "/ycb_models/poses/" + objectNames[i];
    bool found{false};
    tempPose.position = getPosition(baseString + "/position", nh, found);
    if(!found) {
      ROS_ERROR_STREAM("Full position information not found for " << objectNames[i]);
      continue;			// Cannot add this object
    }
    
    // Obtain orientation if possible
    // The default if it was not found is a unit quaternion, so its fine
    tempPose.orientation = getQuaternion(baseString + "orientation", nh, found);

    // Insert pose into objectNames
    validObjectNames.push_back(objectNames[i]);
    objectPoses.push_back(tempPose);
  }

  // Now that we have retrieved all object poses and names, we retrieve their urdfs
  // This is the last step before spawning the objects in gazebo because reading files can
  // be expensive
  std::vector<std::string> ycbModelURDFS; // vector of urdfs
  std::string fileContents;	// String containing urdf
  for(int i = 0; i < validObjectNames.size(); ++i) {
    bool found{false};
    fileContents = loadLocal(validObjectNames[i], found);
    if(!found) {
      ROS_ERROR_STREAM("Could not find/read file for object model: " << validObjectNames[i]);
      continue;
    }

    // Add to parameter server and string vectors
    ros::param::set("/ycb_spawner/urdfs/" + validObjectNames[i], fileContents);
    ycbModelURDFS.push_back(fileContents);
    spawnableObjectNames.push_back(validObjectNames[i]);
    spawnableObjectPoses.push_back(objectPoses[i]);
  }

  // After all objects are placed onto the server, we can spawn them in gazebo
  // First, wait for gazebo service to be available
  ROS_INFO("Waiting for /gazebo/spawn_urdf_model");
  ros::service::waitForService("/gazebo/spawn_urdf_model");

  // Now spawn all models
  ros::ServiceClient spawnModelClient =
    nh.serviceClient<gazebo_msgs::SpawnModel>("/gazebo/spawn_urdf_model");

  gazebo_msgs::SpawnModel spawnModelMsg;
  for(int i = 0; i < spawnableObjectNames.size(); ++i) {
    spawnModelMsg.request.model_xml = ycbModelURDFS[i];
    spawnModelMsg.request.model_name = spawnableObjectNames[i];
    spawnModelMsg.request.initial_pose = spawnableObjectPoses[i];
    if(!spawnModelClient.call(spawnModelMsg))
      ROS_WARN_STREAM("Could not spawn model " << spawnableObjectNames[i]);
  }

  return 0;
}
