#include <ros/ros.h>
#include <string>

#include <gazebo_msgs/SpawnModel.h>
#include <gazebo_msgs/ModelState.h>
#include <gazebo_msgs/ModelStates.h>

double conveyor_width;
double conveyor_segment_length;
double conveyor_length;
double conveyor_speed;
double y, x, z;
std::string conveyor_urdf;

int main(int argc, char** argv) {
  ros::init(argc, argv, "conveyor_sim_node");
  ros::NodeHandle nh;

  if(!nh.getParam("/conveyor/width", conveyor_width) ||
     !nh.getParam("/conveyor/length", conveyor_length) ||
     !nh.getParam("/conveyor/speed", conveyor_speed) ||
     !nh.getParam("/conveyor/description",  conveyor_urdf) ||
     !nh.getParam("/conveyor/seg_length", conveyor_segment_length) ||
     !nh.getParam("/conveyor/y", y) ||
     !nh.getParam("/conveyor/x", x) ||
     !nh.getParam("/conveyor/z", z)) {
    ROS_ERROR("Could not load conveyor parameters");
    return 1;
  }

  ros::service::waitForService("/gazebo/spawn_urdf_model");

  ros::ServiceClient spawnModelClient = nh.serviceClient<gazebo_msgs::SpawnModel>("/gazebo/spawn_urdf_model");

  // Spawn conveyor links
  gazebo_msgs::SpawnModel modelSpawn;
  modelSpawn.request.model_xml = conveyor_urdf;
  modelSpawn.request.initial_pose.orientation.w = 1;
  modelSpawn.request.initial_pose.position.x = x;
  modelSpawn.request.initial_pose.position.z = z;
  modelSpawn.request.initial_pose.position.y = y;
  for(int i = 0; i < (int)(conveyor_length/conveyor_segment_length); ++i) {
    modelSpawn.request.model_name = "segment_" + std::to_string(i);
    spawnModelClient.call(modelSpawn);
    modelSpawn.request.initial_pose.position.y += conveyor_segment_length;
  }

  return 0;
}
