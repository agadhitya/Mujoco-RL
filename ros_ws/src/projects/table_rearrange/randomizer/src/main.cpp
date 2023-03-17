#include "ros/ros.h"
#include "gazebo_msgs/GetLinkState.h"
#include "gazebo_msgs/SetLinkState.h"
#include "geometry_msgs/Pose.h"

#include "randomizer/Rand.h"

#include <math.h>
#include <vector>
#include <string>
#include <random>
#include <iostream>

// Constants
#define PI 3.14159265

// List of items to randomize
std::vector<std::string> items{"mug::mug", "master_chef_can::master_chef_can"};
std::vector<geometry_msgs::Pose> initial_states{};

// Table params
double table_height;		// z
double table_center;		// x-center
double table_width;		// y
double table_length;		// x

// Randomizer
std::default_random_engine gen;

// Server
ros::ServiceClient randGazeboClient;

bool randomizeFunction(randomizer::Rand::Request& req, randomizer::Rand::Response& res) {
  std::uniform_real_distribution<double> lD{-table_width / 2, table_width / 2};
  std::uniform_real_distribution<double> wD{table_center-table_length / 2, table_center+table_length / 2};
  std::uniform_real_distribution<double> qD{0.0, 1.0};

  // Randomize each item
  for(int i = 0; i < items.size(); ++i) {
    gazebo_msgs::SetLinkState nState;
    geometry_msgs::Pose randomPose;

    // Randomize Position
    randomPose.position.x = wD(gen);
    randomPose.position.y = lD(gen);
    randomPose.position.z = table_height + 0.1;
    // Randomize Quaternion
    double u1 = qD(gen);
    double u2 = qD(gen);
    double u3 = qD(gen);
    randomPose.orientation.x = std::sqrt(1-u1)*std::sin(2*PI*u2);
    randomPose.orientation.y = std::sqrt(1-u1)*std::cos(2*PI*u2);
    randomPose.orientation.z = std::sqrt(u1)*std::sin(2*PI*u3);
    randomPose.orientation.w = std::sqrt(u1)*std::cos(2*PI*u3);    

    nState.request.link_state.link_name = items[i];
    nState.request.link_state.pose = randomPose;
    nState.request.link_state.reference_frame = "world";

    randGazeboClient.call(nState);
  }
  
  // TODO: Implement toStart

  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "item_randomizer");
  ros::NodeHandle nh{};

  ros::ServiceServer randServer = nh.advertiseService("randomize", randomizeFunction);
  ros::ServiceClient getStateClient = nh.serviceClient<gazebo_msgs::GetLinkState>("/gazebo/get_link_state");
  randGazeboClient = nh.serviceClient<gazebo_msgs::SetLinkState>("/gazebo/set_link_state");

  // Retrieve args for table dimensions
  if(argc < 5) {
    std::cerr<<"Not enough arguments (expected 4)"<<std::endl;
    return 1;
  }
  table_height = std::stod(argv[1]);
  table_center = std::stod(argv[2]);
  table_width  = std::stod(argv[3]);
  table_length = std::stod(argv[4]);

  // Get starting positions
  for(int i = 0; i < items.size(); ++i) {
    gazebo_msgs::GetLinkState states;
    states.request.link_name = items[i];
    states.request.reference_frame = "world";
    getStateClient.call(states);

    initial_states.push_back(states.response.link_state.pose);
  }

  ros::spin();

  return 0;
}
