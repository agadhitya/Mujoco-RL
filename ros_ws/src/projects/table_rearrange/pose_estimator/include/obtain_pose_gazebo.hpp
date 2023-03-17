#include "ros/ros.h"

#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Quaternion.h"

#include "gazebo_msgs/ModelStates.h"
#include "pose_estimator/PoseEstimation.h"
#include "pose_estimator/SimEstimation.h"

#include <math.h>
#include <random>
#include <vector>
#include <string>

class ObtainPoseGazebo {
public:
  ObtainPoseGazebo(ros::NodeHandle& n) : nh{n}, topic_name{"/pose_estimator/gazebo_pose"}, urd{-1.0, 1.0} {
    setupTopics();

    publishData();
  }
  ~ObtainPoseGazebo() {}	// Default destructor

  // Callbacks
  // Received poses from gazebo
  void receivedGazeboPoses(gazebo_msgs::ModelStates states) {
    object_names = states.name;
    object_poses = states.pose;
    measuredTime = ros::WallTime::now();
  };
  bool getPoses(pose_estimator::PoseEstimation::Request &req,
		pose_estimator::PoseEstimation::Response &res) {
    // Do not need point cloud data
    res.detected_object_names = object_names;
    res.detected_object_poses = object_poses;

    return true;
  }
  bool getSimPose(pose_estimator::SimEstimation::Request& req,
		  pose_estimator::SimEstimation::Response& res) {
    // First, check if object exists
    int objIndex = -1;

    for(int i = 0; i < object_names.size(); ++i) {
      if(object_names[i] == req.model_name) {
	objIndex = i;
	break;
      }
    }

    if(objIndex == -1)
      return false;

    // Object found, get pose
    geometry_msgs::Pose ret = object_poses[objIndex];
    
    // Position Error
    ret.position.x += (urd(gen)*req.err_x);
    ret.position.y += (urd(gen)*req.err_y);
    // ret.position.z += (urd(gen)*req.err_z); No error on z since it is on a conveyor

    // Orientation Error
    double zAngle = urd(gen)*req.err_angle_z;
    geometry_msgs::Quaternion quat;
    quat.z = std::sin(zAngle / 2);
    quat.w = std::cos(zAngle / 2);
    ret.orientation = multQuats(ret.orientation, quat);

    // Simulate wait
    ros::Duration d(req.time_to_wait);
    d.sleep();

    // Done
    res.pose = ret;
    res.time = measuredTime.toSec();
    return true;
  }

private:
  ros::NodeHandle nh;
  ros::Subscriber gazeboSub;
  ros::ServiceServer modelPoseServer, simPoseServer;
  
  std::string topic_name;
  std::default_random_engine gen;
  std::uniform_real_distribution<double> urd;
  std::vector<std::string> object_names;
  std::vector<geometry_msgs::Pose> object_poses;
  ros::WallTime measuredTime;

  // Setup
  void setupTopics() {
    nh.getParam("/table_objects", object_names); // Load desired objects from param server
    for(int i = 0; i < object_names.size(); ++i)
      object_poses.push_back(geometry_msgs::Pose{});

    gazeboSub = nh.subscribe("/gazebo/model_states", 1, &ObtainPoseGazebo::receivedGazeboPoses, this);
    // Temporarily disabled
    // modelPoseServer = nh.advertiseService(topic_name, &ObtainPoseGazebo::getPoses, this);
    simPoseServer = nh.advertiseService("get_object_pose", &ObtainPoseGazebo::getSimPose, this);
  }

  // Actual loop
  void publishData() {
    ros::Rate r(30);
    while(ros::ok()) {
      ros::spinOnce();
      // Publish pose information
      r.sleep();
    }
  }

  // Helper func
  geometry_msgs::Quaternion multQuats(const geometry_msgs::Quaternion& a, const geometry_msgs::Quaternion& b) {
    geometry_msgs::Quaternion ret;

    ret.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
    ret.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
    ret.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
    ret.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;

    return ret;
  }

};
