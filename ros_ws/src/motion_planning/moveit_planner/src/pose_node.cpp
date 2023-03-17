#include <string>

#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <tf/transform_listener.h>

#include "moveit_planner/GetTF.h"

tf::TransformListener* listener;

inline geometry_msgs::Pose stampedTFToPose(const tf::StampedTransform& st) {
  geometry_msgs::Pose ret;

  // Set position
  ROS_INFO_STREAM(st.getOrigin().getY());
  ret.position.x = st.getOrigin().getX();
  ret.position.y = st.getOrigin().getY();
  ret.position.z = st.getOrigin().getZ();

  // Set rotation
  ret.orientation.x = st.getRotation().x();
  ret.orientation.y = st.getRotation().y();
  ret.orientation.z = st.getRotation().z();
  ret.orientation.w = st.getRotation().w();

  // Done
  return ret;
}

bool transformCallback(moveit_planner::GetTF::Request& req,
		       moveit_planner::GetTF::Response& res) {
  ros::spinOnce();
  ros::spinOnce();
  ros::spinOnce();

  int attempts = 5;
  tf::StampedTransform transform;
  ros::Time cur = ros::Time::now();

  ROS_INFO_STREAM("Attempting " << attempts << " times to get the pose");
  for(int i = 0; i < attempts; ++i) {
    // Try to get transform
    try {
      listener->waitForTransform(req.from, req.to, cur, ros::Duration(10.0));
      listener->lookupTransform(req.from, req.to, cur, transform);

      // Success
      ROS_INFO("Obtained pose");
      res.pose = stampedTFToPose(transform);
      return true;
    }
    catch(tf::TransformException &ex) {
      ROS_ERROR("%s", ex.what());
      ROS_ERROR_STREAM("Trying again: " << attempts + 1 << " / " << attempts);

      ros::Duration(0.5).sleep();

      continue;
    }
  }

  // Failed
  return false;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "pose_node");
  ros::NodeHandle nh;

  listener = new tf::TransformListener();
  ros::ServiceServer server = nh.advertiseService("get_transform", transformCallback);

  ros::spin();
  return 0;
}
