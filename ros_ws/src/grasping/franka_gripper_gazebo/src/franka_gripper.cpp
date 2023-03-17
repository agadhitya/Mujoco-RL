#include "franka_gripper.hpp"

#include <std_msgs/Float64.h>

FrankaGripper::FrankaGripper(ros::NodeHandle& nh, std::string f1_topic, std::string f2_topic) : _nh{nh} {
  _grip_server = nh.advertiseService("gazebo_franka_grip", &FrankaGripper::grip, this);
  _finger1_publisher = nh.advertise<std_msgs::Float64>(f1_topic, 1);
  _finger2_publisher = nh.advertise<std_msgs::Float64>(f2_topic, 1);
  _temp_delayed_motion = nh.subscribe("gazebo_franka_delayed", 1, &FrankaGripper::delayedMotion, this);

  ros::spin();
}
FrankaGripper::~FrankaGripper() {}

bool FrankaGripper::grip(franka_gripper_gazebo::GripMsg::Request& req,
			 franka_gripper_gazebo::GripMsg::Response& res) {
  std_msgs::Float64 joint_msg;
  joint_msg.data = req.force;

  // Now output to each joint
  _finger1_publisher.publish(joint_msg);
  _finger2_publisher.publish(joint_msg);

  ros::spinOnce();
  ros::spinOnce();

  return true;
}

void FrankaGripper::delayedMotion(const franka_gripper_gazebo::DelayedMotion& msg) {
  ros::Duration d(msg.delay);
  d.sleep();

  std_msgs::Float64 joint_msg;
  joint_msg.data = msg.force;

  // Now output to each joint
  _finger1_publisher.publish(joint_msg);
  _finger2_publisher.publish(joint_msg);

  ros::spinOnce();
  ros::spinOnce();
}
