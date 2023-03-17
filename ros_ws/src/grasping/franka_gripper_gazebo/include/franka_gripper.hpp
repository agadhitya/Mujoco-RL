#include <ros/ros.h>

#include <string>

#include <franka_gripper_gazebo/GripMsg.h>
#include <franka_gripper_gazebo/DelayedMotion.h>

class FrankaGripper {
public:
  FrankaGripper(ros::NodeHandle& nh, std::string f1_topic, std::string f2_topic);
  ~FrankaGripper();

  bool grip(franka_gripper_gazebo::GripMsg::Request& req,
	    franka_gripper_gazebo::GripMsg::Response& res);

  void delayedMotion(const franka_gripper_gazebo::DelayedMotion& msg);
private:
  ros::NodeHandle& _nh;
  
  ros::ServiceServer _grip_server;
  
  ros::Publisher _finger1_publisher;
  ros::Publisher _finger2_publisher;

  ros::Subscriber _temp_delayed_motion;
};
