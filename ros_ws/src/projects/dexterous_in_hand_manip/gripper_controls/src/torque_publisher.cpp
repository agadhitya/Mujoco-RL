#include "ros/ros.h"
#include <sensor_msgs/JointState.h>
#include <gripper_controls/JointEffort.h>

class TorquePublisher{
public:
  TorquePublisher(){
    sub_ = n_.subscribe("/gripper/joint_states",1, &TorquePublisher::callback, this);
  }
  void callback(const sensor_msgs::JointState& input){
    left_effort = input.effort[1];
    right_effort = input.effort[0];
  }
  float get_effort(int finger){
    if (finger==0)
      return left_effort;
    else
      return right_effort;
  }

private:
  ros::NodeHandle n_;
  ros::Subscriber sub_;
  float left_effort{0};
  float right_effort{0};
};

int main(int argc, char **argv){
  ros::init(argc,argv,"torque_publisher");
  TorquePublisher torque_publisher;
  ros::NodeHandle n;
  ros::Publisher pub = n.advertise<gripper_controls::JointEffort>("/gripper/torque_readings",1000);

  ros::Rate loop_rate(20);
  while(ros::ok()){
    gripper_controls::JointEffort effort_msg;
    effort_msg.left = torque_publisher.get_effort(0);
    effort_msg.right = torque_publisher.get_effort(1);
    pub.publish(effort_msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}
