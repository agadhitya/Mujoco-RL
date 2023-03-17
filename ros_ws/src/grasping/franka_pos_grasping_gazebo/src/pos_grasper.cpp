#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include "franka_pos_grasping_gazebo/GripPos.h"

ros::Publisher finger1_pub;
ros::Publisher finger2_pub;

bool grip_pos_callback(franka_pos_grasping_gazebo::GripPos::Request &req, franka_pos_grasping_gazebo::GripPos::Response &res ){
    std_msgs::Float64 joint_msg;
    joint_msg.data = req.finger_pos;
    finger1_pub.publish(joint_msg);
    finger2_pub.publish(joint_msg);
}

int main(int argc, char **argv){
    ros::init(argc, argv,"pos_grasper");
    ros::NodeHandle n;

    // Advertising service for sending position commands to gripper finger
    // 0.0 -> center position (gripper closed)
    // 0.04 -> end position (gripper open)
    // can provide values in between for different lengths
    // might need to tune values visually
    finger1_pub = n.advertise<std_msgs::Float64>("/panda/panda_finger1_controller/command",1);
    finger2_pub = n.advertise<std_msgs::Float64>("/panda/panda_finger2_controller/command",1);
    ros::ServiceServer service = n.advertiseService("gripPosServer", grip_pos_callback);

    ros::spin();

    return 0;
}