#include <ros/ros.h>
#include "moveit_planner/SetConstraints.h"

int main(int argc, char **argv){

    ros::init(argc, argv, "arm_constraints");
    ros::NodeHandle n;

    ros::ServiceClient constraintClient = n.serviceClient<moveit_planner::SetConstraints>("set_constraints");
    moveit_planner::SetConstraints cons ;
    moveit_msgs::JointConstraint jc;
    moveit_msgs::JointConstraint jc1;

    jc.position = 0.3;
    jc.tolerance_above = 0.4;//3.14159;
    jc.tolerance_below = 0.4;
    jc.joint_name = "panda_joint2";
    jc.weight = 0.5;

    jc1.position = -0.07;
    jc1.tolerance_above = 0.3;
    jc1.tolerance_below = 0.3;
    jc1.joint_name = "panda_joint1";
    jc1.weight = 1.0;

    cons.request.constraints.joint_constraints.push_back(jc1);
    cons.request.constraints.joint_constraints.push_back(jc);

    return 0;
}