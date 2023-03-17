// The arm_move node handles movement for the robot. It pulls services from the moveit_planner node.
#include <ros/ros.h>
#include "moveit_planner/MoveCart.h"
#include "moveit_planner/MovePose.h"

void setup_service_client(ros::NodeHandle n){
    ros::ServiceClient MovePoseClient = n.serviceClient<moveit_planner::MovePose>("move_to_pose");
}

int main(int argc, char **argv){

    ros::init(argc, argv, "movement");
    ros::NodeHandle n;



    //Setting up Service Clients
    //setup_service_client(n);

// Do this when key is moved to some other location, key is currently visible in frame on launch
    // ros::ServiceClient MovePoseClient = n.serviceClient<moveit_planner::MovePose>("move_to_pose");

    // moveit_planner::MovePose pose;
    // n.getParam("pre_pose/x_pos", pose.request.val.position.x);
    // n.getParam("pre_pose/y_pos", pose.request.val.position.y);
    // n.getParam("pre_pose/z_pos", pose.request.val.position.z);
    // n.getParam("pre_pose/w_orn", pose.request.val.orientation.w);
    // n.getParam("pre_pose/x_orn", pose.request.val.orientation.x);
    // n.getParam("pre_pose/y_orn", pose.request.val.orientation.y);
    // n.getParam("pre_pose/z_orn", pose.request.val.orientation.z);
    // ROS_INFO_STREAM(pose.request.val);
    // pose.request.execute = true;

    // MovePoseClient.call(pose);
    // ROS_INFO("Move Pose Client Call");






    return 0;
}
