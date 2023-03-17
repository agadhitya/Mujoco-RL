#include <ros/ros.h>
#include <lock_key/GraspAction.h> // Note: "Action" is appended
#include <actionlib/server/simple_action_server.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include "moveit_planner/GetPose.h"
#include "moveit_planner/MoveCart.h"
#include "moveit_planner/MovePose.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

ros::NodeHandle* n_ptr;
int32_t timeout = 1000;

typedef actionlib::SimpleActionServer<lock_key::GraspAction> Server;

void moveAbs(double x){
	//Waiting for services to be available
	ros::service::waitForService("move_to_pose",timeout);
    ros::ServiceClient moveCartClient = n_ptr->serviceClient<moveit_planner::MoveCart>("cartesian_move");
    moveit_planner::MoveCart cart;
    geometry_msgs::Pose p;

    ROS_INFO("Grasping Object");
}

void execute(const lock_key::GraspGoalConstPtr& goal, Server* as){  // Note: "Action" is not appended here

    moveAbs(goal->x);

    as->setSucceeded();
}

int main(int argc, char **argv){
    ros::init(argc, argv, "grasp_node");
    ros::NodeHandle n;
    n_ptr=&n;
    
    Server server(n, "grasp_server", boost::bind(&execute, _1, &server), false);
    server.start();

    ros::spin();
}