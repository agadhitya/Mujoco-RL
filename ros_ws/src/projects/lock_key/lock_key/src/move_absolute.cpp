#include <ros/ros.h>
#include <lock_key/MoveAbsAction.h> // Note: "Action" is appended
#include <actionlib/server/simple_action_server.h>
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

typedef actionlib::SimpleActionServer<lock_key::MoveAbsAction> Server;

void moveAbs(double x, double y, double z, double roll, double pitch, double yaw){
	//Waiting for services to be available
	ros::service::waitForService("move_to_pose",timeout);
    ros::ServiceClient moveCartClient = n_ptr->serviceClient<moveit_planner::MoveCart>("cartesian_move");
    moveit_planner::MoveCart cart;
    geometry_msgs::Pose p;
    //Convert RPY to quat
    tf2::Quaternion q_new;
    q_new.setRPY(roll, pitch, yaw);
    q_new.normalize();
    // Set goal
    p.position.x = x;
    p.position.y = y;
    p.position.z = z;
    tf2::convert(q_new, p.orientation);
	cart.request.val.push_back(p);
    cart.request.execute = true;
    moveCartClient.call(cart);
    ROS_INFO("Moving to Absolute Position");
}

void execute(const lock_key::MoveAbsGoalConstPtr& goal, Server* as){  // Note: "Action" is not appended here

    moveAbs(goal->x, goal->y, goal->z, goal->roll, goal->pitch, goal->yaw);

    as->setSucceeded();
}

int main(int argc, char **argv){
    ros::init(argc, argv, "move_abs_node");
    ros::NodeHandle n;
    n_ptr=&n;
    
    Server server(n, "move_abs_server", boost::bind(&execute, _1, &server), false);
    server.start();

    ros::spin();
}