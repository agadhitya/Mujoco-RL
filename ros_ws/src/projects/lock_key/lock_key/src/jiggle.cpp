#include <ros/ros.h>
#include <lock_key/JiggleAction.h> // Note: "Action" is appended
#include <actionlib/server/simple_action_server.h>
#include "moveit_planner/GetTF.h"
#include "moveit_planner/MoveCart.h"
#include "moveit_planner/MovePose.h"
#include "lock_key/getWrench.h"
#include "lock_key/getAveWrench.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

ros::NodeHandle* n_ptr;
int32_t timeout = 1000;

typedef actionlib::SimpleActionServer<lock_key::JiggleAction> Server;

void moveRel(double x, double y, double z, double roll, double pitch, double yaw){
	//Waiting for services to be available
	ros::service::waitForService("get_transform",timeout);
	ros::service::waitForService("move_to_pose",timeout);
    ros::ServiceClient getTFClient = n_ptr->serviceClient<moveit_planner::GetTF>("get_transform");
	ros::ServiceClient moveCartClient = n_ptr->serviceClient<moveit_planner::MoveCart>("cartesian_move");
    moveit_planner::GetTF curTF;
	moveit_planner::MoveCart cart;

    //Get current TF from world to EE
    curTF.request.from="map"; //map
    curTF.request.to="panda_link8"; //end_effector_link
    getTFClient.call(curTF);
	geometry_msgs::Pose p;
    p=curTF.response.pose;

	//Convert RPY to quat
	tf2::Quaternion q_orig, q_change, q_new;
    tf2::convert(p.orientation , q_orig);
	q_change.setRPY(roll, pitch, yaw);
	q_new=q_change*q_orig;
	q_new.normalize();
	tf2::convert(q_new, p.orientation);
	
    //Update position with relative changes
	p.position.x += x;
	p.position.y += y;
	p.position.z += z;
	cart.request.val.push_back(p);
    cart.request.execute = true;
    moveCartClient.call(cart);
}

void jiggle(double angle, basic_string<char> axis){
	if (axis=="x"){
		ROS_INFO("Jiggle about X");
		moveRel(0.0,0.0,0.0,-angle,0.0,0.0);
		moveRel(0.0,0.0,0.0,2*angle,0.0,0.0);
		moveRel(0.0,0.0,0.0,-angle,0.0,0.0);
	}
	else if (axis=="y"){
		ROS_INFO("Jiggle about Y");	
		moveRel(0.0,0.0,0.0,0.0,-angle,0.0);
		moveRel(0.0,0.0,0.0,0.0,2*angle,0.0);
		moveRel(0.0,0.0,0.0,0.0,-angle,0.0);
	}
	else if (axis=="z"){
		ROS_INFO("Jiggle about Z");
		moveRel(0.0,0.0,0.0,0.0,0.0,-angle);
		moveRel(0.0,0.0,0.0,0.0,0.0,2*angle);
		moveRel(0.0,0.0,0.0,0.0,0.0,-angle);
	}
	else {
		ROS_INFO("Unknown axis. Please use x, y, or z.");
	}
}

void execute(const lock_key::JiggleGoalConstPtr& goal, Server* as){  // Note: "Action" is not appended here

    jiggle(goal->angle, goal->axis);

    as->setSucceeded();
}

int main(int argc, char **argv){
    ros::init(argc, argv, "jiggleKey");
    ros::NodeHandle n;
    n_ptr=&n;
    
    Server server(n, "jiggle_key", boost::bind(&execute, _1, &server), false);
    server.start();

    ros::spin();
}