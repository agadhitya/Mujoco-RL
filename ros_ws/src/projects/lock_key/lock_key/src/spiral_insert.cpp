#include <ros/ros.h>
#include <lock_key/SpiralInsertAction.h> // Note: "Action" is appended
#include <actionlib/server/simple_action_server.h>
#include "moveit_planner/GetTF.h"
#include "moveit_planner/MoveCart.h"
#include "moveit_planner/MovePose.h"
#include "lock_key/getWrench.h"
#include "lock_key/getAveWrench.h"
#include "std_msgs/Float64.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

ros::NodeHandle* n_ptr;
ros::Publisher* fz_pub_ptr;

std_msgs::Float64 fz_msg;
int32_t timeout = 1000;
double Tx_limit;
double Ty_limit;
double Fx_bias{0}, Fy_bias{0}, Fz_bias{0};
double Tx_bias{0}, Ty_bias{0}, Tz_bias{0};
double minSpiralForce{0.0};
geometry_msgs::Quaternion padlock_goal_or;

typedef actionlib::SimpleActionServer<lock_key::SpiralInsertAction> Server;

bool maxDownForce(double force){
    ros::service::waitForService("getWrench",timeout);
    ros::ServiceClient getWrenchclient = n_ptr->serviceClient<lock_key::getWrench>("getWrench");
    lock_key::getWrench currentWrench;
    getWrenchclient.call(currentWrench);

    vector<double> forces={currentWrench.response.fx-Fx_bias,
                           currentWrench.response.fy-Fy_bias,
                           currentWrench.response.fz-Fz_bias};
    ROS_INFO("Stop if (Fz: %f > %f)",forces[2], force);

	return forces[2]<force;
}

bool maxSpiralForces(double Fd){
    ros::service::waitForService("getWrench",timeout);
    ros::ServiceClient getWrenchclient = n_ptr->serviceClient<lock_key::getWrench>("getWrench");
    lock_key::getWrench currentWrench;
    getWrenchclient.call(currentWrench);

    vector<double> forces={currentWrench.response.fx-Fx_bias,
                           currentWrench.response.fy-Fy_bias,
                           currentWrench.response.fz-Fz_bias};
    vector<double> torques={currentWrench.response.tx-Tx_bias,
                            currentWrench.response.ty-Ty_bias,
                            currentWrench.response.tz-Tz_bias};

    fz_msg.data=forces[2];
    fz_pub_ptr->publish(fz_msg);
   
    ROS_INFO("Spiral. Stop if Fz: %.4f < %.4f", forces[2], minSpiralForce);
    ROS_INFO("Spiral. Stop if Tx: %.4f > %.4f", abs(torques[0]), Tx_limit);
    ROS_INFO("Spiral. Stop if Ty: %.4f > %.4f", abs(torques[1]), Ty_limit);

    return ((forces[2]>minSpiralForce) &&
            (sqrt(pow(torques[0],2))<Tx_limit) &&
            (sqrt(pow(torques[1],2))<Ty_limit)); //If true, keep spiralling
}

void moveAbs(double x, double y, double z, double qw, double qx, double qy, double qz){
	//Waiting for services to be available
	ros::service::waitForService("cartesian_move",timeout);
    ros::ServiceClient moveCartClient = n_ptr->serviceClient<moveit_planner::MoveCart>("cartesian_move");
    moveit_planner::MoveCart cart;
    geometry_msgs::Pose p;
    p.position.x = x;
    p.position.y = y;
    p.position.z = z;
    p.orientation.w = qw;
    p.orientation.x = qx;
    p.orientation.y = qy; 
    p.orientation.z = qz;
	cart.request.val.push_back(p);
    cart.request.execute = true;
    moveCartClient.call(cart);
    ROS_INFO("Moving to Absolute Position");
}

void setGoalOrientation(){
    double roll, pitch, yaw;
    //Retrieve orientation parameters
    n_ptr->getParam("padlock_goal/roll",roll);
    n_ptr->getParam("padlock_goal/pitch",pitch);
    n_ptr->getParam("padlock_goal/yaw",yaw);  
    //Convert RPY orientations to Quaternion
    tf2::Quaternion q_padlock_goal;
    q_padlock_goal.setRPY(roll, pitch, yaw);
    q_padlock_goal.normalize();
    //Update global variable
    tf2::convert(q_padlock_goal, padlock_goal_or);
}

void moveRel(double x, double y, double z){
	//Waiting for services to be available
	ros::service::waitForService("get_transform",timeout);
    ros::service::waitForService("cartesian_move",timeout);
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
    p.orientation=padlock_goal_or;
    //Update position with relative changes
	p.position.x += x;
	p.position.y += y;
    p.position.z += z;

    cart.request.val.push_back(p);
    cart.request.execute = true;
    moveCartClient.call(cart);
}

void moveAbsSpiral(double x, double y, double z, bool useCurrentZ){
	//Waiting for services to be available
	ros::service::waitForService("cartesian_move",timeout);;
	ros::ServiceClient moveCartClient = n_ptr->serviceClient<moveit_planner::MoveCart>("cartesian_move");
    moveit_planner::MoveCart cart;
    geometry_msgs::Pose p;

    //Use current z-position for upcoming command
    if (useCurrentZ){
        ros::ServiceClient getTFClient_spiral_abs = n_ptr->serviceClient<moveit_planner::GetTF>("get_transform");
        moveit_planner::GetTF curTF_spiral_abs;
        curTF_spiral_abs.request.from="map"; //map
        curTF_spiral_abs.request.to="panda_link8"; //end_effector_link
        getTFClient_spiral_abs.call(curTF_spiral_abs);
        double current_z=curTF_spiral_abs.response.pose.position.z;
        p.position.z = current_z;
    } 
    //Or use an absolute z-position from args
    else{
        p.position.z = z;
    }

    //Set remaining orientation/position info
    p.orientation=padlock_goal_or;
	p.position.x = x;
	p.position.y = y;

    //Execute
    cart.request.val.push_back(p);
    cart.request.execute = true;
    moveCartClient.call(cart);
}

void calculateFTBias(){
    // Retrieve current FT readings
    ros::service::waitForService("getAveWrench",timeout);
    ros::ServiceClient getAveWrenchclient = n_ptr->serviceClient<lock_key::getAveWrench>("getAveWrench");
    lock_key::getAveWrench aveWrench;
    getAveWrenchclient.call(aveWrench);   
    // Store in global variables
    Fx_bias=aveWrench.response.fx;
    Fy_bias=aveWrench.response.fy;
    Fz_bias=aveWrench.response.fz;
    Tx_bias=aveWrench.response.tx;
    Ty_bias=aveWrench.response.ty;
    Tz_bias=aveWrench.response.tz;
    ROS_INFO("Successfully retrieved FT sensor biases.");
}

void insertPartSpiral(double Ft, double Fd, double Fi, double delta_max){
	// Implement modified Algorithm 1 from Watson, Miller, and Correll 2020 Paper here.

    // 11. MoveAbs - Move to insertion plane. Should already be
    //               accomplished by controller.cpp

    // 12. MoveRel - Move towards goal (xi,yi) until traveled 
    //               delta_max or MaxDownForce==false
    ROS_INFO("Moving to insertion plane");
    // Retrieve goal orientation
    setGoalOrientation();

    double delta{0.0};
    //Break delta up into smaller steps
    double delta_cmd;
    n_ptr->getParam("spiral/delta_step",delta_cmd); // Distance to move down per step
    while (delta<=delta_max && maxDownForce(Ft)){
        moveRel(0.0,0.0,-delta_cmd);
        ROS_INFO("delta: %f",delta);
        // Update delta. TODO: Use actual feedback rather than cmd
        delta+=delta_cmd;
    }
    if (delta>=delta_max){
        ROS_INFO("Exceeded max insertion plane detection distance.");
    }
    ROS_INFO("Arrived at insertion plane");
    // 13. MoveSpiral - Perform spiral motion while 
    //                  MaxSpiralForces is true
    double x{0}, y{0};           // Current EE position
    double prev_x{0}, prev_y{0}; // Previous EE position
    double nn{1.0}; // Iteration Counter
    double spiral_a, spiral_b, spiral_rad, spiral_rot, phi;
    double spiral_nmax; //Max number of iterations
    double initial_x, initial_y, initial_z; //initial position for spiral

    ROS_INFO("Retrieving spiral parameters");
    n_ptr->getParam("spiral/Tx",Tx_limit);
    n_ptr->getParam("spiral/Ty",Ty_limit);
    n_ptr->getParam("spiral/a",spiral_a);
    n_ptr->getParam("spiral/b",spiral_b);
    n_ptr->getParam("spiral/nmax",spiral_nmax);
    n_ptr->getParam("spiral/rot",spiral_rot);
    n_ptr->getParam("spiral/min_spiral_force",minSpiralForce);
    
    ROS_INFO("Starting Spiral Insertion Motion");

    // Get starting XY position for reference
    ros::ServiceClient getTFClient_spiral = n_ptr->serviceClient<moveit_planner::GetTF>("get_transform");
    moveit_planner::GetTF curTF_spiral;
    curTF_spiral.request.from="map"; //map
    curTF_spiral.request.to="panda_link8"; //end_effector_link
    getTFClient_spiral.call(curTF_spiral);
    initial_x=curTF_spiral.response.pose.position.x;
    initial_y=curTF_spiral.response.pose.position.y;
    initial_z=curTF_spiral.response.pose.position.z;

    while (nn<spiral_nmax && maxSpiralForces(Fd)){
        //Calculate new xy
        phi = sqrt(nn/spiral_nmax)*(spiral_rot*2.0*M_PI);
        spiral_rad=(spiral_a-spiral_b*phi);
        ROS_INFO("nn: %.4f, phi: %.4f", nn, phi);
        x+=spiral_rad*cos(phi);
        y+=spiral_rad*sin(phi);
        ROS_INFO("Spiral X: %.4f, Y: %.4f", x, y);
        //Send relative movement command
        moveAbsSpiral(x+initial_x,y+initial_y,initial_z,0);
        //Update parameters
        nn+=1.0;
    }

    ROS_INFO("Performing final insertion");

    // // 14. MoveRel - Finish insertion
    while (delta<=delta_max && maxDownForce(Fi)){
        moveRel(0.0,0.0,-delta_cmd);
        // Update delta. TODO: Use actual feedback rather than cmd
        delta+=delta_cmd;
    }

    ROS_INFO("Insertion complete!");

}

void execute(const lock_key::SpiralInsertGoalConstPtr& goal, Server* as){  // Note: "Action" is not appended here
    ROS_INFO("Beginning insertion procedure");

    // Ft: Max force when moving to insertion plane
    // Fd: Max wrist Force/Torque during spiral
    // Fi: Max Insertion Force
    // delta_max: acceptable travel in z-axis

    calculateFTBias();
    insertPartSpiral(goal->Ft, goal->Fd, goal->Fi, goal->delta_max);

    as->setSucceeded();
}

int main(int argc, char **argv){
    ros::init(argc, argv, "spiralInsert");
    ros::NodeHandle n;
    ros::Publisher fz_pub = n.advertise<std_msgs::Float64>("fz_bias_removed", 1000);

    n_ptr=&n;
    fz_pub_ptr=&fz_pub;

    Server server(n, "spiral_insert_key", boost::bind(&execute, _1, &server), false);
    server.start();

    ros::spin();
}