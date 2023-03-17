#include <stdlib.h>
#include <string>
#include <math.h>
#include <random>
#include <chrono>

#include <ros/ros.h>

#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <moveit_planner/MoveCart.h>
#include <moveit_planner/MovePose.h>
#include <moveit_planner/MoveAway.h>
#include <moveit_planner/Inv.h>
#include <pose_estimator/SimEstimation.h>
#include <franka_gripper_gazebo/GripMsg.h>
#include <franka_gripper_gazebo/DelayedMotion.h>

#include <Eigen/Dense>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;

//****************************************Default Parameters*****************************************
double interceptionTime = 2;	// How many seconds from mug detection do we begin approach?        *
double approachDist = 0.3;	// How many meters away from the mug should we be when approaching  *
double approachVel = 0.3;	// How quickly should we approach the mug?                          *
double pDelay = 0.0;		// How much time does it take to get the mug pose?                  *
double vDelay = 0.0;		// How much time does it take to get the mug velocity?              *
double pErr = 0.0;		// Error on the position (x/y)                                      *
double aErr = 0.0;		// Error on the mug angle                                           *
double vErr = 0.0;		// Error on the mug velocity                                        *
double allowance = 0.4;		// How much time to wait before moving to grasp?                    *
std::string model = "spawner"; 	// Name of model to get                                             *
//****************************************Default Parameters*****************************************

Eigen::MatrixXd poseToTrans(const geometry_msgs::Pose& pose) {
  Eigen::MatrixXd retTrans(4, 4);
  retTrans = Eigen::MatrixXd::Zero(4, 4);
  Eigen::Quaterniond retQuat;
  retQuat.x() = pose.orientation.x;
  retQuat.y() = pose.orientation.y;
  retQuat.z() = pose.orientation.z;
  retQuat.w() = pose.orientation.w;
  retTrans.topLeftCorner(3, 3) = retQuat.normalized().toRotationMatrix();
  retTrans(0,3) = pose.position.x;
  retTrans(1,3) = pose.position.y;
  retTrans(2,3) = pose.position.z;
  retTrans(3,3) = 1;

  return retTrans;
}

geometry_msgs::Pose getMugGraspPose(const geometry_msgs::Pose& mugPose) {
  Eigen::MatrixXd graspPosition(4, 4);
  graspPosition = Eigen::MatrixXd::Zero(4, 4);
  // Rotation
  graspPosition(0,1) = -1;	// x-axis
  graspPosition(0,0) = 0;	// x-axis
  graspPosition(1,0) = -1;	// y-axis
  graspPosition(1,1) = 0;	// y-axis
  graspPosition(2,2) = -1;	// z-axis
  // Position
  graspPosition(0,3) = -0.0587;
  graspPosition(1,3) = 0;
  graspPosition(2,3) = 0.06069;
  graspPosition(3,3) = 1;

  // Convert from quat + position to transform matrix
  Eigen::MatrixXd objectPose = poseToTrans(mugPose);

  // Multiply to get final transform matrix
  Eigen::MatrixXd finalGraspTransform = objectPose * graspPosition;

  // Convert to quat
  Eigen::Matrix3d rot = finalGraspTransform.topLeftCorner(3, 3);
  Eigen::Quaterniond finalGraspQuat{rot};
  
  // Send final transform to server
  double graspDist = 0.3;
  geometry_msgs::Pose graspPose;
  graspPose.orientation.x = finalGraspQuat.x();
  graspPose.orientation.y = finalGraspQuat.y();
  graspPose.orientation.z = finalGraspQuat.z();
  graspPose.orientation.w = finalGraspQuat.w();
  graspPose.position.x = finalGraspTransform(0,3);
  graspPose.position.y = finalGraspTransform(1,3);
  graspPose.position.z = finalGraspTransform(2,3);

  return graspPose;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "conveyor_picking_main_node");
  ros::NodeHandle nh;

  if(argc != 11) {
    ROS_ERROR("Please provide 10 arguments");
    return 1;
  }
  // Given arguments
  interceptionTime = atof(argv[1]);
  approachDist = atof(argv[2]);
  approachVel = atof(argv[3]);
  pDelay = atof(argv[4]);
  vDelay = atof(argv[5]);
  pErr = atof(argv[6]);
  aErr = atof(argv[7]);
  vErr = atof(argv[8]);
  allowance = atof(argv[9]);
  model = std::string(argv[10]);
  ROS_INFO("Obtained all arguments");

  // Services, Clients, and msgs pre-created to avoid interfering with grasping
  ROS_INFO("Setting up services");
  ros::ServiceClient movePoseClient = nh.serviceClient<moveit_planner::MovePose>("move_to_pose");
  ros::ServiceClient moveAwayClient = nh.serviceClient<moveit_planner::MoveAway>("move_away_point");
  ros::ServiceClient moveCartClient = nh.serviceClient<moveit_planner::MoveCart>("cartesian_move");
  ros::ServiceClient invClient = nh.serviceClient<moveit_planner::Inv>("inverse_kinematics");
  ros::ServiceClient poseClient = nh.serviceClient<pose_estimator::SimEstimation>("get_object_pose");
  ros::ServiceClient gripClient = nh.serviceClient<franka_gripper_gazebo::GripMsg>("gazebo_franka_grip");

  ros::Publisher jointTrajPub = nh.advertise<trajectory_msgs::JointTrajectory>("/panda/gazebo_ros_control/command", 1);
  ros::Publisher delayedMotionPub = nh.advertise<franka_gripper_gazebo::DelayedMotion>("gazebo_franka_delayed", 1);

  moveit_planner::MovePose poseReq;
  moveit_planner::MoveAway awayReq;
  moveit_planner::MoveCart cartReq;
  pose_estimator::SimEstimation estMsg;
  trajectory_msgs::JointTrajectory graspTraj;
  franka_gripper_gazebo::GripMsg gripMsg;
  franka_gripper_gazebo::DelayedMotion delMotion;

  // Make sure gripper is open
  gripMsg.request.force = 0.2;
  gripClient.call(gripMsg);

  // Estimate pose
  ROS_INFO("Obtaining object pose");
  estMsg.request.model_name = model;
  estMsg.request.time_to_wait = pDelay; // WAIT for pDelay before aquiring
  estMsg.request.err_x = pErr;		// Set errors
  estMsg.request.err_y = pErr;
  estMsg.request.err_z = pErr;
  estMsg.request.err_angle_z = aErr;
  if(!poseClient.call(estMsg)) {
    ROS_ERROR("Could not obtain pose");
    return 1;
  }
  
  ROS_INFO_STREAM("Obtained estimated pose of object " <<
		  estMsg.response.pose << " at " <<
		  estMsg.response.time);

  // Estimate velocity
  ROS_INFO("Estimating velocity");
  ros::Duration velDuration(vDelay);
  double actualSpeed;
  nh.getParam("/conveyor/speed", actualSpeed);
  // Create random generators
  std::default_random_engine gen;
  std::uniform_real_distribution<double> urd{-1.0, 1.0};
  // Add error to velocity
  double estSpeed = actualSpeed + urd(gen)*vErr;
  // Wait
  velDuration.sleep();
  ROS_INFO_STREAM("Estimated velocity as " << estSpeed);

  // Going to where the mug is
  geometry_msgs::Pose curPose = estMsg.response.pose;
  curPose.position.y += ((ros::WallTime::now().toSec() - estMsg.response.time) * estSpeed); // Where is it now?
  ROS_INFO_STREAM("Time passed: " << (ros::WallTime::now().toSec() - estMsg.response.time) <<
		  "\nEstimated pose: " << curPose);
  ROS_INFO_STREAM("Projecting mug pose " << interceptionTime << " seconds ahead");
  double curPoseTime = ros::WallTime::now().toSec();
  geometry_msgs::Pose approachPose = curPose; // Starting from where it is now
  approachPose.position.y += (interceptionTime * estSpeed);
  ROS_INFO_STREAM("Obtained future pose " << approachPose);
  // Now, lets see if we can make it there in time
  ROS_INFO("Moving above expected location");
  awayReq.request.execute = false;
  awayReq.request.distance = approachDist;
  awayReq.request.pose = getMugGraspPose(approachPose);
  moveAwayClient.call(awayReq);

  double totalTime = awayReq.response.planning_time + awayReq.response.arrival_time.toSec();
  ROS_INFO_STREAM("Estimated time to arrive: " << totalTime);

  if(totalTime > interceptionTime) {
    ROS_ERROR("Could not arrive at designated location in time");
    return 1;
  }
  else {			// Actually execute motion
    ROS_INFO("Executing motion");
    awayReq.request.execute = true;
    moveAwayClient.call(awayReq);

    double timeToWait = interceptionTime - (ros::WallTime::now().toSec() - curPoseTime);
    ros::Duration d(timeToWait - allowance);
    ROS_INFO_STREAM("Waiting " << timeToWait << " until object reaches position");
    d.sleep();

    ROS_INFO("Descending to grasp & queuing grip");
    geometry_msgs::Pose graspTimePose = getMugGraspPose(approachPose);
    double timeToIntercept = approachDist / approachVel;
    graspTimePose.position.y += (timeToIntercept * estSpeed);
    cartReq.request.val.push_back(graspTimePose);
    cartReq.request.time = timeToIntercept;
    cartReq.request.execute = true;
    delMotion.delay = timeToIntercept;
    delMotion.force = -gripMsg.request.force;
    delayedMotionPub.publish(delMotion); // Publish grip command
    moveCartClient.call(cartReq);	 // Move to interception
    ROS_INFO("Reached grasp position & grasped (hopefully)");

    // Now move up
    awayReq.request.pose = graspTimePose;
    awayReq.request.distance = 0.3;
    moveAwayClient.call(awayReq);
    // DONE!
  }

  return 0;
}
