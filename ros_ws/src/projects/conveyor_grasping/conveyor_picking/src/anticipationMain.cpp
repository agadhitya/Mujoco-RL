#include <stdlib.h>
#include <string>
#include <math.h>
#include <random>
#include <chrono>

#include <ros/ros.h>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <moveit_planner/MoveCart.h>
#include <moveit_planner/MovePose.h>
#include <moveit_planner/MoveJoint.h>
#include <pose_estimator/SimEstimation.h>
#include <franka_gripper_gazebo/GripMsg.h>
#include <franka_gripper_gazebo/DelayedMotion.h>

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
double threshVel = 0.5;		// Maximum relative velocity between arm and object                 *
std::string model = "spawner"; 	// Name of model to get                                             *
//****************************************Default Parameters*****************************************

// Center + radius, minimal representation
struct Region {
  geometry_msgs::Point center;
  double radius;
};
// Start + End regions to mark boundary, also stores velocity bounds
struct URegion {
  Region start;
  Region end;
  double vMin, vMax;
};

// Checks whether object is in region
inline bool isInRegion(const geometry_msgs::Point& pos, const Region& reg) {
  double dX = pos.x - reg.center.x;
  double dY = pos.y - reg.center.y;
  return sqrt(dY*dY) <= reg.radius;
}

// Returns -1 if outside
double getUV(const URegion& ureg, const geometry_msgs::Point& pos, unsigned int steps) {
  // Quick checks
  if(pos.y < ureg.start.center.y - ureg.start.radius) // Behind region
    return -1;
  else if(pos.y > ureg.end.center.y + ureg.end.radius) // Ahead of region
    return -1;
  Region cur = ureg.end;	// Starting from the end of the path
  double stepSize = (ureg.end.center.y - ureg.start.center.y) / ((double) steps);
  double velStepSize = (ureg.vMax - ureg.vMin) / ((double) steps);
  for(int i = 0; i < steps; ++i) {
    cur.center.y -= stepSize;
    if(isInRegion(pos, cur))
      return ureg.vMax - velStepSize * i;
  }

  return -1;			// Not in URegion
}

// End is the front, start is the back (start/end are relative to the direction of y)
URegion getURegion(const Region& initRegion, double vEst, double vErr, double time) {
  URegion ret;

  // Velocities
  ret.vMin = vEst - vErr;
  ret.vMax = vEst + vErr;

  // Init
  ret.start = initRegion;
  ret.end = initRegion;

  // Shift y's
  ret.start.center.y = initRegion.center.y + (ret.vMin) * time;
  ret.end.center.y = initRegion.center.y + (ret.vMax) * time;

  return ret;
}

inline double getArmVel(double objVel, double threshVel) {
  return (0 > (objVel - threshVel)) ? 0 : (objVel - threshVel);
}

std::vector<double> startJointVals; // TODO: Fill this up with joint values

int main(int argc, char** argv) {
  ros::init(argc, argv, "anticipation_grasp_node");
  ros::NodeHandle nh;

  if(argc != 12) {
    ROS_ERROR("Please provide 11 arguments");
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
  threshVel = atof(argv[10]);
  model = std::string(argv[11]);
  ROS_INFO("Obtained all arguments");

  // Services, Clients, and msgs pre-created to avoid interfering with grasping
  ROS_INFO("Setting up services");
  ros::ServiceClient movePoseClient = nh.serviceClient<moveit_planner::MovePose>("move_to_pose");
  ros::ServiceClient moveCartClient = nh.serviceClient<moveit_planner::MoveCart>("cartesian_move");
  ros::ServiceClient poseClient = nh.serviceClient<pose_estimator::SimEstimation>("get_object_pose");
  ros::ServiceClient gripClient = nh.serviceClient<franka_gripper_gazebo::GripMsg>("gazebo_franka_grip");

  ros::Publisher delayedMotionPub = nh.advertise<franka_gripper_gazebo::DelayedMotion>("gazebo_franka_delayed", 1);

  moveit_planner::MovePose poseReq;
  moveit_planner::MoveCart cartReq;
  pose_estimator::SimEstimation estMsg;
  franka_gripper_gazebo::GripMsg gripMsg;
  franka_gripper_gazebo::DelayedMotion delMotion;

  // Make sure gripper is open
  gripMsg.request.force = 0.2;
  gripClient.call(gripMsg);

  // Estimate pose
  ROS_INFO("Obtaining object pose");
  estMsg.request.model_name = model;
  estMsg.request.time_to_wait = pDelay; // WAIT for pDelay before acquiring
  estMsg.request.err_x = pErr;		// Set errors
  estMsg.request.err_y = pErr;
  estMsg.request.err_z = pErr;
  estMsg.request.err_angle_z = aErr;
  if(!poseClient.call(estMsg)) {
    ROS_ERROR("Could not obtain pose");
    return 1;
  }
  
  ROS_INFO_STREAM("Obtained estimated pose of object\n" <<
		  estMsg.response.pose << " at " <<
		  estMsg.response.time);

  // Estimate velocity
  ROS_INFO("Estimating velocity");
  ros::Duration velDuration(vDelay);
  double actualSpeed;
  nh.getParam("/conveyor/speed", actualSpeed);
  // Create random generators
  std::default_random_engine gen;
  gen.seed(ros::WallTime::now().toSec());
  std::uniform_real_distribution<double> urd{-1.0, 1.0};
  // Add error to velocity
  double estSpeed = actualSpeed + urd(gen)*vErr;
  // Keep speed above 0 (SHOULD NEVER HAPPEN)
  estSpeed = (estSpeed > 0) ? estSpeed : 0;
  // Wait
  velDuration.sleep();
  ROS_INFO_STREAM("Estimated velocity as " << estSpeed);

  // Setup region
  Region initRegion;
  initRegion.center = estMsg.response.pose.position;
  initRegion.radius = pErr;

  // Get earliest point in region
  // Where is the object now?
  double initialPoseTime = ros::WallTime::now().toSec();
  ROS_INFO_STREAM("Factoring delays, the time is now " << initialPoseTime);
  double timeFromBeginning = initialPoseTime + interceptionTime; // How much time is x seconds from now?
  ROS_INFO_STREAM("Attempting to catch object at time " << timeFromBeginning << ", which is " <<
		  timeFromBeginning - estMsg.response.time << "s from the measurement of pose");
  URegion interceptionURegion = getURegion(initRegion, estSpeed, vErr, timeFromBeginning - estMsg.response.time);
  ROS_INFO_STREAM("Uncertainty region starts at " << interceptionURegion.start.center.y <<
		  " and ends at " << interceptionURegion.end.center.y <<
		  " with radius " << interceptionURegion.start.radius);
  ROS_INFO_STREAM("Uncertainty region velocity varies between " << interceptionURegion.vMin <<
		  " and " << interceptionURegion.vMax);
  // Farthest point to the right is where the arm should go to:
  double interceptionY = interceptionURegion.end.center.y + interceptionURegion.end.radius;
  ROS_INFO_STREAM("Intercepting object at " << interceptionY);

  // Move to predefined start position
  geometry_msgs::Pose targetPose;
  targetPose.position.x = 0.5 - 0.025;
  targetPose.position.y = interceptionY;
  targetPose.position.z = 0.27;
  targetPose.orientation.x = -0.5;
  targetPose.orientation.y = -0.5;
  targetPose.orientation.z = 0.5;
  targetPose.orientation.w = -0.5;
  poseReq.request.val = targetPose;
  poseReq.request.execute = true;
  movePoseClient.call(poseReq);

  ROS_INFO("Calculating velocity profile");
  double dt = 0.05;		// Seconds between each motion
  double steps = 100;		// Steps when calculating velocity
  bool prevState = false;
  double armVel = 0;
  URegion curURegion = interceptionURegion;
  geometry_msgs::Point curPoint = targetPose.position;
  std::vector<double> armVels;
  std::vector<double> armPos;
  std::vector<double> armTimes;

  // Initial values
  armPos.push_back(curPoint.y - 0.01); // Shift slightly to keep in region
  armVels.push_back(getArmVel(interceptionURegion.vMax, threshVel));
  armTimes.push_back(0);
  double urVel = 0;
  while(urVel != -1) {		// Once the arm doesn't need to move, it won't have to anymore
    curPoint.y = armPos.back();
    // Project state forwards
    curURegion = getURegion(initRegion, estSpeed, vErr,
			    (timeFromBeginning - estMsg.response.time) + dt * armVels.size()); // Region
    armPos.push_back(armPos.back() + armVels.back() * dt); // Position
    urVel = getUV(curURegion, curPoint, steps);
    armVels.push_back(getArmVel(urVel, threshVel));
    armTimes.push_back(dt * armVels.size());
  }

  ROS_INFO("Obtained velocity profile: ");

  // Get average velocity in profile
  double avgVel = 0;
  double nonZeroCount = 0;
  int lastNonZero = 0;
  for(int i = 0; i < armVels.size(); ++i) {
    if(armVels[i] > 0) {
      nonZeroCount++;
      lastNonZero = i;
    }
    avgVel += armVels[i];
  }
  if(nonZeroCount != 0)
    avgVel /= nonZeroCount;

  // Wait until we enter region
  double curTime = ros::WallTime::now().toSec();
  ROS_INFO_STREAM("Waiting " << timeFromBeginning - curTime << " until we enter region");
  ros::Duration intRegionDur(timeFromBeginning - curTime);
  intRegionDur.sleep();

  double timePassed = 0;
  // Got average velocity, sending message
  if(avgVel == 0 || lastNonZero == 0) {
    ROS_INFO("No motion to execute");
    // No motion here
  }
  else {
    ROS_INFO("Executing motion");
    geometry_msgs::Pose endPose = targetPose;
    endPose.position.y = armPos[lastNonZero];
    cartReq.request.val.push_back(endPose);
    cartReq.request.time = dt * lastNonZero;
    cartReq.request.execute = true;
    moveCartClient.call(cartReq);
    timePassed = cartReq.request.time;
  }

  // Wait remainder of time
  ROS_INFO_STREAM("Waiting for " << dt * armPos.size() - timePassed << "s for the region to disappear");
  ros::Duration waitUntilRegion(dt * armPos.size() - timePassed);
  waitUntilRegion.sleep();

  // Close gripper
  gripMsg.request.force = -gripMsg.request.force;
  gripClient.call(gripMsg);

  // DONE! Move up
  ROS_INFO("Done grasping object");
  poseReq.request.val.position.z += 0.1;
  movePoseClient.call(poseReq);

  return 0;
}
