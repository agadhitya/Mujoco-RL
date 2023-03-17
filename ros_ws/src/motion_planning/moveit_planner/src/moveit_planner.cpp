#include "moveit_planner.hpp"
#include "moveit_planner/GetTF.h"

// moveit_msgs includes
#include "moveit_msgs/RobotTrajectory.h"

#include "geometry_msgs/PoseStamped.h"

namespace moveit_planner {
  // Constructor/Destructor
  MoveitPlanner::MoveitPlanner(ros::NodeHandle& nh, const std::string& arm)
    : n{nh}, spinner{0}, armGroup{arm}, moveGroup{armGroup}, curScaling{1.0},
      eef_step{0.005}, jump_threshold{0} {
    spinner.start();

    setupServices();
    setupMoveit();

    ros::waitForShutdown();
  }

  MoveitPlanner::~MoveitPlanner() {}

  bool MoveitPlanner::getPose(geometry_msgs::Pose& pose) {
    // TODO: Performance improvements
    ROS_INFO("Waiting for /get_transform service");
    if(ros::service::waitForService("/get_transform", ros::Duration(5.0))) {
      ROS_INFO("Service found");

      ros::ServiceClient tempClient = n.serviceClient<moveit_planner::GetTF>("/get_transform");
      moveit_planner::GetTF tempMsg;

      tempMsg.request.from = baseLink;
      tempMsg.request.to = endEffector;
      if(tempClient.call(tempMsg)) {
        pose = tempMsg.response.pose;
        return true;
      }
      else {
        ROS_ERROR_STREAM("Could not obtain transform from pose_node");
        return false;
      }
    }
    else {
      ROS_ERROR("Could not find required service, is pose_node running?");
      return false;
    }
  }
  bool MoveitPlanner::getPoseClientCallback(moveit_planner::GetPose::Request& req,moveit_planner::GetPose::Response& res) {
    return getPose(res.pose);
  }

  bool MoveitPlanner::namedStateCallback(moveit_planner::MoveNamedState::Request& req,moveit_planner::MoveNamedState::Response& res){
    // std::map<std::string,double> jointValues = ;
    moveGroup.setJointValueTarget(moveGroup.getNamedTargetValues(req.name));
    auto errCode = moveGroup.plan(curPlan);
    bool success = (errCode == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    if(!success) ROS_WARN_STREAM("ERROR MOVING TO NAMED STATE " << errCode);
    else         moveGroup.execute(curPlan);
    return success;
  }

  bool MoveitPlanner::moveToPose(const geometry_msgs::Pose& p, const bool& exe) {
    moveGroup.setPoseTarget(p);
    auto errCode = moveGroup.plan(curPlan);
    bool success = (errCode == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    if(!success)
      ROS_WARN_STREAM("ERROR MOVING TO POSE " << errCode);
    if(success && exe)
      moveGroup.execute(curPlan);

    return success;
  }
  bool MoveitPlanner::poseClientCallback(moveit_planner::MovePose::Request& req,moveit_planner::MovePose::Response& res) {
    bool success = moveToPose(req.val, req.execute);

    // Set timing info
    res.planning_time = curPlan.planning_time_;
    res.arrival_time =
      curPlan.trajectory_
      .joint_trajectory
      .points.back()
      .time_from_start;

    return success;
  }

  bool MoveitPlanner::moveToJointSpace(const std::vector<double>& jointPositions, bool exe) {
    moveGroup.setJointValueTarget(jointPositions);
    auto errCode = moveGroup.plan(curPlan);
    bool success = (errCode == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    if(!success)
      ROS_WARN_STREAM("ERROR MOVING TO JOINT " << errCode);
    if(success && exe)
      moveGroup.execute(curPlan);

    return success;
  }
  bool MoveitPlanner::jsClientCallback(moveit_planner::MoveJoint::Request& req,moveit_planner::MoveJoint::Response& res) {
    return moveToJointSpace(req.val, req.execute);
  }

  bool MoveitPlanner::cartesianMove(const std::vector<geometry_msgs::Pose>& p, double setTime, const bool& exe) {
    if(p.size() == 0)
      return true;		// Empty waypoints, nothing to execute

    ROS_INFO_STREAM("Beginning trajectory at " << ros::Time::now().toSec());
    moveit_msgs::RobotTrajectory trajectory;
    moveit::planning_interface::MoveGroupInterface::Plan curPlan;
    ROS_INFO_STREAM("Created objects at " << ros::Time::now().toSec());

    moveGroup.computeCartesianPath(p, eef_step, jump_threshold, curPlan.trajectory_);
    ROS_INFO_STREAM("Computed cartesian path at " << ros::Time::now().toSec());
    double start = ros::Time::now().toSec();
    if(setTime != 0) {		// Set the time of arrival
      ROS_INFO("Scaling plan");
      double trajTime = curPlan.trajectory_.joint_trajectory.points.back().time_from_start.toSec();
      double factor = setTime / trajTime; // How much to scale time by
      // Ex: 0->1s->2s => Cartesian arrives @ 2s
      //     We want to arrive at 1s
      //     Multiply by 1/2 -> 0.5:
      //     0->0.5s->1s => We now arrive at 1
      //     For velocity/acceleration we need to divide instead (less time, faster speed)
      for(int i = 0; i < curPlan.trajectory_.joint_trajectory.points.size(); ++i) { // Scale each point
        curPlan.trajectory_.joint_trajectory.points[i].time_from_start *= factor;
        for(int j = 0; j < curPlan.trajectory_.joint_trajectory.points[i].velocities.size(); ++j) { // Scale each joint
          curPlan.trajectory_.joint_trajectory.points[i].velocities[j] /= factor;
          curPlan.trajectory_.joint_trajectory.points[i].accelerations[j] /= factor;
        }
        ROS_INFO_STREAM("Point " << i << " starts at " << curPlan.trajectory_.joint_trajectory.points[i].time_from_start);
      }
    }
    double end = ros::Time::now().toSec();
    ROS_INFO_STREAM("Calculations took " << end - start);

    ROS_INFO_STREAM("Executing path at " << ros::Time::now().toSec());
    if(exe)
      moveGroup.execute(curPlan);
    ROS_INFO_STREAM("Execution done at " << ros::Time::now().toSec());

    // TODO: Check how to get success
    return true;
  }
  bool MoveitPlanner::cartesianMoveCallback(moveit_planner::MoveCart::Request& req,moveit_planner::MoveCart::Response& res) {
    return cartesianMove(req.val, req.time, req.execute);
  }

  bool MoveitPlanner::distanceAwayCallback(moveit_planner::MoveAway::Request& req,moveit_planner::MoveAway::Response& res) {
    geometry_msgs::Quaternion desiredQuat = req.pose.orientation;

    // Calculate z-axis direction
    geometry_msgs::Point desiredZ;
    desiredZ.x = 2*desiredQuat.x*desiredQuat.z + 2*desiredQuat.w*desiredQuat.y;
    desiredZ.y = 2*desiredQuat.y*desiredQuat.z - 2*desiredQuat.w*desiredQuat.x;
    desiredZ.z = 1 - 2*desiredQuat.x*desiredQuat.x - 2*desiredQuat.y*desiredQuat.y;

    // Calculate new pose
    geometry_msgs::Pose targetPose = req.pose;
    targetPose.position.x = targetPose.position.x - desiredZ.x * req.distance;
    targetPose.position.y = targetPose.position.y - desiredZ.y * req.distance;
    targetPose.position.z = targetPose.position.z - desiredZ.z * req.distance;

    // Setup response
    res.awayPose = targetPose;

    // TODO: Standardise timing info
    // Timing info & Planning
    bool success = moveToPose(targetPose, req.execute);
    res.planning_time = curPlan.planning_time_;
    res.arrival_time =
      curPlan.trajectory_
      .joint_trajectory
      .points.back()
      .time_from_start;

    return success;
  }

  bool MoveitPlanner::setVelocityCallback(moveit_planner::SetVelocity::Request& req,moveit_planner::SetVelocity::Response& res) {
    if(req.velScaling <= 0 || req.velScaling > 1) // Invalid scaling
      return false;

    moveGroup.setMaxVelocityScalingFactor(req.velScaling);
    moveGroup.setMaxAccelerationScalingFactor(req.velScaling);
    curScaling = req.velScaling;
    return true;
  }

  bool MoveitPlanner::addCollisionObjects(const std::vector<moveit_msgs::CollisionObject>& collObjs) {
    planningSceneInterface.applyCollisionObjects(collObjs);

    return true;
  }
  bool MoveitPlanner::addCollisionCallback(moveit_planner::AddCollision::Request& req, moveit_planner::AddCollision::Response& res) {
    // Place in a vector to use general private function
    std::vector<moveit_msgs::CollisionObject> collObjects;
    collObjects.push_back(req.collObject);

    // Return whether it was successfull or not
    return addCollisionObjects(collObjects);
  }

  bool MoveitPlanner::setConstraints(const moveit_msgs::Constraints& constraints) {
    moveGroup.setPathConstraints(constraints);
    return true;
  }
  bool MoveitPlanner::setConstraintsCallback(moveit_planner::SetConstraints::Request& req, moveit_planner::SetConstraints::Response& res) {
    return setConstraints(req.constraints);
  }

  bool MoveitPlanner::clearConstraintsCallback(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res){
    moveGroup.clearPathConstraints();
    return true;
  }

  bool MoveitPlanner::ik(const geometry_msgs::Pose& pose, std::vector<double>& results) {
    double timeout = 0.1;	// Time before quitting (no solution found)
    int attempts = 2;		// Number of tries
    bool found = curState->setFromIK(jointModelGroup, pose, attempts, timeout);

    curState->copyJointGroupPositions(jointModelGroup, results);

    return found;
  }
  bool MoveitPlanner::invClientCallback(moveit_planner::Inv::Request& req, moveit_planner::Inv::Response& res) {
    return ik(req.pose, res.joint_vals);
  }

  bool MoveitPlanner::ikWithCollCheck(const geometry_msgs::Pose& pose, std::vector<double>& results) {
    double timeout = 0.1;	// Time before quitting (no solution found)
    int attempts = 2;		// Number of tries
    bool ok = curState->setFromIK(jointModelGroup, pose, attempts, timeout);
    curState->copyJointGroupPositions(jointModelGroup, results);

    // static planning_scene::PlanningScene planning_scene(robotModel);
    collision_result.clear();
    // planning_scene.checkSelfCollision(collision_request, collision_result, *curState);
    planning_scene_monitor::LockedPlanningSceneRO(planningSceneMonitorPtr)->checkCollision(collision_request, collision_result, *curState);
    ok *= !(collision_result.collision);
    // std::cout << "Test 2: Current state is " << collision_result.collision << std::endl;
    return ok;
  }
  bool MoveitPlanner::invCollCheckClientCallback(moveit_planner::Inv::Request& req,moveit_planner::Inv::Response& res) {
    return ikWithCollCheck(req.pose, res.joint_vals);
  }

  bool MoveitPlanner::oneJointWithTimeCallback(moveit_planner::SetJointWithTime::Request& req,moveit_planner::SetJointWithTime::Response& res) {
    moveGroup.setJointValueTarget(req.joint_name,req.joint_angle);
    auto errCode = moveGroup.plan(curPlan);
    bool success = (errCode == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    if(success)
      moveGroup.execute(curPlan);
    else
      ROS_WARN_STREAM("ERROR MOVING THE JOINT " << errCode);

    return success;
  }

  bool MoveitPlanner::rotClientCallback(moveit_planner::MoveQuat::Request& req,moveit_planner::MoveQuat::Response& res) {
    geometry_msgs::PoseStamped curPose = moveGroup.getCurrentPose();
    ROS_INFO_STREAM("Current pose: " << curPose);
    geometry_msgs::Pose targetPose = curPose.pose;
    targetPose.orientation = req.val;

    return moveToPose(targetPose, req.execute);
  }

  // Private
  void MoveitPlanner::setupServices() {
    // TOOD: Setup remaining services
    getPoseClient = n.advertiseService("get_pose", &MoveitPlanner::getPoseClientCallback, this);
    namedStateClient = n.advertiseService("move_to_named_state", &MoveitPlanner::namedStateCallback, this);
    poseClient = n.advertiseService("move_to_pose", &MoveitPlanner::poseClientCallback, this);
    jsClient = n.advertiseService("move_to_joint_space", &MoveitPlanner::jsClientCallback, this);
    cartesianClient = n.advertiseService("cartesian_move", &MoveitPlanner::cartesianMoveCallback, this);
    distanceAwayClient = n.advertiseService("move_away_point", &MoveitPlanner::distanceAwayCallback, this);
    velocityClient = n.advertiseService("set_velocity_scaling", &MoveitPlanner::setVelocityCallback, this);
    addCollClient = n.advertiseService("add_collision_object", &MoveitPlanner::addCollisionCallback, this);
    setConstClient = n.advertiseService("set_constraints", &MoveitPlanner::setConstraintsCallback, this);
    clearConstClient = n.advertiseService("clear_constraints", &MoveitPlanner::clearConstraintsCallback, this);
    invClient = n.advertiseService("inverse_kinematics", &MoveitPlanner::invClientCallback, this);
    invCollCheckClient = n.advertiseService("inverse_kinematics_collision_check", &MoveitPlanner::invCollCheckClientCallback, this);
    oneJointWithTimeClient = n.advertiseService("one_joint_with_time", &MoveitPlanner::oneJointWithTimeCallback, this);

    // Not currently working, need to debug joint_states issue
    // rotClient = n.advertiseService("move_to_orientation", &MoveitPlanner::rotClientCallback, this);

  }

  void MoveitPlanner::setupMoveit() {
    ros::spinOnce();

    endEffector = moveGroup.getEndEffectorLink();

    std::vector<std::string> links = moveGroup.getLinkNames();
    baseLink = links[0];

    // Setup core objects
    robotModelLoader = robot_model_loader::RobotModelLoader{"robot_description"};
    robotModel = robotModelLoader.getModel();
    curState = moveit::core::RobotStatePtr{new moveit::core::RobotState{robotModel}};
    curState->setToDefaultValues();
    jointModelGroup = robotModel->getJointModelGroup(armGroup);

    planningSceneMonitorPtr = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>("robot_description");
    bool success = planningSceneMonitorPtr->requestPlanningSceneState("/get_planning_scene");
    planningSceneMonitorPtr->startSceneMonitor("/move_group/monitored_planning_scene");
  }


}
