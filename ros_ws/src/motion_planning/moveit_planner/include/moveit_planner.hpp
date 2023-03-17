// Header file
#ifndef MOVEIT_PLANNER_H
#define MOVEIT_PLANNER_H

// std includes
#include <string>
#include <vector>

// ros includes
#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>

// custom srv includes
#include "moveit_planner/Inv.h"
#include "moveit_planner/GetPose.h"
#include "moveit_planner/MoveQuat.h"
#include "moveit_planner/MovePose.h"
#include "moveit_planner/MoveCart.h"
#include "moveit_planner/MoveAway.h"
#include "moveit_planner/MoveJoint.h"
#include "moveit_planner/MovePoint.h"
#include "moveit_planner/SetVelocity.h"
#include "moveit_planner/AddCollision.h"
#include "moveit_planner/SetConstraints.h"
#include "moveit_planner/MoveNamedState.h"
#include "moveit_planner/SetJointWithTime.h"
#include "std_srvs/Empty.h"

// moveit includes
#include "moveit_msgs/Constraints.h"
#include "moveit/robot_model/robot_model.h"
#include "moveit/robot_state/robot_state.h"
#include "moveit/robot_trajectory/robot_trajectory.h"
#include "moveit/robot_model_loader/robot_model_loader.h"
#include "moveit/move_group_interface/move_group_interface.h"
#include "moveit/planning_scene/planning_scene.h"
#include "moveit/planning_scene_interface/planning_scene_interface.h"
#include "moveit/planning_scene_monitor/planning_scene_monitor.h"
#include "moveit/trajectory_processing/iterative_time_parameterization.h"

namespace moveit_planner {

  class MoveitPlanner {
  public:
    MoveitPlanner(ros::NodeHandle&, const std::string& arm);
    ~MoveitPlanner();

    // Movement commands
    // bool moveToOrientation(const geometry_msgs::Quaternion& r, bool exe);
    // bool moveToPosition(const geometry_msgs::Point& p, bool exe);
    bool getPose(geometry_msgs::Pose& pose);
    bool moveToPose(const geometry_msgs::Pose& p, const bool& exe);
    bool moveToJointSpace(const std::vector<double>& jointPositions, bool exe);
    bool cartesianMove(const std::vector<geometry_msgs::Pose>& p, double setTime, const bool& exe);
    bool addCollisionObjects(const std::vector<moveit_msgs::CollisionObject>& collObjs);
    bool setConstraints(const moveit_msgs::Constraints& constraints);
    bool ik(const geometry_msgs::Pose& pose, std::vector<double>& results);
    bool ikWithCollCheck(const geometry_msgs::Pose& pose, std::vector<double>& results);

    // Getters/Setters
    std::string getGroup() {return armGroup;};
  private:
    ros::NodeHandle n;
    ros::AsyncSpinner spinner;
    tf::TransformListener listener;
    std::string armGroup, endEffector, baseLink;

    // Setup
    void setupServices();
    void setupMoveit();

    // moveit stuff
    moveit::planning_interface::MoveGroupInterface moveGroup;
    moveit::planning_interface::PlanningSceneInterface planningSceneInterface;
    moveit::planning_interface::MoveGroupInterface::Plan curPlan;
    planning_scene_monitor::PlanningSceneMonitorPtr planningSceneMonitorPtr;
    collision_detection::CollisionResult collision_result;
    collision_detection::CollisionRequest collision_request;
    // Core moveit objects
    robot_model_loader::RobotModelLoader robotModelLoader; // NOTE: Is constructed from robot_description directly
    moveit::core::RobotModelPtr robotModel;
    moveit::core::RobotStatePtr curState;
    moveit::core::JointModelGroup* jointModelGroup;

    std::vector<double> curJointPositions;
    double curScaling;

    // Services/Topics
    ros::ServiceServer getPoseClient;
    ros::ServiceServer poseClient;
    ros::ServiceServer namedStateClient;
    ros::ServiceServer rotClient;
    ros::ServiceServer jsClient;
    ros::ServiceServer cartesianClient;
    ros::ServiceServer distanceAwayClient;
    ros::ServiceServer velocityClient;
    ros::ServiceServer addCollClient;
    ros::ServiceServer setConstClient;
    ros::ServiceServer clearConstClient;
    ros::ServiceServer invClient;
    ros::ServiceServer invCollCheckClient;
    ros::ServiceServer oneJointWithTimeClient;

    // Callbacks
    bool getPoseClientCallback(moveit_planner::GetPose::Request& req,moveit_planner::GetPose::Response& res);
    bool poseClientCallback(moveit_planner::MovePose::Request& req,moveit_planner::MovePose::Response& res);
    bool namedStateCallback(moveit_planner::MoveNamedState::Request& req,moveit_planner::MoveNamedState::Response& res);
    bool rotClientCallback(moveit_planner::MoveQuat::Request& req,moveit_planner::MoveQuat::Response& res);
    bool jsClientCallback(moveit_planner::MoveJoint::Request& req,moveit_planner::MoveJoint::Response& res);
    bool cartesianMoveCallback(moveit_planner::MoveCart::Request& req,moveit_planner::MoveCart::Response& res);
    bool distanceAwayCallback(moveit_planner::MoveAway::Request& req,moveit_planner::MoveAway::Response& res);
    bool setVelocityCallback(moveit_planner::SetVelocity::Request& req,moveit_planner::SetVelocity::Response& res);
    bool addCollisionCallback(moveit_planner::AddCollision::Request& req,moveit_planner::AddCollision::Response& res);
    bool setConstraintsCallback(moveit_planner::SetConstraints::Request& req,moveit_planner::SetConstraints::Response& res);
    bool clearConstraintsCallback(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);
    bool invClientCallback(moveit_planner::Inv::Request& req,moveit_planner::Inv::Response& res);
    bool invCollCheckClientCallback(moveit_planner::Inv::Request& req,moveit_planner::Inv::Response& res);
    bool oneJointWithTimeCallback(moveit_planner::SetJointWithTime::Request& req,moveit_planner::SetJointWithTime::Response& res);

    // Misc
    bool checkSuccess();
    double eef_step;
    double jump_threshold;
  };
}

#endif
