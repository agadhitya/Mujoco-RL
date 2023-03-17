#include <ros/ros.h>
#include <moveit_planner.hpp>
#include <geometry_msgs/Pose.h>
#include <arm_controls/PoseChange.h>
#include <arm_controls/MoveStraight.h>
#include <gazebo_msgs/GetModelState.h>
#include <gazebo_msgs/GetLinkState.h>
#include <moveit_msgs/CollisionObject.h>
#include <shape_msgs/SolidPrimitive.h>


// ArmController class implementation

class ArmController{

private:
  ros::NodeHandle n_;

  // store current pose of the arm
  geometry_msgs::Pose pose;

  // update pose variable by reading it from ros topic
  void update_pose(){
    ros::service::waitForService("gazebo/get_link_state");
    ros::ServiceClient client_pose = n_.serviceClient<gazebo_msgs::GetLinkState>("gazebo/get_link_state");
    gazebo_msgs::GetLinkState srv;
    srv.request.link_name = "panda_link7";
    if (client_pose.call(srv))
    {
      ROS_INFO("Pose update");
    }
    else
    {
      ROS_ERROR("Failed to call service");
    }
    geometry_msgs::Pose arm_pose{srv.response.link_state.pose};
    // An offset is required
    arm_pose.position.x += 0.107;
    pose = arm_pose;
  }

public:

  // Reset arm joints to 0 rad
  bool reset_arm(arm_controls::PoseChange::Request &req, arm_controls::PoseChange::Response &res){
    ros::service::waitForService("move_to_joint_space");
    ros::ServiceClient client_pose = n_.serviceClient<moveit_planner::MoveJoint>("move_to_joint_space");
    moveit_planner::MoveJoint srv;
    float goal [7]{0,0,0,0,0,0,0};
    for (int i = 0; i < 7; i++) {
      srv.request.val.push_back(goal[i]);
    }
    srv.request.execute = req.execute;
    if (client_pose.call(srv)){
      update_pose();
      ROS_INFO("Arm reset");
      return true;
    }
    else{
      ROS_ERROR("Arm reset failed");
      return false;
    }
  }

  // initialize arm to a predefined pose
  bool initialize_pose(arm_controls::PoseChange::Request &req, arm_controls::PoseChange::Response &res){

    // Cartesian initialization
    ros::service::waitForService("move_to_pose");
    ros::ServiceClient client_pose = n_.serviceClient<moveit_planner::MovePose>("move_to_pose");
    moveit_planner::MovePose srv;
    geometry_msgs::Pose goal;
    n_.getParam("arm_control_params/init_pose/x_pos", goal.position.x);
    n_.getParam("arm_control_params/init_pose/y_pos", goal.position.y);
    n_.getParam("arm_control_params/init_pose/z_pos", goal.position.z);
    n_.getParam("arm_control_params/init_pose/x_orn", goal.orientation.x);
    n_.getParam("arm_control_params/init_pose/y_orn", goal.orientation.y);
    n_.getParam("arm_control_params/init_pose/z_orn", goal.orientation.z);
    n_.getParam("arm_control_params/init_pose/w_orn", goal.orientation.w);

    //  Joint space initialization
    // ros::service::waitForService("move_to_joint_space");
    // ros::ServiceClient client_pose = n_.serviceClient<moveit_planner::MoveJoint>("move_to_joint_space");
    // moveit_planner::MoveJoint srv;
    // float goal [7];
    // n_.getParam("arm_control_params/init_joint_ref/J1", goal[0]);
    // n_.getParam("arm_control_params/init_joint_ref/J2", goal[1]);
    // n_.getParam("arm_control_params/init_joint_ref/J3", goal[2]);
    // n_.getParam("arm_control_params/init_joint_ref/J4", goal[3]);
    // n_.getParam("arm_control_params/init_joint_ref/J5", goal[4]);
    // n_.getParam("arm_control_params/init_joint_ref/J6", goal[5]);
    // n_.getParam("arm_control_params/init_joint_ref/J7", goal[6]);
    // for (int i = 0; i < 7; i++) {
    //   srv.request.val.push_back(goal[i]);
    // }

    srv.request.val = goal;
    srv.request.execute = req.execute;
    if (client_pose.call(srv)){
      pose = goal;
      // update_pose();
      ROS_INFO("Grasp pose reached");
      return true;
    }
    else{
      ROS_ERROR("Grasp pose failed");
      return false;
    }
  }

  // go to a predefined grasping pose
  bool grasp_pose(arm_controls::PoseChange::Request &req, arm_controls::PoseChange::Response &res){
    // Cartesian method
    ros::service::waitForService("move_to_pose");
    ros::ServiceClient client_pose = n_.serviceClient<moveit_planner::MovePose>("move_to_pose");
    moveit_planner::MovePose srv;
    geometry_msgs::Pose goal;
    n_.getParam("arm_control_params/grasp_pose/x_pos", goal.position.x);
    n_.getParam("arm_control_params/grasp_pose/y_pos", goal.position.y);
    n_.getParam("arm_control_params/grasp_pose/z_pos", goal.position.z);
    n_.getParam("arm_control_params/grasp_pose/x_orn", goal.orientation.x);
    n_.getParam("arm_control_params/grasp_pose/y_orn", goal.orientation.y);
    n_.getParam("arm_control_params/grasp_pose/z_orn", goal.orientation.z);
    n_.getParam("arm_control_params/grasp_pose/w_orn", goal.orientation.w);
    // Joint space method
    // ros::service::waitForService("move_to_joint_space");
    // ros::ServiceClient client_pose = n_.serviceClient<moveit_planner::MoveJoint>("move_to_joint_space");
    // moveit_planner::MoveJoint srv;
    // float goal [7];
    // n_.getParam("arm_control_params/grasp_joint_ref/J1", goal[0]);
    // n_.getParam("arm_control_params/grasp_joint_ref/J2", goal[1]);
    // n_.getParam("arm_control_params/grasp_joint_ref/J3", goal[2]);
    // n_.getParam("arm_control_params/grasp_joint_ref/J4", goal[3]);
    // n_.getParam("arm_control_params/grasp_joint_ref/J5", goal[4]);
    // n_.getParam("arm_control_params/grasp_joint_ref/J6", goal[5]);
    // n_.getParam("arm_control_params/grasp_joint_ref/J7", goal[6]);
    // for (int i = 0; i < 7; i++) {
    //   srv.request.val.push_back(goal[i]);
    // }
    srv.request.val = goal;
    srv.request.execute = req.execute;
    if (client_pose.call(srv)){
      pose = goal;
      // update_pose();
      ROS_INFO("Grasp pose reached");
      return true;
    }
    else{
      ROS_ERROR("Grasp pose failed");
      return false;
    }
  }

  // go to a predefined resting pose
  bool rest_pose(arm_controls::PoseChange::Request &req, arm_controls::PoseChange::Response &res){
    // Cartesian method
    ros::service::waitForService("move_to_pose");
    ros::ServiceClient client_pose = n_.serviceClient<moveit_planner::MovePose>("move_to_pose");
    moveit_planner::MovePose srv;
    geometry_msgs::Pose goal;
    n_.getParam("arm_control_params/rest_pose/x_pos", goal.position.x);
    n_.getParam("arm_control_params/rest_pose/y_pos", goal.position.y);
    n_.getParam("arm_control_params/rest_pose/z_pos", goal.position.z);
    n_.getParam("arm_control_params/rest_pose/x_orn", goal.orientation.x);
    n_.getParam("arm_control_params/rest_pose/y_orn", goal.orientation.y);
    n_.getParam("arm_control_params/rest_pose/z_orn", goal.orientation.z);
    n_.getParam("arm_control_params/rest_pose/w_orn", goal.orientation.w);
    srv.request.val = goal;
    srv.request.execute = req.execute;
    if (client_pose.call(srv)){
      pose = goal;
      ROS_INFO("Rest pose reached");
      return true;
    }
    else{
      ROS_ERROR("Rest pose failed");
      return false;
    }
  }

  // move up by input distance
  bool move_up(arm_controls::MoveStraight::Request &req, arm_controls::MoveStraight::Response &res){
    ros::service::waitForService("move_to_pose");
    ros::ServiceClient client_pose = n_.serviceClient<moveit_planner::MovePose>("move_to_pose");
    moveit_planner::MovePose srv;
    geometry_msgs::Pose goal;
    goal = pose;
    goal.position.z += req.val;
    srv.request.val = goal;
    srv.request.execute = true;
    if (client_pose.call(srv)){
      pose = goal;
      ROS_INFO("Moved up");
      return true;
    }
    else{
      ROS_ERROR("Moving up failed");
      return false;
    }
  }

  // move down by input distance
  bool move_down(arm_controls::MoveStraight::Request &req, arm_controls::MoveStraight::Response &res){
    ros::service::waitForService("move_to_pose");
    ros::ServiceClient client_pose = n_.serviceClient<moveit_planner::MovePose>("move_to_pose");
    moveit_planner::MovePose srv;
    geometry_msgs::Pose goal;
    goal = pose;
    goal.position.z -= req.val;
    srv.request.val = goal;
    srv.request.execute = true;
    if (client_pose.call(srv)){
      pose = goal;
      ROS_INFO("Moved down");
      return true;
    }
    else{
      ROS_ERROR("Moving down failed");
      return false;
    }
  }

  // move left by input distance
  bool move_left(arm_controls::MoveStraight::Request &req, arm_controls::MoveStraight::Response &res){
    ros::service::waitForService("move_to_pose");
    ros::ServiceClient client_pose = n_.serviceClient<moveit_planner::MovePose>("move_to_pose");
    moveit_planner::MovePose srv;
    geometry_msgs::Pose goal;
    goal = pose;
    goal.position.x += req.val;
    srv.request.val = goal;
    srv.request.execute = true;
    if (client_pose.call(srv)){
      pose = goal;
      ROS_INFO("Moved left");
      return true;
    }
    else{
      ROS_ERROR("Moving left failed");
      return false;
    }
  }

  // move right by input distance
  bool move_right(arm_controls::MoveStraight::Request &req, arm_controls::MoveStraight::Response &res){
    ros::service::waitForService("move_to_pose");
    ros::ServiceClient client_pose = n_.serviceClient<moveit_planner::MovePose>("move_to_pose");
    moveit_planner::MovePose srv;
    geometry_msgs::Pose goal;
    goal = pose;
    goal.position.x -= req.val;
    srv.request.val = goal;
    srv.request.execute = true;
    if (client_pose.call(srv)){
      pose = goal;
      ROS_INFO("Moved right");
      return true;
    }
    else{
      ROS_ERROR("Moving right failed");
      return false;
    }
  }

  // move forward by input distance
  bool move_forward(arm_controls::MoveStraight::Request &req, arm_controls::MoveStraight::Response &res){
    ros::service::waitForService("move_to_pose");
    ros::ServiceClient client_pose = n_.serviceClient<moveit_planner::MovePose>("move_to_pose");
    moveit_planner::MovePose srv;
    geometry_msgs::Pose goal;
    goal = pose;
    goal.position.y -= req.val;
    srv.request.val = goal;
    srv.request.execute = true;
    if (client_pose.call(srv)){
      pose = goal;
      ROS_INFO("Moved forward");
      return true;
    }
    else{
      ROS_ERROR("Moving forward failed");
      return false;
    }
  }

  // move back by input distance
  bool move_back(arm_controls::MoveStraight::Request &req, arm_controls::MoveStraight::Response &res){
    ros::service::waitForService("move_to_pose");
    ros::ServiceClient client_pose = n_.serviceClient<moveit_planner::MovePose>("move_to_pose");
    moveit_planner::MovePose srv;
    geometry_msgs::Pose goal;
    goal = pose;
    goal.position.y += req.val;
    srv.request.val = goal;
    srv.request.execute = true;
    if (client_pose.call(srv)){
      pose = goal;
      ROS_INFO("Moved back");
      return true;
    }
    else{
      ROS_ERROR("Moving back failed");
      return false;
    }
  }
};


int main(int argc, char **argv){
  ros::init(argc, argv, "arm_control_server");
  ros::NodeHandle n;

  ArmController armctrl;
  ROS_INFO("ArmController object constructed");

  // advertise services
  ros::ServiceServer arm_reset = n.advertiseService("reset_arm", &ArmController::reset_arm, &armctrl);
  ros::ServiceServer pose_init = n.advertiseService("initialize_arm_pose", &ArmController::initialize_pose, &armctrl);
  ros::ServiceServer pose_grasp = n.advertiseService("grasp_pose", &ArmController::grasp_pose, &armctrl);
  ros::ServiceServer pose_rest = n.advertiseService("rest_pose", &ArmController::rest_pose, &armctrl);
  ros::ServiceServer move_up = n.advertiseService("move_up", &ArmController::move_up, &armctrl);
  ros::ServiceServer move_down = n.advertiseService("move_down", &ArmController::move_down, &armctrl);
  ros::ServiceServer move_left = n.advertiseService("move_left", &ArmController::move_left, &armctrl);
  ros::ServiceServer move_right = n.advertiseService("move_right", &ArmController::move_right, &armctrl);
  ros::ServiceServer move_forward = n.advertiseService("move_forward", &ArmController::move_forward, &armctrl);
  ros::ServiceServer move_back = n.advertiseService("move_back", &ArmController::move_back, &armctrl);

  ros::spin();
  return 0;
}
