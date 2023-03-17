#include <ros/ros.h>
#include <moveit_planner.hpp>
#include <manipulation_planning/ArmPose.h>
#include <std_srvs/Empty.h>
class FrankaActions
{
private:
  ros::NodeHandle n_;
  ros::Duration dur;
public:
    bool move_pose(manipulation_planning::ArmPose::Request &req, manipulation_planning::ArmPose::Response &res){
      ros::service::waitForService("move_to_pose");
      ros::ServiceClient client_pose = n_.serviceClient<moveit_planner::MovePose>("move_to_pose");
      moveit_planner::MovePose srv;
      geometry_msgs::Pose goal;
      srv.request.val = req.pose;
      srv.request.execute = 1;
      if (client_pose.call(srv)){
        ROS_INFO("Pose reached");
        return true;
      }
      else{
        ROS_ERROR("Pose failed");
        return false;
      }
    }
    bool initialize_arm(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res){
      ros::service::waitForService("move_to_pose");
      ros::ServiceClient client_pose = n_.serviceClient<moveit_planner::MovePose>("move_to_pose");
      moveit_planner::MovePose srv;
      geometry_msgs::Pose goal;
      n_.getParam("arm_action_params/init_pose/x_pos", goal.position.x);
      n_.getParam("arm_action_params/init_pose/y_pos", goal.position.y);
      n_.getParam("arm_action_params/init_pose/z_pos", goal.position.z);
      n_.getParam("arm_action_params/init_pose/x_orn", goal.orientation.x);
      n_.getParam("arm_action_params/init_pose/y_orn", goal.orientation.y);
      n_.getParam("arm_action_params/init_pose/z_orn", goal.orientation.z);
      n_.getParam("arm_action_params/init_pose/w_orn", goal.orientation.w);
      srv.request.val = goal;
      srv.request.execute = 1;
      if (client_pose.call(srv)){
        ROS_INFO("Pose reached");
        return true;
      }
      else{
        ROS_ERROR("Pose failed");
        return false;
      }
    }
};

int main(int argc, char **argv){
  ros::init(argc, argv, "franka_actions_server");
  ros::NodeHandle n;

  ros::service::waitForService("add_collision_object");
  ros::ServiceClient add_coll = n.serviceClient<moveit_planner::AddCollision>("add_collision_object");
  moveit_planner::AddCollision srv;

  // moveit_msgs::CollisionObject floor_table;
  // floor_table.header.frame_id = "/world";
  // floor_table.id = "floor";
  //
  //   /* A default pose */
  // geometry_msgs::Pose pose1;
  // pose1.position.x = 0;
  // pose1.position.y = 0.0;
  // pose1.position.z = 0;
  // pose1.orientation.w = 1.0;
  //
  // /* Define a box to be attached */
  // shape_msgs::SolidPrimitive primitive1;
  // primitive1.type = primitive1.BOX;
  // primitive1.dimensions.resize(3);
  // primitive1.dimensions[0] = 2;
  // primitive1.dimensions[1] = 0.5;
  // primitive1.dimensions[2] = 0.05;
  //
  // floor_table.primitives.push_back(primitive1);
  // floor_table.primitive_poses.push_back(pose1);
  //
  // srv.request.collObject = floor_table;
  //
  // if (add_coll.call(srv)){
  //   ROS_INFO("Collision added");
  // }
  // else{
  //   ROS_ERROR("Collision adding failed");
  // }

  moveit_msgs::CollisionObject table;
  table.header.frame_id = "/world";
  table.id = "table";

    /* A default pose */
  geometry_msgs::Pose pose2;
  n.getParam("arm_action_params/table_pose/x_pos", pose2.position.x);
  n.getParam("arm_action_params/table_pose/y_pos", pose2.position.y);
  n.getParam("arm_action_params/table_pose/z_pos", pose2.position.z);
  n.getParam("arm_action_params/table_pose/x_orn", pose2.orientation.x);
  n.getParam("arm_action_params/table_pose/y_orn", pose2.orientation.y);
  n.getParam("arm_action_params/table_pose/z_orn", pose2.orientation.z);
  n.getParam("arm_action_params/table_pose/w_orn", pose2.orientation.w);

  /* Define a box to be attached */
  shape_msgs::SolidPrimitive primitive2;
  primitive2.type = primitive2.BOX;
  primitive2.dimensions.resize(3);
  n.getParam("arm_action_params/table_dims/x", primitive2.dimensions[0]);
  n.getParam("arm_action_params/table_dims/y", primitive2.dimensions[1]);
  n.getParam("arm_action_params/table_dims/z", primitive2.dimensions[2]);

  table.primitives.push_back(primitive2);
  table.primitive_poses.push_back(pose2);

  srv.request.collObject = table;

  if (add_coll.call(srv)){
    ROS_INFO("Collision added");
  }
  else{
    ROS_ERROR("Collision adding failed");
  }


  FrankaActions franka_acts;
  ros::ServiceServer srv_pose = n.advertiseService("franka_act/move_pose", &FrankaActions::move_pose, &franka_acts);
  ros::ServiceServer srv_init = n.advertiseService("franka_act/init_arm", &FrankaActions::initialize_arm, &franka_acts);
  ros::spin();
  return 0;
}
