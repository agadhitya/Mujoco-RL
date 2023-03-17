// The movement class defines several services from the MoveIt! packages
// as well as other useful methods integral to the experimental framework
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <piercing_experiments_msgs/PrePose.h>
#include <piercing_experiments_msgs/CartesianPath.h>
class Movement
{
    public:
        Movement(ros::NodeHandle& _n) : n(_n), move_group("arm") {
            //Name the movegroup as "arm" for lock and key and as "panda_arm" for piercing experiments
            ros::AsyncSpinner spinner(2);
            spinner.start();
            geometry_msgs::PoseStamped ps = move_group.getCurrentPose();
            ROS_INFO_STREAM(ps);
            prepose_server = n.advertiseService("prepose", &Movement::prepose_callback, this);
            cartesian_server = n.advertiseService("cartesian_path", &Movement::cartesian_path_callback, this);
            ros::spinOnce();
            ros::waitForShutdown;
        
        }

        bool prepose_callback(piercing_experiments_msgs::PrePose::Request& req, piercing_experiments_msgs::PrePose::Response& res){
            move_group.setPoseTarget(req.target_1);
            moveit::planning_interface::MoveGroupInterface::Plan myPlan;
            bool success = (move_group.plan(myPlan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
            move_group.execute(myPlan);
            
            return success;
            
            
        }
        bool cartesian_path_callback(piercing_experiments_msgs::CartesianPath::Request& req, piercing_experiments_msgs::CartesianPath::Response& res){
            
            geometry_msgs::Pose pose  = move_group.getCurrentPose().pose;
            
            std::vector<geometry_msgs::Pose> waypoints;
            waypoints.push_back(pose);
            
            pose.position.z -= req.final_target.position.z;  //Add for x and y if needed
            waypoints.push_back(pose);
        
            move_group.setMaxVelocityScalingFactor(req.vel);
            moveit_msgs::RobotTrajectory trajectory;
            double jump_threshold; double eef_step;

            n.getParam("jmp_thrsh", jump_threshold);
            n.getParam("eef_stp", eef_step);
            
            ROS_INFO("moving");
            double fraction  = move_group.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
            ROS_INFO("Cartesian Path %.2f%% acheived", fraction*100.0);

            moveit::planning_interface::MoveGroupInterface::Plan myPlan;
            myPlan.trajectory_ = trajectory;
            move_group.execute(myPlan);

            if(fraction == 1.0)
                return true;
            else
                return false;

        }
        
    private:
        ros::NodeHandle n;
        moveit::planning_interface::MoveGroupInterface move_group;

        ros::ServiceServer prepose_server;
        ros::ServiceServer cartesian_server;
};