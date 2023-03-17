//This node calls services for positioning the robot and then piercing the foam

#include <ros/ros.h>
#include <piercing_experiments_msgs/PrePose.h>
#include <piercing_experiments_msgs/CartesianPath.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "main_pierce");
    ros::NodeHandle n;

    int32_t timeout; n.getParam("timeout", timeout);
    ros::service::waitForService( "prepose", timeout);
    ros::service::waitForService("cartesian_path", timeout);

    ros::ServiceClient prepose_client = n.serviceClient<piercing_experiments_msgs::PrePose>("prepose");
    piercing_experiments_msgs::PrePose pose;

     n.getParam("pre_pose/x_pos", pose.request.target_1.position.x);
     n.getParam("pre_pose/y_pos", pose.request.target_1.position.y);
     n.getParam("pre_pose/z_pos", pose.request.target_1.position.z);
     n.getParam("pre_pose/w_orn", pose.request.target_1.orientation.w);
     n.getParam("pre_pose/x_orn", pose.request.target_1.orientation.x);
     n.getParam("pre_pose/y_orn", pose.request.target_1.orientation.y);
     n.getParam("pre_pose/z_orn", pose.request.target_1.orientation.z);

    bool success = true;
     if(prepose_client.call(pose)){    
         ROS_INFO("Moving to PrePose");
         success = true;
     }
     else{   
         ROS_ERROR("Couldn't move to PrePose");
         ros::shutdown();
         return 1;
     }
    if(success){
        ros::ServiceClient cartesian_client = n.serviceClient<piercing_experiments_msgs::CartesianPath>("cartesian_path");
        piercing_experiments_msgs::CartesianPath ftarget;

        ftarget.request.final_target.position.x = 0.0;
        ftarget.request.final_target.position.y = 0.0;
        ftarget.request.final_target.position.z = 0.02;  //Add these guys to YAML
        
        n.getParam("vel_scaling", ftarget.request.vel); //Velocity Scaling Factor for cartesianComputePath
        ROS_INFO("Piercing");
        cartesian_client.call(ftarget);
        ROS_INFO("Done Piercing");
    }

    ros::spin();

    return 0;
}