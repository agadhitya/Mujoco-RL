// MARKER ID 72 -> End Effector
// MARKER ID 97 -> Base

#include <ros/ros.h>

#include "geometry_msgs/Point.h"
#include "owi_robot/generatecmd.h"
#include "owi_robot/writecmd.h"

geometry_msgs::Point curPose_;

geometry_msgs::Point get_error(geometry_msgs::Point goalPose){

    geometry_msgs::Point err;
    err.x = goalPose.x - curPose_.x;
    err.y = goalPose.y - curPose_.y;
    err.z = goalPose.z - curPose_.z;

    return err;
}

geometry_msgs::Point get_goal_pose(){

    geometry_msgs::Point goalPose;
    goalPose.x = 0.23;  //Read from YAML
    goalPose.y = -0.14;
    goalPose.z = -0.17;

    return goalPose;
}

void get_cur_pose(const geometry_msgs::Point &msg){

    curPose_.x = msg.x;
    curPose_.y = msg.y;
    curPose_.z = msg.z;

}

int main(int argc, char **argv){
    
    ros::init(argc, argv, "owi_control");
    ros::NodeHandle n;

    ros::Subscriber sub = n.subscribe("object_position",1,get_cur_pose);
    
    geometry_msgs::Point goal = get_goal_pose();
    geometry_msgs::Point err = get_error(goal);
    
    //TO DO: compute error magnitude
    
    ros::ServiceClient gen_cmd_client = n.serviceClient<owi_robot::generatecmd>("cmdGenServer");
    owi_robot::generatecmd gencmd;
    
    if(err.x>1e2){
        if (err.x>0){
            //Move base right
            gencmd.request.motor_num = 4;
            gencmd.request.direction = true;
            ros::service::call("generatecmd", gencmd);
        }
        else if(err.x<0){
            //Move base left
            gencmd.request.motor_num = 4;
            gencmd.request.direction = false;
            ros::service::call("cmdGenServer", gencmd);
        }
        //Send generatecmd response to writecmd
    }
    else{
        ROS_INFO("DONE!");
    }
    

    
    return 0;
}