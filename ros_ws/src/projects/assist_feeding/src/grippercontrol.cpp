#include <ros/ros.h>
#include "assist_feeding/gripper_open.h"
#include "assist_feeding/gripper_close.h"


int main(int argc, char** argv){

ros::init(argc, argv, "gripper_controller");
ros::NodeHandle n;

ros::ServiceClient client = n.serviceClient<assist_feeding::gripper_close>("gripper_close");


    return 0;
}