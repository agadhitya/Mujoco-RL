#include <ros/ros.h>
#include <std_srvs/Empty.h>
int main(int argc, char **argv){

    ros::init(argc, argv, "setup");
    ros::NodeHandle n;
    int32_t timeout = 1000;
    ros::service::waitForService("gazebo/unpause_physics", timeout);

    ros::ServiceClient unpauseGazebo = n.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
    std_srvs::Empty emptySrv;
    unpauseGazebo.call(emptySrv);
    return 0;
}