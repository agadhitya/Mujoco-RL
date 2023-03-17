#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "pcl/point_types.h"
#include "rrbot_pcl/cloudMsg.h"

sensor_msgs::PointCloud2 curImg;

void camSubCallback(const sensor_msgs::PointCloud2& img){
    curImg = img;
}

bool getCloud(rrbot_pcl::cloudMsg::Request &req, rrbot_pcl::cloudMsg::Response &res){
    res.pc2 = curImg;
    ROS_INFO("Sending image to client");

    return true;
}

int main(int argc, char **argv){
    ros::init(argc, argv, "get_cloud");
    ros::NodeHandle n;

    ros::Subscriber cam_sub = n.subscribe("camera/depth/points", 1, camSubCallback);
    ros::ServiceServer getCloudServ = n.advertiseService("get_cloud", getCloud);
    ROS_INFO("Pointcloud Received");

    ros::spin();
    return 0;
}