#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include "lock_key/imgCapture.h"

sensor_msgs::PointCloud2 curImg;

void img_callback(const sensor_msgs::PointCloud2& img){
    curImg = img;
}

bool imgcap_service_callback(lock_key::imgCapture::Request &req, lock_key::imgCapture::Response &res){
res.pc2 = curImg;
ROS_INFO("Sending Image to Client");

return true;
}

int main(int argc, char **argv){
    
    ros::init(argc, argv, "imgCapture");
    ros::NodeHandle n;

    ros::Subscriber camera = n.subscribe("/panda_camera/depth/points",1,img_callback);
    ros::ServiceServer imgcap = n.advertiseService("imgCaptureServer", imgcap_service_callback);
    
    ros::spin();


    return 0;
}