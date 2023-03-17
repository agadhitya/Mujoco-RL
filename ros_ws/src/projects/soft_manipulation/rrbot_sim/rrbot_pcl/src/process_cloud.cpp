#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "rrbot_pcl/processCloud.h"

#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

bool processCloud(rrbot_pcl::processCloud::Request &req, rrbot_pcl::processCloud::Response &res){
    ROS_INFO("Received processing request");
    
    // Convert sensor msgs pc2 to pcl pointcloud2
    pcl::PCLPointCloud2::Ptr pcl_pc2 (new pcl::PCLPointCloud2 ());
    std::string frameID = "/camera_optical_link";
    pcl_conversions::toPCL(req.pc2, *pcl_pc2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(*pcl_pc2, *cloud);

    ROS_INFO("Downsampling");
    // Downsampling
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize(0.001f, 0.001f, 0.001f);
    sor.filter(*cloud_downsampled);

    ROS_INFO("Filtering");
    // Filtering
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (cloud_downsampled);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0, 3.1);
    pass.filter (*cloud_filtered);

    // Convert Cloud Cluster to PCL for debugging filtered cloud
    sensor_msgs::PointCloud2 ret;
    pcl::PCLPointCloud2 temppc2;
    pcl::toPCLPointCloud2(*cloud_filtered, temppc2);
    pcl_conversions::fromPCL(temppc2, ret);
    res.pc2 = ret;

    ROS_INFO("DONE");

    return true;
}
int main(int argc, char** argv){
    ros::init(argc, argv, "process_cloud");
    ros::NodeHandle n;

    ros::ServiceServer processCloudServ = n.advertiseService("process_cloud", processCloud);

    ros::spin();
    return 0;
}