// Project to 2D and return an image that can be used by python numpy

#include <ros/ros.h>
#include "sensor_msgs/PointCloud2.h"
#include "rrbot_pcl/projectCloud.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/filters/passthrough.h"
#include "sensor_msgs/Image.h"
#include "pcl/octree/octree.h"
#include "pcl/octree/octree_pointcloud_pointvector.h"

bool projectCloud(rrbot_pcl::projectCloud::Request &req, rrbot_pcl::projectCloud::Response &res){
    ROS_INFO("Received projection request");
    
    // Convert sensor msgs pc2 to pcl pointcloud2
    pcl::PCLPointCloud2::Ptr pcl_pc2 (new pcl::PCLPointCloud2 ());
    std::string frameID = "/camera_optical_link";
    pcl_conversions::toPCL(req.pc2, *pcl_pc2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(*pcl_pc2, *cloud);

    for(int i = 0; i < cloud->points.size(); i++)
        cloud->points[i].z = 3.1;

    pcl::octree::OctreePointCloudPointVector<pcl::PointXYZRGB> octree(req.res);
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();
    ROS_INFO("Beginning getOccupiedVoxelCenters");
    std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB>> occVoxels;
    octree.getOccupiedVoxelCenters(occVoxels);
    ROS_INFO("Getting bounding box");
    double xmin, ymin, zmin;
    double xmax, ymax, zmax;
    octree.getBoundingBox(xmin, ymin, zmin,
    			xmax, ymax, zmax);
    ROS_INFO("Inserting data");
    res.x.push_back(xmin);
    res.y.push_back(ymin);
    res.z.push_back(zmin);
    res.x.push_back(xmax);
    res.y.push_back(ymax);
    res.z.push_back(zmax);

    ROS_INFO("Populating data");
    for(const auto& pt : occVoxels) {
      res.x.push_back(pt.x);
      res.y.push_back(pt.y);
      res.z.push_back(pt.z);
    }
    ROS_INFO("DONE");



    return true;
}

int main(int argc, char** argv){

    ros::init(argc, argv, "project_cloud_node");
    ros::NodeHandle n;

    ros::ServiceServer projectCloudServ = n.advertiseService("project_cloud", projectCloud);

    ros::spin();
    return 0;
}