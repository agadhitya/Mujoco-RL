#include "ros/ros.h"
#include "rrbot_pcl/concaveHull.h"
#include "pcl_conversions/pcl_conversions.h"
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/concave_hull.h>

bool concaveHull(rrbot_pcl::concaveHull::Request &req, rrbot_pcl::concaveHull::Response &res){

    // Convert sensor msgs pc2 to pcl pointcloud2
    pcl::PCLPointCloud2::Ptr pcl_pc2 (new pcl::PCLPointCloud2 ());
    std::string frameID = "/camera_optical_link";
    pcl_conversions::toPCL(req.pc2, *pcl_pc2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(*pcl_pc2, *cloud);

    // Projecting all points to a plane
    for(int i = 0; i < cloud->points.size(); i++)
        cloud->points[i].z = 3.1; 

    // Find concave hull
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZRGB>);
    std::vector< pcl::Vertices > polygons;
    pcl::ConcaveHull<pcl::PointXYZRGB> chull;
    chull.setInputCloud (cloud);
    chull.setAlpha (0.01);
    chull.reconstruct (*cloud_hull, polygons);

    std::cerr << "Concave hull has: " << cloud_hull->points.size () << " data points." << std::endl;
    // Convert Cloud Cluster to PCL for debugging filtered cloud
    sensor_msgs::PointCloud2 ret;
    pcl::PCLPointCloud2 temppc2;
    pcl::toPCLPointCloud2(*cloud_hull, temppc2);
    pcl_conversions::fromPCL(temppc2, ret);
    res.pc2 = ret;


    return true;
}

int main(int argc, char** argv){

    ros::init(argc, argv, "concave_hull");
    ros::NodeHandle n;

    ros::ServiceServer concaveHullServ = n.advertiseService("concave_hull", concaveHull);

    ros::spin();
    return 0;
}