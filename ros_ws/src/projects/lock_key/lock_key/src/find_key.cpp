#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include "lock_key/findKey.h"
#include <iostream>
// PCL Includes for conversion
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
// Downsampling
#include <pcl/filters/voxel_grid.h>
// Segmentation
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
// Centroid
#include <pcl/common/centroid.h>

bool key_service_callback(lock_key::findKey::Request &req, lock_key::findKey::Response &res){
    ROS_INFO("FINDING KEY");
    // Convert sensor_msgs to pcl pointcloud2
    // TO DO: Put in a function
    pcl::PCLPointCloud2::Ptr pcl_pc2 (new pcl::PCLPointCloud2 ());
    std::string frameID = "/panda_camera_optical_link";
    pcl_conversions::toPCL(req.pc2, *pcl_pc2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(*pcl_pc2, *cloud);

    //Downsampling the pointcloud
    // TO DO: Put in a function
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);  
    // std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height 
    //    << " data points (" << pcl::getFieldsList (*cloud) << ").";

    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize(0.002f, 0.002f, 0.002f);
    sor.filter(*cloud_filtered);

    // std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height 
    //    << " data points (" << pcl::getFieldsList (*cloud_filtered) << ").";

    // Lets Try Segmentation
    // TO DO: Put in a function

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices); 
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZRGB> ());
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.001);

    int i=0, nr_points = (int) cloud_filtered->points.size ();
    while (cloud_filtered->points.size () > 0.3 * nr_points){
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0){
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud (cloud_filtered);
        extract.setIndices (inliers);
        extract.setNegative (false);

        // Get the points associated with the planar surface
        extract.filter (*cloud_plane);
        std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*cloud_f);
        *cloud_filtered = *cloud_f;
    }

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud (cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (700);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract (cluster_indices);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it){
        
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
        // std::stringstream ss;
        // ss << "cloud_cluster_" << j << ".pcd";
        //writer.write<pcl::PointXYZRGB> (ss.str (), *cloud_cluster, false); //*
        // j++;
    }

    // Convert Cloud Cluster to PCL
    // TO DO: put in function
    sensor_msgs::PointCloud2 ret;
    pcl::PCLPointCloud2 temppc2;
    pcl::toPCLPointCloud2(*cloud_filtered, temppc2);
    pcl_conversions::fromPCL(temppc2, ret);

    

    // Find Position of Centroid of the key
    // TO DO: put in function
    pcl::CentroidPoint<pcl::PointXYZ> centroid;
    // Loop over points in segmented cloud
    for (int i = 0; i < cloud_cluster->points.size(); i++){
        centroid.add(pcl::PointXYZ (cloud_cluster->points[i].x, cloud_cluster->points[i].y, cloud_cluster->points[i].z));
    }
    pcl::PointXYZ key_cent;
    centroid.get(key_cent);
    // TO DO: Insert color check for lock/ key
    ROS_INFO_STREAM(key_cent);
    // Fill up the return object
    ret.header.stamp = ros::Time();
    ret.header.frame_id = frameID;
    res.pc2 = ret;
    res.p.header.frame_id = frameID;
    res.p.point.x = key_cent.x;
    res.p.point.y = key_cent.y;
    res.p.point.z = key_cent.z;
    return true;

}

int main(int argc, char **argv){

    ros::init(argc, argv, "findKey");
    ros::NodeHandle n;
    
    ros::ServiceServer key = n.advertiseService("findKeyServer", key_service_callback);

    ros::spin();
}