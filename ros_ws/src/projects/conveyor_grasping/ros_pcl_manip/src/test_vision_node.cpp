#include <vector>

#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/PointCloud2.h>

#include "ros_pcl_manip/Downsample.h"
#include "ros_pcl_manip/SegmentPlane.h"
#include "ros_pcl_manip/CorrGroup.h"
#include "ros_pcl_manip/LoadFile.h"
#include "ros_pcl_manip/ToFile.h"

sensor_msgs::PointCloud2 img;

void img_callback(const sensor_msgs::PointCloud2& msg) {
  img = msg;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "test_vision_node");
  ros::NodeHandle nh;

  ros::Subscriber imgSub = nh.subscribe("/panda_camera/depth/points", 1, img_callback);
  while(img.data.size() == 0)
    ros::spinOnce();		// Spin once to get an image

  // Send downsample request
  ROS_INFO("Downsampling");
  ros::ServiceClient downClient = nh.serviceClient<ros_pcl_manip::Downsample>("downsample");
  ros_pcl_manip::Downsample dsReq;
  dsReq.request.cloud = img;
  dsReq.request.size = 0.01;
  downClient.call(dsReq);
  ros::Publisher down_pub = nh.advertise<sensor_msgs::PointCloud2>("downsample_result", 1);

  // Send plane seg request
  ROS_INFO("Segmenting plane");
  img = sensor_msgs::PointCloud2();
  while(img.data.size() == 0)
    ros::spinOnce();		// Spin once to get an image
  ros::ServiceClient planeClient = nh.serviceClient<ros_pcl_manip::SegmentPlane>("segment_plane");
  ros_pcl_manip::SegmentPlane planeReq;
  planeReq.request.cloud = img;
  planeClient.call(planeReq);
  ros::Publisher plane_pub = nh.advertise<sensor_msgs::PointCloud2>("segment_plane_result", 1);
  ros::Publisher neg_plane_pub = nh.advertise<sensor_msgs::PointCloud2>("segment_plane_negative_result", 1);

  // Send file request
  ROS_INFO("Loading file");
  ros::ServiceClient loadClient = nh.serviceClient<ros_pcl_manip::LoadFile>("load_pcd_file");
  ros_pcl_manip::LoadFile fileReq;
  fileReq.request.filepath = "/home/fadi/mer_lab/ros_ws/src/projects/table_rearrange/ycb_models/mesh/mug/mug.pcd";
  fileReq.request.frame = "map";
  loadClient.call(fileReq);
  ros::Publisher mug_pub = nh.advertise<sensor_msgs::PointCloud2>("mug_pcd", 1);

  // Send correspondence request
  ROS_INFO("Grouping mug");
  img = sensor_msgs::PointCloud2();
  while(img.data.size() == 0)
    ros::spinOnce();		// Spin once to get an image
  // ros::ServiceClient corrClient = nh.serviceClient<ros_pcl_manip::CorrGroup>("correspondence_grouping");
  // ros_pcl_manip::CorrGroup corrReq;
  // corrReq.request.model = fileReq.response.cloud;
  // corrReq.request.scene = img;
  // corrReq.request.invariant = false;
  // corrClient.call(corrReq);
  // ROS_INFO_STREAM("Received models: " << corrReq.response.detectedModels.size());
  // PARAMETERS:
  // --scene_ss 0.005 --model_ss 0.001 --cg_thresh 5 --cg_size 0.02 --descr_rad 0.0125 --rf_rad 0.015
  ros::ServiceClient toPCDClient = nh.serviceClient<ros_pcl_manip::ToFile>("save_to_pcd");
  ros_pcl_manip::ToFile toReq;
  toReq.request.cloud = img;
  toReq.request.filepath = "/home/fadi/scene.pcd";
  toPCDClient.call(toReq);
  toReq.request.cloud = planeReq.response.negativeCloud;
  toReq.request.filepath = "/home/fadi/negative.pcd";
  toPCDClient.call(toReq);

  ROS_INFO("Publishing");
  while(ros::ok()) {
    down_pub.publish(dsReq.response.cloud);
    plane_pub.publish(planeReq.response.plane);
    neg_plane_pub.publish(planeReq.response.negativeCloud);
    // mug_pub.publish(fileReq.response.cloud);
    ros::spinOnce();
  }

  return 0;
}
