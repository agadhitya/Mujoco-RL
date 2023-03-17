#include "ros/ros.h"

#include <conveyor_picking/FindMug.h>

#include <ros_pcl_manip/ToFile.h>
#include <ros_pcl_manip/LoadFile.h>
#include <ros_pcl_manip/CorrGroup.h>
#include <ros_pcl_manip/SegmentPlane.h>

ros_pcl_manip::LoadFile loadMugReq;
ros::ServiceClient* segPlaneClient;
ros::ServiceClient* corrClient;
ros::ServiceClient* toFileClient;

bool loadMug(conveyor_picking::FindMug::Request& req,
	     conveyor_picking::FindMug::Response& res) {
  ROS_INFO("Backing up given point cloud");
  ros_pcl_manip::ToFile toFileReq;
  toFileReq.request.cloud = req.cloud;
  toFileReq.request.filepath = "/home/fadi/backup_cloud.pcd";
  toFileClient->call(toFileReq);

  ROS_INFO("Segmenting single plane");
  ros_pcl_manip::SegmentPlane segReq;
  segReq.request.cloud = req.cloud;
  segPlaneClient->call(segReq);
  toFileReq.request.cloud = segReq.response.negativeCloud;
  toFileReq.request.filepath = "/home/fadi/single_negative.pcd";
  toFileClient->call(toFileReq);

  // For now, do nothing
  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "find_mug_node");
  ros::NodeHandle nh;

  ros::ServiceClient loadMugClient = nh.serviceClient<ros_pcl_manip::LoadFile>("load_pcd_file");
  loadMugReq.request.frame = "world";
  loadMugReq.request.filepath = "/home/fadi/mer_lab/ros_ws/src/projects/table_rearrange/ycb_models/mesh/mug/mug.pcd";
  if(!loadMugClient.call(loadMugReq)) {
    ROS_ERROR("Could not load mug pcd");
    return 1;
  }

  ros::ServiceServer loadMugServer = nh.advertiseService("load_mug", loadMug);
  segPlaneClient = new ros::ServiceClient(nh.serviceClient<ros_pcl_manip::SegmentPlane>("segment_plane"));
  corrClient = new ros::ServiceClient(nh.serviceClient<ros_pcl_manip::CorrGroup>("correspondence_grouping"));
  toFileClient = new ros::ServiceClient(nh.serviceClient<ros_pcl_manip::ToFile>("save_to_pcd"));

  ros::spin();
  return 0;
}
