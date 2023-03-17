#include <vector>

#include "ros/ros.h"

#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"

#include "pcl_recorder/GetPointCloud.h"
#include "pcl_recorder/BeginPointCloudRange.h"
#include "pcl_recorder/EndPointCloudRange.h"

#include "octomap_msgs/conversions.h"
#include "octomap_msgs/Octomap.h"
#include "octomap/octomap.h"

constexpr double recResolution = 0.01;
constexpr double maxRecTime = 60;
bool isRecording = false;
double recFreq = 1.0;
ros::Time beginRecTime;
octomap::OcTree tree{recResolution};
octomap::Pointcloud tempCloud;

sensor_msgs::PointCloud2 curCloud;
std::vector<sensor_msgs::PointCloud2> cloudCluster;
// TODO: Implement collecting a group of point clouds over time

// CONVERSIONS-----------------------------------------------------
void pointCloud2ToOctomap(const sensor_msgs::PointCloud2& cloud, octomap::Pointcloud& octomapCloud){
  octomapCloud.reserve(cloud.data.size() / cloud.point_step);
 
  sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud, "z");
 
  for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z){
    // Check if the point is invalid
    if (!std::isnan (*iter_x) && !std::isnan (*iter_y) && !std::isnan (*iter_z))
      octomapCloud.push_back(*iter_x, *iter_y, *iter_z);
  }
}
void pointsOctomapToPointCloud2(const octomap::point3d_list& points, sensor_msgs::PointCloud2& cloud){
  // make sure the channel is valid
  std::vector<sensor_msgs::PointField>::const_iterator field_iter = cloud.fields.begin(), field_end =
    cloud.fields.end();
  bool has_x, has_y, has_z;
  has_x = has_y = has_z = false;
  while (field_iter != field_end) {
    if ((field_iter->name == "x") || (field_iter->name == "X"))
      has_x = true;
    if ((field_iter->name == "y") || (field_iter->name == "Y"))
      has_y = true;
    if ((field_iter->name == "z") || (field_iter->name == "Z"))
      has_z = true;
    ++field_iter;
  }
 
  if ((!has_x) || (!has_y) || (!has_z))
    throw std::runtime_error("One of the fields xyz does not exist");
 
  sensor_msgs::PointCloud2Modifier pcd_modifier(cloud);
  pcd_modifier.resize(points.size());
 
  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
 
  for (octomap::point3d_list::const_iterator it = points.begin(); it != points.end(); ++it, ++iter_x, ++iter_y, ++iter_z) {
    *iter_x = it->x();
    *iter_y = it->y();
    *iter_z = it->z();
  }
}
// CONVERSIONS-----------------------------------------------------

void recvPCLData(const sensor_msgs::PointCloud2ConstPtr& pclPtr) {
  curCloud = *pclPtr;
}

bool reqPointCloud(pcl_recorder::GetPointCloud::Request& req,
		   pcl_recorder::GetPointCloud::Response& res) {
  res.data = curCloud;

  return true;
}

bool startRecMap(pcl_recorder::BeginPointCloudRange::Request& req,
		 pcl_recorder::BeginPointCloudRange::Response& res) {
  if(isRecording) {
    ROS_INFO("Call to start_recording_map while already recording, dumping data");
    isRecording = false;
  }
  tree.clear();
  recFreq = req.frequency;
  res.limitSeconds = maxRecTime;
  beginRecTime = ros::Time::now();
  return true;
}

bool endRecMap(pcl_recorder::EndPointCloudRange::Request& req,
	       pcl_recorder::EndPointCloudRange::Response& res) {
  isRecording = false;
  tree.clear();
  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "pcl_recorder_node");
  ros::NodeHandle nh{};

  // Setup connections
  ros::Subscriber pclSub =
    nh.subscribe<sensor_msgs::PointCloud2>("/panda_camera/depth/points", 1, recvPCLData);
  ros::ServiceServer pclServer =
    nh.advertiseService("get_point_cloud", reqPointCloud);
  ros::ServiceServer startRecServer =
    nh.advertiseService("start_recording_map", startRecMap);
  ros::ServiceServer endRecServer =
    nh.advertiseService("end_recording_map", endRecMap);

  // Spin
  while(ros::ok()) {
    ros::spinOnce();
    if(isRecording) {
      pointCloud2ToOctomap(curCloud, tempCloud);
      // Insert point cloud w/o transformation
      tree.insertPointCloud(tempCloud, octomath::Vector3{}, octomath::Pose6D{0, 0, 0, 0, 0, 0});
      if(ros::Time::now().toSec() - beginRecTime.toSec() >= maxRecTime) { // Exceeded time
	// Dump data
	tree.clear();
	isRecording = false;
	ROS_INFO("Too long before calling end, clearing tree and stopping recording");
      }
      ros::Duration(1/recFreq).sleep(); // Sleep for specified rec duration
    }
  }

  return 0;
}
