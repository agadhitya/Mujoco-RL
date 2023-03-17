#include <vector>

// ROS includes
#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/PointCloud2.h>

// PCL includes
#include <pcl/point_types.h>
#include <pcl/correspondence.h>
#include <pcl/filters/passthrough.h>

// SRV includes
#include <ros_pcl_manip/ToFile.h>
#include <ros_pcl_manip/LoadFile.h>
#include <ros_pcl_manip/CorrGroup.h>
#include <ros_pcl_manip/Downsample.h>
#include <ros_pcl_manip/PassFilter.h>
#include <ros_pcl_manip/SegmentPlane.h>

typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;
typedef pcl::SHOT352 DescriptorType;
typedef pcl::PointCloud<PointType> CloudType;
typedef pcl::PointCloud<NormalType> NormalCloud;
typedef pcl::PointCloud<DescriptorType> DescriptorCloud;
typedef CloudType::Ptr CloudPtr;
typedef NormalCloud::Ptr NormalCloudPtr;
typedef DescriptorCloud::Ptr DescriptorCloudPtr;
typedef CloudType::ConstPtr CloudConstPtr;
typedef pcl::PassThrough<PointType> PassThroughType;

struct CorrespondenceParams {
  float model_ss{0.01f};
  float scene_ss{0.03f};
  float rf_rad{0.015f};
  float descr_rad{0.02f};
  float cg_size{0.01f};
  float cg_thresh{5.0f};
};

class ROS_PCL {
public:
  ROS_PCL(ros::NodeHandle& nh);
  ~ROS_PCL();

  // Non-service methods
  CloudPtr downsample(CloudPtr in_cloud, float leaf_size);
  CloudPtr seg_plane(CloudPtr in_cloud, bool remove=false, float leaf_size=0.0f);
  double getResolution(CloudConstPtr cloud);
  CorrespondenceParams scaleInvariant(const CorrespondenceParams& p, double modelRes);
  NormalCloudPtr computeNormals(CloudConstPtr in_cloud);
  CloudPtr extractKeypoints(CloudPtr in_cloud, float search);
  DescriptorCloudPtr computeDescriptors(CloudPtr in_cloud, CloudPtr in_keypoints, NormalCloudPtr in_normal, float search);
  pcl::CorrespondencesPtr getCorrespondences(DescriptorCloudPtr model_descs, DescriptorCloudPtr scene_descs);
  std::vector<geometry_msgs::Pose> geometricConsistency(CloudPtr model_keypts, CloudPtr scene_keypts,
							pcl::CorrespondencesPtr msc, float size, float thresh);
  std::vector<geometry_msgs::Pose> correspondenceGrouping(CloudPtr scene, CloudPtr model,
							  bool cloud_res, CorrespondenceParams params);
  CloudPtr filter_cloud(CloudPtr in_cloud, std::string field_name,
			double lower, double upper, bool remove=false);

  // Conversions
  CloudPtr from_pc2(const sensor_msgs::PointCloud2& pc2);
  sensor_msgs::PointCloud2 to_pc2(CloudPtr in_cloud);

  // Service methods
  bool downsample_service(ros_pcl_manip::Downsample::Request& req,
			  ros_pcl_manip::Downsample::Response& res);
  bool segment_plane_service(ros_pcl_manip::SegmentPlane::Request& req,
			     ros_pcl_manip::SegmentPlane::Response& res);
  bool cor_group_service(ros_pcl_manip::CorrGroup::Request& req,
			 ros_pcl_manip::CorrGroup::Response& res);
  bool filter_service(ros_pcl_manip::PassFilter::Request& req,
		      ros_pcl_manip::PassFilter::Response& res);
  bool load_service(ros_pcl_manip::LoadFile::Request& req,
		    ros_pcl_manip::LoadFile::Response& res);
  bool save_service(ros_pcl_manip::ToFile::Request& req,
		    ros_pcl_manip::ToFile::Response& res);
private:
  ros::NodeHandle& _nh;
  ros::ServiceServer _down_server;
  ros::ServiceServer _plane_server;
  ros::ServiceServer _cor_server;
  ros::ServiceServer _load_server;
  ros::ServiceServer _save_server;
  ros::ServiceServer _pass_server;

  void _setup_services();
};
