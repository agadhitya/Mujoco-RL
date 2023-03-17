#include <iostream>
#include <stdlib.h>
#include <string>

#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl_ros/point_cloud.h>

std::string mode;

class ImageConverter{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  ros::Subscriber rgbd_sub_;
  image_transport::Publisher rgb_pub_;

public:
  ImageConverter(): it_(nh_){
    // Subscrive to input video feed and publish output video feed
    std::string stream;
    if(mode == "FRANKA") stream = "/camera/depth/color/points";
    else                 stream = "/camera/depth/points";

    rgbd_sub_ = nh_.subscribe(stream, 1, &ImageConverter::cloud_cb, this);
    rgb_pub_ = it_.advertise("/camera/depth/rgb", 1);

    std::cout << "Subscribing from : " << stream << std::endl;
    std::cout << "Publishing to : " << "/camera/depth/rgb" << std::endl;
  }

  void cloud_cb(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud_msg){

    int projIndex;

    cv::Mat rgb = cv::Mat::zeros(cloud_msg->height,cloud_msg->width,CV_64FC1);
    for(int x = 0; x < cloud_msg->height ; x++){
      for(int y = 0 ; y < cloud_msg->width ; y++){
        projIndex = x*(cloud_msg->width)+y;
        cv::Vec3b &color = rgb.at<cv::Vec3b>(cv::Point(y,x));
        color[0] = cloud_msg->points[projIndex].b;
        color[1] = cloud_msg->points[projIndex].g;
        color[2] = cloud_msg->points[projIndex].r;
      }
    }
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rgb).toImageMsg();
    rgb_pub_.publish(msg);
  }
};

int main (int argc, char** argv){

  if(argc != 2){
    std::cout << "ERROR. Incorrect number of arguments." << std::endl;
    return(-1);
  }

  mode = argv[1];
  if(mode != "FRANKA" && mode != "SIMULATION"){
    std::cout << "Enter only FRANKA or SIMULATION" << std::endl;
    return(-1);
  }

  // Initialize ROS
  ros::init (argc, argv, "RBGD_to_RGB");
  ImageConverter ic;
  ros::spin();
  return 0;
}
