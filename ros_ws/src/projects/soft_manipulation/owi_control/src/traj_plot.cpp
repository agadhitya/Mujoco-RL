#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <vector>
#include <image_transport/image_transport.h>
#include <opencv2/core/types.hpp>

#include "geometry_msgs/PointStamped.h"
#include "sensor_msgs/image_encodings.h"

cv::Mat inImage;
std::vector<float> traj_points_X;
std::vector<float> traj_points_Y;
ros::Publisher traj_pub;

void trajPointsCallback(const geometry_msgs::PointStamped &msg){
    traj_points_X.push_back(msg.point.x);
    traj_points_Y.push_back(msg.point.y);
}

void trajPlotCallback(const sensor_msgs::ImageConstPtr& img){
    //convert msg to imae matrix
    cv_bridge::CvImagePtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& e){
        ROS_ERROR("cv_bridge exception: %s",e.what());
        return;
    }
    const cv::Scalar red(0,0,255);
    // for loop to loop over trajpoints vector
        // cv::line(trajpoints[i-1], trajpoints[i])
    for(int i = 1; i <= traj_points_X.size(); i++){
        cv::Point pt1 = cv::Point(traj_points_X[i-1], traj_points_Y[i-1]);
        cv::Point pt2 = cv::Point(traj_points_X[i], traj_points_Y[i]);

        cv::line(cv_ptr->image, pt1, pt2, red, 1, 16, 0);
    }
    // Convert back to ROS Msg and publish
    traj_pub.publish(cv_ptr->toImageMsg());

}

int main(int argc, char **argv){
    ros::init(argc, argv, "traj_plotter");
    ros::NodeHandle n;

    //subscribe to ee pose
    ros::Subscriber traj_sub = n.subscribe("/aruco_simple/pixel2",1,trajPointsCallback);
    //callback: everytime new pose is received, pushback into a vector
    
    //subscribe to image
    ros::Subscriber img_sub = n.subscribe("/aruco_simple/result",1,trajPlotCallback);
    //callback: draw line between successive pairwise points in the vector
    
    // publish the new image
    traj_pub = n.advertise<sensor_msgs::Image>("/traj_output",1);
    ros::spin();
    return 0;
}