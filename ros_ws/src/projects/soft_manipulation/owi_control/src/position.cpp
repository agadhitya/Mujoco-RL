#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <vector>
#include <image_transport/image_transport.h>
#include <opencv2/core/types.hpp>

#include "geometry_msgs/PointStamped.h"
#include "sensor_msgs/image_encodings.h"

ros::NodeHandle* nh;
cv::Mat inImage;

float traj_point_X;
float traj_point_Y;
float start_X;
float start_Y;
float goal_X;
float goal_Y;

ros::Publisher circle_pub;


geometry_msgs::Point get_start_pose(){

    geometry_msgs::Point startPose;
    nh->getParam("owi_controller/start_x",startPose.x);
    nh->getParam("owi_controller/start_y",startPose.y);
    startPose.z = 0;

    return startPose;
}

geometry_msgs::Point get_goal_pose(){

    geometry_msgs::Point goalPose;
    nh->getParam("owi_controller/goal_x",goalPose.x);
    nh->getParam("owi_controller/goal_y",goalPose.y);
    goalPose.z = 0;

    return goalPose;
}


void trajPointCallback(const geometry_msgs::PointStamped &msg){
    traj_point_X = msg.point.x;
    traj_point_Y = msg.point.y;
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
    const cv::Scalar green(0,255,0);
    const cv::Scalar blue(255,0,0);

    cv::Point center1(traj_point_X, traj_point_Y);  // end-effector posn
    cv::Point center2(start_X,start_Y);             // start posn
    cv::Point center3(goal_X,goal_Y);

    int rad = 4;
    int thickness = 1;
    cv::circle(cv_ptr->image, center1, rad, red, thickness);
    cv::circle(cv_ptr->image, center2, rad, green, thickness);
    cv::circle(cv_ptr->image, center3, rad, blue, thickness);

    // Convert back to ROS Msg and publish
    circle_pub.publish(cv_ptr->toImageMsg());

}

int main(int argc, char **argv){
    ros::init(argc, argv, "position_robot");
    nh = new ros::NodeHandle;

    geometry_msgs::Point start = get_start_pose();
    start_X = start.x;
    start_Y = start.y;
    
    geometry_msgs::Point goal = get_goal_pose();
    goal_X = goal.x;
    goal_Y = goal.y;

    //subscribe to ee pose
    ros::Subscriber traj_sub = nh->subscribe("/aruco_simple/pixel2",1,trajPointCallback);
    //callback: everytime new pose is received, pushback into a vector
    
    //subscribe to image
    ros::Subscriber img_sub = nh->subscribe("/aruco_simple/result",1,trajPlotCallback);
    //callback: draw marker at end effector and at start position
    
    // publish the new image
    circle_pub = nh->advertise<sensor_msgs::Image>("/posn_output",1);
    ros::spin();
    return 0;
}