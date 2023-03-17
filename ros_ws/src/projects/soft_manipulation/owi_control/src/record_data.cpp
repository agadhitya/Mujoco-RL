#include "ros/ros.h"
#include "rosbag/bag.h"
#include <time.h>

#include "std_msgs/Float32.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"
#include "sensor_msgs/Image.h"
#include "geometry_msgs/PointStamped.h"
#include "geometry_msgs/Point.h"
#include "std_msgs/Bool.h"

std_msgs::Float32 w1;
geometry_msgs::Point pwm;
sensor_msgs::Image img;
geometry_msgs::Point pose;
std_msgs::Bool flag;
std_msgs::Float64 action_msg;
geometry_msgs::Point base_px;
geometry_msgs::Point ee_px;
geometry_msgs::Point elbow_px;

void flagCheck(const std_msgs::Bool &msg){
    flag.data = msg.data;
}

void rec_pos(const geometry_msgs::PointStamped &msg){
    pose.x = msg.point.x;
    pose.y = msg.point.y;
}

void rec_img(const sensor_msgs::Image image){
    img = image;
}

void rec_vel(const std_msgs::Float32 &msg){
    w1.data = msg.data;
}

void rec_pwm(const std_msgs::Float64MultiArray &msg){
    pwm.x = msg.data.at(0);
    pwm.y = msg.data.at(1);
}

void rec_action(const std_msgs::Float64 &msg){
    action_msg.data = msg.data;
}

void rec_base(const geometry_msgs::PointStamped &msg){
    base_px.x = msg.point.x;
    base_px.y = msg.point.y;
}

void shut_down(const std_msgs::Float64 &msg){
    if(msg.data == 1){
        ROS_INFO("RECORDING COMPLETE");
        ros::shutdown();
    }
}

void rec_ee(const geometry_msgs::PointStamped &msg){
    ee_px.x = msg.point.x;
    ee_px.y = msg.point.y;
}

void rec_elbow(const geometry_msgs::PointStamped &msg){
    elbow_px.x = msg.point.x;
    elbow_px.y = msg.point.y;
}

int main(int argc, char** argv){

    rosbag::Bag bag;
    bag.open("/home/abhinav/freq_test.bag", rosbag::bagmode::Write);

    ros::init(argc, argv, "record_freq");
    ros::NodeHandle n;

    // ros::Subscriber sub1 = n.subscribe("angular_vel",1,rec_vel);
    // ros::Subscriber sub6 = n.subscribe("action",1,rec_action);
    ros::Subscriber sub2 = n.subscribe("pwm",1,rec_pwm);
    ros::Subscriber sub3 = n.subscribe("usb_cam/image_raw",1,rec_img);
    ros::Subscriber sub4 = n.subscribe("aruco_simple/pixel3",1,rec_pos);
    ros::Subscriber sub5 = n.subscribe("newFrame",1,flagCheck);
    ros::Subscriber sub7 = n.subscribe("aruco_simple/pixel",1,rec_base);
    ros::Subscriber sub8 = n.subscribe("aruco_simple/pixel2",1,rec_ee);
    ros::Subscriber sub9 = n.subscribe("aruco_simple/pixel4",1,rec_elbow);

    ros::Subscriber shutdown_sub = n.subscribe("shutdown",1,shut_down);

    ros::Rate r{30};
    while(ros::ok()){
        ros::spinOnce();
        ros::Time t = ros::Time::now();
        // bag.write("angular_vel", t, w1);
        // bag.write("action", t, action_msg);
        bag.write("pwm", t, pwm);
        bag.write("pixel3", t,pose);
        bag.write("image", t, img);
        bag.write("newFrame", t,flag);
        bag.write("base_pos",t, base_px);
        bag.write("end_effector",t,ee_px);
        bag.write("pixel4",t,elbow_px);

        r.sleep();
    }

    bag.close();

    ros::spin();
    return 0;
}