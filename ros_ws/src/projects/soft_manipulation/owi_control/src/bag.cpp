#include "ros/ros.h"
#include "rosbag/bag.h"
#include <time.h>

#include "sensor_msgs/Image.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"

std_msgs::Float64 err_X;
std_msgs::Float64 err_Y;
std_msgs::Float64 err_Z;
sensor_msgs::Image img;

void rec_err(const std_msgs::Float64MultiArray::ConstPtr& msg){
    err_X.data = msg->data.at(0);
    err_Y.data = msg->data.at(1);
    err_Z.data = msg->data.at(2);
}

void rec_img(const sensor_msgs::Image image){

    img = image;

}


int main(int argc, char** argv){
    rosbag::Bag bag;
    bag.open("/home/abhinav/test2.bag", rosbag::bagmode::Write);

    ros::init(argc, argv, "record_util");
    ros::NodeHandle n;

    ros::Subscriber sub1 = n.subscribe("error",1,rec_err);
    ros::Subscriber sub2 = n.subscribe("usb_cam/image_raw",1,rec_img);
    ros::Rate r{30};
    while(ros::ok()){
        ros::Time t = ros::Time::now();

        bag.write("err_X", t, err_X);
        bag.write("err_Y", t, err_Y);
        bag.write("err_Z", t, err_Z);
        bag.write("image", t, img);

        ros::spinOnce();
        r.sleep();
    }

    bag.close();

    ros::spin();
    return 0;
}