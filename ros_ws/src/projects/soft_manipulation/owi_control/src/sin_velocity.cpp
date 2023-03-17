#include "ros/ros.h"
#include <math.h>

#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"

#define PI 3.14159265/180

int main(int argc, char **argv){

    ros::init(argc, argv, "sin_vel");
    ros::NodeHandle n;

    ros::Publisher sin_pub = n.advertise<std_msgs::Float64MultiArray>("pwm",1);
    ros::Publisher action_pub = n.advertise<std_msgs::Float64>("action",1);
    ros::Publisher shutdown_pub = n.advertise<std_msgs::Float64>("shutdown",1);

    float count = 0;
    float vel;
    float shutdown_counter = 0;;
    std_msgs::Float64 shutdown_msg;
    shutdown_msg.data = 0;
    std_msgs::Float64MultiArray vel_msg;
    vel_msg.data.resize(2);

    std_msgs::Float64 action_msg;

    ros::Rate r{30};

    while(ros::ok()){
        if(count <=360){
            vel = sin(count*PI);
            
            vel_msg.data[0] = vel/2;
            vel_msg.data[1] = 0;
            
            sin_pub.publish(vel_msg);
            if(vel_msg.data[0] >=0)
            {
                action_msg.data = 1.0;
            }
            else
            {
                action_msg.data = 2.0;
            }
            action_pub.publish(action_msg);
            shutdown_pub.publish(shutdown_msg);
            count += 1.5;
        }
        else{
            ROS_INFO("DONE!");
            action_msg.data = -1.0;
            action_pub.publish(action_msg);
            shutdown_counter += 1;
            if(shutdown_counter >= 50){
                shutdown_msg.data = 1.0;
                shutdown_pub.publish(shutdown_msg);
                
                ros::shutdown();
            }
        }
            ros::spinOnce();
            r.sleep();
    }
        ros::spin();
    return 0;
}