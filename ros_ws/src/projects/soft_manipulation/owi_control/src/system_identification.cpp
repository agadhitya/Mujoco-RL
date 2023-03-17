#include "ros/ros.h"

#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"

int main(int argc, char **argv){

ros::init(argc, argv, "system_id");
ros::NodeHandle n;

ros::Duration(2.0).sleep(); // Sleep for 2 seconds

ros::Publisher shutdown_pub = n.advertise<std_msgs::Float64>("shutdown",1);
// This is done to wait for the visual-feedback system to acquire the aruco markers

ros::Publisher vel_pub = n.advertise<std_msgs::Float64MultiArray>("pwm",1);

int frames = 0;

std_msgs::Float64MultiArray vel_msg;    //Velocity Message for Pub
vel_msg.data.resize(2);
n.getParam("sys_id/j1_vel", vel_msg.data[0]);
n.getParam("sys_id/j2_vel", vel_msg.data[1]);

std_msgs::Float64 shutdown_msg;
shutdown_msg.data = 0.0;

ros::Rate r{30};

while(ros::ok()){
    frames += 1;
    
    vel_pub.publish(vel_msg);
    shutdown_pub.publish(shutdown_msg);

    if(frames == 40){

        vel_msg.data[0] = 0.0;
        vel_msg.data[1] = 0.0;
        vel_pub.publish(vel_msg);

        shutdown_msg.data = 1.0;
        shutdown_pub.publish(shutdown_msg);

    }

    ros::spinOnce();
    r.sleep();
}

ros::spin();
return 0;
}
