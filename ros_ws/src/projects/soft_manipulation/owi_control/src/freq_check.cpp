#include "ros/ros.h"
#include "std_msgs/Float32.h"
#include "geometry_msgs/Point.h"
#include "std_msgs/Bool.h"

#define rad 0.0174533

ros::WallTime prevTime = ros::WallTime::now();
float old_theta1 = 0;
float prev_angular_vel = 0;
float cur_angular_vel = 0;
float cur_angular_acc = 0;
ros::Publisher pub;
bool flag;

void frameCallback(const std_msgs::Bool &msg){
    flag = msg.data;
}

void diff(const geometry_msgs::Point &msg){
       
    ros::WallTime curTime = ros::WallTime::now();
    ros::WallDuration d = curTime - prevTime;
    float dt = d.toSec();
    // ROS_INFO("DURATION: %f", dt);

    float theta1 = msg.x * rad;

    if((theta1 - old_theta1) != 0){
        // ROS_INFO("DIFF");
        // ROS_INFO_STREAM(theta1 - old_theta1);
        cur_angular_vel = (theta1 - old_theta1)/dt;
        // ROS_INFO("VEL");
        // ROS_INFO_STREAM(cur_angular_vel);
        cur_angular_acc = (cur_angular_vel - prev_angular_vel)/dt;
        old_theta1 = theta1;
    }
    
    else{
        cur_angular_vel = prev_angular_vel + cur_angular_acc*dt;
        old_theta1 += cur_angular_vel*dt;
    }

    prevTime = curTime;
    prev_angular_vel = cur_angular_vel;

    std_msgs::Float32 angular_msg;
    angular_msg.data = cur_angular_vel;
    pub.publish(angular_msg);

}

int main(int argc, char **argv){

    ros::init(argc, argv, "freq_response");
    ros::NodeHandle n;
    //Check if new frame is available
    ros::Subscriber flag_sub = n.subscribe("newFrame", 1, frameCallback);
    //subscribe to angles
    ros::Subscriber sub = n.subscribe("theta_vals", 1, diff);
    // callback: (cur angle - old angle)/(timestamp cur - timestamp old)
    
    // publish this angular velocity
    pub = n.advertise<std_msgs::Float32>("angular_vel",1);
    
    //record angular velocity
    //plot angular velocity




    ros::spin();
    return 0;
}