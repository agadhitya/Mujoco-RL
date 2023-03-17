#include "ros/ros.h"
#include <math.h>

#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"

int main(int argc, char **argv){

    ros::init(argc, argv, "const_vel");
    ros::NodeHandle n;

    ros::Publisher sin_pub = n.advertise<std_msgs::Float64MultiArray>("pwm",1);
    ros::Publisher action_pub = n.advertise<std_msgs::Float64>("action",1);
    ros::Publisher shutdown_pub = n.advertise<std_msgs::Float64>("shutdown",1);

    float count = 0;
    float vel;
    std_msgs::Float64MultiArray vel_msg;
    vel_msg.data.resize(2);

    std_msgs::Float64 action_msg;
    std_msgs::Float64 shutdown_msg;

    ros::Rate r{30};
    while(ros::ok()){
        if(count <= 150){        // running experiment for 20s at 30 fps
            n.getParam("/const_vel_exp/vel", vel);
            //vel = 0.5;          // add from YAML
            vel_msg.data[0] = vel;
            vel_msg.data[1] = 0;
            sin_pub.publish(vel_msg);
            count += 1;
            
            action_msg.data = 1;
            action_pub.publish(action_msg);

            shutdown_msg.data = 0.0;
            shutdown_pub.publish(shutdown_msg);
        }
        else{
            vel_msg.data[0] = 0;
            sin_pub.publish(vel_msg);
            action_msg.data = -1;
            action_pub.publish(action_msg);
            ROS_INFO("DONE");
            count += 1;
            if(count >200){
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