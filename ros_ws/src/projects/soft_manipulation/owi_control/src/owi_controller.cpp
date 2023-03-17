#include "ros/ros.h"
#include <eigen3/Eigen/Dense>
#include <math.h>

#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/Float64.h"
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/PointStamped.h"

#define PI 3.14159265/180

ros::NodeHandle* nh;
ros::Publisher pub;
ros::Publisher err_pub;
ros::Publisher shutdown_pub;
Eigen::MatrixXd Jr(2,2);
// float count = 0;

geometry_msgs::Point curPose_;

geometry_msgs::Point get_goal_pose(){

    geometry_msgs::Point goalPose;
    nh->getParam("owi_controller/goal_x",goalPose.x);
    nh->getParam("owi_controller/goal_y",goalPose.y);
    goalPose.z = 0;

    return goalPose;
}

void get_cur_pose(const geometry_msgs::PointStamped &msg){

    curPose_.x = msg.point.x;
    curPose_.y = msg.point.y;
    curPose_.z = msg.point.z;   
    
    geometry_msgs::Point goal = get_goal_pose();
    
    Eigen::Vector3d f(curPose_.x, curPose_.y, curPose_.z); // cur pose vector
    Eigen::Vector3d f_d(goal.x, goal.y, goal.z);  // goal pose vector
    Eigen::Vector2d err;
    Eigen::Vector2d t_dot;      // velocity in task space
    Eigen::Vector2d j_dot;      // velocity in joint space
    Eigen::Vector2d j_dot_fit;  // joint velocities fit to range
    err(0) = f_d(0) - f(0);
    err(1) = f_d(1) - f(1);

    int k = 1;
    
    // Send robot joint velocities if error is greater than 1 px in either direction
    std_msgs::Float64 shutdown_msg;
    float err_norm = sqrt((err(0)*err(0)) + (err(1)*err(1)));
    printf("err_x: %f", err(0));
    printf("err_y: %f", err(1));
    printf("err: %f\n", err_norm);
    if(err_norm >= 3){

        Eigen::MatrixXd Kp(2,2); //Gains matrix, this incorporates the image Jacobian since it is constant for a 2D case
        Kp << k, 0,
              0, k;

        // Robot jacobian is updated using the update_jacobian()
        // Inverting this updated jacobian

        Eigen::MatrixXd Jr_inv(2,2);
        Jr_inv = Jr.inverse();

        t_dot = -Kp * err;

        j_dot = Jr_inv * t_dot;

        // Fix joint velocities to be between (-0.5 : 0.5) while preserving their ratio
        j_dot_fit(0) = j_dot(0)/(2*(abs(j_dot(0)) + abs(j_dot(1))));
        j_dot_fit(1) = j_dot(1)/(2*(abs(j_dot(0)) + abs(j_dot(1))));

        std_msgs::Float64MultiArray vel_msg;    //Velocity Message for Pub
        vel_msg.data.resize(2);
    
        vel_msg.data[0] = j_dot_fit(0);
        vel_msg.data[1] = j_dot_fit(1);

        pub.publish(vel_msg);

        std_msgs::Float64MultiArray err_msg;
        err_msg.data.resize(3);
    
        err_msg.data[0] = err(0);
        err_msg.data[1] = err(1);
        err_msg.data[2] = 0;

        err_pub.publish(err_msg);

        shutdown_msg.data = 0;
        shutdown_pub.publish(shutdown_msg);
    }
    else{
        
        std_msgs::Float64MultiArray vel_msg;    //Velocity Message for Pub
        vel_msg.data.resize(2);

            vel_msg.data[0] = 0;
            vel_msg.data[1] = 0;
            pub.publish(vel_msg);

            shutdown_msg.data = 1.0;
            shutdown_pub.publish(shutdown_msg);

            ros::Duration(2.0).sleep(); // waiting for record note to shutdown
            ros::shutdown();
    }
}

void update_Jr(const geometry_msgs::Point &msg){

    float th1 = msg.x;
    float th2 = msg.y;

    float l1 = 0.092;    // length of link1 in m
    float l2 = 0.09;   // length of link2 in m

    float jr1 = - l1*sin(th1*PI) - l2*sin((th1+th2)*PI);
    float jr2 = - l2*sin((th1 + th2)*PI);
    float jr3 = l1*cos(th1*PI) + l2*cos((th1 + th2)*PI);
    float jr4 = l2*cos((th1 + th2)*PI);
   
    Jr << jr1, jr2,
          jr3, jr4;


}

int main(int argc, char **argv){
    
    ros::init(argc, argv, "owi_controller");
    nh = new ros::NodeHandle;
    ros::Duration(2.0).sleep(); // Sleep for 2 seconds
    // This is done to wait for the visual-feedback system to acquire the aruco markers

    ros::Subscriber sub = nh->subscribe("aruco_simple/pixel2",1,get_cur_pose);
    ros::Subscriber theta_sub = nh->subscribe("theta_vals",1,update_Jr);
    shutdown_pub = nh->advertise<std_msgs::Float64>("shutdown",1);

    ros::Rate r{30};
    while(ros::ok()){
        pub = nh->advertise<std_msgs::Float64MultiArray>("pwm",1);
        err_pub = nh->advertise<std_msgs::Float64MultiArray>("error",1);

        ros::spinOnce();
        r.sleep();
    }

    
    ros::spin();
    return 0;
}