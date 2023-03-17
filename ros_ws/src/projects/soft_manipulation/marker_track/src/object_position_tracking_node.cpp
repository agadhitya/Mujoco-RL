#include "ros/ros.h"
#include <tf/tf.h>
#include <math.h>

#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "std_msgs/Float32.h"

#define PI 57.2957795786

float base[3], ee[3], elbow[3], l2[4]; 

float Orientation_Z;
float Orientation_Reference_Z; 

void baseCallback(const geometry_msgs::PointStamped &msg1)
{
	base[0]=msg1.point.x; // - 0.0057;  // Adding an offset to the marker1
	base[1]=msg1.point.y;
	base[2]=0;
	for(int i= 0;i<3;i++)
	{
		printf("pose1: ");
		printf("%f\n",base[i]);
	}
	// tf::Quaternion q(
    //     msg1.orientation.x,
    //     msg1.orientation.y,
    //     msg1.orientation.z,
    //     msg1.orientation.w);
	// tf::Matrix3x3 m(q);
    // 	double roll, pitch, yaw;
    // 	m.getRPY(roll, pitch, yaw);
    // 	Orientation_Reference_Z=yaw;
    	
}

void eeCallback(const geometry_msgs::PointStamped &msg2)
{
	ee[0]=msg2.point.x;
	ee[1]=msg2.point.y;
	ee[2]=0;
	for(int i= 0;i<3;i++)
	{
		printf("pose2: ");
		printf("%f\n", ee[i]);
	}
	// tf::Quaternion q(
    //     msg2.orientation.x,
    //     msg2.orientation.y,
    //     msg2.orientation.z,
    //     msg2.orientation.w);
	// tf::Matrix3x3 m(q);
    // 	double roll, pitch, yaw;
    // 	m.getRPY(roll, pitch, yaw);
    // 	Orientation_Z=yaw;
}

void elbowCallback(const geometry_msgs::PointStamped &msg3){
	elbow[0]=msg3.point.x;
	elbow[1]=msg3.point.y;
	elbow[2]=0;
	for(int i= 0;i<3;i++)
	{
		printf("pose3: ");
		printf("%f\n", elbow[i]);
	}
}

void l2Callback(const geometry_msgs::PointStamped &msg4){
	l2[0]=msg4.point.x;
	l2[1]=msg4.point.y;
	l2[2]=0;
	for(int i= 0;i<3;i++)
	{
		printf("pose4: ");
		printf("%f\n", l2[i]);
	}
}

int main(int argc, char **argv)
{
	float x[3];
	float elbow_marker[3];
	ros::init(argc, argv, "listener");

	ros::NodeHandle n;

	ros::Subscriber sub1 = n.subscribe("/aruco_simple/pixel", 1, baseCallback);
	ros::Subscriber sub2 = n.subscribe("/aruco_simple/pixel2", 1, eeCallback);
	ros::Subscriber sub3 = n.subscribe("/aruco_simple/pixel3", 1, elbowCallback);
	ros::Subscriber sub4 = n.subscribe("/aruco_simple/pixel4", 1, l2Callback);
	ros::Publisher theta_pub = n.advertise<geometry_msgs::Point>("/theta_vals", 1);

	// ros::Publisher pub = n.advertise<geometry_msgs::Point>("/object_position", 1000);
	// ros::Publisher pub1 = n.advertise<std_msgs::Float32>("/object_orientation", 1000);
	// ros::Publisher elbow_pub = n.advertise<geometry_msgs::Point>("/elbow_position", 1000);
	
	// Computing angles
	float theta1 = 0;
	float theta2 = 0;
	
	ros::Rate r(30);
	while(ros::ok())
	{


		if(elbow[0]!= base[0]){
			if(elbow[0]<base[0]){
				theta1 = atan((elbow[1] - base[1])/(elbow[0] - base[0])) * PI; //Since x is opposite dirn
			}
			else
			{
				theta1 = 180 + atan((elbow[1] - base[1])/(elbow[0] - base[0])) * PI;
			}
			
		}
		else
		{
			theta1 = 90;	
		}

		if(ee[0] != l2[0]){
			if(l2[0] > ee[0]){
				theta2 = (atan((ee[1] - l2[1])/(ee[0] - l2[0])) * PI) - theta1; //Since x is opp dirn
			}
			else
			{
				theta2 = 180 + (atan((ee[1] - l2[1])/(ee[0] - l2[0])) * PI) - theta1;
			}
		}
		else
		{
			theta2 = 90;
		}
		geometry_msgs::Point angle_msg;
		angle_msg.x = theta1;
		angle_msg.y = theta2;
		angle_msg.z = 0.0;

		theta_pub.publish(angle_msg);

		ros::spinOnce();
		r.sleep();
	}

	return 0;
}
