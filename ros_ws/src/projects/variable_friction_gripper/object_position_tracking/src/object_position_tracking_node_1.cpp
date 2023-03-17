#include "ros/ros.h"
#include <tf/tf.h>
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "std_msgs/Float32.h"

float x1[3], x2[3]; 
float Orientation_Z;
float Orientation_Reference_Z; 

void Callback1(const geometry_msgs::Pose &msg1)
{
	x1[0]=msg1.position.x;
	x1[1]=msg1.position.y;
	x1[2]=msg1.position.z;
	for(int i= 0;i<3;i++)
	{
		printf("pose1: ");
		printf("%f\n",x1[i]);
	}
	tf::Quaternion q(
        msg1.orientation.x,
        msg1.orientation.y,
        msg1.orientation.z,
        msg1.orientation.w);
	tf::Matrix3x3 m(q);
    	double roll, pitch, yaw;
    	m.getRPY(roll, pitch, yaw);
    	Orientation_Reference_Z=yaw;
    	
}


void Callback2(const geometry_msgs::Pose &msg2)
{
	x2[0]=msg2.position.x;
	x2[1]=msg2.position.y;
	x2[2]=msg2.position.z;
	for(int i= 0;i<3;i++)
	{
		printf("pose2: ");
		printf("%f\n", x2[i]);
	}
	tf::Quaternion q(
        msg2.orientation.x,
        msg2.orientation.y,
        msg2.orientation.z,
        msg2.orientation.w);
	tf::Matrix3x3 m(q);
    	double roll, pitch, yaw;
    	m.getRPY(roll, pitch, yaw);
    	Orientation_Z=yaw;
}

int main(int argc, char **argv)
{
	float x[3];
	ros::init(argc, argv, "listener");

	ros::NodeHandle n;

	ros::Subscriber sub1 = n.subscribe("/aruco_simple/pose", 1000, Callback1);
	ros::Subscriber sub2 = n.subscribe("/aruco_simple/pose2", 1000, Callback2);
	ros::Publisher pub = n.advertise<geometry_msgs::Point>("/object_position_2", 1000);
	ros::Publisher pub1 = n.advertise<std_msgs::Float32>("/object_orientation", 1000);
	ros::Rate r(45);
	while(ros::ok())
	{
		for (int i=0; i<3; i++)
		{
			x[i] = x1[i]-x2[i];
		}

		geometry_msgs::Point msg;
		msg.x = -x[0];
		msg.y = x[1];
		msg.z = x[2];
		std_msgs::Float32 Block_orientation;
		Block_orientation.data= Orientation_Z - Orientation_Reference_Z;


		//ROS_INFO("publishing the pose difference");



		pub.publish(msg);
		pub1.publish(Block_orientation);
		for(int i= 0;i<3;i++)
		{
			printf("***POSE***: ");
			printf("%f\n", x[i]);
		}

		ros::spinOnce();
		r.sleep();
	}

	return 0;
}
