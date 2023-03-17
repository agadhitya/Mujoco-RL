#include "dynamixel_motors/dynamixel_node.hpp"
#include "ros/ros.h"

//Creating a class for fingers with length , width and friction_coefficient as its attributes..

class finger
{
 public:
	float length;
	float width;
	float friction_coefficient;
 
	finger()
	{
 	 std::cout<<"Constructor initiated";
         length=0;
 	 width=0;
	 friction_coefficient=0;
	}
	finger(float l, float w, float fc)
	{
	 ROS_INFO("Constructor with parameters initiated");
 	 length=l;
	 width=w;
 	 friction_coefficient=fc;
	}
};
