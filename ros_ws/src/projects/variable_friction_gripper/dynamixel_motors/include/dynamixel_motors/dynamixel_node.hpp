#ifndef DYN_OPENHAND_NODE_HPP
#define DYN_OPENHAND_NODE_HPP

#include "dynamixel_motors/dynamixel_motor.hpp"
#include "ros/ros.h"
#include "common_msgs_gl/SendIntArray.h"
#include "common_msgs_gl/SendDoubleArray.h"
#include "common_msgs_gl/GetDoubleArray.h"
#include "common_msgs_gl/GetBoolArray.h"

class DynamixelNode{
	int number_of_motors_ = 0;
	ros::NodeHandle node_handle_;
	ros::ServiceServer srvsrvr_set_operating_mode_, srvsrvr_cmd_pos_, srvsrvr_cmd_pos_ind_, srvsrvr_cmd_torque_, srvsrvr_cmd_torque_ind_, srvsrvr_read_pos_, srvsrvr_read_velocity_, srvsrvr_read_current_, srvsrvr_is_moving_, srvsrvr_cmd_vel_, srvsrvr_cmd_vel_ind_, srvsrv_set_profile_velocity_;
	DynamixelMotor** dyn_motors_;
	
	bool callbackSetOperatingMode(common_msgs_gl::SendIntArray::Request& req, common_msgs_gl::SendIntArray::Response& res);
	bool callbackCmdPos(common_msgs_gl::SendDoubleArray::Request& req,common_msgs_gl::SendDoubleArray::Response& res);
	bool callbackCmdPosInd(common_msgs_gl::SendDoubleArray::Request& req,common_msgs_gl::SendDoubleArray::Response& res);
	bool callbackCmdTorqueInd(common_msgs_gl::SendDoubleArray::Request& req,common_msgs_gl::SendDoubleArray::Response& res);
	bool callbackCmdTorque(common_msgs_gl::SendDoubleArray::Request& req,common_msgs_gl::SendDoubleArray::Response& res);
	bool callbackReadPos(common_msgs_gl::GetDoubleArray::Request& req,common_msgs_gl::GetDoubleArray::Response& res);
	bool callbackReadVel(common_msgs_gl::GetDoubleArray::Request& req,common_msgs_gl::GetDoubleArray::Response& res);
	bool callbackReadTorque(common_msgs_gl::GetDoubleArray::Request& req,common_msgs_gl::GetDoubleArray::Response& res);
	bool callbackIsMoving(common_msgs_gl::GetBoolArray::Request& req,common_msgs_gl::GetBoolArray::Response& res);
	bool callbackCmdVel(common_msgs_gl::SendDoubleArray::Request& req,common_msgs_gl::SendDoubleArray::Response& res);
	bool callbackCmdVelInd(common_msgs_gl::SendDoubleArray::Request& req,common_msgs_gl::SendDoubleArray::Response& res);
	bool callbackSetVel(common_msgs_gl::SendDoubleArray::Request& req, common_msgs_gl::SendDoubleArray::Response& res);
public:

	DynamixelNode(std::string dyn_type, std::vector<int> ids, std::string port_name = "/dev/ttyUSB0", std::vector<int> direction = {1,1}, std::vector<int> operating_mode = {3,3}, float protocol_version = 2.0, int baud_rate = 57600, std::vector<double> current_limits = {1.0,1.0});

};
#endif
