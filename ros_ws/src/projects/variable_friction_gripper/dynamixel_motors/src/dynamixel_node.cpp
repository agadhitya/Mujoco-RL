#include "dynamixel_motors/dynamixel_node.hpp"
#include "dynamixel_motors/dyn_xm_motor.hpp"

DynamixelNode::DynamixelNode(std::string dyn_type, std::vector<int> ids, std::string port_name, std::vector<int> direction, std::vector<int> operating_mode, float protocol_version, int baud_rate, std::vector<double> current_limits){
	number_of_motors_ = ids.size();
	if(dyn_type == "XM"){
		dyn_motors_ = new DynamixelMotor*[number_of_motors_];
		for(size_t i = 0; i<number_of_motors_; i++)
			dyn_motors_[i] = new DynXMMotor(ids[i], port_name, direction[i], operating_mode[i], protocol_version, baud_rate);
	}
	else{
		std::runtime_error("[DynamixelNode] The classes for dynamixel are not implemented for the selected type.");
	}
	
	for(size_t i = 0; i < number_of_motors_; i++){
		dyn_motors_[i]->setCurrentLimit(current_limits[i]);
	}

	srvsrvr_set_operating_mode_ = node_handle_.advertiseService("set_operating_mode", &DynamixelNode::callbackSetOperatingMode, this);
	srvsrvr_cmd_pos_ = node_handle_.advertiseService("cmd_pos", &DynamixelNode::callbackCmdPos, this);
	srvsrvr_cmd_torque_ = node_handle_.advertiseService("cmd_torque", &DynamixelNode::callbackCmdTorque, this);
	srvsrvr_cmd_pos_ind_ = node_handle_.advertiseService("cmd_pos_ind", &DynamixelNode::callbackCmdPosInd, this);
	srvsrvr_cmd_torque_ind_ = node_handle_.advertiseService("cmd_torque_ind", &DynamixelNode::callbackCmdTorqueInd, this);
	srvsrvr_read_pos_ = node_handle_.advertiseService("read_pos", &DynamixelNode::callbackReadPos, this);
	srvsrvr_read_current_ = node_handle_.advertiseService("read_current", &DynamixelNode::callbackReadTorque, this);
	srvsrvr_is_moving_ = node_handle_.advertiseService("is_moving", &DynamixelNode::callbackIsMoving, this);
	srvsrv_set_profile_velocity_ = node_handle_.advertiseService("set_vel", &DynamixelNode::callbackSetVel, this);
	srvsrvr_cmd_vel_ = node_handle_.advertiseService("cmd_vel", &DynamixelNode::callbackCmdVel, this);
	srvsrvr_cmd_vel_ind_ = node_handle_.advertiseService("cmd_vel_ind", &DynamixelNode::callbackCmdVelInd, this);
}
bool DynamixelNode::callbackSetVel(common_msgs_gl::SendDoubleArray::Request& req, common_msgs_gl::SendDoubleArray::Response& res){
	for (size_t i = 0; i < number_of_motors_; i++){
		dyn_motors_[i]-> setVelocity(req.data[i]);
	}
	return true;
}

bool DynamixelNode::callbackIsMoving(common_msgs_gl::GetBoolArray::Request& req,common_msgs_gl::GetBoolArray::Response& res){
	res.data.resize(number_of_motors_);
	for (size_t i = 0; i < number_of_motors_; i++){
			res.data[i] = dyn_motors_[i]->isMoving();
	}
	return true;
}

bool DynamixelNode::callbackSetOperatingMode(common_msgs_gl::SendIntArray::Request& req, common_msgs_gl::SendIntArray::Response& res){
	if (req.data.size() == 1)
		for (size_t i = 0; i < number_of_motors_; i++){
			dyn_motors_[i]->setOperatingMode(req.data[0]);
		}
	else
		for (size_t i = 0; i < number_of_motors_; i++){
			dyn_motors_[i]->setOperatingMode(req.data[i]);
		}
        return true;

}

bool DynamixelNode::callbackCmdPos(common_msgs_gl::SendDoubleArray::Request& req,common_msgs_gl::SendDoubleArray::Response& res){
	for (size_t i = 0; i < number_of_motors_; i++){
		dyn_motors_[i]->moveTo(req.data[i]);
	}
	return true;
}

bool DynamixelNode::callbackCmdVel(common_msgs_gl::SendDoubleArray::Request& req, common_msgs_gl::SendDoubleArray::Response& res){
	for(size_t i=0; i< number_of_motors_;i++){
		dyn_motors_[i]->sendVelocity(req.data[i]);
	}
	return true;
}

bool DynamixelNode::callbackCmdTorque(common_msgs_gl::SendDoubleArray::Request& req,common_msgs_gl::SendDoubleArray::Response& res){
	for (size_t i = 0; i < number_of_motors_; i++){
		dyn_motors_[i]->applyTorque(req.data[i]);
	}
	return true;

}

bool DynamixelNode::callbackCmdPosInd(common_msgs_gl::SendDoubleArray::Request& req,common_msgs_gl::SendDoubleArray::Response& res){
	dyn_motors_[((int)req.data[0])]->moveTo(req.data[1]);
	return true;
}

bool DynamixelNode::callbackCmdVelInd(common_msgs_gl::SendDoubleArray::Request& req, common_msgs_gl::SendDoubleArray::Response& res){
	dyn_motors_[((int)req.data[0])]->sendVelocity(req.data[1]);
	return true;
}

bool DynamixelNode::callbackCmdTorqueInd(common_msgs_gl::SendDoubleArray::Request& req,common_msgs_gl::SendDoubleArray::Response& res){
	dyn_motors_[((int)req.data[0])]->applyTorque(req.data[1]);
	return true;
}


bool DynamixelNode::callbackReadPos(common_msgs_gl::GetDoubleArray::Request& req,common_msgs_gl::GetDoubleArray::Response& res){
	res.data.resize(number_of_motors_);
	for(size_t i = 0; i < number_of_motors_; i++)
		res.data[i] = dyn_motors_[i]->readPos();
	return true;
}

bool DynamixelNode::callbackReadVel(common_msgs_gl::GetDoubleArray::Request& req, common_msgs_gl::GetDoubleArray::Response& res){
	res.data.resize(number_of_motors_);
	for(size_t i = 0; i <number_of_motors_; i++)
		res.data[i] = dyn_motors_[i]->readVelocity();
	return true;
}

bool DynamixelNode::callbackReadTorque(common_msgs_gl::GetDoubleArray::Request& req,common_msgs_gl::GetDoubleArray::Response& res){
	res.data.resize(number_of_motors_);
	for(size_t i = 0; i < number_of_motors_; i++)
		res.data[i] = dyn_motors_[i]->readCurrent();
	return true;
}
