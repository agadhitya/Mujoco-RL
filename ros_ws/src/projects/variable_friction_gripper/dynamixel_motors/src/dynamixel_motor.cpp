#include "dynamixel_motors/dynamixel_motor.hpp"

DynamixelMotor::DynamixelMotor(){}

void DynamixelMotor::initialize(int id, std::string port_name, int direction, int operating_mode, float protocol_version, int baud_rate){
	port_handler_ = dynamixel::PortHandler::getPortHandler(port_name.c_str());
	packet_handler_ = dynamixel::PacketHandler::getPacketHandler(protocol_version);
	if (!port_handler_->openPort()){
		std::runtime_error("[DynamixelModel] Failed to open the port. Aborting...");
	}
	if (!port_handler_->setBaudRate(baud_rate)){
		std::runtime_error("[DynamixelModel] Failed to set the baud rate. Aborting...");
	}
	id_ = id;
	disable();
	mode_ = getOperatingMode();
	setDriveMode(direction);
	setOperatingMode(operating_mode);
	enable();
}
DynamixelMotor::~DynamixelMotor(){
	port_handler_->closePort();
}
