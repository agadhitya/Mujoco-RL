#include "dynamixel_motors/dyn_xm_motor.hpp"
#include <cmath>
DynXMMotor::DynXMMotor(){
	initialize();
}

DynXMMotor::DynXMMotor(int id, std::string port_name, int drive_mode, int operating_mode, float protocol_version, int baud_rate){
	initialize(id, port_name, drive_mode, operating_mode, protocol_version, baud_rate);
}

void DynXMMotor::enable(){
	int no_of_trials = 0;
	do{
		dxl_comm_result_ = packet_handler_->write1ByteTxRx(port_handler_, id_, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error_);
		no_of_trials++;
	}while (dxl_comm_result_ != COMM_SUCCESS && no_of_trials<3);

	if (dxl_comm_result_ != COMM_SUCCESS){
		packet_handler_->getTxRxResult(dxl_comm_result_);
		std::runtime_error("[DynXMMotor] Cannot communicate with the dynamixel.");
	}
	else if (dxl_error_ != 0){
		packet_handler_->getRxPacketError(dxl_error_);
		std::runtime_error("[DynXMMotor] Error while setting dynamixel torque mode (enable).");
	}
}
void DynXMMotor::setDriveMode(int direction){
	int no_of_trials = 0;
	do{
		dxl_comm_result_ = packet_handler_->write1ByteTxRx(port_handler_, id_, ADDR_DRIVE_MODE, (int8_t)direction, &dxl_error_);
		no_of_trials++;
	}while (dxl_comm_result_ != COMM_SUCCESS && no_of_trials<3);

	if (dxl_comm_result_ != COMM_SUCCESS){
		packet_handler_->getTxRxResult(dxl_comm_result_);
		std::runtime_error("[DynXMMotor] Cannot communicate with the dynamixel.");
	}
	else if (dxl_error_ != 0){
		packet_handler_->getRxPacketError(dxl_error_);
		std::runtime_error("[DynXMMotor] Error while setting setting the direction (enable).");
	}
}

void DynXMMotor::setCurrentLimit(double limit){
	disable();
	int no_of_trials = 0;
	int16_t limit_int = limit*1193;
	do{
		dxl_comm_result_ = packet_handler_->write2ByteTxRx(port_handler_, id_, ADDR_CURRENT_LIMIT, (int16_t)limit_int, &dxl_error_);
		no_of_trials++;
	}while (dxl_comm_result_ != COMM_SUCCESS && no_of_trials<3);

	if (dxl_comm_result_ != COMM_SUCCESS){
		packet_handler_->getTxRxResult(dxl_comm_result_);
		std::runtime_error("[DynXMMotor] Cannot communicate with the dynamixel.");
	}
	else if (dxl_error_ != 0){
		packet_handler_->getRxPacketError(dxl_error_);
		std::runtime_error("[DynXMMotor] Error while setting setting the direction (enable).");
	}
	enable();
}

bool DynXMMotor::isMoving(){
	int no_of_trials = 0;
	int8_t move_bool;
	do{
		dxl_comm_result_ = packet_handler_->read1ByteTxRx(port_handler_, id_, ADDR_MOVING, (uint8_t*)&move_bool, &dxl_error_);
		no_of_trials++;
	}while (dxl_comm_result_ != COMM_SUCCESS && no_of_trials<3);

	if (dxl_comm_result_ != COMM_SUCCESS){
		packet_handler_->getTxRxResult(dxl_comm_result_);
		std::runtime_error("[DynXMMotor] Cannot communicate with the dynamixel.");
	}
	else if (dxl_error_ != 0){
		packet_handler_->getRxPacketError(dxl_error_);
		std::runtime_error("[DynXMMotor] Error while setting setting the direction (enable).");
	}
	if ((int)move_bool == 1)
		return true;
	else
		return false;
}

void DynXMMotor::disable(){
	int no_of_trials = 0;
	do{
		dxl_comm_result_ = packet_handler_->write1ByteTxRx(port_handler_, id_, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error_);
		no_of_trials++;
	}while (dxl_comm_result_ != COMM_SUCCESS && no_of_trials<3);

	if (dxl_comm_result_ != COMM_SUCCESS){
		packet_handler_->getTxRxResult(dxl_comm_result_);
		std::runtime_error("[DynXMMotor] Cannot communicate with the dynamixel.");
	}
	else if (dxl_error_ != 0){
		packet_handler_->getRxPacketError(dxl_error_);
		std::runtime_error("[DynXMMotor] Error while setting dynamixel torque mode (enable).");
	}
}

void DynXMMotor::moveTo(double goal_position){
	goal_position = goal_position*4095;
	if(getOperatingMode() != POSITION_CONTROL_MODE){
		printf("[DynXMMotor] Position control is initiated, but dynamixel is not in position control mode. Neglecting...");
		return;
	}
	
	int no_of_trials = 0;
	do{

		dxl_comm_result_ = packet_handler_->write4ByteTxRx(port_handler_, id_, ADDR_GOAL_POSITION, goal_position, &dxl_error_);
		no_of_trials++;
	}while (dxl_comm_result_ != COMM_SUCCESS && no_of_trials<3);

    if (dxl_comm_result_ != COMM_SUCCESS){
	packet_handler_->getTxRxResult(dxl_comm_result_);
	std::runtime_error("[DynXMMotor] Cannot communicate with the dynamixel.");
	}
    else if (dxl_error_ != 0){
		packet_handler_->getRxPacketError(dxl_error_);
		std::runtime_error("[DynXMMotor] Error while sending position reference.");
	}

}
void DynXMMotor::setVelocity(double velocity){
	if(getOperatingMode() != POSITION_CONTROL_MODE){
		printf("[DynXMMotor] Position control is initiated, but dynamixel is not in position control mode. Neglecting...");
		return;
	}
	int no_of_trials = 0;
	do{
		dxl_comm_result_ = packet_handler_->write4ByteTxRx(port_handler_, id_, ADDR_PROFILE_VELOCITY, velocity, &dxl_error_);
		no_of_trials++;
	}while (dxl_comm_result_ != COMM_SUCCESS && no_of_trials <3);

	if (dxl_comm_result_ != COMM_SUCCESS){
		packet_handler_->getTxRxResult(dxl_comm_result_);
		std::runtime_error("[DynXMMotor] Cannot communicate with the dynamixel.");
	}
	else if (dxl_error_ != 0){
		packet_handler_->getRxPacketError(dxl_error_);
		std::runtime_error("[DynXMMotor Error while setting velocity. ");
	}
}

void DynXMMotor::sendVelocity(double goal_velocity){
	if(getOperatingMode() != VELOCITY_CONTROL_MODE){
		printf("[DynXMMotor] Velocity control is initiated, but dynamixel is not in Velocity control mode. Neglecting...");
		return;
	}
	int no_of_trials = 0;
	do{
		dxl_comm_result_ = packet_handler_->write4ByteTxRx(port_handler_, id_, ADDR_GOAL_VELOCITY, goal_velocity, &dxl_error_);
		no_of_trials++;
	}while (dxl_comm_result_ != COMM_SUCCESS && no_of_trials<3);

	if (dxl_comm_result_ != COMM_SUCCESS){
	packet_handler_->getTxRxResult(dxl_comm_result_);
		std::runtime_error("[DynXMMotor] Cannot communicate with the dynamixel.");
	}
    	else if (dxl_error_ != 0){
		packet_handler_->getRxPacketError(dxl_error_);
		std::runtime_error("[DynXMMotor] Error while sending velocity reference.");
	}
		
} 
void DynXMMotor::applyTorque(double goal_torque){
	goal_torque = goal_torque * 1193;
	if(getOperatingMode() != CURRENT_CONTROL_MODE){
		printf("[DynXMMotor] Current control is initiated, but dynamixel is not in current control mode. Neglecting...");
		return;
	}

	int no_of_trials = 0;
	do{
		dxl_comm_result_ = packet_handler_->write2ByteTxRx(port_handler_, id_, ADDR_GOAL_CURRENT, goal_torque, &dxl_error_);
		no_of_trials++;
	}while (dxl_comm_result_ != COMM_SUCCESS && no_of_trials<3);

	if (dxl_comm_result_ != COMM_SUCCESS){
		packet_handler_->getTxRxResult(dxl_comm_result_);
		std::runtime_error("[DynXMMotor] Cannot communicate with the dynamixel.");
	}
	else if (dxl_error_ != 0){
		packet_handler_->getRxPacketError(dxl_error_);
		std::runtime_error("[DynXMMotor] Error while sending torque reference.");
	}
}

void DynXMMotor::setOperatingMode(int mode){
	disable();
	int no_of_trials = 0;
	do{
		dxl_comm_result_ = packet_handler_->write1ByteTxRx(port_handler_, id_, ADDR_OPERATING_MODE, mode, &dxl_error_);
		no_of_trials++;
	}while (dxl_comm_result_ != COMM_SUCCESS && no_of_trials<3);

	if (dxl_comm_result_ != COMM_SUCCESS){
		packet_handler_->getTxRxResult(dxl_comm_result_);
		std::runtime_error("[DynXMMotor] Cannot communicate with the dynamixel.");
	}
	else if (dxl_error_ != 0){
		packet_handler_->getRxPacketError(dxl_error_);
		std::runtime_error("[DynXMMotor] Error while setting operation mode.");
	}
//	resetOperationModeReference(mode);
	enable();
}

void DynXMMotor::resetOperationModeReference(int mode){
	if (mode == CURRENT_CONTROL_MODE)
		applyTorque(readCurrent());
	else if (mode == POSITION_CONTROL_MODE)
		moveTo(readPos());
	else if (mode == VELOCITY_CONTROL_MODE)
		moveTo(readVelocity()); 		//Velocity
}

void DynXMMotor::setToTorqueControlMode(){
	setOperatingMode(CURRENT_CONTROL_MODE);
}

void DynXMMotor::setToPositionControlMode(){
	setOperatingMode(POSITION_CONTROL_MODE);
}

void DynXMMotor::setToVelocityControlMode(){
	setOperatingMode(VELOCITY_CONTROL_MODE);
}

int DynXMMotor::getOperatingMode(){
	int8_t mode;
	int no_of_trials = 0;
	do{
		dxl_comm_result_ = packet_handler_->read1ByteTxRx(port_handler_, id_, ADDR_OPERATING_MODE, (uint8_t*)&mode, &dxl_error_);
		no_of_trials++;
	}while (dxl_comm_result_ != COMM_SUCCESS && no_of_trials<3);
	if (dxl_comm_result_ != COMM_SUCCESS){
		packet_handler_->getTxRxResult(dxl_comm_result_);
		std::runtime_error("[DynXMMotor] Cannot communicate with the dynamixel.");
	}
	else if (dxl_error_ != 0){
		packet_handler_->getRxPacketError(dxl_error_);
		std::runtime_error("[DynXMMotor] Error while reading operation mode.");
	}
	return (int)mode;
}

double DynXMMotor::readPos(){
	int32_t current_pos;
	int no_of_trials = 0;
	do{
		dxl_comm_result_ = packet_handler_->read4ByteTxRx(port_handler_, id_, ADDR_PRESENT_POSITION, (uint32_t*)&current_pos, &dxl_error_);
		no_of_trials++;
	}while (dxl_comm_result_ != COMM_SUCCESS && no_of_trials<3);
	if (dxl_comm_result_ != COMM_SUCCESS){
		packet_handler_->getTxRxResult(dxl_comm_result_);
		std::runtime_error("[DynXMMotor] Cannot communicate with the dynamixel.");
	}
	else if (dxl_error_ != 0){
		packet_handler_->getRxPacketError(dxl_error_);
		std::runtime_error("[DynXMMotor] Error while reading operation mode.");
	}
	return ((int)current_pos)/4095.0;
}

double DynXMMotor::readVelocity(){
	int32_t current_velocity;
	int no_of_trials = 0;
	do{
		dxl_comm_result_ = packet_handler_->read4ByteTxRx(port_handler_, id_, ADDR_PRESENT_VELOCITY, (uint32_t*)&current_velocity, &dxl_error_);
		no_of_trials++;
	}while (dxl_comm_result_ != COMM_SUCCESS && no_of_trials<3);

	if (dxl_comm_result_ != COMM_SUCCESS){
		packet_handler_->getTxRxResult(dxl_comm_result_);
		std::runtime_error("[DynXMMotor] Cannot communicate with the dynamixel.");
	}
	else if (dxl_error_ != 0){
		packet_handler_->getRxPacketError(dxl_error_);
		std::runtime_error("[DynXMMotor] Error while reading operation mode.");
	}
	//return ((int)current_velocity)/4095.0;
		
}
double DynXMMotor::readCurrent(){
	int16_t current_curr;
	int no_of_trials = 0;
	do{
		dxl_comm_result_ = packet_handler_->read2ByteTxRx(port_handler_, id_, ADDR_PRESENT_CURRENT, (uint16_t*)&current_curr, &dxl_error_);
		no_of_trials++;
	}while (dxl_comm_result_ != COMM_SUCCESS && no_of_trials<3);
	if (dxl_comm_result_ != COMM_SUCCESS){
		packet_handler_->getTxRxResult(dxl_comm_result_);
		std::runtime_error("[DynXMMotor] Cannot communicate with the dynamixel.");
	}
	else if (dxl_error_ != 0){
		packet_handler_->getRxPacketError(dxl_error_);
		std::runtime_error("[DynXMMotor] Error while reading operation mode.");
	}
	return ((int)current_curr)*0.001*2.69;
}
