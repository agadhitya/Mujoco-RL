#ifndef DYN_XM_MOTOR_HPP
#define DYN_XM_MOTOR_HPP


#include "dynamixel_motors/dyn_xm_control_table.h"
#include "dynamixel_motors/dynamixel_motor.hpp"
class DynXMMotor : public DynamixelMotor{
	int mode_ = 0;
	uint8_t dxl_error_ = 0; 
	int dxl_comm_result_ = COMM_TX_FAIL;

public:
	DynXMMotor();	
	DynXMMotor(int id, std::string port_name = "/dev/ttyUSB0", int drive_mode = 1, int operating_mode = 1, float protocol_version = 2.0, int baud_rate = 57600);
	void enable();
	void disable();
	bool isMoving();
	void moveTo(double goal_position);
	void sendVelocity(double goal_velocity);
	void setVelocity(double velocity);
	void applyTorque(double goal_torque);
	void setToTorqueControlMode();
	void setToPositionControlMode();
	void setToVelocityControlMode();
	void setOperatingMode(int mode);
	void setDriveMode(int direction);
	void setCurrentLimit(double limit);
	void resetOperationModeReference(int mode);
	double readCurrent();
	double readPos();
	double readVelocity();
	int getOperatingMode();
};
#endif
