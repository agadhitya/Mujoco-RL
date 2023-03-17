#ifndef DYNAMIXEL_MOTOR_HPP
#define DYNAMIXEL_MOTOR_HPP

#include <string>
#include "dynamixel_sdk/dynamixel_sdk.h"                                  // Uses Dynamixel SDK library

class DynamixelMotor{
protected:
	dynamixel::PortHandler *port_handler_;
	dynamixel::PacketHandler *packet_handler_;
	int mode_ = 0;
	int id_ = 1;
	DynamixelMotor();
	~DynamixelMotor();
public:
	void initialize(int id = 1, std::string port_name = "/dev/ttyUSB0",int drive_mode = 1, int operating_mode = 1, float protocol_version = 2.0, int baud_rate = 57600);
	virtual void setDriveMode(int direction) = 0;
	virtual void enable() = 0;
	virtual void disable() = 0;
	virtual void moveTo(double goal_position) = 0;
	virtual void sendVelocity(double goal_velocity) = 0;
	virtual void applyTorque(double goal_torque) = 0;
	virtual void setToTorqueControlMode() = 0;
	virtual void setToPositionControlMode() = 0;
	virtual void setToVelocityControlMode() = 0;
	virtual void setOperatingMode(int mode) = 0;
	virtual void setCurrentLimit(double limit) = 0;
	virtual void setVelocity(double velocity) = 0;
	virtual double readPos() = 0;
	virtual double readCurrent() = 0;
	virtual double readVelocity() = 0;
	virtual int getOperatingMode() = 0;
	virtual bool isMoving() = 0;
};
#endif
