#ifndef XM_CONTROL_TABLE_H
#define XM_CONTROL_TABLE_H

#define ADDR_MODEL_NUMBER 			0		// 2 bytes (R)
#define ADDR_MODEL_INFORMATION 		2		// 4 bytes (R)
#define ADDR_VERSION_OF_FIRMWARE 	6		// 1 byte (R)
#define ADDR_ID 					7		// 1 bytes (RW)
#define ADDR_BAUD_RATE				8 		// 1 byte (RW)
#define ADDR_RETURN_DELAY_TIME		9 		// 1 byte (RW)
#define ADDR_DRIVE_MODE				10 		// 1 byte (RW)	(direction of rotationm, takes 0 or 1)
#define ADDR_OPERATING_MODE			11 		// 1 byte (RW)
#define ADDR_SECONDARY_ID			12 		// 1 byte (RW) (for grouping dynamixels)
#define ADDR_PROTOCOL_VERSION		13 		// 1 byte (RW)
#define ADDR_HOMING_OFFSET			20 		// 4 bytes (RW)
#define ADDR_MOVING_THRESHOLD		24 		// 4 bytes (RW) (This value helps to determine whether the Dynamixel is in motion or not. When the absolute value of Present Velocity(128) is greater than the Moving Threshold(24), Moving(122) is set to ‘1’, otherwise it is cleared to ‘0’.)
#define ADDR_TEMPERATURE_LIMIT		31 		// 1 byte (RW)
#define ADDR_MAX_VOLTAGE_LIMIT		32 		// 2 bytes (RW)
#define ADDR_MIN_VOLTAGE_LIMIT		34 		// 2 bytes (RW)
#define ADDR_PWM_LIMIT				36 		// 2 bytes (RW)
#define ADDR_CURRENT_LIMIT			38 		// 2 bytes (RW)
#define ADDR_ACCELERATION_LIMIT		40 		// 4 bytes (RW)
#define ADDR_VELOCITY_LIMIT			10 		// 4 bytes (RW)
#define ADDR_MAX_POSITION_LIMIT		48 		// 4 bytes (RW)
#define ADDR_MIX_POSITION_LIMIT		52 		// 4 bytes (RW)
#define ADDR_SHUTDOWN				63 		// 1 byte (RW)
#define ADDR_TORQUE_ENABLE			64 		// 1 byte (RW)
#define ADDR_LED					65 		// 1 byte (RW)
#define ADDR_STATUS_RETURN_LEVEL	68		// 1 byte (RW)
#define ADDR_REGISTERED_INSTRUCTION	69 		// 1 byte (R)
#define ADDR_HARDWARE_ERROR_STATUS	70 		// 1 byte (R)
#define ADDR_VELOCITY_I_GAIN		76 		// 2 bytes (RW)
#define ADDR_VELOCITY_P_GAIN		78 		// 2 bytes (RW)
#define ADDR_POSITION_D_GAIN		80 		// 2 bytes (RW)
#define ADDR_POSITION_I_GAIN		82 		// 2 bytes (RW)
#define ADDR_POSITION_P_GAIN		84 		// 2 bytes (RW)
#define ADDR_FEEDFORWARD_2ND_GAIN	88		// 2 bytes (RW)
#define ADDR_FEEDFORWARD_1ST_GAIN	90 		// 2 bytes (RW)
#define ADDR_BUS_WATCHDOG			98 		// 1 byte (RW)
#define ADDR_GOAL_PWM				100		// 2 bytes (RW)
#define ADDR_GOAL_CURRENT			102		// 2 bytes (RW)
#define ADDR_GOAL_VELOCITY			104		// 4 bytes (RW)
#define ADDR_PROFILE_ACCELERATION	108		// 4 bytes (RW)
#define ADDR_PROFILE_VELOCITY		112		// 4 bytes (RW)
#define ADDR_GOAL_POSITION			116		// 4 bytes (RW)
#define ADDR_REALTIME_TICK			120		// 2 bytes (R)
#define ADDR_MOVING					122		// 1 byte (R)
#define ADDR_MOVING_STATUS			123		// 1 byte (R)
#define ADDR_PRESENT_PWM			124		// 2 bytes (RW)
#define ADDR_PRESENT_CURRENT		126		// 2 bytes (R)
#define ADDR_PRESENT_VELOCITY		128		// 4 bytes (R)
#define ADDR_PRESENT_POSITION		132		// 4 bytes (R)
#define ADDR_VELOCITY_TRAJECTORY	136		// 4 bytes (R)
#define ADDR_POSITION_TRAJECTORY	140		// 4 bytes (R)
#define ADDR_PRESENT_INPUT_VOLTAGE	144		// 2 bytes (R)
#define ADDR_PRESENT_TEMPERATURE	146		// 1 byte (R)

#define TORQUE_ENABLE							1                   // Value for enabling the torque
#define TORQUE_DISABLE							0                   // Value for disabling the torque
#define CURRENT_CONTROL_MODE					0
#define VELOCITY_CONTROL_MODE					1
#define POSITION_CONTROL_MODE					3
#define EXTENDED_POSITION_CONTROL_MODE			4
#define CURRENT_BASED_POSITION_CONTROL_MODE		5
#define PWM_CONTROL_MODE						16
#endif
