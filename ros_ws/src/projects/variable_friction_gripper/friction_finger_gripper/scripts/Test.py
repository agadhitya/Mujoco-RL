#!/usr/bin/env python

import rospy
from common_msgs_gl.srv import *
import numpy as np
import time


def set_velocity(num, velocity):
	rospy.wait_for_service("set_vel")
	try:
		client_set_profile_velocity = rospy.ServiceProxy('set_vel', SendDoubleArray)
		resp1 = client_set_profile_velocity([num, velocity])
		return 1
	except rospy.ServiceException, e:
		print "Setting Velocity Failed"

def set_actuator_modes(size, modes):
	rospy.wait_for_service("set_operating_mode")
	try:
		client_operating_mode = rospy.ServiceProxy('set_operating_mode', SendIntArray)
		resp1 = client_operating_mode(modes)
		return 1
	except rospy.ServiceException, e:
		print "Actuator modes service call failed"


def command_position(num,position):
	rospy.wait_for_service('cmd_pos_ind')
	try:
		client_position = rospy.ServiceProxy('cmd_pos_ind', SendDoubleArray)
		resp1 = client_position([num, position])
		return 1

	except rospy.ServiceException, e:
		print "Position Service call failed"

def command_velocity(num, velocity):
	rospy.wait_for_service('cmd_vel_ind')
	try:
		client_velocity = rospy.ServiceProxy('cmd_vel_ind', SendDoubleArray)
		resp1 = client_velocity([num, velocity])
		return 1
	except rospy.ServiceException, e:
		print "Velocity Service call failed"

def command_torque(num, torque):
	rospy.wait_for_service('cmd_torque_ind')
	try:
		client_torque = rospy.ServiceProxy('cmd_torque_ind', SendDoubleArray)
		resp1 = client_torque([num, torque])
		return 1

	except rospy.ServiceException, e:
		print "Torque Service Call Failed"

def set_friction_right(friction_surface):
	rospy.wait_for_service('Friction_surface_Left')
	try:
		client_right_surface = rospy.ServiceProxy('Friction_surface_Left', SendBool)
		resp1 = client_right_surface(friction_surface)
		return 1

	except rospy.ServiceException, e:
		print "Right Friction Surface failed"

def set_friction_left(friction_surface):
	rospy.wait_for_service('Friction_surface_Right')
	try:
		client_left_surface = rospy.ServiceProxy('Friction_surface_Right', SendBool)
		resp1 = client_left_surface(friction_surface)
		return 1

	except rospy.ServiceException, e:
		print "Left Friction Surface Failed"


def slide_left_down(p):
	modes = [3, 0]			# Set modes - Left -> Position, Right -> Torque (3 -> Position, 0 -> Torque)
	set_modes = set_actuator_modes(2, modes)

	set_friction_l = set_friction_right(0)
	set_friction_r = set_friction_left(1)

	send_v = set_velocity(0, 10)
	send_pos = command_position(0, p)

	send_torque = command_torque(1, 0.12)

	


def slide_left_up(p):
	modes = [0, 3]
	set_modes = set_actuator_modes(2, modes)

	set_friction_l = set_friction_right(0)
	set_friction_r = set_friction_left(1)

	send_v = set_velocity(1, 10)
	send_pos = command_position(1, p)

	send_torque = command_torque(0, 0.12)

	

def slide_right_down(p):
	modes = [0, 3]
	set_modes = set_actuator_modes(2, modes)

	set_friction_l = set_friction_right(1)
	set_friction_r = set_friction_left(0)
	
	send_v = set_velocity(1, 10)
	send_pos = command_position(1, p)
	
	send_torque = command_torque(0, 0.12)
	
	

def slide_right_up(p):
	modes = [3, 0]
	
	set_modes = set_actuator_modes(2, modes)
	
	set_friction_l = set_friction_right(1)
	set_friction_r = set_friction_left(0)

	send_v = set_velocity(0, 10)
	send_pos = command_position(0, p)

	send_torque = command_torque(1, 0.12)


def setV(v):
	modes = [1, 0]
	set_modes = set_actuator_modes(2, modes)
	send_vel = command_velocity(0, v)
	send_torque = command_torque(1, 0.025)
	print 123
	

if __name__ == '__main__':
	
	slide_left_up(0.4)
	# modes = [3, 3]
	# set_modes = set_actuator_modes(2, modes)
	# send_v = set_velocity(1, 10)
	# send_pos = command_position(1, 0.4)

	#setV(10)

	# start_time = time.time()
	# slide_right_up(0.78)
	# print time.time()-start_time
	'''set_friction_left(1)
	time.sleep(2)
	set_friction_left(0)
	time.sleep(2)
	
	set_modes = set_actuator_mode(2, [3,3])
	for i in np.arange(0.85,0.75,-0.01):
		send_pos = command_position(1, i)	
	'''
	