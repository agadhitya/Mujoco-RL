#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
import time
from friction_finger_gripper.srv import*
from common_msgs_gl.srv import *
from std_msgs.msg       import Float32

import roslib
roslib.load_manifest("rosparam")
import rosparam

paramlist  = rosparam.load_file('/home/joshua/ff_ws/src/friction_finger_gripper/config/ff_parameters.yaml')


# Parameters for converting angles to encoder values
for params, ns in paramlist:
	rosparam.upload_params(ns,params)
	method = params['method']
	start_x = params['start_x']
	start_y = params['start_y']
	end_x = params['end_x']
	end_y = params['end_y']
	theta_goal = params['theta_goal']
	Tolerance = params['Tolerance']
	hold_left = params['hold_left']
	hold_right = params['hold_right']


# class High_level:
# 	def __init__(self):
# 		p1 = None
# 		p2 = None
# 		self.pub = rospy.Publisher('state', Int32, queue_size=10000)
# 		self.Motor_value_pub = rospy.Publisher('state', Motor_position,queue_size=10000)
		
# 		#loop rate
# 		rate = rospy.Rate(20)

def Hold_object_1(p1, p2):

	p = [p1, p2]
	rospy.wait_for_service('Hold_object')
	try:
		client_hold = rospy.ServiceProxy('Hold_object', Holdcommand)
		resp1 = client_hold(p1, p2)
		return 1
	except rospy.ServiceException, e:
		return None

def Home_position(p1, p2):

	p = [p1, p2]
	rospy.wait_for_service('Home_pos')
	try:
		client_hold = rospy.ServiceProxy('Home_pos', Holdcommand)
		resp1 = client_hold(p1, p2)
		return 1
	except rospy.ServiceException, e:
		return None

def Visual_servoing():

	rospy.wait_for_service('Visual_servoing')
	try:
		vs = rospy.ServiceProxy('Visual_servoing', Visual_servo_goal)
		resp1 = vs(end_x*100, end_y*100, theta_goal, Tolerance)
		return 1
	except rospy.ServiceException, e:
		print "Calling Visual servoing failed"

def Motion_planner():

	rospy.wait_for_service('Motion_planner')
	try:
		mp = rospy.ServiceProxy('Motion_planner', SendBool)
		resp1 = mp(1)
		return 1
	except rospy.ServiceException, e:
		print "Calling Motion Planner"

def publish_state():
	global start
	pub = rospy.Publisher('state', Int32, queue_size = 2)
	rospy.init_node('state', anonymous = True)
	rate = rospy.Rate(10)
	pub.publish(start)
	if (start == 0):
		Hold_object_1(0.80, 0.52)
		start = 1
		time.sleep(1)
		pub.publish(start)

		if method == 1:
			print 'motion planner started'
			# Motion_planner()

		elif method == 2:
			print 'visual servoing started'
			v = Visual_servoing()
			if v== 1:
				print 'reached'
				start = 0

		elif method == 3:
			Motion_planner()
			Visual_servoing()

	

if __name__ == '__main__':
	global start 
	start = 0
	try:
		publish_state()
	except rospy.ROSInterruptException:
		pass


	
	start = 0

