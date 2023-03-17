#!/usr/bin/env python

import numpy as np
from mpmath import *
from sympy import *
import scipy.optimize as opt
import rospy
from std_msgs.msg import Int32
import time
from friction_finger_gripper.srv import*
# from visualservoing.srv import *

from geometry_msgs.msg import Point
from common_msgs_gl.srv import *
from std_msgs.msg       import Float32
from common_msgs_gl.msg import Motor_position

import roslib
roslib.load_manifest("rosparam")
import rosparam


TOLERANCE=0.3
MAX_ITERATIONS=40
JACOBIAN_STEPS=1

# Geometric Parameters
wp = 6.2
w0 = 2.5
fw = 1.8

paramlist  = rosparam.load_file('/home/joshua/ff_ws/src/friction_finger_gripper/config/ff_parameters.yaml')


# Parameters for converting angles to encoder values
for params, ns in paramlist:
	rosparam.upload_params(ns,params)
	a_left = params['a_left']
	b_left = params['b_left']
	a_right = params['a_right']
	b_right = params['b_right']



######################################### FUNCTIONS TO SET ACTUATOR CONTROL MODES AND FRICTION SURFACES #####################################################
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


############################################## ANGLES, ENCODER VALUES CONVERSION ################################################
def angle_conversion(angle, flag):

	angle = 180. * angle / np.pi

	# 0-> Left, 1-> Right
	if(flag == 1):
		n_angle =  a_right*angle+ b_right
	else:
		n_angle = a_left*angle + b_left

	print 'motor value = ', n_angle
	return (n_angle)

def encoder_gripper_angle_conversion(enc,flag):

	# 0-> Left, 1-> Right
	if(flag==1):
		theta= (enc - b_right)/a_right
	else:
		theta= (enc - b_left)/a_left
	return theta


############################################## VISUAL SERVOING CLASS ##########################################################
class visual_servoing:

	def __init__(self):
		rospy.init_node('Visual_Servoing')

		# Friction configuration of the finger
		self.finger_state = None		# 1-> Slide left Down, 2-> Slide Left Up, 3-> Slide Right Down, 4 -> Slide Right Up, 5-> rotate

		# Desired Coordinates
		self.X_d = np.array([0,0])
		self.theta_d=0

		self.x = None
		self.y = None
		self.t1 = None
		self.t2 = None
		self.d1 = None
		self.d2 = None
		self.action = None

		self.Block_orientation = None
		'''
		#publishers for recording the data
		self.pub = rospy.Publisher('Action', Int32, queue_size=10000)
		self.Motor_value_pub=rospy.Publisher('Finger_motor_position',Motor_position,queue_size=10000)
		
		#loop rate
		rate = rospy.Rate(20)
		'''

	def goal_update(self, goal_x,goal_y,goal_theta):
		self.X_d = np.array([goal_x, goal_y])
		self.theta_d = goal_theta


	################################### OBJECT POSITION AND ORIENTATION ##############################################
	def callbackObjectPosition(self, msg):
		self.x = msg.x * 100.
		self.y = msg.y * 100.
		#print 'position =', self.x, self.y
		'''
		values=Motor_position()
		val=read_pos()
		values.Left=val[0]
		values.Right=val[1]
		self.Motor_value_pub.publish(values)
		'''

	def callbackObjectOrientation(self, msg):
		#Radians to degree conversion(Values lies between -180 to 180)
		if(msg.data>=0 and msg.data<=3.14):
			self.Block_orientation=msg.data*180/np.pi
		else:
			self.Block_orientation=180+ abs(msg.data)*180/np.pi


	def listener_object_pose(self):
		rospy.Subscriber("/object_position", Point, self.callbackObjectPosition)
		rospy.Subscriber("/object_orientation", Float32, self.callbackObjectOrientation)


	################################# SLIDING AND ROTATION ACTIONS #######################################################

	def slide_left_down(self, p):
		
		if self.finger_state != 1:
			modes = [3, 0]			# Set modes - Left -> Position, Right -> Torque (3 -> Position, 0 -> Torque)
			set_modes = set_actuator_modes(2, modes)
			set_friction_l = set_friction_right(0)
			set_friction_r = set_friction_left(1)
			time.sleep(1)
			self.finger_state = 1

		send_torque = command_torque(1, 0.15)
		send_pos = command_position(0, p)


	def slide_left_up(self, p):
		
		if self.finger_state != 2:
			modes = [0, 3]
			set_modes = set_actuator_modes(2, modes)
			set_friction_l = set_friction_right(0)
			set_friction_r = set_friction_left(1)
			time.sleep(1)
			self.finger_state = 1

		send_torque = command_torque(0, 0.15)
		send_pos = command_position(1, p)

	def slide_right_down(self, p):
		
		if self.finger_state != 3:
			modes = [0, 3]
			set_modes = set_actuator_modes(2, modes)
			set_friction_l = set_friction_right(1)
			set_friction_r = set_friction_left(0)
			time.sleep(1)
			self.finger_state = 2

		send_torque = command_torque(0, 0.15)
		send_pos = command_position(1, p)

	def slide_right_up(self, p):
		
		if self.finger_state != 4:
			modes = [3, 0]
			set_modes = set_actuator_modes(2, modes)
			set_friction_l = set_friction_right(1)
			set_friction_r = set_friction_left(0)
			time.sleep(1)
			self.finger_state = 2

		send_torque = command_torque(1, 0.15)
		send_pos = command_position(0, p)    

    ################################# FORWARD KINEMATICS CALCULATION #####################################################
	def translateLeft(self):
        # Center Coordinates
		x_square = wp + (self.d2 + a / 2.) * np.cos(np.float64(self.t2)) - (fw + b / 2.) * np.sin(np.float64(self.t2))
		y_square = (self.d2 + a / 2.) * np.sin(np.float64(self.t2)) + (fw + b / 2.) * np.cos(np.float64(self.t2))

		R = np.array([[cos(self.t2), -sin(self.t2), x_coord], [sin(self.t2), cos(self.t2), y_coord], [0,0,1]])
		coords = np.dot(R, np.array([[a/2, a/2, -a/2, -a/2], [-b/2, b/2, b/2, -b/2],[1,1,1,1]]))

		PoC = np.argmax(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))	# Point of contact of the object with the finger
		alpha = max(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))

		self.d1 = sqrt((coords[0, contact])**2 + coords[1, contact]**2 - fw**2)
		self.t1 = alpha + np.arctan2(fw, self.d1)

		 
	def translateRight(self):
		# Center Coordinates of square
		x_square = (self.d1 + a / 2.) * np.cos(self.t1) + (b / 2. + fw) * np.sin(self.t1)
		y_square = (self.d1 + a / 2.) * np.sin(self.t1) - (b / 2. + fw) * np.cos(self.t1)
		

		R = np.array([[cos(self.t1), -sin(self.t1), x_coord], [sin(self.t1), cos(self.t1), y_coord], [0, 0, 1]])
		coords = np.dot(R, np.array([[a/2, a/2, -a/2, -a/2], [-b/2, b/2, b/2, -b/2], [1, 1, 1, 1]]))

		PoC = np.argmin(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))
		alpha = min(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))
		self.d2 = sqrt((coords[0, PoC]-wp)**2 + coords[1, PoC]**2 - fw**2)

		self.t2 = alpha - np.arctan2(fw, self.d2)


	'''

    #################### Solving using Inverse kinematics using Symbolic Variables ####################################
    
    def ik_leftFinger(self):
        t2_sol, d2_sol = symbols('t2_sol d2_sol')
        eqn1 = (d2_sol + self.w0 / 2.) * cos(t2_sol) - (self.fw + self.w0 / 2.) * sin(t2_sol)
        eqn2 = (d2_sol + self.w0 / 2.) * sin(t2_sol) + (self.fw + self.w0 / 2.) * cos(t2_sol)
        eqn3 = (self.x - self.wp)**2 + self.y**2 - eqn1**2 - eqn2**2
        sold2 = solve(eqn3, d2_sol)
        solt2 = solve(eqn1.subs(d2_sol, sold2[1]) - (self.x - self.wp), t2_sol)
        print 'd2, t2 = ', sold2[1], solt2[1]
        d2v = np.array([sold2[1] * cos(solt2[1]), sold2[1] * sin(solt2[1])])
        w0v = np.array([self.w0 * sin(solt2[1]), -self.w0 * cos(solt2[1])])
        wpv = np.array([self.wp, 0.])
        f1v = np.array([self.fw * sin(solt2[1]), -self.fw * cos(solt2[1])])
        av = d2v - f1v - w0v + wpv

        self.d1 = sqrt(float((av * av).sum() - self.fw * self.fw))
        self.t1 = np.arctan2(float(av[1]), float(av[0])) + np.arctan2(float(self.fw), float(self.d1))
        self.d2 = float(sold2[1])
        self.t2 = float(solt2[1])


    def ik_rightFinger(self):
        t1_sol, d1_sol = symbols('t1_sol d1_sol')
        eqn1 = (d1_sol + self.w0 / 2.) * cos(t1_sol) + (self.fw + self.w0 / 2.) * sin(t1_sol)
        eqn2 = (d1_sol + self.w0 / 2.) * sin(t1_sol) - (self.fw + self.w0 / 2.) * cos(t1_sol)
        eqn3 = self.x**2 + self.y**2 - eqn1**2 - eqn2**2
        sold1 = solve(eqn3, d1_sol)
        solt1 = solve(eqn1.subs(d1_sol, sold1[1]) - self.x, t1_sol)
        print 't2, d2 = ', sold1[1], solt1[1]
        d1v = np.array([sold1[1] * cos(solt1[1]), sold1[1] * sin(solt1[1])])
        w0v = np.array([self.w0 * sin(solt1[1]), -self.w0 * cos(solt1[1])])
        wpv = np.array([self.wp, 0.])
        f1v = np.array([self.fw * sin(solt1[1]), -self.fw * cos(solt1[1])])
        av = d1v + f1v + w0v - wpv
        self.t1 = float(solt1[1])
        self.d1 = float(sold1[1])
        self.d2 = sqrt((av * av).sum() - self.fw * self.fw)
        self.t2 = np.arctan2(float(av[1]), float(av[0])) - np.arctan2(float(self.fw), float(self.d2))

	'''
    ####################### Solving Inverse Kinematics Numerically ####################################


	##### INVERSE KINEMATICS FOR SLIDING ALONG THE RIGHT FINGER #####
	def left_equations_d(self, variables):
		d2_sol = variables
		eqn3 = (self.x - wp)**2 + self.y**2 - (d2_sol + w0 / 2.)**2 - (fw + w0 / 2.)**2
		return (eqn3[0])


	def solve_d2(self):

		initial_guess_d2 = 5.8

		solution = opt.fsolve(self.left_equations_d, initial_guess_d2) 
		sold2 =  solution[0]

		self.d2 = sold2


	def solve_left(self, variables):
		t2_sol = variables[0]
		self.solve_d2() 
		eqn1 = (self.d2 + w0 / 2.) * cos(t2_sol) - (fw + w0 / 2.) * sin(t2_sol) - (self.x - wp)
		return eqn1


	def ik_leftFinger_n(self):

		initial_guess_t1 = np.pi/3
		solution_t2 = opt.fsolve(self.solve_left, initial_guess_t1)

		for i in range(1,5):
			initial_guess_t1 = np.pi * i / 5
			solution = opt.fsolve(self.solve_left, initial_guess_t1, full_output=True)
			if solution[2]==1 and solution[0] > 0 and solution[0] < np.pi:
				solt2 = solution_t2[0]
				self.t2 = float(solt2)

		R = np.array([[cos(self.t2), -sin(self.t2), self.x], [sin(self.t2), cos(self.t2), self.y], [0,0,1]])
		coords = np.dot(R, np.array([[a/2, a/2, -a/2, -a/2], [-b/2, b/2, b/2, -b/2],[1,1,1,1]]))

		PoC = np.argmax(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))	# Point of contact of the object with the finger
		alpha = max(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))

		self.d1 = sqrt((coords[0, contact])**2 + coords[1, contact]**2 - fw**2)
		self.t1 = alpha + np.arctan2(fw, self.d1)
	


    ##### INVERSE KINEMATICS FOR SLIDING ALONG THE RIGHT FINGER #####
	def right_equations_d(self, variables):

		d1_sol = variables
        
		eqn3 = self.x**2 + self.y**2 - (d1_sol + w0 / 2.)**2 - (fw + w0 / 2.)**2
		return (eqn3[0])

    
	def solve_d1(self):
        
		initial_guess_d1 = 5.8
        
		solution = opt.fsolve(self.right_equations_d, initial_guess_d1, xtol= 1e-5)
		sold1 =  solution[0]

		self.d1 = sold1


	def right_equations_theta(self, variables):
        
		t1_sol = variables[0]
        
		self.solve_d1()
		eqn1 = (self.d1 + w0 / 2.) * cos(t1_sol) + (fw + w0 / 2.) * sin(t1_sol)
		return eqn1
    
	def ik_rightFinger_n(self):
    
		initial_guess_t1 = np.pi/3
		solution_t1 = opt.fsolve(self.right_equations_theta, initial_guess_t1, xtol= 1e-5)
        
		for i in range(1,5):
			initial_guess_t1 = np.pi * i / 5
			solution = opt.fsolve(self.solve_left, initial_guess_t1, full_output=True)
			if solution[2]==1 and solution[0] > 0 and solution[0] < np.pi:
				self.t1 = float(solution_t1[0])


		R = np.array([[cos(self.t1), -sin(self.t1), x_coord], [sin(self.t1), cos(self.t1), y_coord], [0, 0, 1]])
		coords = np.dot(R, np.array([[a/2, a/2, -a/2, -a/2], [-b/2, b/2, b/2, -b/2], [1, 1, 1, 1]]))

		PoC = np.argmin(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))
		alpha = min(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))
		self.d2 = sqrt((coords[0, PoC]-wp)**2 + coords[1, PoC]**2 - fw**2)

		self.t2 = alpha - np.arctan2(fw, self.d2)


	
	###################### Visual Servoing Control #######################
	def controller(self, goal_x,goal_y, goal_theta, TOLERANCE = 0.5):

		print "Visual Servoing Initializing"
		print "Tolerance = ", TOLERANCE

		# Time interval 
		dt_1 = 0.75
		Kp = 0.25
		pos = np.zeros((2,4))
		errors = np.zeros(4)
		self.goal_update(goal_x,goal_y,goal_theta)
		time.sleep(1)
		self.listener_object_pose()
        
		time.sleep(1)
		#self.ik_rightFinger()
		if (self.theta_d== 90):
			self.anticlockwise()
		if (self.theta_d == -90):
			self.clockwise()
		if (self.theta_d == 180):
			self.anticlockwise()
			self.anticlockwise()
		X = np.array([self.x, self.y])
		e = self.X_d - X
		i = 0
		print "Visual Servoing started"

		while norm(e) > TOLERANCE:
			start_time_action = time.time()

			X = np.array([self.x, self.y])
			e = X - self.X_d

			# Check if object has reached the desired position
			if norm(e) < TOLERANCE:
				print 'x, y = ', self.x, self.y
				print 'reached'
				return

			# Estimate the expected error if object slides on right finger 
			#self.ik_rightFinger()
			self.ik_rightFinger_n()
			J_right = np.matrix([[-(self.d1 + w0 / 2.0) * sin(self.t1) + (w0 / 2.0 + fw) * cos(self.t1)], [(self.d1 + w0 / 2.0) * cos(self.t1) + (w0 / 2.0 + fw) * sin(self.t1)]], dtype='float')
			dtheta_right = -np.linalg.pinv(J_right) *  (e.reshape(e.shape[0], 1))
			X_right = np.array([(self.d1 + w0 / 2.0) * cos(self.t1 + dtheta_right[0, 0]) + (w0 / 2.0 + fw) * sin(self.t1 + dtheta_right[0, 0]), (self.d1 + w0 / 2.0) * sin(self.t1 + dtheta_right[0, 0]) - (w0 / 2.0 + fw) * cos(self.t1 + dtheta_right[0, 0])])
			e_right = norm(X_right - self.X_d)

			# Estimate the expected error if object slides on left finger
			#self.ik_leftFinger()
			J_left = np.matrix([[-(self.d2 + w0 / 2.0) * sin(self.t2) - (w0 / 2.0 + fw) * cos(self.t2)], [(self.d2 + w0 / 2.0) * cos(self.t2) + (w0 / 2.0 + fw) * sin(self.t2)]], dtype='float')
			dtheta_left = -np.linalg.pinv(J_left) *  (e.reshape(e.shape[0], 1))
			X_left = np.array([wp + (self.d2 + w0 / 2.0) * np.cos(self.t2 + dtheta_left[0,0]) - (w0 / 2.0 + fw) * sin(self.t2 + dtheta_left[0,0]), (self.d2 + w0 / 2.0) * sin(self.t2 + dtheta_left[0,0]) + (w0 + fw) * cos(self.t2 + dtheta_left[0,0])])
			e_left = norm(X_left - self.X_d)

			if(dtheta_left[0,0]<0.2):
				dtheta_left[0,0] = 0.2

			elif(dtheta_left[0,0]>=0.75):
				dtheta_left[0,0] = 0.75

			if(dtheta_right[0,0]<0.2):
				dtheta_right[0,0] = 0.2

			elif(dtheta_right[0,0]>0.75):
				dtheta_right[0,0] = 0.75


			# Executing the action which has the lowest estimated error
			
			if e_left < e_right:

				self.t2 = self.t2 + Kp*dtheta_left[0, 0]
				self.translateLeft()

				if dtheta_left[0, 0] > 0:
					self.slide_left_down(angle_conversion(self.t1,0))
					# self.action = 6
					# self.pub.publish(self.action)
                    
                   
				else:
					self.slide_left_up(angle_conversion(self.t2,1))
					# self.action = 8
					# self.pub.publish(self.action)
                    
			else: 

				self.t1 = self.t1 + Kp*dtheta_right[0, 0]
				self.translateRight()     

				if dtheta_right[0, 0] < 0:
					self.slide_right_down(angle_conversion(self.t2,1))
					# self.action = 7
					# self.pub.publish(self.action)
                    
				else:
					self.slide_right_up(angle_conversion(self.t1,0))    
					# self.action = 9
					# self.pub.publish(self.action)
                    
                    

			


			X = np.array([self.x, self.y])
			e = X - self.X_d
			print 'time for executing action', time.time() - start_time_action


def Visual_Servoing(req):
	print '***************'
	v = visual_servoing()
	if req.goal_theta == 0:
		v.controller(req.goal_x,req.goal_y,req.goal_theta,req.Tol)

	if req.goal_theta == 180:
		# Find Cartesian coordinates of the extreme position 
		d1 = 7
		t1 = 0.87
		x_square = wp + (d1 + w0 / 2.) * np.cos(t1) - (w0 / 2. + fw) * np.sin(t1)
		y_square = (d1 + w0 / 2.) * np.sin(t1) + (w0 / 2. + fw) * np.cos(t1)

		v.controller(x_square,y_square,0, 1.5)
		v.controller(x_square,y_square, 90, 200)
		v.controller(x_square,y_square,0, 1.5)
		v.controller(req.goal_x,req.goal_y,90,req.Tol)

	if req.goal_theta == 90:
		# Find Cartesian coordinates of the extreme position
		d1 = 7
		t1 = 0.87
		x_square = wp + (d1 + w0 / 2.) * np.cos(t1) - (w0 / 2. + fw) * np.sin(t1)
		y_square = (d1 + w0 / 2.) * np.sin(t1) + (w0 / 2. + fw) * np.cos(t1)


		v.controller(x_square,y_square,0, 1.5)
		v.controller(req.goal_x,req.goal_y,req.goal_theta,req.Tol)

	if req.goal_theta == -90:
		# Find Cartesian coordinates of the extreme position
		d2 = 7
		t2 = 2.267
		x_square = (d2 + w0 / 2.) * np.cos(np.float64(t2)) + (fw + w0 / 2.) * np.sin(np.float64(t2))
		y_square = (d2 + w0 / 2.) * np.sin(np.float64(t2)) - (fw + w0 / 2.) * np.cos(np.float64(t2))

		v.controller(x_square,y_square,0, 1.5)
		v.controller(req.goal_x,req.goal_y,req.goal_theta,req.Tol)
	return 1


def Visual_Sevoing_server():
	rospy.init_node('Visual_Servoing')
	vs=rospy.Service('Visual_servoing',Visual_servo_goal,Visual_Servoing)
	rospy.spin()


if __name__ == '__main__':
	Visual_Sevoing_server()
   
