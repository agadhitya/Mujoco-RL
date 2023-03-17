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
w0 = 1
fw = 1.8
# Vertices of the shape
X_vertices = np.array([0.9207, 0.9207, 0, -0.8201, -0.8201, 0])
Y_vertices = np.array([-0.92913, 0.920691, 1.319, 0.7472, -0.8617, -1.362])
n = X_vertices.shape[0]
print n
contact = 1

distances = np.array([])
x_d = np.array([])
y_d = np.array([])

gamma = np.array([])

for i in range(n-1):

	gamma = np.append(gamma, np.arctan2(Y_vertices[i+1]- Y_vertices[i], X_vertices[i+1]- X_vertices[i]))
	distances = np.append(distances, (X_vertices[i+1]*Y_vertices[i] - Y_vertices[i+1]*X_vertices[i])/sqrt((Y_vertices[i+1]-Y_vertices[i])**2 + (X_vertices[i+1]-X_vertices[i])**2))
	x_d = np.append(x_d, (Y_vertices[i]*X_vertices[i+1]-X_vertices[i]*Y_vertices[i+1])*(X_vertices[i+1]-X_vertices[i])*(Y_vertices[i+1]-Y_vertices[i])/((Y_vertices[i+1] - Y_vertices[i])**2 + (X_vertices[i+1]-X_vertices[i])**2))
	if(Y_vertices[i+1]-Y_vertices[i] != 0):
		y_d = np.append(y_d, (X_vertices[i+1] - X_vertices[i])*x_d[0]/(Y_vertices[i+1]-Y_vertices[i]))
	else:
		y_d = np.append(y_d, Y_vertices[i])

gamma = np.append(gamma, np.arctan2(Y_vertices[0]- Y_vertices[n-1], X_vertices[0]- X_vertices[n-1]))
distances = np.append(distances, (X_vertices[0]*Y_vertices[n-1] - Y_vertices[0]*X_vertices[n-1])/sqrt((Y_vertices[0]-Y_vertices[n-1])**2 + (X_vertices[0]-X_vertices[n-1])**2))
x_d = np.append(x_d, (Y_vertices[n-1]*X_vertices[0]-X_vertices[n-1]*Y_vertices[0])*(X_vertices[0]-X_vertices[n-1])*(Y_vertices[0]-Y_vertices[n-1])/((Y_vertices[0] - Y_vertices[n-1])**2 + (X_vertices[0]-X_vertices[n-1])**2))

if(Y_vertices[0]-Y_vertices[n-1] != 0):
	y_d = np.append(y_d, (X_vertices[0] - X_vertices[n-1])*x_d[0]/(Y_vertices[0]-Y_vertices[n-1]))
else:
	y_d = np.append(y_d, Y_vertices[0])

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

def set_velocity(num, velocity):
	rospy.wait_for_service("set_vel")
	try:
		client_set_profile_velocity = rospy.ServiceProxy('set_vel', SendDoubleArray)
		resp1 = client_set_profile_velocity([num, velocity])
		return 1
	except rospy.ServiceException, e:
		print "Setting Velocity Failed"


def set_friction_right(friction_surface):
	rospy.wait_for_service('Friction_surface_Right')
	try:
		client_right_surface = rospy.ServiceProxy('Friction_surface_Right', SendBool)
		resp1 = client_right_surface(friction_surface)
		return 1

	except rospy.ServiceException, e:
		return None
		# print "Right Friction Surface failed"

def set_friction_left(friction_surface):
	rospy.wait_for_service('Friction_surface_Left')
	try:
		client_left_surface = rospy.ServiceProxy('Friction_surface_Left', SendBool)
		resp1 = client_left_surface(friction_surface)
		return 1

	except rospy.ServiceException, e:
		return None
		# print "Left Friction Surface Failed"


def read_pos():
    rospy.wait_for_service('read_pos')
    try:
        read_position_handler = rospy.ServiceProxy('read_pos', GetDoubleArray)
        values = read_position_handler()
        #print values.data
        return values.data
    except rospy.ServiceException, e:
        print ("Service call failed: %s" % e)


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

		self.contact_right = 0
		self.contact_left = 0
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
			set_friction_l = set_friction_right(1)
			set_friction_r = set_friction_left(0)
			time.sleep(1)
			self.finger_state = 1
			send_v = set_velocity(0, 10)
		send_pos = command_position(0, p)
		send_torque = command_torque(1, 0.12)

	def slide_left_up(self, p):
		
		if self.finger_state != 2:
			modes = [0, 3]
			set_modes = set_actuator_modes(2, modes)
			set_friction_l = set_friction_right(1)
			set_friction_r = set_friction_left(0)
			time.sleep(1)
			self.finger_state = 2
			send_v = set_velocity(1, 10)
		send_pos = command_position(1, p)
		send_torque = command_torque(0, 0.12)
		

	def slide_right_down(self, p):
		
		if self.finger_state != 3:
			modes = [0, 3]
			set_modes = set_actuator_modes(2, modes)
			set_friction_l = set_friction_right(0)
			set_friction_r = set_friction_left(1)
			time.sleep(1)
			self.finger_state = 3
			send_v = set_velocity(1, 10)
		send_pos = command_position(1, p)
		send_torque = command_torque(0, 0.12)
		

	def slide_right_up(self, p):
		
		if self.finger_state != 4:
			modes = [3, 0]
			set_modes = set_actuator_modes(2, modes)
			set_friction_l = set_friction_right(0)
			set_friction_r = set_friction_left(1)
			time.sleep(1)
			self.finger_state = 4
			send_v = set_velocity(0, 10)
		send_pos = command_position(0, p)
		send_torque = command_torque(1, 0.12)
		    

    ################################# FORWARD KINEMATICS CALCULATION #####################################################
	def translateLeft(self):

		x_coord = wp + (self.d2 + norm(X_vertices[self.contact_right] - x_d[self.contact_right], Y_vertices[self.contact_right] - y_d[self.contact_right]))*np.cos(self.t2) - (distances[self.contact_right] +  fw)*np.sin(self.t2)
		y_coord = (self.d2 + norm(X_vertices[self.contact_right] - x_d[self.contact_right], Y_vertices[self.contact_right] - y_d[self.contact_right]))*np.sin(self.t2) + (distances[self.contact_right] + fw)*np.cos(self.t2)

		R = np.array([[cos(-gamma[self.contact_right] + self.t2), -sin(-gamma[self.contact_right] + self.t2), x_coord], [sin(-gamma[self.contact_right] + self.t2), cos(-gamma[self.contact_right] + self.t2), y_coord], [0, 0, 1]])
		coords = np.dot(R, np.array([X_vertices, Y_vertices, np.ones((1,n))]))

		alpha = max(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))
		self.contact_left = np.argmax(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))

		self.d1 = sqrt((coords[0, contact_left] - wp)**2 + coords[1, contact_left]**2 -fw**2)
		self.t1 = alpha + np.arctan2(fw, self.d1)


	def translateRight(self):

		# Coordinates of the center of the rectangle
		x_coord = (self.d1 + norm(X_vertices[self.contact_left]-x_d[self.contact_left], Y_vertices[self.contact_left]-y_d[self.contact_left]))*np.cos(self.t1) + (distances[self.contact_left] + fw)*np.sin(self.t1)
		y_coord = (self.d1 + norm(X_vertices[self.contact_left]-x_d[self.contact_left], Y_vertices[self.contact_left]-y_d[self.contact_left]))*np.sin(self.t1) - (distances[self.contact_left] + fw)*np.sin(self.t1)

		R = np.array([[cos(-alpha[self.contact_left] + self.t1), -sin(-alpha[self.contact_left] + self.t1), x_coord], [sin(-alpha[self.contact_left] + self.t1), cos(-alpha[self.contact_left] + self.t1), y_coord], [0, 0, 1]])
		coords = np.dot(R, np.array([X_vertices, Y_vertices, np.ones((1,n))]))

		alpha = min(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))
		self.contact_right = np.argmin((np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float))))
		
		self.d2 = sqrt((coords[0, self.contact_right] - wp)**2 + coords[1, self.contact_right]**2 - fw**2)
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
		eqn3 = (self.x - wp)**2 + self.y**2 - (d2_sol + norm(X_vertices[self.contact_right] - x_d[self.contact_right], Y_vertices[self.contact_right] - y_d[self.contact_right]))**2 - (fw + distances[self.contact_right])**2
		return (eqn3[0])

	def solve_d2(self):

		initial_guess_d2 = 5.8

		solution = opt.fsolve(self.left_equations_d, initial_guess_d2) 
		sold2 =  solution[0]

		self.d2 = sold2

	def left_equations_theta(self, variables):
		t2_sol = variables[0]
		self.solve_d2() 
		eqn1 = (self.d2 +  norm(X_vertices[self.contact_right] - x_d[self.contact_right], Y_vertices[self.contact_right] - y_d[self.contact_right])) * cos(t2_sol) - (fw + distances[self.contact_right]) * sin(t2_sol) - (self.x - wp)
		return eqn1

	def solve_t1_slide_left(self, variables, coords):
		solt2 = variables[0]
		eqn = coords[1]*cos(solt2) - coords[0]*sin(solt2) + fw
		return eqn

	def ik_leftFinger_n(self):

		for i in range(1,5):
			initial_guess_t1 = np.pi * i / 5
			solution = opt.fsolve(self.left_equations_theta, initial_guess_t1, full_output=True)
			if solution[2]==1 and solution[0] > 0 and solution[0] < np.pi:
				solt2 = solution[0]
				self.t2 = float(solt2)

		R = np.array([[cos(-gamma[self.contact_right] + self.t2), -sin(-gamma[self.contact_right] + self.t2), self.x], [sin(-gamma[self.contact_right] + self.t2), cos(-gamma[self.contact_right] + self.t2), self.y], [0, 0, 1]])
		coords = np.dot(R, np.concatenate(([X_vertices], [Y_vertices], np.ones((1,n)))))
		
		beta = -9999
		contact_left = -1

		for i in range(n):
			initial_guess_t1 = np.pi/2	
			solution = opt.fsolve(self.solve_t1_slide_left, initial_guess_t1, args = coords[:,i], full_output=True)
			if (solution[2]==1 and solution[0] > 0 and solution[0] < np.pi and solution[0] > beta):
				beta = solution[0]
				self.contact_left = i

		self.t1 = beta[0]
		self.d1 = sqrt((coords[0, self.contact_left] - wp)**2 + coords[1, self.contact_left]**2 -fw**2)
		print 'left ', self.d1, self.d2, self.t1, self.t2

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
		eqn1 = (self.d1 + norm(X_vertices[self.contact_left]-x_d[self.contact_left], Y_vertices[self.contact_left]-y_d[self.contact_left])) * cos(t1_sol) + (fw + distances[self.contact_left]) * sin(t1_sol) - self.x
		return eqn1
    
 	def solve_t2_slide_right(self, variables, coords):
		solt2 = variables[0]
		eqn = coords[1]*cos(solt2) - (coords[0] - wp)*sin(solt2) - fw
		return eqn

	def ik_rightFinger_n(self):
    
		for i in range(1,5):
			initial_guess_t1 = np.pi * i / 5
			solution = opt.fsolve(self.right_equations_theta, initial_guess_t1, full_output=True)
			if solution[2]==1 and solution[0] > 0 and solution[0] < np.pi:
				self.t1 = float(solution[0])
        
		
		R = np.array([[cos(-gamma[self.contact_left] + self.t1), -sin(-gamma[self.contact_left] + self.t1), self.x], [sin(-gamma[self.contact_left] + self.t1), cos(-gamma[self.contact_left] + self.t1), self.y], [0, 0, 1]])
		coords = np.dot(R, np.concatenate(([X_vertices], [Y_vertices], np.ones((1,n)))))


		beta = 99999
		self.contact_right = -1
		
		for i in range(n):
			initial_guess_t2 = np.pi/2
			solution = opt.fsolve(self.solve_t2_slide_right, initial_guess_t2, args = coords[:,i], full_output=True)
			if (solution[2]==1 and solution[0] > 0 and solution[0] < np.pi and solution[0] < beta):
				beta = solution[0]
				contact_right = i
		
		self.t2 = beta[0]	
		self.d2 = sqrt((coords[0, self.contact_right] - wp)**2 + coords[1, self.contact_right]**2 - fw**2)
		print 'right ', self.d1, self.d2, self.t1, self.t2


    ###################### Visual Servoing Control #######################
	def controller(self, goal_x,goal_y, goal_theta, TOLERANCE = 0.5):

		print "Visual Servoing Initializing"
		print "Tolerance = ", TOLERANCE

		# Time interval 
		dt_1 = 0.1
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
			# if norm(e) < TOLERANCE:
			# 	theta = read_pos()
				
			# 	if(self.finger_state == 1):
			# 		self.slide_left_down(theta[0])
			# 	elif(self.finger_state == 2):
			# 		self.slide_left_up(theta[1])
			# 	elif(self.finger_state == 3):
			# 		self.slide_right_down(theta[1])
			# 	else:
			# 		self.slide_right_up(theta[0])
				
			# 	print 'x, y = ', self.x, self.y
			# 	print 'reached'
			# 	return
				# Read the actuator angles and go to the position after it is reached
			# Estimate the expected error if object slides on right finger 
			self.ik_rightFinger_n()
			# self.ik_leftFinger_n()
			# print 'd1, d2, t1, t2', self.d1, self.d2, self.t1, self.t2

			J_right = np.matrix([[-(self.d1 + norm(X_vertices[self.contact_left]-x_d[self.contact_left], Y_vertices[self.contact_left]-y_d[self.contact_left])) * sin(self.t1) + (distances[self.contact_left] + fw) * cos(self.t1)], [(self.d1 + norm(X_vertices[self.contact_left]-x_d[self.contact_left], Y_vertices[self.contact_left]-y_d[self.contact_left])) * cos(self.t1) + (distances[self.contact_left] + fw) * sin(self.t1)]], dtype='float')
			dtheta_right = -2.5*np.linalg.pinv(J_right) *  (e.reshape(e.shape[0], 1))
			X_right = np.array([(self.d1 + norm(X_vertices[self.contact_left]-x_d[self.contact_left], Y_vertices[self.contact_left]-y_d[self.contact_left])) * cos(self.t1 + dtheta_right[0, 0]) + (distances[self.contact_left] + fw) * sin(self.t1 + dtheta_right[0, 0]), (self.d1 + norm(X_vertices[self.contact_left]-x_d[self.contact_left], Y_vertices[self.contact_left]-y_d[self.contact_left])) * sin(self.t1 + dtheta_right[0, 0]) - (distances[self.contact_left]+ fw) * cos(self.t1 + dtheta_right[0, 0])])
			e_right = norm(X_right - self.X_d)
			# Estimate the expected error if object slides on left finger
			# self.ik_leftFinger()
			self.ik_leftFinger_n()
			J_left = np.matrix([[-(self.d2 + norm(X_vertices[self.contact_right] - x_d[self.contact_right], Y_vertices[self.contact_right] - y_d[self.contact_right])) * sin(self.t2) - (distances[self.contact_right] + fw) * cos(self.t2)], [(self.d2 +norm(X_vertices[self.contact_right] - x_d[self.contact_right], Y_vertices[self.contact_right] - y_d[self.contact_right])) * cos(self.t2) + (distances[self.contact_right] + fw) * sin(self.t2)]], dtype='float')
			dtheta_left = -2.5*np.linalg.pinv(J_left) *  (e.reshape(e.shape[0], 1))
			X_left = np.array([wp + (self.d2 + norm(X_vertices[self.contact_right] - x_d[self.contact_right], Y_vertices[self.contact_right] - y_d[self.contact_right])) * np.cos(self.t2 + dtheta_left[0,0]) - (distances[self.contact_right] + fw) * sin(self.t2 + dtheta_left[0,0]), (self.d2 + norm(X_vertices[self.contact_right] - x_d[self.contact_right], Y_vertices[self.contact_right] - y_d[self.contact_right])) * sin(self.t2 + dtheta_left[0,0]) + (distances[self.contact_right] + fw) * cos(self.t2 + dtheta_left[0,0])])
			e_left = norm(X_left - self.X_d)


			print 'dtheta_left', dtheta_left, 'dtheta_right', dtheta_right
			
			####### SET LIMITS FOR DELTA THETAS ########
			# if(abs(dtheta_left[0,0])<0.001):
			# 	dtheta_left[0,0] = np.sign(dtheta_left[0,0])*0.005

			# if(abs(dtheta_left[0,0])>=0.01):
			# 	dtheta_left[0,0] = np.sign(dtheta_left[0,0])*0.01

			# if(abs(dtheta_right[0,0])<0.001):
			# 	dtheta_right[0,0] = np.sign(dtheta_left[0,0])*0.005

			# if(abs(dtheta_right[0,0])>0.75):
			# 	dtheta_right[0,0] = np.sign(dtheta_left[0,0])*0.01
			
			 
			# Executing the action which has the lowest estimated error
			if e_left < e_right:

				# self.t2 = self.t2 + dtheta_left[0, 0]
				# self.translateLeft()

				if dtheta_left[0, 0] > 0:
					if self.finger_state != 1:
						theta = read_pos()
						theta_ref = theta[0]
					theta_ref = theta_ref - Kp*dt_1*abs(dtheta_left[0,0])
					self.slide_left_down(theta_ref)
					# theta_ref = theta_ref - 0.005
					# self.slide_left_down(theta_ref)
					# print 'action = slide left down, theta_ref = ', theta_ref
					# self.action = 6
					# self.slide_left_down(angle_conversion(self.t1,0))
					# theta= read_pos()
					# self.slide_left_down(theta[0]-0.01)
					# self.slide_left_down(angle_conversion(self.t1,0))
					# self.action = 6
					# self.pub.publish(self.action)
                    
                   
				else:
					if self.finger_state != 2:
						theta = read_pos()
						theta_ref = theta[1]
					theta_ref = theta_ref - Kp*dt_1*abs(dtheta_left[0,0])
					self.slide_left_up(theta_ref)
					# theta_ref = theta_ref - 0.005
					# self.slide_left_up(theta_ref)
					# print 'action = slide left up, theta_ref = ', theta_ref
					# self.action = 8
					# theta= read_pos()
					# self.slide_left_up(theta[1]-0.01)
					# self.slide_left_up(angle_conversion(self.t2,1))
					# self.action = 8
					# self.pub.publish(self.action)
                    
			else: 

				# self.t1 = self.t1 + dtheta_right[0, 0]
				# self.translateRight()     

				if dtheta_right[0, 0] < 0:
					if self.finger_state != 3:
						theta = read_pos()
						theta_ref = theta[1]
					theta_ref = theta_ref - Kp*dt_1*abs(dtheta_right[0,0])
					self.slide_right_down(theta_ref)
					# theta_ref = theta_ref - 0.005
					# self.slide_right_down(theta_ref)
					# print 'action = slide right down, theta_ref = ', theta_ref
					# self.action = 7
					# theta = read_pos()
					# self.slide_right_down(theta[1]-0.01)
					# self.slide_right_down(angle_conversion(self.t2,1))
					# self.action = 7
					# self.pub.publish(self.action)
                    
				else:
					if self.finger_state != 4:
						theta = read_pos()
						theta_ref = theta[0]    
					theta_ref = theta_ref - Kp*dt_1*abs(dtheta_right[0,0])
					self.slide_right_up(theta_ref)
					# theta_ref = theta_ref - 0.005
					# self.slide_right_up(theta_ref)
					# print 'action = slide right up, theta_ref = ', theta_ref
					# self.action = 9
					# theta = read_pos()
					# self.slide_right_up(theta[0]-0.01)
					# self.slide_right_up(angle_conversion(self.t1,0))    
					# self.action = 9
					# self.pub.publish(self.action)
                    
                   
			X = np.array([self.x, self.y])
			e = X - self.X_d
			# print 'total time', time.time() - start_time_action
		# theta = read_pos()
		# self.stop_action()

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
   
