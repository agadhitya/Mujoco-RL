from mpmath  import *
from sympy import *
import numpy as np
import rospy
from friction_finger_gripper.srv import*
import std_msgs.msg


from geometry_msgs.msg import Point
from common_msgs_gl.srv import *
from std_msgs.msg       import Float32


import roslib
roslib.load_manifest("rosparam")
import rosparam



a = 2
b = 3
wp = 6
global pos_x
global pos_y

pos_x = None
pos_y = None

#flag -0 (left)
#flag -1 (right)

paramlist  = rosparam.load_file('/home/joshua/ff_ws/src/friction_finger_gripper/config/ff_parameters.yaml')

for params, ns in paramlist:
    rosparam.upload_params(ns,params)
    a_left = params['a_left']
    b_left = params['b_left']
    a_right = params['a_right']
    b_right = params['b_right']


def angle_conversion(angle, flag):
    
    angle = 180. * angle / np.pi
    # 0-> Left, 1-> Right
    if(flag == 1):
        n_angle =  a_right*angle+ b_right
    else:
        n_angle = a_left*angle + b_left

    print 'motor value = ', n_angle
    return (n_angle)


def slide_left_finger_down(p):
	#Call Left_Slide_Down(till left most position) Assume t1 = pi/6
	rospy.wait_for_service('Slide_Left_Finger_Down')
	try:
		slide_left_down = rospy.ServiceProxy('Slide_Left_Finger_Down',PositionCommand)
		resp1 = slide_left_down(p)
	except rospy.ServiceException, e:
		print ("Service call failed: %s"%e)

def slide_left_finger_up(p):
	rospy.wait_for_service('Slide_Left_Finger_Up')
	try:
		slide_left_up = rospy.ServiceProxy('Slide_Left_Finger_Up',PositionCommand)
		resp1 = slide_left_up(p)
	except rospy.ServiceException, e:
		print ("Service call failed: %s"%e)

def slide_right_finger_down(p):
	#Call Left_Slide_Down(till left most position) Assume t1 = pi/6
	rospy.wait_for_service('Slide_Right_Finger_Down')
	try:
		slide_right_down = rospy.ServiceProxy('Slide_Right_Finger_Down',PositionCommand)
		resp1 = slide_right_down(p)
	except rospy.ServiceException, e:
		print ("Service call failed: %s"%e)

def slide_right_finger_up(p):
	rospy.wait_for_service('Slide_Right_Finger_Up')
	try:
		slide_right_up = rospy.ServiceProxy('Slide_Right_Finger_Up',PositionCommand)
		resp1 = slide_right_up(p)
	except rospy.ServiceException, e:
		print ("Service call failed: %s"%e)


def rotate_object_anticlockwise(p):
	rospy.wait_for_service('Rotate_anticlockwise')
	try:
		rotate_anti = rospy.ServiceProxy('Rotate_anticlockwise',PositionCommand)
		resp1 = rotate_anti(p)
	except rospy.ServiceException, e:
		print ("Service call failed: %s"%e)

def rotate_object_clockwise(p): #0.35
	rospy.wait_for_service('Rotate_clockwise')
	try:
		Rotate_clockwise = rospy.ServiceProxy('Rotate_clockwise',PositionCommand)
		resp1 = Rotate_clockwise(p)
	except rospy.ServiceException, e:
		print ("Service call failed: %s"%e)



def translationRight(t1, d1):
	x_coord = d1*cos(t1) + a/2*sin(t1)
	y_coord = d1*sin(t1) - a/2*cos(t1)

	R = np.array([[cos(t1), -sin(t1), x_coord], [sin(t1), cos(t1), y_coord], [0, 0, 1]])
	coords = np.dot(R, np.array([[a/2, a/2, -a/2, -a/2], [-b/2, b/2, b/2, -b/2], [1, 1, 1, 1]]))

	t2 = min(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))
	contact = np.argmin(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))
	d2 = sqrt((coords[0, contact] - wp)**2 + coords[1, contact]**2)

	return t2, d2

def translationLeft(t2, d2):
	x_coord = wp + d2*cos(t2) - a/2*sin(t2)
	y_coord = d2*sin(t2) + d2*sin(t2)

	R = np.array([[cos(t2), -sin(t2), x_coord], [sin(t2), cos(t2), y_coord], [0, 0, 1]])
	coords = np.dot(R, np.array([[a/2, a/2, -a/2, -a/2], [-b/2, b/2, b/2, -b/2], [1, 1, 1, 1]]))

	t1 = max(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))
	contact = np.argmax(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))
	d1 = sqrt((coords[0, contact])**2 + coords[1, contact]**2)

	return t1, d1

def ik_finger(x, y):
	t1, d1 = symbols('t1 d1')
	eqn1 = d1*cos(t1) + b*sin(t1)/2
	eqn2 = d1*sin(t1) - b*cos(t1)/2

	eqn3 = x**2 + y**2 - eqn1**2 - eqn2**2

	sold1 = solve(eqn3, d1)

	solt1 = solve(eqn1.subs(d1, sold1[1])-x, t1)
	
	R = np.array([[cos(solt1[1]), -sin(solt1[1]), x], [sin(solt1[1]), cos(solt1[1]), y], [0,0,1]])
	coords = np.dot(R, np.array([[a/2, a/2, -a/2, -a/2], [-b/2, b/2, b/2, -b/2], [1, 1, 1, 1]]))

	
	t2 = min(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))
	contact = np.argmin(np.arctan2(coords[1,:].ravel().astype(float), coords[0,:].ravel().astype(float)))
	d2 = sqrt((coords[0, contact] - wp)**2 + coords[1, contact]**2)

	return solt1[1], t2, sold1[1], d2

def finger_planning(x_desired, y_desired):
	print 'start'
	global pos_x, pos_y
	t1_d, t2_d, d1_d, d2_d = ik_finger(x_desired, y_desired)
	print pos_x, pos_y
	t1, t2, d1, d2 = ik_finger(pos_x, pos_y)


	if(d1_d > d1 and d2_d > d2):
		while ((d1_d - d1) > 0.001 or (d2_d - d2)> 0.001):
			while (d1 < d1_d and t1 > 45*pi/180):
				t2 = t2 - 0.01
				t1, d1 = translationLeft(t2, d2)
				print 'slide_right_finger_up', t1, d1
			slide_right_finger_up(angle_conversion(t1,0))

			while (d2 < d2_d and t2 < 135*pi/180):
				t1  = t1 + 0.01
				t2, d2 = translationRight(t1, d1)
				print 'slide_left_finger_up', d1
			slide_left_finger_up(angle_conversion(t2,1))
	else:	
		while ((d1_d - d1) < -0.001 or (d2_d - d2) < -0.001):
			while (d2 > d2_d and t2 < 135*pi/180):
				t1  = t1 - 0.01
				t2, d2 = translationRight(t1, d1)
				print 'slide_left_finger_up', d1
			slide_right_finger_down(angle_conversion(t2,1))

			while (d1 > d1_d and t1 > 45*pi/180):
				t2 = t2 - 0.01
				t1, d1 = translationLeft(t2, d2)
				print 'slide_left_finger_down', d2
			slide_left_finger_down(angle_conversion(t1,0))



def listener_object_pose():
    rospy.init_node('Position')
    rospy.Subscriber("/object_position", Point, callbackObjectPosition)
    #rospy.spin()


def callbackObjectPosition(msg):
	global pos_x, pos_y
   	pos_x = msg.x*100
	pos_y = msg.y*100


if __name__ == "__main__":

	x_desired = 2.5
	y_desired = 10
	listener_object_pose()
	finger_planning(x_desired, y_desired)
