
from mpmath  import *
from sympy import *
import numpy as np
import rospy
from friction_finger_gripper.srv import*
import std_msgs.msg


#flag -0 (left)
#flag -1 (right)
def angle_conversion(angle,flag): 
	if(flag==1):
		n_angle= (angle*180/3.14-21)*0.003246 +0.5626
		print("Gripper_angle=",angle*180/3.14)
		
	else:
		n_angle = (angle*180/3.14-152)*(-0.002113)+0.3794
		print("Gripper_angle=",angle*180/3.14)
	print("n_angle",n_angle)
	return (n_angle)
		
def ik_finger(x, y, w0, wp, fw):
	t2, d2 = symbols('t2 d2')
	eqn1 = (d2 - w0/2)*cos(t2) + (fw + w0/2)*sin(t2)
	eqn2 = (d2 - w0/2)*sin(t2) - (fw + w0/2)*cos(t2)
	eqn3 = x**2 + y**2 - eqn1**2 - eqn2**2
	sold2 = solve(eqn3 , d2)
	solt2 = solve(eqn1.subs(d2,sold2[1])-x,t2)
	
	d2v = np.array([sold2[1]* cos(solt2[1]), sold2[1]* sin(solt2[1])] )
	w0v = np.array([w0* sin(solt2[1]), -w0* cos(solt2[1])])
	wpv = np.array([wp, 0])
	f1v = np.array([fw* sin(solt2[1]), -fw* cos(solt2[1])])
	av = d2v + f1v+ w0v - wpv

	d1 = sqrt((av*av).sum() - fw*fw)
	t1 = np.arctan2(float(av[1]), float(av[0])) - np.arctan2(float(fw), float(d1))
	return t1, solt2[1], d1, sold2[1]

def finger_translation1(w0, wp, fw, t2, d2):
	# Center Coordinates 
	x_square = (d2 - w0/2)*np.cos(np.float64(t2)) + (fw + w0/2)*np.sin(np.float64(t2))
	y_square = (d2 - w0/2)*np.sin(np.float64(t2)) - (fw + w0/2)*np.cos(np.float64(t2))

	# Calculate theta2, d2
	d2v = np.array([d2* np.cos(np.float64(t2)), d2* np.sin(np.float64(t2))] )
	w0v = np.array([w0* np.sin(np.float64(t2)), -w0* np.cos(np.float64(t2))])
	wpv = np.array([wp, 0])
	f1v = np.array([fw* np.sin(np.float64(t2)), -fw* np.cos(np.float64(t2))])
	av = d2v + f1v+ w0v - wpv

	d1 = np.sqrt((av*av).sum() - fw*fw)
	t1 = np.arctan2(av[1], av[0]) - np.arctan2(fw, d1)
	return(t1, d1)

def finger_translation2(w0, wp, fw, t1, d1):
	# Center Coordinates of square
	x_square = wp + (d1 - w0/2)* np.cos(t1) - (w0/2 + fw)* np.sin(t1)
	y_square = (d1 - w0/2)* np.sin(t1) + (w0/2 + fw)* np.cos(t1)
	# Calculate theta1, d1
	d1v = np.array([d1* np.cos(t1), d1* np.sin(t1)])
	w0v = np.array([w0* np.sin(t1), w0* np.cos(t1)])
	wpv = np.array([wp, 0])	
	f2v = np.array([fw* np.sin(t1), fw* np.cos(t1)])
	av = d1v - w0v - f2v + wpv
	d2 = np.sqrt((av*av).sum() - fw*fw)
	t2 = np.arctan2(av[1], av[0]) + np.arctan2(fw, d2)
	return(t2, d2)









def finger_planning(x, y, w0, wp, fw, t1, t2, d1, d2):
	# Calculate Desired Position Parameters
	t1_d, t2_d, d1_d, d2_d = ik_finger(x, y, w0, wp, fw)
	print (d1_d, float(d2_d), t1_d, t2_d)
	# Path Planning
	d1t = d1
	d2t = d2
	
	if (d1_d > d1 and d2_d > d2):
		while ((d1_d - d1) > 0.001 or (d2_d - d2)> 0.001):
			print "d1 = ", d1
			print "d2 = ", d2
			print "t1 = ", t1
			print "t2 = ", t2
			print 'Left_Slide_Up start'
			
			while (d1 < d1_d and t1 < 152*pi/180):
				t2 = t2 + 0.01
				t1, d1 = finger_translation1(w0, wp, fw, t2, d2)				
			
			print("t2=",t2)
			print("t1=",t1)
			print("d1=",d1)
			print("d2=",d2)
			
			slide_left_finger_up(angle_conversion(3.14-t2,1))
			
			print 'Left_Slide_Up end'
			
			print 'Right_Slide_Up start'
			while (d2 < d2_d and t2 > 21*pi/180):
				t1  = t1 - 0.01
				t2, d2 = finger_translation2(w0, wp, fw, t1, d1)
				
			slide_right_finger_up(angle_conversion(3.14-t1,0))	
			print ("t1=",t1)
			print ("t2=",t2)
			print ("d2=",d2)		
			print("d1=",d1)		
			print 'Right_Slide_Up end'
			
			
			#slide_left_finger_up(angle_conversion(t2,1))
			d1t = d1
			d2t = d2
			#print (d1_d, d1, d2_d, d2)

		t1, d1 = finger_translation1(w0, wp, fw, t2_d, d2)
		print "d1 =", d1
		print "d2 =", d2
		print "d1_d =", d1_d
		print "d2_d =", d2_d
		# Call Left_Slide_Up(t1_d, t2)
		slide_right_finger_up(angle_conversion(3.14-t1_d,0))

	else:	
		while ((d1_d - d1) < -0.001 or (d2_d - d2) < -0.001):
			print("t1=",t1)
			print("t2=",t2)
			print("d1=",d1)
			print("d2=",d2)
			print ('right finger down start')
			while (d2 > d2_d and t1 < 152*pi/180):
				t1  = t1 + 0.01
				t2, d2 = finger_translation2(w0, wp, fw, t1, d1)
			
			print("t1=",t1)
			print("t2=",t2)
			print("d1=",d1)
			print("d2=",d2)
			slide_right_finger_down(angle_conversion(3.14-t2,1))
			print ('right finger down end')
			print ('left finger down start')
			print ('d2 =', d2)
			print ('d2_d =', d2_d)
			
			while (d1 > d1_d and t2 > 21*pi/180):
				t2 = t2 - 0.01
				t1, d1 = finger_translation1(w0, wp, fw, t2, d2)
			print ('left finger down end')
			print("t1=",t1)
			print("t2=",t2)
			print("d1=",d1)
			print("d2=",d2)
			slide_left_finger_down(angle_conversion(3.14-t1,0))
			d1t = d1
			d2t = d2
		t1_d, d1 = finger_translation1(w0, wp, fw, t2_d, d2)
		# Call Left_Slide_Down(t1_d, t2)
		slide_left_finger_down(angle_conversion(3.14-t1_d,0))
	#print (t1_d, t1, t2_d, t2)

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







if __name__ == "__main__":
	#Initial position of the object in the fingers 
	d1 = 9
	
	#Starting angle of the fingers
	t1 = np.pi/2
	
	#width of finger
	fw=1.5
	#Width of object
	w0 = 2.5
        #Width of palm
	wp = 7
	#Desired position of the object
	x=3
	y=7  		
        n = 3 # n = 1 -> Anticlockwise rotation, n = 0 -> Clockwise
	t2, d2= finger_translation2(w0, wp, fw, t1, d1)
	#Hold object
	#rospy.wait_for_service('Hold_object')
	try:
		hold_object = rospy.ServiceProxy('Hold_object',Holdcommand)
		#req = friction_finger_gripper.srv.PositionCommandRequest(0.52,0.64)
		#resp1 = hold_object(req)
		resp1 = hold_object(0.52,0.74)
	except rospy.ServiceException, e:
		print ("Service call failed: %s"%e)
	
	if n == 1:	
		'''while (t1 > 21*pi/180):
			t1  = t1 - 0.01
			t2, d2 = finger_translation2(w0, wp, fw, t1, d1)
		'''
		t1f = np.arccos(((d2 - w0)**2 + w0**2 - (d1+w0)**2 - wp**2)/(2*wp*(d1 + w0))) 
		# Call Rotate_anticlockwise(t2f)
		

		rotate_object_anticlockwise(angle_conversion(3.14-t1f,1))

		t2= t1f

		d1 = d1 + w0
		d2 = d2 - w0
	if n == 0:
		''''while (t2 < 152*pi/180):
			t2 = t2 + 0.01
			t1, d1 = finger_translation1(w0, wp, fw, t2, d2)			'''	

		#Call Right_Slide_Down(till right most position) Assume t2 = 5*pi/6
		#slide_right_finger_down((t2*180/pi - 23.5)*0.00218 + 0.6993)

	
		#t2, d2 = finger_translation1(w0, wp, t1, d1)
		t2f = np.pi - np.arccos((((d1-w0)**2 + w0**2 - wp**2 - (d2 + w0)**2)/(2*wp*(d2+w0))))
		print (t2f)
		t1 = t2f
		
		d1 = d1 - w0
		d2 = d2 + w0

		# Call Rotate_clockwise(t1f)
		rotate_object_clockwise(angle_conversion(3.14-t2f,0))

	finger_planning(x, y, w0, wp,fw, t1, t2, d1, d2)
