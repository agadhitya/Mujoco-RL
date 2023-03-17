#!/usr/bin/env python

import rospy
import cv2
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt
import time

import roslib
roslib.load_manifest("rosparam")
import rosparam


paramlist  = rosparam.load_file('/home/joshua/ff_ws/src/friction_finger_gripper/config/ff_parameters.yaml')
for params, ns in paramlist:
    rosparam.upload_params(ns,params)
    start_x = params['start_x']
    start_y = params['start_y']
    end_x = params['end_x']
    end_y = params['end_y']
    a_x = params['a_x']
    b_x = params['b_x']
    a_y = params['a_y']
    b_y = params['b_y']



bridge = CvBridge()
X = []
Y = []
start=[start_x, start_y]
end=[end_x, end_y]
#end=[-0.0244, 0.07125]
#end=[0.0724, 0.07125]
#end = [0.0275, 0.10947]

colour = []
class trajectory:
    def __init__(self):
        self.X = []
        self.Y = []
        self.cv_image = Image()
        self.x = None
        self.y = None
        self.action = ''
        self.start = None

    def listener(self):
        rospy.Subscriber("/object_position", Point, self.callback_position)
        rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        rospy.Subscriber("/Action", Int32, self.callback_action)
        rospy.Subscriber("/State", Int32, self.callback_state)

    def callback_state(self, s):
        self.start = s.data


    def callback(self, data):
        global colour
        self.cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        (rows, cols, channels) = self.cv_image.shape
        # if cols > 60 and rows > 60:
        #     cv2.circle(self.cv_image, (50, 50), 10, 255)
        #     length = len(self.X)
        # cv2.line(cv_image, (0, 0), (50, 50), (255, 255, 0), 2)
        length = len(self.X)
        # pts = self.coords.reshape((-1, 1, 2))
        for i in range(2, length - 1):
            # print self.X[0], self.Y[0]
            # print self.X[i], self.Y[i]
            if i < 100:
                continue
            if colour[i] == 1:
                cv2.line(self.cv_image, (self.X[i], self.Y[i]), (self.X[i - 1], self.Y[i - 1]), (0, 0, 255), 5, lineType=8)
            elif colour[i] == 2:
                cv2.line(self.cv_image, (self.X[i], self.Y[i]), (self.X[i - 1], self.Y[i - 1]), (0, 255, 0), 5, lineType=8)
            else:
                cv2.line(self.cv_image, (self.X[i], self.Y[i]), (self.X[i - 1], self.Y[i - 1]), (255, 255, 0), 5, lineType=8)
        #cv2.line(self.cv_image, (self.X[length - 1], self.Y[length - 1]), (self.X[length - 2], self.Y[length - 2]), (255, 255, 0), 2, lineType=8)

        # cv2.polylines(cv_image, [pts], True, (0, 255, 255))
        # cv2.line(self.cv_image, (10, 10), (20, 20), (255, 255, 0), 2)
        # cv2.line(self.cv_image, (10, 10), (10, 20), (255, 255, 0), 2)
        cv2.circle(self.cv_image, self.XY_to_pixel_conversion(start[0],start[1]), 9,(255, 0, 255),-1)
        cv2.circle(self.cv_image, self.XY_to_pixel_conversion(end[0],end[1]), 9, (100, 60, 255),-1)
        cv2.circle(self.cv_image, self.XY_to_pixel_conversion(0,0), 9, (100, 255, 100),-1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.cv_image, self.action, (50,30), font, 1, (0, 0, 10), 2)
        cv2.imshow("Image window", self.cv_image)
        cv2.waitKey(3)

    def XY_to_pixel_conversion(self,x,y):
        x = int(x * a_x + b_x)
        y = int(y * a_y  +  b_y)
        return x,y

    def callback_position(self, msg):
        # print 123
        global X, Y, X_rotation, Y_rotation, colour
        # self.x = int(msg.x * 1893.283 + 265.703)
        # self.y = int(-msg.y * 2016.301  +  317.598)
        self.x,self.y=self.XY_to_pixel_conversion(msg.x,msg.y)
        self.X.append(self.x)
        self.Y.append(self.y)
        if self.action == 'Anti-clockwise Rotation' or self.action == 'Clockwise Rotation':
            colour.append(1)
        elif self.action == 'Visual Servoing - Left Slide down' or self.action == 'Visual Servoing - Right Slide down' or self.action == 'Visual Servoing - Left Slide up' or self.action =='Visual Servoing - Right Slide up' :
            colour.append(2)
        else:
            colour.append(0)

        #print "Length of array=", len(self.X)
        X = self.X
        Y = self.Y
        # self.coords.append([self.x, self.y])
        # print self.x, self.y
    
    def callback_action(self, act):

        if (act.data == 0):
            self.action = 'Left Slide down'
        if (act.data == 1):
            self.action = 'Right Slide down'
        if (act.data == 2):
            self.action = 'Left Slide up'
        if (act.data == 3):
            self.action = 'Right Slide up'
        if (act.data == 4):
            self.action = 'Anti-clockwise Rotation'
        if (act.data == 5):
            self.action = 'Clockwise Rotation'
        if (act.data == 6):
            self.action = 'Visual Servoing - Left Slide down'
        if (act.data == 7):
            self.action = 'Visual Servoing - Left Slide up'
        if (act.data == 8):
            self.action = 'Visual Servoing - Right Slide down'
        if (act.data == 9):
            self.action = 'Visual Servoing - Right Slide up'

    def traj_publisher(self):
        pub = rospy.Publisher("/object_trajectory", Image,queue_size = 50)
        rate = rospy.Rate(30) # 10hz
        self.listener()
        rospy.wait_for_message("object_position", Point)
	rospy.wait_for_message("aruco_simple/result", Image)
        while not rospy.is_shutdown():
            pub.publish(bridge.cv2_to_imgmsg(self.cv_image, "bgr8"))
            rate.sleep()


'''
def callback(data):
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    (rows, cols, channels) = cv_image.shape
    if cols > 60 and rows > 60:
        cv2.circle(cv_image, (50, 50), 10, 255)
    # cv2.line(cv_image, (X[len(X) - 1], Y[len(X) - 1]), (X[len(X) - 2], Y[len(X) - 2]), (255, 255, 0), 2)
    # print len(X)
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)


def callback_position(msg):
    print 123
    x = msg.x * 100
    y = msg.y * 100
    X.append(x)
    Y.append(y)
    print len(X)


def listener():
    rospy.Subscriber("/aruco_simple/result", Image, callback)
    rospy.Subscriber("/aruco_simple/pose1", Point, callback_position)

'''


def main():
    
    global X, Y
    rospy.init_node('Image', anonymous=True)
    t = trajectory()
    t.traj_publisher()
    #rospy.spin()
    # plt.show()


if __name__ == '__main__':
    main()
