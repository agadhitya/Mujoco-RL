#! /usr/bin/env python
"""Data capture node for the Deep CNN network"""

import rospy
# from trajectory_msgs.msg import JointTrajectory
# from trajectory_msgs.msg import JointTrajectoryPoint
# from control_msgs.msg import JointTrajectoryControllerState
# from std_srvs.srv import Empty
import argparse
import glob
import datetime
import time
import numpy as np
import cv2 as cv
import csv
import rospy
from std_msgs.msg import String 
from sensor_msgs.msg import Image 
from gazebo_msgs.msg import LinkStates 
from rospy.numpy_msg import numpy_msg 
from std_msgs.msg import Float64 
from encoderless_robots.msg import EEPoseGoals 
from encoderless_robots.msg import EffortValues

running_joints_val = [] 
depth_image_data = None 
state_error_data = None 
value_to_move = [0.0, 0.0, 0.0, 0.0, 0.0, 1.66, 0.0] 
effort_data = [] 
timer = 0.0
i = 0 

# def generate(frame,joint_data,ee_pose_data):
#     global i
#     i += 1
    
#     # frame = np.frombuffer(img_data.data, dtype=np.uint8).reshape(img_data.height, img_data.width, -1)
#     # cv.namedWindow("win",cv.WINDOW_AUTOSIZE)
#     cv.imwrite('/home/ajay/kaam/DR/dataset3/img%s.jpg'%i,frame)
#     # jointValues = data.position
#     # q1 = jointValues[0]
#     # q2 = jointValues[1]
#     with open('/home/ajay/kaam/DR/dataset3/joint%s.csv'%i, 'w') as csvFile:
#         writer = csv.writer(csvFile, quoting=csv.QUOTE_NONE)
#         writer.writerow(joint_data)
#         csvFile.close()
#     with open('/home/ajay/kaam/DR/dataset3/ee_pose%s.csv'%i, 'w') as csvFile:
#         writer = csv.writer(csvFile, quoting=csv.QUOTE_NONE)
#         writer.writerow(ee_pose_data)
#         csvFile.close()
#     print('Generating datsets sample -- ',i)
#     # time.sleep(0.20)


start_flag = 1


def effort_values(data):
    # if i == 5000:
    #     rospy.signal_shutdown('dataset generated')
    global depth_image_data, start_flag, effort_data, timer
    # print start_flag
    if (rospy.get_time() - timer) < 8:
        effort_data.append(data.data[0])
        effort_data.append(data.data[1])
        effort_data.append(data.data[2])
        effort_data.append(data.data[3])
        effort_data.append(data.data[4])
        effort_data.append(data.data[5])
        effort_data.append(data.data[6])
            

        with open('/home/ajay/kaam/DR/dataset4/effort_values1.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile, quoting=csv.QUOTE_NONE)
            writer.writerow(effort_data)
            csvFile.close()
        del effort_data[:]

def cb_depth_image(data):
    # if i == 5000:
    #     rospy.signal_shutdown('dataset generated')
    # print("in data")
    

    global depth_image_data, start_flag, timer, i

    curr = rospy.get_time()
    if (curr - timer) == 8:
        start_flag = 1
    
    if start_flag == 1:
        depth_image_data = data
        framed = np.frombuffer(depth_image_data.data, dtype=np.float32).reshape(depth_image_data.height, depth_image_data.width)

        framed = framed / 3.0
        framed = framed * 255
        framed = framed.astype(np.uint8)
        cv.imwrite('/home/ajay/kaam/DR/dataset4/img%s.jpg'%i,framed)
        start_flag = 0
        i = i+1
    
def cb_wait(data):
    # if i == 5000:
    #     rospy.signal_shutdown('dataset generated')
    global start_flag, timer
    # if start_flag:
    timer = rospy.get_time()

# def cb_ee_data(data):
#     if i == 5000:
#         rospy.signal_shutdown('dataset generated')
#     global ee_pose_data
#     del ee_pose_data[:]
#     # print "aasdcjhs\dh"
#     ee_pose_data.append(data.pose[11].position.x)
#     ee_pose_data.append(data.pose[11].position.y)
#     ee_pose_data.append(data.pose[11].position.z)
#     ee_pose_data.append(data.pose[11].orientation.w)
#     ee_pose_data.append(data.pose[11].orientation.x)
#     ee_pose_data.append(data.pose[11].orientation.y)
#     ee_pose_data.append(data.pose[11].orientation.z)

def myhook():
    print ('dataset generated')

if __name__ == '__main__':
    global start_flag
    
    rospy.init_node('effort_data_record_node', anonymous=True)
    print("init done")
    while (not rospy.is_shutdown()):
        t = rospy.get_time()
        
        if start_flag:
            rospy.Subscriber(name='/relaxed_ik/ee_pose_goals', data_class=EEPoseGoals,
                            callback=cb_wait,
                            queue_size=10)
        print rospy.get_time()
        # rospy.Subscriber(name='/gazebo/link_states', data_class=LinkStates,
        #                  callback=cb_ee_data,
        #                  queue_size=10)
        
        rospy.Subscriber(name='/camera_link/depth/image_raw', data_class=numpy_msg(Image),
                         callback=cb_depth_image,
                         queue_size=10)

        rospy.Subscriber(name='/chatter', data_class=EffortValues,
                         callback=effort_values,
                         queue_size=10)
        # if i == 5000:
        #     rospy.signal_shutdown('dataset generated')
        

        rospy.spin()

# print datetime.datetime.now()