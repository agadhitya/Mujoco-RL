#! /usr/bin/env python
"""Publishes joint trajectory to move robot to given trajectory"""


import rospy
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from std_srvs.srv import Empty
import argparse
import glob
import datetime
import time
import numpy as np
import cv2 as cv
import os
import csv
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float64
from relaxed_ik.msg import JointAngles

# rospy.init_node('keyboard_control_node', anonymous=True)
# joint_pub = rospy.Publisher('/panda/arm_controller/command', JointTrajectory, queue_size=1)






def moveJoint(jointcmds, prefix='panda', nbJoints=7):
    jointCmd = JointTrajectory()
    point = JointTrajectoryPoint()
    jointCmd.header.stamp = rospy.Time.now() + rospy.Duration.from_sec(0.0)
    point.time_from_start = rospy.Duration.from_sec(5.0)
    # temp = [1.1158527428028382,0.6550765123812701,-0.6846453664099972,-0.598807004486682,-0.039304507129546806,1.937493448059687,0.6636270731508924]
    for i in range(0, nbJoints):
        jointCmd.joint_names.append(prefix + '_joint' + str(i + 1))
        point.positions.append(jointcmds.angles.data[i])
        point.velocities.append(0)
        point.accelerations.append(0)
        point.effort.append(0)
    jointCmd.points.append(point)
    rate = rospy.Rate(100)
    count = 0
    while (count < 5):
        joint_pub.publish(jointCmd)
        count = count + 1
        rate.sleep()
        print(jointCmd)
        # print("pubs")

# def make_trajectory():
#     for i in range(500):
#         angle = i * np.pi / 180
#         value_to_move = [np.sin(angle), np.cos(angle), 0.0, 0.0, 0.0, 1.66, 0.0]
#         moveJoint(value_to_move)


if __name__ == '__main__':
    print("init done")
    rospy.init_node('keyboard_control_node', anonymous=True)
    joint_pub = rospy.Publisher('/panda/arm_controller/command', JointTrajectory, queue_size=1)
    while (not rospy.is_shutdown()):
    
        rospy.Subscriber(name='/relaxed_ik/joint_angle_solutions', data_class=JointAngles,
                                          callback=moveJoint,
                                          queue_size=1)

        rospy.spin()
    # rospy.Subscriber('/relaxed_ik/joint_angle_solutions',)

