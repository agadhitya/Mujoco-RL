#!/usr/bin/python

import readchar
import rospy
from geometry_msgs.msg import PoseStamped, Vector3Stamped, QuaternionStamped, Pose
from std_msgs.msg import Bool
from relaxed_ik.msg import EEPoseGoals
import RelaxedIK.Utils.transformations as T
import time
from random import seed
from random import random

# seed(1)

# value = random(0, 1)
# print (value)

rospy.init_node('keyboard_ikgoal_driver')

ik_goal_r_pub = rospy.Publisher('/ik_goal_r', PoseStamped, queue_size=5)
ik_goal_l_pub = rospy.Publisher('/ik_goal_l', PoseStamped, queue_size=5)
goal_pos_pub = rospy.Publisher('vive_position', Vector3Stamped)
goal_quat_pub = rospy.Publisher('vive_quaternion', QuaternionStamped)
ee_pose_goals_pub = rospy.Publisher(
    '/relaxed_ik/ee_pose_goals', EEPoseGoals, queue_size=5)
quit_pub = rospy.Publisher('/relaxed_ik/quit', Bool, queue_size=5)

pos_stride_x = 0.06
rot_stride = 0.055

position_r = [0, 0, 0]
rotation_r = [1, 0, 0, 0]

position_l = [0, 0, 0]
rotation_l = [1, 0, 0, 0]

loop_counter = 6
timer = 0.1

seq = 1
rate = rospy.Rate(1000)
while not rospy.is_shutdown():

    pose = PoseStamped()
    pose.pose.position.x = position_r[0]
    pose.pose.position.y = position_r[1]
    pose.pose.position.z = position_r[2]

    pose.pose.orientation.w = rotation_r[0]
    pose.pose.orientation.x = rotation_r[1]
    pose.pose.orientation.y = rotation_r[2]
    pose.pose.orientation.z = rotation_r[3]
    ik_goal_r_pub.publish(pose)

    pose = PoseStamped()
    pose.pose.position.x = position_l[0]
    pose.pose.position.y = position_l[1]
    pose.pose.position.z = position_l[2]

    pose.pose.orientation.w = rotation_l[0]
    pose.pose.orientation.x = rotation_l[1]
    pose.pose.orientation.y = rotation_l[2]
    pose.pose.orientation.z = rotation_l[3]
    ik_goal_l_pub.publish(pose)

    # position_r[2] = 2.40

    ee_pose_goals = EEPoseGoals()
    pose_r = Pose()
    pose_r.position.x = position_r[0]
    pose_r.position.y = position_r[1]
    pose_r.position.z = position_r[2]

    pose_r.orientation.w = rotation_r[0]
    pose_r.orientation.x = rotation_r[1]
    pose_r.orientation.y = rotation_r[2]
    pose_r.orientation.z = rotation_r[3]

    pose_l = Pose()
    pose_l.position.x = position_l[0]
    pose_l.position.y = position_l[1]
    pose_l.position.z = position_l[2]

    pose_l.orientation.w = rotation_l[0]
    pose_l.orientation.x = rotation_l[1]
    pose_l.orientation.y = rotation_l[2]
    pose_l.orientation.z = rotation_l[3]
    ee_pose_goals.ee_poses.append(pose_r)
    ee_pose_goals.ee_poses.append(pose_l)

    key = readchar.readkey()

    if key == 'c':
        rospy.signal_shutdown()
    elif key == 's':
        while (True):
            value_x = random()
            value_y = random()
            value_z = random()
            scaled_value_x = -1.02 + (value_x * (1.02 - (-1.02)))
            scaled_value_y = -2.4 + (value_y * (2.4 - (-2.4)))
            scaled_value_z = -0.30 + (value_z * (2.0 - (-0.30)))
            final_value = [scaled_value_x, scaled_value_y, scaled_value_z]
            current_value = [position_r[0], position_r[1], position_r[2]]
            desired_value = [final_value[i]-current_value[i] for i in range(3) ]
            print (current_value)
            print(final_value)
            print (desired_value)
            list_l = [-1 if i < 0 else 1 for i in desired_value]

            print(list_l)
            mul_flag_x = int(abs(desired_value[0]/0.06))
            mul_flag_y = int(abs(desired_value[1]/0.06))
            mul_flag_z = int(abs(desired_value[2]/0.06))
            mul_flag = [mul_flag_x,mul_flag_y,mul_flag_z]
            print (mul_flag)
            max_pad = max(mul_flag)
            desired_array_x = [list_l[0]*0.06]*mul_flag_x
            desired_array_x.extend([0.0]*(max_pad-mul_flag_x))
            desired_array_y = [list_l[1]*0.06]*mul_flag_y
            desired_array_y.extend([0.0]*(max_pad-mul_flag_y))
            desired_array_z = [list_l[2]*0.06]*mul_flag_z
            desired_array_z.extend([0.0]*(max_pad-mul_flag_z))
            # if (desired_value[0]/0.06) < 0 or (desired_value[0]/0.06) or (desired_value[0]/0.06):
            #     array_x = [-0.06]*int(desired_value[0]/0.06)
            desired_array = [desired_array_x, desired_array_y, desired_array_z]

            # print('********')
            # print(desired_array)
            for i in range(max_pad):
                # if position_r[2] < 2.4 and position_r[2] > -0.36:
                position_r[2] += desired_array[2][i]
                # if position_r[1] < 2.4 and position_r[1] > -2.4:
                position_r[1] += desired_array[1][i]
                # if position_r[0] < 1.02 and position_r[0] > -1.02:
                position_r[0] += desired_array[0][i]
                pose_r.position.x = position_r[0]
                pose_r.position.y = position_r[1]
                pose_r.position.z = position_r[2]
                # euler = list(T.euler_from_quaternion(rotation_r))
                # euler[1] -= rot_stride
                # euler[2] += rot_stride
                # rotation_r = T.quaternion_from_euler(euler[0],euler[1],euler[2])
                # pose_r.orientation.w = rotation_r[0]
                # pose_r.orientation.x = rotation_r[1]
                # pose_r.orientation.y = rotation_r[2]
                # pose_r.orientation.z = rotation_r[3]
                # pose_r.position.z = position_r[2]
                # if position_r[2] > 2.4 or position_r[2] < -0.36:
                    
                ee_pose_goals.header.seq = seq
                seq += 1
                ee_pose_goals_pub.publish(ee_pose_goals)
                
                time.sleep(timer)
        # position_r = [0, 0, 0]

    q = Bool()
    q.data = False
    quit_pub.publish(q)
