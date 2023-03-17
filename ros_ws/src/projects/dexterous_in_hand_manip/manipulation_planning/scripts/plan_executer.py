#!/usr/bin/env python

import rospy
import time
import numpy as np
from gripper_controls.srv import PositionCommand, Holdcommand
from manipulation_planning.srv import PlanExe

def hold_object(left,right):
    rospy.wait_for_service('Hold_object')
    try:
        ho = rospy.ServiceProxy('Hold_object', Holdcommand)
        resp = ho(left,right)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def slide_left_finger_down(val):
    rospy.wait_for_service('Slide_Left_Finger_Down')
    try:
        slfd = rospy.ServiceProxy('Slide_Left_Finger_Down', PositionCommand)
        resp = slfd(val)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def slide_left_finger_up(val):
    rospy.wait_for_service('Slide_Left_Finger_Up')
    try:
        slfu = rospy.ServiceProxy('Slide_Left_Finger_Up', PositionCommand)
        resp = slfu(val)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def slide_right_finger_down(val):
    rospy.wait_for_service('Slide_Right_Finger_Down')
    try:
        srfd = rospy.ServiceProxy('Slide_Right_Finger_Down', PositionCommand)
        resp = srfd(val)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def slide_right_finger_up(val):
    rospy.wait_for_service('Slide_Right_Finger_Up')
    try:
        srfu = rospy.ServiceProxy('Slide_Right_Finger_Up', PositionCommand)
        resp = srfu(val)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def rotate_clockwise(val):
    rospy.wait_for_service('Rotate_clockwise')
    try:
        rc = rospy.ServiceProxy('Rotate_clockwise', PositionCommand)
        resp = rc(val)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def rotate_anticlockwise(val):
    rospy.wait_for_service('Rotate_anticlockwise')
    try:
        rac = rospy.ServiceProxy('Rotate_anticlockwise', PositionCommand)
        resp = rac(val)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def angle_conversion(theta,finger):
    if finger=="left":
        return theta - np.pi/2
    elif finger=="right":
        return np.pi/2 - theta
    else:
        print("Error in angle conversion")
        return 0

def execute_plan():

    list_of_lists = list()
    # with open("catkin_ws/src/FrictionFinger/Motion_planner/Test_results/data1.txt") as f:
    with open("manipulation_plans/test_3/data1.txt") as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            list_of_lists.append(inner_list)

    action_list = list()
    theta_l = list()
    theta_r = list()
    x_pos = list()
    y_pos = list()

    for row in list_of_lists:
        action_list.append(float(row[2]))
        theta_l.append(float(row[3]))
        theta_r.append(float(row[4]))
        x_pos.append(float(row[5]))
        y_pos.append(float(row[6]))

    for step in range(0,len(action_list)):
        action = action_list[step]
        if step != len(action_list)-1:
            next_action = action_list[step+1]
            if action==next_action:
                continue

        if action==0:
            command = angle_conversion(theta_l[step],'left')
            print("Left down: {}".format(command))
            slide_left_finger_down(command)
        elif action==1:
            command = angle_conversion(theta_r[step],'right')
            print("Right down: {}".format(command))
            slide_right_finger_down(command)
        elif action==2:
            command = angle_conversion(theta_r[step],'right')
            print("Left up: {}".format(command))
            slide_left_finger_up(command)
        elif action==3:
            command = angle_conversion(theta_l[step],'left')
            print("Right up: {}".format(command))
            slide_right_finger_up(command)
        elif action==4:
            command = angle_conversion(theta_r[step],'right')
            print("Anticlockwise: {}".format(command))
            rotate_anticlockwise(command)
        elif action==5:
            command = angle_conversion(theta_l[step],'left')
            print("Clockwise: {}".format(command))
            rotate_clockwise(command)
        else:
            print("Error in action")
            break

        time.sleep(2)

def handle_plan_execution(req):
    hold_object(0,0)
    execute_plan()
    print("Plan executed.")

def plan_execution_server():
    rospy.init_node('plan_execution_server')
    s = rospy.Service('execute_plan', PlanExe, handle_plan_execution)
    print("Ready to execute plan.")
    rospy.spin()

if __name__ == "__main__":

    # rospy.init_node('Plan_executer')
    # hold_object(0,0)
    # execute_plan()
    # time.sleep(5)
    plan_execution_server()
