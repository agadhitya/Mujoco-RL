#!/usr/bin/env python

import rospy
import time
import numpy as np
from gripper_controls.srv import Holdcommand
from manipulation_planning.srv import HandAct, ArmAct, PlanExe, PlanExeResponse
from gazebo_msgs.srv import GetModelState

def hold_object(left,right):
    rospy.wait_for_service('Hold_object')
    try:
        ho = rospy.ServiceProxy('Hold_object', Holdcommand)
        resp = ho(left,right)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def slide_left_finger_down(val):
    rospy.wait_for_service('plan_exe/slide_left_down')
    try:
        slfd = rospy.ServiceProxy('plan_exe/slide_left_down', HandAct)
        resp = slfd(val)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def slide_left_finger_up(val):
    rospy.wait_for_service('plan_exe/slide_left_up')
    try:
        slfu = rospy.ServiceProxy('plan_exe/slide_left_up', HandAct)
        resp = slfu(val)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def slide_right_finger_down(val):
    rospy.wait_for_service('plan_exe/slide_right_down')
    try:
        srfd = rospy.ServiceProxy('plan_exe/slide_right_down', HandAct)
        resp = srfd(val)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def slide_right_finger_up(val):
    rospy.wait_for_service('plan_exe/slide_right_up')
    try:
        srfu = rospy.ServiceProxy('plan_exe/slide_right_up', HandAct)
        resp = srfu(val)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def rotate_clockwise(val):
    rospy.wait_for_service('plan_exe/rotate_cw')
    try:
        rc = rospy.ServiceProxy('plan_exe/rotate_cw', HandAct)
        resp = rc(val)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def rotate_anticlockwise(val):
    rospy.wait_for_service('plan_exe/rotate_acw')
    try:
        rac = rospy.ServiceProxy('plan_exe/rotate_acw', HandAct)
        resp = rac(val)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def move_up(val):
    rospy.wait_for_service('plan_exe/move_contact_up')
    try:
        mcu = rospy.ServiceProxy('plan_exe/move_contact_up', ArmAct)
        print(val)
        resp = mcu(val)
        return resp.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def move_down(val):
    rospy.wait_for_service('plan_exe/move_contact_down')
    try:
        mcd = rospy.ServiceProxy('plan_exe/move_contact_down', ArmAct)
        resp = mcd(val)
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
    # with open("manipulation_plans/test_4/data1.txt") as f:
    with open("alp/mr_motion_planning/run_2020-08-16_15-47/data.txt") as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            list_of_lists.append(inner_list)

    action_list = list()
    theta_l = list()
    theta_r = list()

    for row in list_of_lists[1:]:
        action_list.append(row[6])
        if row[7] == "None":
            theta_l.append(None)
        else:
            theta_l.append(float(row[7]))
        if row[8] == "None]":
            theta_r.append(None)
        else:
            theta_r.append(float(row[8][:-1]))

    command = 0.5
    for step in range(0,len(action_list)):
        action = action_list[step]
        if step != len(action_list)-1:
            next_action = action_list[step+1]
            if action==next_action and not (action=="'Move down'" or action=="'Move up'"):
                continue
            elif action==next_action and (action=="'Move down'" or action=="'Move up'"):
                command += 0.5
                continue

        if action=="'Slide left down'":
            command = angle_conversion(theta_l[step],'left')
            print("Left down: {}".format(command))
            slide_left_finger_down(command)
        elif action=="'Slide right down'":
            command = angle_conversion(theta_r[step],'right')
            print("Right down: {}".format(command))
            slide_right_finger_down(command)
        elif action=="'Slide left up'":
            command = angle_conversion(theta_r[step],'right')
            print("Left up: {}".format(command))
            slide_left_finger_up(command)
        elif action=="'Slide right up'":
            command = angle_conversion(theta_l[step],'left')
            print("Right up: {}".format(command))
            slide_right_finger_up(command)
        elif action=="'Rotate ccw'":
            command = angle_conversion(theta_r[step],'right')
            print("Anticlockwise: {}".format(command))
            rotate_anticlockwise(command)
        elif action=="'Rotate cw'":
            command = angle_conversion(theta_l[step],'left')
            print("Clockwise: {}".format(command))
            rotate_clockwise(command)
        elif action=="'Move up'":
            print("Moving up: {}".format(command))
            command = command/100.0
            move_up(command)
            command = 0
        elif action=="'Move down'":
            print("Moving down: {}".format(command))
            command = command/100.0
            move_down(command)
            command = 0
        else:
            print("Error in action")
            break

        # time.sleep(2)

def compute_result():
    rospy.wait_for_service('gazebo/get_model_state')
    try:
        gms = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        resp = gms("object","panda_link7")
        x = resp.pose.position.x
        z = resp.pose.position.z
        zVal = 8.5 - (((0.033 - x)*(8.5 - 2.5))/(0.033 - (-0.027)))
        dVal = 9.5 - (((0.28 - z)*(9.5 - 3.0))/(0.28 - 0.22))
        return zVal, dVal
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def handle_plan_execution(req):
    hold_object(-0.0,-0.0)
    z_init,d_init = compute_result()
    print("z_init: {}, d_init: {}".format(z_init,d_init))
    execute_plan()
    print("Plan executed.")
    z_final,d_final = compute_result()
    print("z_final: {}, d_final: {}".format(z_final,d_final))
    return PlanExeResponse(z_final,d_final)

def plan_execution_server():
    rospy.init_node('plan_execution_server')
    s = rospy.Service('execute_plan', PlanExe, handle_plan_execution)
    print("Ready to execute plan.")
    rospy.spin()

if __name__ == "__main__":

    plan_execution_server()
