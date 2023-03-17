#!/usr/bin/env python

import rospy
import time
import numpy as np
from friction_finger_gripper.srv import*
from geometry_msgs.msg import Point
from common_msgs_gl.srv import *
from std_msgs.msg       import Float32
from std_msgs.msg       import Int32
from common_msgs_gl.msg import Motor_position
from gazebo_msgs.srv    import *
import tf.transformations as tr
from geometry_msgs.msg import Pose
from manipulation_planning.srv import ArmPose
from gripper_controls.srv import SetFriction
# from gripper_controls.srv import Holdcommand
# from manipulation_planning.srv import HandAct, ArmAct, PlanExe, PlanExeResponse
# from gazebo_msgs.srv import GetModelState

import roslib
roslib.load_manifest("rosparam")
import rosparam


paramlist  = rosparam.load_file('/home/asahin//catkin_ws/src/FrictionFinger/friction_finger_gripper/config/beg.yaml')

for params, ns in paramlist:
    rosparam.upload_params(ns,params)
    a_left = params['a_left']
    b_left = params['b_left']
    a_right = params['a_right']
    b_right = params['b_right']
    # hold_pos_left = params['hold_left']
    # hold_pos_right = params['hold_right']

sleep_time = 0.1
angle_diff = -0.01

global finger_state
finger_state = -1

def hold_object(p1, p2):
    rospy.wait_for_service('Hold_object')
    try:
        hold = rospy.ServiceProxy('Hold_object', Holdcommand)
        resp1 = hold(p1, p2)
    except rospy.ServiceException as e:
        print ("Service call failed: %s"%e)

def release_object(p1, p2):
    rospy.wait_for_service('Home_position')
    try:
        h = rospy.ServiceProxy('Home_position', Holdcommand)
        resp1 = h(p1, p2)
        return 1
    except rospy.ServiceException as e:
        return None

def get_pose():
    rospy.wait_for_service('gazebo/get_link_state')
    try:
        gls = rospy.ServiceProxy('gazebo/get_link_state', GetLinkState)
        resp = gls("panda_link7","")
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

    res = resp.link_state.pose
    res.position.x = res.position.x + 0.107

    return res

def move_up(val):
    print("testing move up",val)
    # get ee pose: through gazebo/get_link_state, link name: panda_link7
    # rospy.wait_for_service('gazebo/get_link_state')
    # try:
    #     gls = rospy.ServiceProxy('gazebo/get_link_state', GetLinkState)
    #     resp = gls("panda_link7","")
    # except rospy.ServiceException as e:
    #     print("Service call failed: %s"%e)
    #
    # ref = resp.link_state.pose
    # ref.position.x = resp.link_state.pose.position.x + 0.107
    ref = get_pose()
    ref.position.z = ref.position.z + val
    try:
        mtp = rospy.ServiceProxy('franka_act/move_pose', ArmPose)
        resp = mtp(ref)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def move_down(val):
    print("testing move down", val)
    # get ee pose: through gazebo/get_link_state, link name: panda_link7ref = get_pose()
    ref = get_pose()
    ref.position.z = ref.position.z - val
    try:
        mtp = rospy.ServiceProxy('franka_act/move_pose', ArmPose)
        resp = mtp(ref)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def pose_to_pq(msg):
    """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.position.x, msg.position.y, msg.position.z])
    q = np.array([msg.orientation.x, msg.orientation.y,
                  msg.orientation.z, msg.orientation.w])
    return p, q

def get_transform(theta,d,a,alpha):
    T = np.eye(4)
    T[0,0] = np.cos(theta)
    T[0,1] = -np.sin(theta)*np.cos(alpha)
    T[0,2] = np.sin(theta)*np.sin(alpha)
    T[0,3] = a*np.cos(theta)
    T[1,0] = np.sin(theta)
    T[1,1] = np.cos(theta)*np.cos(alpha)
    T[1,2] = -np.cos(theta)*np.sin(alpha)
    T[1,3] = a*np.sin(theta)
    T[2,1] = np.sin(alpha)
    T[2,2] = np.cos(alpha)
    T[2,3] = d
    # print(T)
    return T

def pivot(theta_finger,dx,theta_pivot,z):
    print("test_pivoting",theta_finger,dx,theta_pivot,z)
    # get ee pose: through gazebo/get_link_state, link name: panda_link7
    # rospy.wait_for_service('gazebo/get_link_state')
    # try:
    #     gls = rospy.ServiceProxy('gazebo/get_link_state', GetLinkState)
    #     resp = gls("panda_link7","")
    #     p, q = pose_to_pq(resp.link_state.pose)
    #     norm = np.linalg.norm(q)
    #     if np.abs(norm - 1.0) > 1e-3:
    #         raise ValueError(
    #             "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
    #                 str(q), np.linalg.norm(q)))
    #     elif np.abs(norm - 1.0) > 1e-6:
    #         q = q / norm
    #     g = tr.quaternion_matrix(q)
    #     g[0:3, -1] = p
    #     p_ee = g
    #     # print(resp.link_state.pose.orientation,tr.quaternion_from_matrix(p_ee))
    # except rospy.ServiceException as e:
    #     print("Service call failed: %s"%e)
    p, q = pose_to_pq(get_pose())
    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-3:
        raise ValueError(
            "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                str(q), np.linalg.norm(q)))
    elif np.abs(norm - 1.0) > 1e-6:
        q = q / norm
    g = tr.quaternion_matrix(q)
    g[0:3, -1] = p
    p_ee = g
    # print(p_ee)

    # calculate transformation with theta_pivot = 0
    offset = 0.070
    T_ee_palm = get_transform(-np.pi/2,offset,0,np.pi/2)
    T_palm_contact = get_transform(theta_finger,0,dx/100,np.pi/2)
    T_contact_object = get_transform(0-np.pi/2,0,z,0)
    # T_ee_palm = get_transform(0,0,offset,0)
    # T_palm_contact = get_transform(theta_finger-np.pi/2,0,dx,np.pi/2)
    # T_contact_object = get_transform(0,z,0,0)
    T_ee_object = np.matmul(T_ee_palm,np.matmul(T_palm_contact,T_contact_object))
    # print(T_ee_object)
    # get object pose: p_object = T*p_ee
    p_object = np.matmul(p_ee,T_ee_object)
    # for theta_pivot in range(0,theta_pivot)
    p_ee_path = []
    angles = np.arange(0,theta_pivot+np.pi/18,np.pi/18)
    for angle in angles:
        # T_contact_object = get_transform(angle,z,0,0)
        T_contact_object = get_transform(angle-np.pi/2,0,z,0)
        T_ee_object = np.matmul(T_ee_palm,np.matmul(T_palm_contact,T_contact_object))
        # p_ee_path.append(np.matmul(np.linalg.inv(T_ee_object),p_object))
        p_ee_path.append(np.matmul(p_object,np.linalg.inv(T_ee_object)))

    franka_ref = []
    for item in p_ee_path:
        ref = Pose()
        ref.position.x = item[0,3]
        ref.position.y = item[1,3]
        ref.position.z = item[2,3]
        q = tr.quaternion_from_matrix(item)
        ref.orientation.x = q[0]
        ref.orientation.y = q[1]
        ref.orientation.z = q[2]
        ref.orientation.w = q[3]
        franka_ref.append(ref)

    # print(franka_ref)
    # set friction left and right to low

    # set_friction_l = set_friction_right(0)
    # set_friction_r = set_friction_left(0)

    rospy.wait_for_service('set_friction')
    sf = rospy.ServiceProxy('set_friction', SetFriction)
    resp = sf(0,False)
    resp = sf(1,False)
    # move_to_pose: p_ee
    rospy.wait_for_service('franka_act/move_pose')
    for ref in franka_ref:
        try:
            mtp = rospy.ServiceProxy('franka_act/move_pose', ArmPose)
            resp = mtp(ref)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
    # set friction high

    # set_friction_l = set_friction_right(1)
    # set_friction_r = set_friction_left(1)
    rospy.wait_for_service('set_friction')
    sf = rospy.ServiceProxy('set_friction', SetFriction)
    resp = sf(0,True)
    resp = sf(1,True)
    # move back to first p_ee through the same points

    for ref in franka_ref[::-1]:
        try:
            mtp = rospy.ServiceProxy('franka_act/move_pose', ArmPose)
            resp = mtp(ref)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

def angle_conversion(angle, flag):
    # angle = 180. * angle / np.pi
    # 0-> Left, 1-> Right
    print ('angle = ', angle)
    if(flag == 'right'):
        n_angle =  a_right*angle+ b_right
    else:
        n_angle = a_left*angle + b_left
    return (n_angle)

def set_actuator_modes(size, modes):
    rospy.wait_for_service("set_operating_mode")
    try:
        client_operating_mode = rospy.ServiceProxy('set_operating_mode', SendIntArray)
        resp1 = client_operating_mode(modes)
        return 1
    except rospy.ServiceException as e:
        print ("Actuator modes service call failed")

def command_position(num,position):
    rospy.wait_for_service('cmd_pos_ind')
    try:
        client_position = rospy.ServiceProxy('cmd_pos_ind', SendDoubleArray)
        resp1 = client_position([num, position])
        return 1

    except rospy.ServiceException as e:
        print ("Position Service call failed")

def command_torque(num, torque):
    rospy.wait_for_service('cmd_torque_ind')
    try:
        client_torque = rospy.ServiceProxy('cmd_torque_ind', SendDoubleArray)
        resp1 = client_torque([num, torque])
        return 1

    except rospy.ServiceException as e:
        print ("Torque Service Call Failed")

def set_friction_right(friction_surface):
    rospy.wait_for_service('Friction_surface_Right')
    try:
        client_right_surface = rospy.ServiceProxy('Friction_surface_Right', SendBool)
        resp1 = client_right_surface(friction_surface)
        return 1

    except rospy.ServiceException as e:
        return None

def set_friction_left(friction_surface):
    rospy.wait_for_service('Friction_surface_Left')
    try:
        client_left_surface = rospy.ServiceProxy('Friction_surface_Left', SendBool)
        resp1 = client_left_surface(friction_surface)
        return 1

    except rospy.ServiceException as e:
        return None

def read_pos():
    rospy.wait_for_service('read_pos')
    try:
        read_position_handler = rospy.ServiceProxy('read_pos', GetDoubleArray)
        values = read_position_handler()
        #print values.data
        return values.data
    except rospy.ServiceException as e:
        print ("Service call failed: %s" % e)

def slide_left_finger_down(p):
    global finger_state
    if finger_state != 1:
        set_friction_l = set_friction_right(1)
        set_friction_r = set_friction_left(1)

        modes = [3, 0]          # Set modes - Left -> Position, Right -> Torque (3 -> Position, 0 -> Torque)
        set_modes = set_actuator_modes(2, modes)
        # send_pos = command_position(0, p)
        send_torque = command_torque(1, 0.10)
        time.sleep(0.5)
        set_friction_l = set_friction_right(1)
        set_friction_r = set_friction_left(0)
        time.sleep(1)
        finger_state = 1
        # send_v = set_velocity(0, 10)
    theta = read_pos()

    for t1 in np.arange(theta[0], p, angle_diff):
        send_pos = command_position(0, t1)
        send_torque = command_torque(1, 0.10)
        time.sleep(sleep_time)

def slide_left_finger_up(p):
    global finger_state
    if finger_state != 2:
        set_friction_l = set_friction_right(1)
        set_friction_r = set_friction_left(1)

        modes = [0, 3]
        set_modes = set_actuator_modes(2, modes)
        time.sleep(0.5)
        # send_pos = command_position(0, p)
        send_torque = command_torque(1, 0.10)
        set_friction_l = set_friction_right(1)
        set_friction_r = set_friction_left(0)
        time.sleep(1)
        finger_state = 2
        # send_v = set_velocity(1, 10)

    theta = read_pos()
    for t2 in np.arange(theta[1], p, angle_diff):
        send_pos = command_position(1, t2)
        send_torque = command_torque(0, 0.10)
        time.sleep(sleep_time)

def slide_right_finger_down(p):
    global finger_state
    if finger_state != 3:
        set_friction_l = set_friction_right(1)
        set_friction_r = set_friction_left(1)

        modes = [0, 3]
        # set_modes = set_actuator_modes(2, modes)
        # send_pos = command_position(0, p)

        send_torque = command_torque(1, 0.10)
        time.sleep(0.5)
        set_friction_l = set_friction_right(0)
        set_friction_r = set_friction_left(1)
        time.sleep(1)
        finger_state = 3
        # send_v = set_velocity(1, 10)

    theta = read_pos()
    for t2 in np.arange(theta[1], p, angle_diff):
        send_pos = command_position(1, t2)
        send_torque = command_torque(0, 0.10)
        time.sleep(sleep_time)

def slide_right_finger_up(p):
    global finger_state
    if finger_state != 4:
        set_friction_l = set_friction_right(1)
        set_friction_r = set_friction_left(1)

        modes = [3, 0]
        set_modes = set_actuator_modes(2, modes)
        # send_pos = command_position(0, p)
        send_torque = command_torque(1, 0.10)
        time.sleep(0.5)
        set_friction_l = set_friction_right(0)
        set_friction_r = set_friction_left(1)
        time.sleep(1)
        finger_state = 4
        # send_v = set_velocity(0, 10)
    theta = read_pos()
    for t1 in np.arange(theta[0], p, angle_diff):
        send_pos = command_position(0, t1)
        send_torque = command_torque(1, 0.10)
        time.sleep(sleep_time)

def rotate_object_clockwise(p):
    global finger_state
    if finger_state != 5:
        set_friction_l = set_friction_right(1)
        set_friction_r = set_friction_left(1)

        modes = [3, 0]
        set_modes = set_actuator_modes(2, modes)
        send_torque = command_torque(1, 0.12)
        time.sleep(0.5)
        set_friction_l = set_friction_right(1)
        set_friction_r = set_friction_left(1)
        time.sleep(1)
        finger_state = 5

    theta = read_pos()
    for t1 in np.arange(theta[0], p, angle_diff):
        send_pos = command_position(0, t1)
        send_torque = command_torque(1, 0.12)
        time.sleep(sleep_time)

def rotate_object_anticlockwise(p):
    global finger_state
    if finger_state != 6:
        set_friction_l = set_friction_right(1)
        set_friction_r = set_friction_left(1)
        modes = [0, 3]
        set_modes = set_actuator_modes(2, modes)
        send_torque = command_torque(1, 0.12)
        time.sleep(0.5)
        set_modes = set_actuator_modes(2, modes)
        set_friction_l = set_friction_right(1)
        set_friction_r = set_friction_left(1)
        time.sleep(1)
        finger_state = 6

    theta = read_pos()
    for t2 in np.arange(theta[1], p, angle_diff):
        send_pos = command_position(1, t2)
        send_torque = command_torque(0, 0.12)
        time.sleep(sleep_time)

def execute_plan():

    list_of_lists = list()
    # with open("catkin_ws/src/FrictionFinger/Motion_planner/Test_results/data1.txt") as f:
    # with open("manipulation_plans/test_4/data1.txt") as f:
    with open("/home/asahin/alp/hardware_planning/data.txt") as f:
    # with open("/home/asahin/alp/hardware_planning/test_data.txt") as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            list_of_lists.append(inner_list)

    action_list = list()
    command_inst = list()
    # theta_l = list()
    # theta_r = list()

    for row in list_of_lists[1:]:
        action_list.append(row[1].strip('('))
        instructions = list()
        for i in range(2,len(row)):
            instructions.append(float(row[i].strip('[').strip(']').strip('(').strip(')').strip(']')))
        command_inst.append(instructions)
        # if row[7] == "None":
        #     theta_l.append(None)
        # else:
        #     theta_l.append(float(row[7]))
        # if row[8] == "None]":
        #     theta_r.append(None)
        # else:
        #     theta_r.append(float(row[8][:-1]))

    for step in range(0,len(action_list)):
        action = action_list[step]
        if step != len(action_list)-1:
            next_action = action_list[step+1]
            if action==next_action:
                continue
            # if action==next_action and not (action=="'Move down'" or action=="'Move up'"):
            #     continue
            # elif action==next_action and (action=="'Move down'" or action=="'Move up'"):
            #     command += 0.5
            #     continue

        if action=="'Slide left down'":
            command = angle_conversion(command_inst[step][0],'left')
            print("Left down: {}".format(command))
            slide_left_finger_down(command)
        elif action=="'Slide right down'":
            command = angle_conversion(command_inst[step][1],'right')
            print("Right down: {}".format(command))
            slide_right_finger_down(command)
        elif action=="'Slide left up'":
            command = angle_conversion(command_inst[step][1],'right')
            print("Left up: {}".format(command))
            slide_left_finger_up(command)
        elif action=="'Slide right up'":
            command = angle_conversion(command_inst[step][0],'left')
            print("Right up: {}".format(command))
            slide_right_finger_up(command)
        elif action=="'Rotate ccw'":
            command = angle_conversion(command_inst[step][1],'right')
            print("Anticlockwise: {}".format(command))
            rotate_object_anticlockwise(command)
        elif action=="'Rotate cw'":
            command = angle_conversion(command_inst[step][0],'left')
            print("Clockwise: {}".format(command))
            rotate_object_clockwise(command)
        elif action=="'Move up'":
            command = command_inst[step][2]
            print("Moving up: {}".format(command))
            command = command/100.0
            # move_up(command)
        elif action=="'Move down'":
            command = command_inst[step][2]
            print("Moving down: {}".format(command))
            command = command/100.0
            # move_down(command)
        elif action=="'Pivot'":
            theta_finger = command_inst[step][0]
            dx = command_inst[step][1]
            theta_pivot = command_inst[step][2]
            z = command_inst[step][3]
            print("Pivoting")
            # pivot(theta_finger,dx,theta_pivot,z)
        else:
            print("Error in action")
            break

        # if action=="'Slide left down'":
        #     command = angle_conversion(theta_l[step],'left')
        #     print("Left down: {}".format(command))
        #     slide_left_finger_down(command)
        # elif action=="'Slide right down'":
        #     command = angle_conversion(theta_r[step],'right')
        #     print("Right down: {}".format(command))
        #     slide_right_finger_down(command)
        # elif action=="'Slide left up'":
        #     command = angle_conversion(theta_r[step],'right')
        #     print("Left up: {}".format(command))
        #     slide_left_finger_up(command)
        # elif action=="'Slide right up'":
        #     command = angle_conversion(theta_l[step],'left')
        #     print("Right up: {}".format(command))
        #     slide_right_finger_up(command)
        # elif action=="'Rotate ccw'":
        #     command = angle_conversion(theta_r[step],'right')
        #     print("Anticlockwise: {}".format(command))
        #     rotate_object_anticlockwise(command)
        # elif action=="'Rotate cw'":
        #     command = angle_conversion(theta_l[step],'left')
        #     print("Clockwise: {}".format(command))
        #     rotate_object_clockwise(command)
        # elif action=="'Move up'":
        #     print("Moving up: {}".format(command))
        #     command = command/100.0
        #     move_up(command)
        #     command = 0
        # elif action=="'Move down'":
        #     print("Moving down: {}".format(command))
        #     command = command/100.0
        #     move_down(command)
        #     command = 0
        # elif action=="'Pivot'":
        #     print("Moving down: {}".format(command))
        #     command = command/100.0
        #     move_down(command)
        #     command = 0
        # else:
        #     print("Error in action")
        #     break

if __name__ == "__main__":
    rospy.init_node('control_loop')
    hold_object(angle_conversion(1.57,'left'),angle_conversion(1.57,'right'))
    # move_down(0.055)
    execute_plan()
    # time.sleep(5)
    release_object(angle_conversion(2.17,'left'),angle_conversion(0.97,'right'))
    # rospy.spin()
