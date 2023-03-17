import time
import math

import Queue

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#from mpmath import *
from sympy import *
import numpy as np

import scipy.optimize as opt
import matplotlib.pyplot as plt
from numpy import exp
import timeit

F1 = open('Test_results/Test_details.txt', 'w')
FINGER_END = 13
FINGER_START = 6.5
SLIDING_RESOLUTION = 0.1
ROTATION_SLIDE_RESOLUTION=0.2
PALM_WIDTH = 5
OBJECT_SIZE= 2.0
TH1_MAX= 2.485 #142.5 degrees
TH2_MIN= 0.65 #37.5
FINGER_WIDTH=1
K=0.1
NUMBER_OF_NODES_EXPANDED=0
PHI=3.14
W_S=1
R_S=1000
W_P=1
Test_no=0
t=0
current_start=0
current_goal=0
current_w_s=0
current_w_p=0
left_position=0
right_position=0
orientation_correction_done=False
start_l=0
start_r=0
OFFSET=0  #to correct the plotting during rotation  actions

def angle_conversion(angle, flag):
    if (flag == 1):
        n_angle = (angle * 180 / 3.14 - 21) * 0.003246 + 0.5626
        print("Gripper_angle=", angle * 180 / 3.14)

    else:
        n_angle = (angle * 180 / 3.14 - 152) * (-0.002113) + 0.3794
        print("Gripper_angle=", angle * 180 / 3.14)
    print("n_angle", n_angle)
    return (n_angle)


def ik_finger(x, y, w0, wp, fw):
    t2, d2 = symbols('t2 d2')
    eqn1 = (d2 - w0 / 2) * cos(t2) + (fw + w0 / 2) * sin(t2)
    eqn2 = (d2 - w0 / 2) * sin(t2) - (fw + w0 / 2) * cos(t2)
    eqn3 = x ** 2 + y ** 2 - eqn1 ** 2 - eqn2 ** 2
    sold2 = solve(eqn3, d2)
    solt2 = solve(eqn1.subs(d2, sold2[1]) - x, t2)

    d2v = np.array([sold2[1] * cos(solt2[1]), sold2[1] * sin(solt2[1])])
    w0v = np.array([w0 * sin(solt2[1]), -w0 * cos(solt2[1])])
    wpv = np.array([wp, 0])
    f1v = np.array([fw * sin(solt2[1]), -fw * cos(solt2[1])])
    av = d2v + f1v + w0v - wpv

    d1 = sqrt((av * av).sum() - fw * fw)
    t1 = np.arctan2(float(av[1]), float(av[0])) - np.arctan2(float(fw), float(d1))
    return t1, solt2[1], d1, sold2[1]

def calculate_th1(th2, d2):
    # Calculate theta2, d2
    d2v = np.array([d2 * np.cos(np.float64(th2)), d2 * np.sin(np.float64(th2))])
    w0v = np.array([OBJECT_SIZE * np.sin(np.float64(th2)), -OBJECT_SIZE * np.cos(np.float64(th2))])
    wpv = np.array([PALM_WIDTH, 0])
    f1v = np.array([FINGER_WIDTH * np.sin(np.float64(th2)), -FINGER_WIDTH * np.cos(np.float64(th2))])
    av = d2v - f1v - w0v + wpv

    d1 = np.sqrt((av * av).sum() - FINGER_WIDTH * FINGER_WIDTH)
    th1 = np.arctan2(av[1], av[0]) + np.arctan2(FINGER_WIDTH, d1)

    return th1

def calculate_th2(th1, d1):
    d1v = np.array([d1 * np.cos(th1), d1 * np.sin(th1)])
    w0v = np.array([OBJECT_SIZE * np.sin(th1), -OBJECT_SIZE * np.cos(th1)])
    wpv = np.array([PALM_WIDTH, 0])
    f2v = np.array([FINGER_WIDTH * np.sin(th1), -FINGER_WIDTH * np.cos(th1)])
    av = d1v + w0v + f2v - wpv

    d2 = np.sqrt((av * av).sum() - FINGER_WIDTH * FINGER_WIDTH)
    th2 = np.arctan2(av[1], av[0]) - np.arctan2(FINGER_WIDTH, d2)

    return th2

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    '''
    Python 2 implementation of Python 3.5 math.isclose()
    https://hg.python.org/cpython/file/tip/Modules/mathmodule.c#l1993
    '''
    # sanity check on the inputs
    if rel_tol < 0 or abs_tol < 0:
        raise ValueError("tolerances must be non-negative")

    # short circuit exact equality -- needed to catch two infinities of
    # the same sign. And perhaps speeds things up a bit sometimes.
    if a == b:
        return True

    # This catches the case of two infinities of opposite sign, or
    # one infinity and one finite number. Two infinities of opposite
    # sign would otherwise have an infinite relative tolerance.
    # Two infinities of the same sign are caught by the equality check
    # above.
    if math.isinf(a) or math.isinf(b):
        return False

    # now do the regular computation
    # this is essentially the "weak" test from the Boost library
    diff = math.fabs(b - a)
    result = (((diff <= math.fabs(rel_tol * b)) or
               (diff <= math.fabs(rel_tol * a))) or
              (diff <= abs_tol))
    return result


def action_right_equations(variables) :
    (th1,th2) = variables
    eqn1 = FINGER_WIDTH*sin(th1)+FINGER_WIDTH*sin(th2)+left_position * cos(th1) + OBJECT_SIZE * sin(th1) - PALM_WIDTH - right_position * cos(th2)
    eqn2 =-FINGER_WIDTH*cos(th1)-FINGER_WIDTH*cos(th2)+left_position * sin(th1) - OBJECT_SIZE * cos(th1) - right_position * sin(th2)
    return [eqn1, eqn2]

def action_left_equations(variables) :
    global left_position
    global right_position


    (th1, th2) = variables
    eqn1 = FINGER_WIDTH * sin(th1) + FINGER_WIDTH * sin(th2) + left_position * cos(th1) + OBJECT_SIZE * sin(th2) - PALM_WIDTH - right_position * cos(th2)
    eqn2 = -FINGER_WIDTH * cos(th1) - FINGER_WIDTH * cos(th2) + left_position * sin(th1) - OBJECT_SIZE * cos(th2) - right_position * sin(th2)
    return [eqn1, eqn2]

def theta_conversion(left, right, action_name):
    global left_position
    global right_position

    left_position =left
    right_position=right
    if (action_name == "r_plus" or action_name == "r_minus"):
        for i in range(31):
            initial_guess=(i/10.0,i/10.0)
            solution = opt.fsolve(action_right_equations, initial_guess, full_output=True)
            if solution[2]==1 and solution[0][0]>0 and solution[0][0]<3.14 and solution[0][1]<3.14 and solution[0][1]>0:
                return solution[0]

                #solution = opt.fsolve(action_right_equations, (0.1, 1.0))

        # print "right"
        # print "left",left_position,"right",right_position
        # #print solution
        return (None,None)
    elif (action_name == "l_plus" or action_name == "l_minus"):
        for i in range(31):
            initial_guess=(i/10.0,i/10.0)
            solution = opt.fsolve(action_left_equations, initial_guess, full_output=True)
            if solution[2] == 1 and solution[0][0] > 0 and solution[0][0] < 3.14 and solution[0][1] < 3.14 and solution[0][1] > 0:
                return solution[0]

                #solution = opt.fsolve(action_right_equations, (0.1, 1.0))

        # print "right"
        # print "left",left_position,"right",right_position
        # #print solution
        return (None,None)
    elif (action_name=="rotate_clockwise"):
        solution= np.pi - np.arccos((((right_position-OBJECT_SIZE)**2 + OBJECT_SIZE**2 - PALM_WIDTH**2 - (left_position + OBJECT_SIZE)**2)/(2*PALM_WIDTH*(left_position+OBJECT_SIZE))))
        print ("-----------------------------------------------")
        print (left_position,right_position,solution)
        return (solution)

    elif (action_name=="rotate_anticlockwise"):
        solution=np.arccos(((left_position - OBJECT_SIZE)**2 + OBJECT_SIZE**2 - (right_position+OBJECT_SIZE)**2 - PALM_WIDTH**2)/(2*PALM_WIDTH*(right_position + OBJECT_SIZE)))

        return (solution)




left_position=0
right_position=0
def limit_check(left_pos,right_pos,action):
    global left_position
    global right_position
    left_position=left_pos
    right_position=right_pos
    #print action

    if(action=="l_plus" or action=="l_minus" or action=="r_plus" or action=="r_minus"):
        if(left_position<FINGER_END and left_position>FINGER_START and right_position<FINGER_END and right_position>FINGER_START):
            sol=theta_conversion(left_position, right_position, action)
            TH2_MAX = calculate_th2(TH1_MAX, left_position)
            TH1_MIN = calculate_th1(TH2_MIN, right_position)
            th1=sol[0]
            th2 = sol[1]
            # print left_pos
            # print right_pos
            # print action
            # print "th2_max=",TH2_MAX
            # print "th1_min=",TH1_MIN
            # print "th1=",th1
            # print "th2=",th2

            # if(isclose(left_pos,9.7,rel_tol=1e-09, abs_tol=0.0)) and (isclose(right_pos, 11.3, rel_tol=1e-09, abs_tol=0.0)):
            #    print finger_to_cartesian(left_pos,right_pos,action,sol)


            if(th1<=TH1_MAX and th1>=TH1_MIN and th2>=TH2_MIN and th2<=TH2_MAX):
                return True
            else:
                print ("range_limit_exceeding for sliding")
                return False
        else:
             return False

    elif action=="rotate_clockwise":
        if (left_position+OBJECT_SIZE < FINGER_END and left_position+OBJECT_SIZE > FINGER_START and right_position-OBJECT_SIZE < FINGER_END and right_position-OBJECT_SIZE > FINGER_START):
            th1=theta_conversion(left_position, right_position, action)
            th2=calculate_th2(th1,left_position)
            TH2_MAX=calculate_th2(TH1_MAX,left_position)
            TH1_MIN=calculate_th1(TH2_MIN,right_position)

            print ("rotate_action_clock(th1)=",th1)

            if(th1<=TH1_MAX and th1>=TH1_MIN and th2>=TH2_MIN and th2<=TH2_MAX):
                return True
            else:
                    print ("range_limit_exceeding for rotate clockwise")
                    return False
        else:
            return False

    elif action=="rotate_anticlockwise":
        if (left_position - OBJECT_SIZE < FINGER_END and left_position - OBJECT_SIZE > FINGER_START and right_position + OBJECT_SIZE < FINGER_END and right_position + OBJECT_SIZE > FINGER_START):
            th2=theta_conversion(left_position, right_position, action)
            th1 = calculate_th1(th2, right_position)
            TH2_MAX = calculate_th2( TH1_MAX,left_position)
            TH1_MIN = calculate_th1(TH2_MIN,right_position)
            print ("rotate_action_anticlock(th2)=", th2)


            if(th2>=TH2_MIN  and th2<=TH2_MAX and th1<=TH1_MAX and th1>=TH1_MIN):
                return True
            else:
                    print ("range_limit_exceeding for rotate anticlockwise")
                    return False
        else:
            return False




class node:


    def __init__(self,res,pose_l,pose_r,pose_o,p,a):
        self.orientation = pose_o
        self.parent = p
        self.action = a
        self.g = 0
        self.h = 0
        self.f = 0
        self.theta=[0,0]



        if (a=="l_plus"):
            self.position_l = pose_l + res
            self.position_r=pose_r
            (self.theta)= theta_conversion(self.position_l, self.position_r, a)
        elif(a=="l_minus"):
            self.position_l = pose_l - res
            self.position_r = pose_r
            (self.theta) = theta_conversion(self.position_l, self.position_r, a)
        elif(a=="r_plus"):
            self.position_r = pose_r + res
            self.position_l = pose_l
            (self.theta) = theta_conversion(self.position_l, self.position_r, a)
        elif(a=="r_minus"):
            self.position_r = pose_r - res
            self.position_l = pose_l
            (self.theta) = theta_conversion(self.position_l, self.position_r, a)
        elif(a=="rotate_clockwise"):
            self.position_r = pose_r -OBJECT_SIZE
            self.position_l = pose_l +OBJECT_SIZE
            self.orientation= pose_o+90
            self.theta[0] = theta_conversion(pose_l, pose_r, a)
            self.theta[1] = calculate_th2(self.theta[0], pose_l)

           # (self.theta) = theta_conversion(self.position_l, self.position_r-0.1, "r_plus")
        elif(a=="rotate_anticlockwise"):
            self.position_r = pose_r + OBJECT_SIZE
            self.position_l = pose_l - OBJECT_SIZE
            self.orientation=pose_o-90
            self.theta[1] = theta_conversion(pose_l, pose_r, a)
            self.theta[0] = calculate_th1(self.theta[1], pose_r)
            #(self.theta) = theta_conversion(self.position_l, self.position_r-0.1, "r_plus")
        else:
            self.position_r = pose_r
            self.position_l = pose_l
            self.orientation= pose_o
            self.theta=None

    def update(self,g,goal_l,goal_r,goal_orientation,parent):

        if (abs(goal_orientation-self.orientation) and self.action=='rotate_anticlockwise' and self.action=='rotate_clockwise') :
            self.g = self.g + 90 +5
        else:
            self.g = g + 0.1;
        self.h=(abs(goal_l-self.position_l)+abs(goal_r-self.position_r))*W_S + abs(goal_orientation-self.orientation)

        if (self.action == parent):
             self.h = W_P*self.h ;

        self.f = self.g+self.h
        print (self.action,".h=",self.h)
        print ("actual_cost=",self.f)



    def find_neighbours(self, COST_T, COST_R,goal_l,goal_r,goal_orientation):
        global orientation_correction_done
        neighbours = []
        actions=("l_plus","l_minus","r_plus","r_minus","rotate_clockwise","rotate_anticlockwise")

        if (limit_check(self.position_l + SLIDING_RESOLUTION,self.position_r ,actions[0])):
            action="l_plus"
            l_plus=node(SLIDING_RESOLUTION,self.position_l,self.position_r,self.orientation,self,action)
            neighbours.append(l_plus)


        if (limit_check(self.position_l - SLIDING_RESOLUTION,self.position_r,actions[1])):


            action="l_minus"
            l_minus=node(SLIDING_RESOLUTION,self.position_l,self.position_r,self.orientation,self,action)

            neighbours.append(l_minus)

        if (limit_check(self.position_l,self.position_r - SLIDING_RESOLUTION,actions[3])):


            action = "r_minus"
            r_minus=node(SLIDING_RESOLUTION,self.position_l,self.position_r,self.orientation,self,action)

            neighbours.append(r_minus)

        if (limit_check(self.position_l,self.position_r + SLIDING_RESOLUTION,actions[2])):

            action = "r_plus"
            r_plus=node(SLIDING_RESOLUTION,self.position_l,self.position_r,self.orientation,self,action)
            neighbours.append(r_plus)


        if(abs(self.orientation-goal_orientation)):

            if (limit_check(self.position_l,self.position_r ,actions[4])):

                action = "rotate_clockwise"
                rotate_clockwise=node(SLIDING_RESOLUTION,self.position_l,self.position_r,self.orientation,self,action)
                neighbours.append(rotate_clockwise)

            if (limit_check(self.position_l,self.position_r ,actions[5])):

                action = "rotate_anticlockwise"
                rotate_anticlockwise=node(SLIDING_RESOLUTION,self.position_l,self.position_r,self.orientation,self,action)
                neighbours.append(rotate_anticlockwise)





        return neighbours


def A_star(start, goal):
    final_path = []
    global orientation_correction_done
    global NUMBER_OF_NODES_EXPANDED,t
    #print "Start and Goal state validation"
    # if limit_check(start.position_l, start.position_r, "r_plus") and  limit_check(start.position_l, start.position_r, "l_plus"):
    #     print"start_valid"
    # else:
    #     print"Invalid startstate"
    #     return None
    # if limit_check(goal.position_l, goal.position_r, "r_plus") and  limit_check(goal.position_l, goal.position_r, "l_plus"):
    #     print"Goal_valid"
    # else:
    #     print"Invalid Goalstate"
    #     return None

    #Record the start time
    start_time = time.time()

    # Priority queue to store the unexpanded nodes
    open_list=Queue.PriorityQueue()
    open_list.put((start.f,start))

    # List to store the expanded nodes
    closed_list=[]

    expanded=0

    while(1):

        cur = (open_list.get())[1]
        for exp_nodes in closed_list:
            if isclose(exp_nodes[0],cur.position_l, rel_tol=1e-09, abs_tol=0.0) and isclose(exp_nodes[1],cur.position_r, rel_tol=1e-09, abs_tol=0.0) and isclose(exp_nodes[2],cur.orientation, rel_tol=1e-09, abs_tol=0.0):
                #print "already in closed list"
                expanded=1
                break

        if(expanded):
            expanded=0
            continue
        #Debug statements
        print ("action=",cur.action)
        print ("l=",(cur.position_l))
        print ("r=",(cur.position_r))
        print ("total_cost=",cur.f)
        exp= [cur.position_l,cur.position_r,cur.orientation]
        closed_list.append(exp)
        NUMBER_OF_NODES_EXPANDED=NUMBER_OF_NODES_EXPANDED+1


        # if((isclose(cur.orientation, goal.orientation, rel_tol=1e-09, abs_tol=0.0)) and not orientation_correction_done):
        #     end_time = time.time()
        #
        #         # print (cur.position_l)
        #         # print (cur.position_r)
        #         # print (goal.position_l)
        #         # print (goal.position_r)
        #     print "Orientation correction done"
        #     print "Time_taken", end_time - start_time
        #     final_path=backtrace(cur)
        #     orientation_correction_done=True
        #     start = node(0, cur.position_l, cur.position_r, cur.orientation, None, None)
        #     open_list=Queue.PriorityQueue()
        #     open_list.put((start.f,start))
        #     closed_list=[]
            #return A_star(start, goal)

        if isclose(cur.position_l, goal.position_l, rel_tol=1e-09, abs_tol=0.0) and isclose(cur.position_r,
                                                                                                     goal.position_r,
                                                                                                     rel_tol=1e-09,
                                                                                                        abs_tol=0.0)and (isclose(cur.orientation, goal.orientation, rel_tol=1e-09, abs_tol=0.0)):
            end_time = time.time()
            print ("position correction done")
            t=end_time - start_time
            print ("Time_taken", t)
            final_path=final_path+(backtrace(cur))
            return final_path



        neighbours=cur.find_neighbours(1,0,goal.position_l,goal.position_r,goal.orientation)
        for nod in neighbours:
            flag=True
            for exp_nodes in closed_list:
                if(exp_nodes[0] == nod.position_l) and (exp_nodes[1] == nod.position_r) and exp_nodes[2]==nod.orientation:
                    flag=False

            if(flag):
                nod.update(cur.g,goal.position_l,goal.position_r,goal.orientation,cur.action)
                open_list.put((nod.f,nod))

        #cur = queue.pop(0)

        #res = backtrace(cur)






def backtrace(cur):
    global start_l,start_r,i,NUMBER_OF_NODES_EXPANDED,t,start,goal,W_S,W_P
    path = []
    R_position = []
    L_position = []
    theta = []
    while not cur.parent == None:
        path.append(cur.action)
        print ("node=", cur.action, "cost=", cur.f, "left=", cur.position_l, "right=", cur.position_r)
        L_position.append(cur.position_l)
        R_position.append(cur.position_r)
        theta.append(cur.theta)
        cur = cur.parent
    # Add the start point
    initial_theta = theta_conversion(start_l, start_r, 'r_plus')
    L_position.append(start_l)
    R_position.append(start_r)
    path.append('r_plus')
    theta.append(initial_theta)


    path.reverse()
    L_position.reverse()
    R_position.reverse()
    theta.reverse()
    SOLUTION_LENGTH=len(path)
    print ("Number of nodes expanded=", NUMBER_OF_NODES_EXPANDED)
    print("Length of solution=", SOLUTION_LENGTH)
    print (path)

    global F1

    data = [Test_no, current_start,current_goal,W_S,W_P,NUMBER_OF_NODES_EXPANDED,SOLUTION_LENGTH,t]
    F1.write(str(data) + "\n")

    # #writing the initial data
    Filename='Test_results/data'+str(Test_no)+'.txt'
    F = open(Filename, 'w')

    # initial_theta=theta_conversion(start_l, start_r,'r_plus')
    # x, y = finger_to_cartesian(start_l, start_r-0.1, 'r_plus', initial_theta)
    # data = [start_l,start_r, 3, initial_theta[0],initial_theta[1], x, y, 0]
    # F.write(str(data) + "\n")

    #Writing other data
    for i in range(len(L_position)):
        if (path[i] == 'l_minus'):
            x, y = finger_to_cartesian(L_position[i], R_position[i], path[i], theta[i])
            a = 0;
        if (path[i] == 'r_minus'):
            x, y = finger_to_cartesian(L_position[i], R_position[i], path[i], theta[i])
            a = 1;
        if (path[i] == 'l_plus'):
            x, y = finger_to_cartesian(L_position[i], R_position[i], path[i], theta[i])
            a = 2;
        if (path[i] == 'r_plus'):
            x, y = finger_to_cartesian(L_position[i], R_position[i], path[i], theta[i])
            a = 3;
        if (path[i] == 'rotate_clockwise'):
            a = 4;
            x, y = finger_to_cartesian(L_position[i], R_position[i], 'r_plus', theta[i])
        if (path[i] == 'rotate_anticlockwise'):
            x, y = finger_to_cartesian(L_position[i], R_position[i], 'l_plus', theta[i])
            a = 5;

        data = [L_position[i], R_position[i], a, theta[i][0], theta[i][1], x, y, 0]
        F.write(str(data) + "\n")

    F.close()
    plot(L_position, R_position, theta, path)

    return path

def finger_to_cartesian(L,R,A,th):
    if A=="r_plus" or A=="r_minus":
        x_square = (L - OBJECT_SIZE/2.0)*np.cos(np.float64(th[0])) + (FINGER_WIDTH + OBJECT_SIZE/2.0)*np.sin(np.float64(th[0]))
        # x_square = (R - (OBJECT_SIZE/2.0))
        y_square = (L - OBJECT_SIZE/2.0)*np.sin(np.float64(th[0])) - (FINGER_WIDTH + OBJECT_SIZE/2.0)*np.cos(np.float64(th[0]))


    elif A=="l_plus" or A=="l_minus":
        x_square = PALM_WIDTH + (R - OBJECT_SIZE/2.0)* np.cos(th[1]) - (OBJECT_SIZE/2.0 + FINGER_WIDTH)* np.sin(th[1])
        y_square = (R - OBJECT_SIZE/2.0)* np.sin(th[1]) + (OBJECT_SIZE/2.0 + FINGER_WIDTH)* np.cos(th[1])


    return x_square,y_square

def orientation_solvet1(t2,d1,d2,fw,w0,wp):
    t1 = symbols('t1')
    # eqn1 = np.array([[d1 * cos(t1)], [d1 * sin(t1)]]) + np.array([[fw * sin(t1)], [fw * cos(t1)]])
    eqn1 = np.array([[d1 * cos(t1)], [d1 * sin(t1)]]) + np.array([[fw * sin(t1)], [-fw * cos(t1)]]) + np.array([[fw * sin(t2)], [-fw * cos(t2)]]) - np.array([[(d2 + w0) * cos(t2) + wp], [(d2 + w0) * sin(t2)]])
    # print t2
    # eqn1 = sin(t1)

    a = simplify(np.dot(np.transpose(eqn1), eqn1))
    # print np.asscalar(simplify(np.dot(np.transpose(eqn1), eqn1)))
    eqn2 = a[0, 0] - 2 * (w0**2)
    solt1 = solve(eqn2, t1)
    t1 = solt1[1]
    ph = np.pi / 4 + np.arctan2(float((d2 + w0) * sin(t2) + fw * cos(t2) - (d1 * sin(t1) - fw * cos(t1))), float((d2 + w0) * cos(t2) - fw * sin(t2) + wp - (d1 * cos(t1) + fw * sin(t1))))
    return t1, ph


def orientation_solvet2(t1,d1,d2,fw,w0,wp):
    t2 = symbols('t2')
    eqn1 = np.array([[(d1 + w0) * cos(t1)], [(d1 + w0) * sin(t1)]]) + np.array([[fw * sin(t1)], [-fw * cos(t1)]]) + np.array([[fw * sin(t2)], [-fw * cos(t2)]]) - np.array([[(d2) * cos(t2) + wp], [(d2) * sin(t2)]])
    a = simplify(np.dot(np.transpose(eqn1), eqn1))
    eqn2 = a[0, 0] - 2 * (w0**2)
    solt2 = solve(eqn2, t2)
    return solt2[0]
    #ph = np.pi/4 + atan2()


def xy_rotation_clockwise(d1,d2,t1,t2,t1f,fw,w0,wp):
    print ("d1=", d1)
    print ("d2=", d2)
    # d1 = d1 - w0
    # d2 = d2 - w0

    # print ((d1 + w0)**2 + (w0 + 2 * fw)**2 - (d2 - w0)**2 - wp**2) / (2 * wp * (d2 - w0))
    #np.pi - np.arccos((((right_position - OBJECT_SIZE) ** 2 + OBJECT_SIZE ** 2 - PALM_WIDTH ** 2 - (
               # left_position + OBJECT_SIZE) ** 2) / (2 * PALM_WIDTH * (left_position + OBJECT_SIZE))))
    #t1f = np.pi-np.arccos(((d2-w0)**2 + w0**2 - wp**2 -(d1+w0)**2 / (2 * wp * (d1 + w0))))
    right_position=d2
    left_position=d1
    # t1f = np.pi - np.arccos((((right_position - OBJECT_SIZE) ** 2 + OBJECT_SIZE ** 2 - PALM_WIDTH ** 2 - (
    #             left_position + OBJECT_SIZE) ** 2) / (2 * PALM_WIDTH * (left_position + OBJECT_SIZE))))
    x=[]
    y=[]
    theta1=[]
    theta2=[]
    print (t1f)
    while t1 >= t1f:
        t2 = orientation_solvet2(t1,d1,d2,fw,w0,wp)
        t2 = float(t2)
        object_x = ((d1) * np.cos(t1) + fw * np.sin(t1) - fw * np.sin(t2) + (d2-w0) * np.cos(t2) + wp) / 2
        object_y = ((d1 ) * np.sin(t1) - fw * np.cos(t1) + fw * np.cos(t2) + (d2-w0) * np.sin(t2)) / 2
        t1 = t1 - 0.05
        x.append(object_x)
        y.append(object_y)
        theta1.append(t1)
        theta2.append(t2)
    return x,y,theta1,theta2

    #xf1 = wp + (d2 + w0) * cos


def xy_rotation_anticlockwise(d1,d2,t1,t2,t2f,fw,w0,wp):
    print ("d1=",d1)
    print ("d2=",d2)
    d1 = d1-w0
    d2 = d2-w0
    t2f = np.arccos(((d1 - w0)**2 + (w0 + 2 * fw)**2 - (d2 + w0)**2 - wp**2) / (2 * wp * (d2 + w0)))
    print (t2f)
    x=[]
    y=[]
    theta1 = []
    theta2 = []
    while t2 <= t2f:
        t1, ph = orientation_solvet1(t2,d1,d2,fw,w0,wp)
        t1 = float(t1)
        ph = float(ph)
        #object_x = (d1 * np.cos(t1) + fw * np.sin(t1) - fw * np.sin(t2) + (d2 + w0) * np.cos(t2) + wp) / 2
        #object_y = (d1 * np.sin(t1) - fw * np.cos(t1) + fw * np.cos(t2) + (d2 + w0) * np.sin(t2)) / 2
        object_x = (d1 * np.cos(t1) + fw * np.sin(t1) - fw * np.sin(t2) + (d2 + w0) * np.cos(t2) + wp) / 2
        object_y = (d1 * np.sin(t1) - fw * np.cos(t1) + fw * np.cos(t2) + (d2 + w0) * np.sin(t2)) / 2
        t2 = t2 + 0.05
        x.append(object_x)
        y.append(object_y)
        theta1.append(t1)
        theta2.append(t2)
    return x, y, theta1, theta2




def plot(L,R,theta,A):
    global current_w_p,current_w_s
    OFFSET=0
    n=len(R)
    X=[]
    Y=[]
    th1=[]
    th2=[]
    count=[]

    # print "size of l=",len(L)
    # print "size of R=", len(R)
    # print "size of A=", len(A)
    for i in range(n):
        print ("Action=", A[i])
        # if(A[i]=="rotate_clockwise" ):
        #     print "hello"
        #
        #     R[i]=R[i]-0.1
        #     L[i]=L[i]
        #     A[i]="r_plus"
        #     count.append(i)
        # elif(A[i]=="rotate_anticlockwise"):
        #     R[i] = R[i]-0.1
        #     L[i] = L[i]
        #     A[i] = "r_plus"
        #     count.append(i)
        if(A[i]=="rotate_clockwise" ):
            if i==0:
                j=i
            else:
                j=i-1

            x_r,y_r,t1,t2=xy_rotation_clockwise(L[i]-OBJECT_SIZE,R[i]+OBJECT_SIZE,theta[j][0],theta[j][1],theta[i][0],FINGER_WIDTH,OBJECT_SIZE,PALM_WIDTH)
            print ("------------------------------------------------------------------------------")
            print ("l=", L[i] + OBJECT_SIZE)
            print ("r=", R[i] - OBJECT_SIZE)
            print ("th1=", "th2=", theta[j][0], theta[j][1])
            print (x_r, y_r)
            X=X+x_r
            Y=Y+y_r
            th1=th1+t1
            th2 = th2 + t2
            print ("X rotation----------------------------------------------")
            print (X)
            print ("Y rotation----------------------------------------------")
            print (Y)
            count.append((i+OFFSET,len(x_r)))
            OFFSET += len(x_r)
        elif (A[i] == "rotate_anticlockwise"):
            if i == 0:   #if the rotate action is the first action
                j = i
            else:           #if the rotate actions is the last
                j = i - 1


            x_r, y_r,t1,t2 = xy_rotation_anticlockwise(L[i]+OBJECT_SIZE, R[i]-OBJECT_SIZE, theta[j][0], theta[j][1],theta[i][0],FINGER_WIDTH,OBJECT_SIZE,PALM_WIDTH)
            print ("------------------------------------------------------------------------------")
            print ("l=",L[i]+OBJECT_SIZE)
            print ("r=",R[i]-OBJECT_SIZE)
            print ("th1=","th2=",theta[j][0], theta[j][1])
            print (x_r, y_r)
            X = X + x_r
            Y = Y + y_r
            th1 = th1 + t1
            th2 = th2 + t2
            count.append((i+OFFSET, len(x_r)))
            OFFSET += len(x_r)
        else:
            x,y=finger_to_cartesian(L[i],R[i],A[i],theta[i])
            X.append(x)
            Y.append(y)
            th1.append(theta[i][0])
            th2.append(theta[i][1])

            print  ("L=",L[i],"R=",R[i])
            print ("theta1=",theta[i][0],"theta2=",theta[i][1])
            print ("x=",x,"y=",y)

    if(len(X)>0):
        # plotting the points
        plt.plot(X, Y,'b-',label='Sliding action')
        # for j in range(len(count)):
        #     print "X=", [X[count[j] - 1], X[count[j]]], "Y=", [Y[count[j] - 1], Y[count[j]]]
        #     plt.plot([X[count[j]-1],X[count[j]]],[Y[count[j]-1],Y[count[j]]],'r')
        for j in range(len(count)):
            print ("*********************************************************************************88")
            print (X[count[j][0]:count[j][0] + count[j][1]], Y[count[j][0]:count[j][0] + count[j][1]])
            plt.plot(X[count[j][0]:count[j][0]+count[j][1]],Y[count[j][0]:count[j][0]+count[j][1]],'r',label='Rotation action')


        plt.plot(X[len(X)-1],Y[len(Y)-1],'g*',label='Goal state',markersize=12)

        plt.plot(X[1],Y[1],'yo',label='Start state',markersize=12)

        plt.xlim([-10, 10])
        plt.ylim([0, 15])

        # naming the x axis
        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')

        # giving a title to my graph
        plt.title('Planned path for the object')
        plt.legend()

        textstr = '\n'.join((
            r'$\beta=%.1f$' % (current_w_p,),
            r'$\epsilon=%.1f$' % (current_w_s,)))
        props = dict( facecolor='white', alpha=0.5)
        plt.text(-9.5, 14.5, textstr, fontsize=14,
                verticalalignment='top', bbox=props)
        #Legend
        # # legend_elements = [Line2D([0], [0], color='b', lw=1, label='Sliding Action'),
        # #                    Line2D([0], [0], marker='o', color='w', label='Start',
        # #                           markerfacecolor='y', markersize=10),
        # #                    Line2D([0], [0], marker='*', color='w', label='Goal',
        # #                           markerfacecolor='g', markersize=10),
        #                   ]

        # legend_elements = [Line2D([0], [0], color='b', lw=4, label='Line'),
        #                    Line2D([0], [0], marker='o', color='w', label='Scatter',
        #                           markerfacecolor='g', markersize=15),
        #                    ]

        #plt.legend(handles=legend_elements)

        # function to show the plot
        Filename='Test_results/data'+str(Test_no)+'.png'
        plt.savefig(Filename)
        plt.clf()
        plt.close()





def high_level_plan(start, goal,w_s,w_p,i):
    #l = A_star(start, goal)
    global W_S, W_P, Test_no
    W_S=w_s
    W_P=w_p
    Test_no=i
    l= A_star(start, goal)


# Main function to run standalone
if __name__=="__main__":

    global current_start,current_goal,current_w_p,current_w_s,NUMBER_OF_NODES_EXPANDED
    start_list = [(7, 7, 0), (7, 7, 0), (7, 7, 0)]
    #start_list = [(7, 7, 0)]
    #goal_list = [(12, 12, 0)]
    goal_list = [(12.0, 12.0, 0), (8.5,11.3, -90), (11.3, 8.5, -90), ]
    #start_list=[(7,7.5,90)]

    #goal_list=[(7.5,10,0)]

    w_s_list = [1,2]

    w_p_list = [0.1]

    #start_list=[(8,7,0)]

    #goal_list=[(7.5,7.5,0)]

    #w_s_list=[1]

    #w_p_list=[1]

    T_N=0
    for start,goal in zip(start_list,goal_list):
        current_start=start
        current_goal=goal
        for w_s in w_s_list:
            for w_p in w_p_list:
                NUMBER_OF_NODES_EXPANDED=0
                T_N=T_N+1
                current_w_s=w_s
                current_w_p=w_p
                start_node = node(0, start[0], start[1], start[2], None, None)
                goal_node = node(0, goal[0],goal[1], goal[2], None, None)
                high_level_plan(start_node, goal_node, w_s,w_p,T_N)
    F1.close()

    # global start_l
    # global start_r
    #
    # start_l = 7
    # start_r = 7
    # start = node(0, start_l, start_r, 0, None, None)
    #
    # goal = node(0, 7.2,7.2, 0, None, None)
    #
    # high_level_plan(start, goal)


