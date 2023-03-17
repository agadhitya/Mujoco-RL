import time
import math

import Queue

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
OBJECT_SIZE= 1.0
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
OFFSET=0


# Vertices of the shape
X_vertices = np.array([OBJECT_SIZE/2, OBJECT_SIZE, OBJECT_SIZE/2, -OBJECT_SIZE/2, -OBJECT_SIZE, -OBJECT_SIZE])
Y_vertices = np.array([float(OBJECT_SIZE*sqrt(3)/2), 0, -float(OBJECT_SIZE*sqrt(3)/2), -float(OBJECT_SIZE*sqrt(3)/2), 0, float(OBJECT_SIZE*sqrt(3)/2)])
n = X_vertices.shape[0]
contact = 1

distances = np.array([])
x_d = np.array([])
y_d = np.array([])

gamma = np.array([])

for i in range(n-1):

	gamma = np.append(gamma, np.arctan2(Y_vertices[i+1]- Y_vertices[i], X_vertices[i+1]- X_vertices[i]))
	distances = np.append(distances, (X_vertices[i+1]*Y_vertices[i] - Y_vertices[i+1]*X_vertices[i])/sqrt((Y_vertices[i+1]-Y_vertices[i])**2 + (X_vertices[i+1]-X_vertices[i])**2))
	x_d = np.append(x_d, (Y_vertices[i]*X_vertices[i+1]-X_vertices[i]*Y_vertices[i+1])*(X_vertices[i+1]-X_vertices[i])*(Y_vertices[i+1]-Y_vertices[i])/((Y_vertices[i+1] - Y_vertices[i])**2 + (X_vertices[i+1]-X_vertices[i])**2))
	if(Y_vertices[i+1]-Y_vertices[i] != 0):
		y_d = np.append(y_d, (X_vertices[i+1] - X_vertices[i])*x_d[0]/(Y_vertices[i+1]-Y_vertices[i]))
	else:
		y_d = np.append(y_d, Y_vertices[i])

gamma = np.append(gamma, np.arctan2(Y_vertices[0]- Y_vertices[n-1], X_vertices[0]- X_vertices[n-1]))
distances = np.append(distances, (X_vertices[0]*Y_vertices[n-1] - Y_vertices[0]*X_vertices[n-1])/sqrt((Y_vertices[0]-Y_vertices[n-1])**2 + (X_vertices[0]-X_vertices[n-1])**2))
x_d = np.append(x_d, (Y_vertices[n-1]*X_vertices[0]-X_vertices[n-1]*Y_vertices[0])*(X_vertices[0]-X_vertices[n-1])*(Y_vertices[0]-Y_vertices[n-1])/((Y_vertices[0] - Y_vertices[n-1])**2 + (X_vertices[0]-X_vertices[n-1])**2))

if(Y_vertices[0]-Y_vertices[n-1] != 0):
	y_d = np.append(y_d, (X_vertices[0] - X_vertices[n-1])*x_d[0]/(Y_vertices[0]-Y_vertices[n-1]))
else:
	y_d = np.append(y_d, Y_vertices[0])


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):

    if rel_tol < 0 or abs_tol < 0:
        raise ValueError("tolerances must be non-negative")

    if a == b:
        return True

    if math.isinf(a) or math.isinf(b):
        return False

    diff = math.fabs(b - a)
    result = (((diff <= math.fabs(rel_tol * b)) or
               (diff <= math.fabs(rel_tol * a))) or
              (diff <= abs_tol))
    return result

def solve_t2_slide_right(variables, coords):
	solt2 = variables[0]
	eqn = coords[1]*cos(solt2) - (coords[0] - PALM_WIDTH)*sin(solt2) - FINGER_WIDTH
	return eqn

def calculate_th2(th1, d1):
    x_coord = (d1 + OBJECT_SIZE/2.0) * np.cos(th1) + (FINGER_WIDTH + OBJECT_SIZE*sqrt(3)/2) * np.sin(th1)
    y_coord = (d1 + OBJECT_SIZE/2.0) * np.sin(th1) - (FINGER_WIDTH + OBJECT_SIZE*sqrt(3)/2) * np.cos(th1)

    R = np.array([[cos(-gamma[0] + th1), -sin(-gamma[0] + th1), x_coord], [sin(-gamma[0] + th1), cos(-gamma[0] + th1), y_coord], [0, 0, 1]])
    coords = np.dot(R, np.concatenate(([X_vertices], [Y_vertices], np.ones((1, n)))))

    beta = 99999
    contact_right = -1

    for i in range(n):
        initial_guess_t2 = np.pi / 2
        solution = opt.fsolve(solve_t2_slide_right, initial_guess_t2, args=coords[:, i], full_output=True)
        if (solution[2] == 1 and solution[0] > 0 and solution[0] < np.pi and solution[0] < beta):
            beta = solution[0]
            contact_right = i

    th2 = beta[0]
    d2 = sqrt((coords[0, contact_right] - PALM_WIDTH) ** 2 + coords[1, contact_right] ** 2 - FINGER_WIDTH ** 2)

    return th2

def solve_t1_slide_left(variables, coords):
	solt2 = variables[0]
	eqn = coords[1]*cos(solt2) - coords[0]*sin(solt2) + FINGER_WIDTH
	return eqn

def calculate_th1(th2, d2):
    x_coord = PALM_WIDTH + (d2 + OBJECT_SIZE / 2.0) * np.cos(th2) - (sqrt(3) * OBJECT_SIZE / 2.0 + FINGER_WIDTH) * np.sin(th2)
    y_coord = (d2 + OBJECT_SIZE / 2.0) * np.sin(th2) + (sqrt(3) * OBJECT_SIZE / 2.0 + FINGER_WIDTH) * np.cos(th2)

    R = np.array([[cos(-gamma[0] + th2), -sin(-gamma[0] + th2), x_coord],
                  [sin(-gamma[0] + th2), cos(-gamma[0] + th2), y_coord],
                  [0, 0, 1]])
    coords = np.dot(R, np.concatenate(([X_vertices], [Y_vertices], np.ones((1, n)))))

    beta = -9999
    contact_left = -1

    for i in range(n):
        initial_guess_t1 = np.pi / 2
        solution = opt.fsolve(solve_t1_slide_left, initial_guess_t1, args=coords[:, i], full_output=True)
        if (solution[2] == 1 and solution[0] > 0 and solution[0] < np.pi and solution[0] > beta):
            beta = solution[0]
            contact_left = i

    t1 = beta[0]
    d1 = sqrt((coords[0, contact_left] - PALM_WIDTH) ** 2 + coords[1, contact_left] ** 2 - FINGER_WIDTH** 2)

    return t1

def limit_check(left_pos, right_pos, action):
    global left_position
    global right_position
    left_position = left_pos
    right_position = right_pos

    if (action == "l_plus" or action == "l_minus" or action == "r_plus" or action == "r_minus"):
        if (left_position < FINGER_END and left_position > FINGER_START and right_position < FINGER_END and right_position > FINGER_START):
            sol = theta_conversion(left_position, right_position, action)
            TH2_MAX = calculate_th2(TH1_MAX, left_position)
            TH1_MIN = calculate_th1(TH2_MIN, right_position)
            th1 = sol[0]
            th2 = sol[1]

            if (th1 <= TH1_MAX and th1 >= TH1_MIN and th2 >= TH2_MIN and th2 <= TH2_MAX):
                return True
            else:
                return False
        else:
            return False

    elif action=="rotate_clockwise":
        if (left_position+OBJECT_SIZE < FINGER_END and left_position+OBJECT_SIZE > FINGER_START and right_position-OBJECT_SIZE < FINGER_END and right_position-OBJECT_SIZE > FINGER_START):
            th1=theta_conversion(left_position, right_position, action)
            th2=calculate_th2(th1,left_position)
            TH2_MAX=calculate_th2(TH1_MAX,left_position)
            TH1_MIN=calculate_th1(TH2_MIN,right_position)

            if(th1<=TH1_MAX and th1>=TH1_MIN and th2>=TH2_MIN and th2<=TH2_MAX):
                return True
            else:
                return False
        else:
            return False

    elif action=="rotate_anticlockwise":
        if (left_position - OBJECT_SIZE < FINGER_END and left_position - OBJECT_SIZE > FINGER_START and right_position + OBJECT_SIZE < FINGER_END and right_position + OBJECT_SIZE > FINGER_START):
            th2=theta_conversion(left_position, right_position, action)
            th1 = calculate_th1(th2, right_position)
            TH2_MAX = calculate_th2( TH1_MAX,left_position)
            TH1_MIN = calculate_th1(TH2_MIN,right_position)

            if(th2>=TH2_MIN  and th2<=TH2_MAX and th1<=TH1_MAX and th1>=TH1_MIN):
                return True
            else:
                    return False
        else:
            return False


def orientation_solvet1(t2,d1,d2,fw,w0,wp):
    t1 = symbols('t1')
    eqn1 = np.array([[d1 * cos(t1)], [d1 * sin(t1)]]) + np.array([[fw * sin(t1)], [-fw * cos(t1)]]) + np.array([[fw * sin(t2)], [-fw * cos(t2)]]) - np.array([[(d2 + w0) * cos(t2) + wp], [(d2 + w0) * sin(t2)]])
    a = simplify(np.dot(np.transpose(eqn1), eqn1))
    eqn2 = a[0, 0] - 2 * (w0 ** 2)
    solt1 = solve(eqn2, t1)
    t1 = solt1[1]
    return t1

def orientation_solvet2(t1, d1, d2, fw, w0, wp):
    t2 = symbols('t2')
    eqn1 = np.array([[(d1 + w0) * cos(t1)], [(d1 + w0) * sin(t1)]]) + np.array([[fw * sin(t1)], [-fw * cos(t1)]]) + np.array([[fw * sin(t2)], [-fw * cos(t2)]]) - np.array([[(d2) * cos(t2) + wp], [(d2) * sin(t2)]])
    a = simplify(np.dot(np.transpose(eqn1), eqn1))
    eqn2 = a[0, 0] - 4*(w0 ** 2)
    solt2 = solve(eqn2, t2)
    return solt2[0]

def xy_rotation_clockwise(d1, d2, t1, t2, t1f, fw, w0, wp):
    x = []
    y = []
    theta1 = []
    theta2 = []

    while t1 >= t1f:
        t2 = float(orientation_solvet2(t1, d1, d2, fw, w0, wp))
        object_x = ((d1+w0) * np.cos(t1) + fw * np.sin(t1) - fw * np.sin(t2) + d2 * cos(t2) + wp)/2
        object_y = ((d1+w0) * np.sin(t1) - fw* np.cos(t1) + fw*np.cos(t2) + d2 * sin(t2))/2
        t1 = t1 - 0.05

        x.append(object_x)
        y.append(object_y)
        theta1.append(t1)
        theta2.append(t2)

    return x, y, theta1, theta2

def xy_rotation_anticlockwise(d1, d2, t1, t2, t2f, fw, w0, wp):
    t2f = np.arccos(((d1 - w0)**2 + (w0 + 2 * fw)**2 - (d2 + w0)**2 - wp**2) / (2 * wp * (d2 + w0)))
    x = []
    y = []
    theta1 = []
    theta2 = []
    while t2 <= t2f:
        t1 = orientation_solvet1(t2, d1, d2, fw, w0, wp)
        object_x = (d1 * np.cos(t1) + fw * np.sin(t1) - fw * np.sin(t2) + (d2 + w0) * np.cos(t2) + wp) / 2
        object_y = (d1 * np.sin(t1) - fw * np.cos(t1) + fw * np.cos(t2) + (d2 + w0) * np.sin(t2)) / 2
        t2 = t2 + 0.05
        x.append(object_x)
        y.append(object_y)
        theta1.append(t1)
        theta2.append(t2)
    return x, y, theta1, theta2

def action_right_equations(variables):
    global left_position
    global right_position

    (th1, th2) = variables

    eqn1 = left_position*cos(th1) + (FINGER_WIDTH + 2*OBJECT_SIZE*cos(np.pi/6))*sin(th1) + (FINGER_WIDTH)*sin(th2) - right_position*cos(th2) - PALM_WIDTH
    eqn2 = left_position*sin(th1) - (FINGER_WIDTH + 2*OBJECT_SIZE*cos(np.pi/6))*cos(th1) - FINGER_WIDTH*cos(th2) - right_position*sin(th2)
    return [eqn1, eqn2]

def action_left_equations(variables) :
    global left_position
    global right_position

    (th1, th2) = variables

    eqn1 = right_position * cos(th2) - (FINGER_WIDTH + 2*OBJECT_SIZE*cos(np.pi/6)) * sin(th2) - FINGER_WIDTH * sin(th2) - left_position * cos(th1) + PALM_WIDTH
    eqn2 = right_position * sin(th2) + (FINGER_WIDTH + 2*OBJECT_SIZE*cos(np.pi/6)) * cos(th2) + FINGER_WIDTH * cos(th1) - left_position * sin(th1)
    return [eqn1, eqn2]

def theta_conversion(left, right, action_name):
    global left_position
    global right_position

    left_position = left
    right_position = right

    if (action_name == "r_plus" or action_name == "r_minus"):

        for i in range(31):
            initial_guess=(i/10.0,i/10.0)
            solution = opt.fsolve(action_right_equations, initial_guess, full_output=True)
            if solution[2]==1 and solution[0][0]>0 and solution[0][0]<3.14 and solution[0][1]<3.14 and solution[0][1]>0:
                print 'solution', solution[0]
                return solution[0]

        return (None, None)

    elif (action_name == "l_plus" or action_name == "l_minus"):

        for i in range(31):
            initial_guess=(i/10.0,i/10.0)
            solution = opt.fsolve(action_left_equations, initial_guess, full_output=True)
            if solution[2] == 1 and solution[0][0] > 0 and solution[0][0] < 3.14 and solution[0][1] < 3.14 and solution[0][1] > 0:
                return solution[0]

        return (None,None)

    elif (action_name=="rotate_clockwise"):
        solution= np.pi - np.arccos((((right_position-OBJECT_SIZE)**2 + OBJECT_SIZE**2 - PALM_WIDTH**2 - (left_position + OBJECT_SIZE)**2)/(2*PALM_WIDTH*(left_position+OBJECT_SIZE))))
        return (solution)

    elif (action_name=="rotate_anticlockwise"):
        solution=np.arccos(((left_position - OBJECT_SIZE)**2 + OBJECT_SIZE**2 - (right_position+OBJECT_SIZE)**2 - PALM_WIDTH**2)/(2*PALM_WIDTH*(right_position + OBJECT_SIZE)))
        return (solution)

def plot(L,R,theta,A):
    global current_w_p,current_w_s
    OFFSET=0
    n=len(R)
    X=[]
    Y=[]
    th1=[]
    th2=[]
    count=[]

    for i in range(n):

        if(A[i]=="rotate_clockwise" ):
            if i==0:
                j=i
            else:
                j=i-1

            x_r,y_r,t1,t2=xy_rotation_clockwise(L[i]-OBJECT_SIZE,R[i]+OBJECT_SIZE,theta[j][0],theta[j][1],theta[i][0],FINGER_WIDTH,OBJECT_SIZE,PALM_WIDTH)

            X=X+x_r
            Y=Y+y_r
            th1=th1+t1
            th2 = th2 + t2

            count.append((i+OFFSET,len(x_r)))
            OFFSET += len(x_r)
        elif (A[i] == "rotate_anticlockwise"):
            if i == 0:   #if the rotate action is the first action
                j = i
            else:           #if the rotate actions is the last
                j = i - 1


            x_r, y_r,t1,t2 = xy_rotation_anticlockwise(L[i]+OBJECT_SIZE, R[i]-OBJECT_SIZE, theta[j][0], theta[j][1],theta[i][0],FINGER_WIDTH,OBJECT_SIZE,PALM_WIDTH)

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

    if(len(X)>0):
        plt.plot(X, Y,'b-',label='Sliding action')

        for j in range(len(count)):
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


        # function to show the plot
        Filename='Test_results/data'+str(Test_no)+'.png'
        plt.savefig(Filename)
        plt.clf()
        plt.close()

class node:

    def __init__(self, res, pose_l, pose_r, pose_o, p, a):
        self.orientation = pose_o
        self.parent = p
        self.action = a

        self.g = 0
        self.h = 0
        self.f = 0

        self.theta = [0, 0]

        if(a == 'l_plus'):
            self.position_l = pose_l + res
            self.position_r = pose_r
            (self.theta) = theta_conversion(self.position_l, self.position_r, a)

        elif (a == "l_minus"):
            self.position_l = pose_l - res
            self.position_r = pose_r
            (self.theta) = theta_conversion(self.position_l, self.position_r, a)

        elif (a == "r_plus"):
            self.position_r = pose_r + res
            self.position_l = pose_l
            (self.theta) = theta_conversion(self.position_l, self.position_r, a)

        elif (a == "r_minus"):
            self.position_r = pose_r - res
            self.position_l = pose_l
            (self.theta) = theta_conversion(self.position_l, self.position_r, a)

        elif (a == "rotate_clockwise"):
            self.position_r = pose_r - OBJECT_SIZE
            self.position_l = pose_l + OBJECT_SIZE
            self.orientation = pose_o + 90
            self.theta[0] = theta_conversion(pose_l, pose_r, a)
            self.theta[1] = calculate_th2(self.theta[0], pose_l)

        elif (a == "rotate_anticlockwise"):
            self.position_r = pose_r + OBJECT_SIZE
            self.position_l = pose_l - OBJECT_SIZE
            self.orientation = pose_o - 90
            self.theta[1] = theta_conversion(pose_l, pose_r, a)
            self.theta[0] = calculate_th1(self.theta[1], pose_r)

        else:
            self.position_r = pose_r
            self.position_l = pose_l
            self.orientation= pose_o
            self.theta=None

    def update(self, g, goal_l, goal_r, goal_orientation, parent):

        if (abs(goal_orientation - self.orientation) and self.action == 'rotate_anticlockwise' and self.action == 'rotate_clockwise'):
            self.g = self.g + 90 + 5
        else:
            self.g = g + 0.1;
        self.h = (abs(goal_l - self.position_l) + abs(goal_r - self.position_r)) * W_S + abs(goal_orientation - self.orientation)

        if (self.action == parent):
            self.h = W_P * self.h;

        self.f = self.g + self.h
        print(self.action, ".h=", self.h)
        print("actual_cost=", self.f)


    def find_neighbors(self, COST_T, COST_R, goal_l, goal_r, goal_orientation):
        global orientation_correction_done
        neighbors = []
        actions = ("l_plus", "l_minus", "r_plus", "r_minus", "rotate_clockwise", "rotate_anticlockwise")

        if (limit_check(self.position_l + SLIDING_RESOLUTION, self.position_r, actions[0])):
            action = "l_plus"
            l_plus = node(SLIDING_RESOLUTION, self.position_l, self.position_r, self.orientation, self, action)
            neighbors.append(l_plus)

        if (limit_check(self.position_l - SLIDING_RESOLUTION, self.position_r, actions[1])):
            action = "l_minus"
            l_minus = node(SLIDING_RESOLUTION, self.position_l, self.position_r, self.orientation, self, action)

            neighbors.append(l_minus)

        if (limit_check(self.position_l, self.position_r - SLIDING_RESOLUTION, actions[3])):
            action = "r_minus"
            r_minus = node(SLIDING_RESOLUTION, self.position_l, self.position_r, self.orientation, self, action)

            neighbors.append(r_minus)

        if (limit_check(self.position_l, self.position_r + SLIDING_RESOLUTION, actions[2])):
            action = "r_plus"
            r_plus = node(SLIDING_RESOLUTION, self.position_l, self.position_r, self.orientation, self, action)
            neighbors.append(r_plus)

        if (abs(self.orientation - goal_orientation)):

            if (limit_check(self.position_l, self.position_r, actions[4])):
                action = "rotate_clockwise"
                rotate_clockwise = node(SLIDING_RESOLUTION, self.position_l, self.position_r, self.orientation, self,
                                        action)
                neighbors.append(rotate_clockwise)

            if (limit_check(self.position_l, self.position_r, actions[5])):
                action = "rotate_anticlockwise"
                rotate_anticlockwise = node(SLIDING_RESOLUTION, self.position_l, self.position_r, self.orientation,
                                            self, action)
                neighbors.append(rotate_anticlockwise)

        return neighbors

def A_star(start, goal):
    final_path = []
    global orientation_correction_done
    global NUMBER_OF_NODES_EXPANDED, t

    open_list = Queue.PriorityQueue()
    open_list.put((start.f, start))

    closed_list = []

    expanded = 0

    while(1):

        cur = (open_list.get())[1]
        for exp_nodes in closed_list:
            if isclose(exp_nodes[0], cur.position_l, rel_tol = 1e-9, abs_tol = 0.0) and isclose(
                    exp_nodes[1], cur.position_r, rel_tol = 1e-9) and isclose(
                exp_nodes[2, cur.orientation], rel_tol = 1e-9, abs_tol = 0.0):
                expanded = 1
                break

        if (expanded):
            expanded = 0
            continue

        exp = [cur.position_l, cur.position_r, cur.orientation]
        closed_list.append(exp)
        NUMBER_OF_NODES_EXPANDED = NUMBER_OF_NODES_EXPANDED + 1

        if isclose(cur.position_l, goal.position_l, rel_tol = 1e-9) and isclose(
                cur.position_r, goal.position_r, rel_tol = 1e-9, abs_tol = 0.0) and (isclose(
            cur.orientation, goal.orientation, rel_tol = 1e-9, abs_tol = 0.0)):
            final_path = final_path + (backtrace(cur))
            return final_path

        neighbors = cur.find_neighbors(1, 0, goal.position_l, goal.position_r, goal.orientation)
        for nod in neighbors:
            flag = True
            for exp_nodes in closed_list:
                if(exp_nodes[0] == nod.position_l) and (exp_nodes[1] == nod.position_r) and (exp_nodes[2] == nod.orientation):
                    flag = False

            if(flag):
                nod.update(cur.g, goal.position_l, goal.position_r, goal.orientation, cur.action)
                open_list.put((nod.f, nod))

def backtrace(cur):
    global start_l, start_r, i, NUMBER_OF_NODES_EXPANDED, t, start, goal, W_S, W_P
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

    initial_theta = theta_conversion(start_l, start_r, 'r_plus')
    L_position.append(start_l)
    R_position.append(start_r)
    path.append('r_plus')
    theta.append(initial_theta)

    path.reverse()
    L_position.reverse()
    R_position.reverse()
    theta.reverse()
    SOLUTION_LENGTH = len(path)

    global F1

    data = [Test_no, current_start,current_goal,W_S,W_P,NUMBER_OF_NODES_EXPANDED,SOLUTION_LENGTH,t]
    F1.write(str(data) + "\n")

    Filename='Test_results/data'+str(Test_no)+'.txt'
    F = open(Filename, 'w')

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

def finger_to_cartesian(L, R, A, th):
    if (A == 'r_plus' or A == 'r_minus'):
        x_square = (L + OBJECT_SIZE/2.0) * np.cos(np.float64(th[0])) + (FINGER_WIDTH + OBJECT_SIZE*sqrt(3)/2) * np.sin(np.float64(th[0]))
        y_square = (L + OBJECT_SIZE/2.0) * np.sin(np.float64(th[0])) - (FINGER_WIDTH + OBJECT_SIZE*sqrt(3)/2) * np.cos(np.float64(th[0]))

    elif(A == 'l_plus' or A == 'l_minus'):
        x_square = PALM_WIDTH + (R + OBJECT_SIZE/2.0) * np.cos(th[1]) - (sqrt(3)*OBJECT_SIZE/2.0 + FINGER_WIDTH)* np.sin(th[1])
        y_square = (R + OBJECT_SIZE/2.0) * np.sin(th[1]) + (sqrt(3)*OBJECT_SIZE/2.0 + FINGER_WIDTH)* np.cos(th[1])

    return x_square, y_square

def high_level_plan(start, goal, w_s, w_p, i):
    global W_S, W_P, Test_no
    W_S = w_s
    W_P = w_p
    Test_no = i
    l = A_star(start, goal)

# Main function to run standalone
if __name__=="__main__":

    global current_start,current_goal,current_w_p,current_w_s,NUMBER_OF_NODES_EXPANDED

    start_list = [(7.0, 7.0, -90)]
    goal_list = [(7.9, 7.2, 90)]

    w_s_list = [2]
    w_p_list = [0.1]

    T_N = 0
    for start,goal in zip(start_list,goal_list):
        current_start=start
        current_goal=goal
        for w_s in w_s_list:
            for w_p in w_p_list:
                NUMBER_OF_NODES_EXPANDED=0
                T_N=T_N+1
                current_w_s = w_s
                current_w_p = w_p
                start_node = node(0, start[0], start[1], start[2], None, None)
                goal_node = node(0, goal[0], goal[1], goal[2], None, None)
                high_level_plan(start_node, goal_node, w_s, w_p, T_N)

    F1.close()
