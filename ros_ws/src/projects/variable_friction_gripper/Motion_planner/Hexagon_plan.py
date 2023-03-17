

def limit_check(left_pos, right_pos, contact, action):
    global left_position
    global right_position

    left_position = left_pos
    right_position = right_pos

    if (action == "l_plus" or action == "l_minus" or action == "r_plus" or action == "r_minus"):
        if (left_position < FINGER_END and left_position > FINGER_START and right_position < FINGER_END and right_position > FINGER_START):
            sol = theta_conversion(left_position, right_position, contact, action)
            TH2_MAX = calculate_th2(TH1_MAX, left_position, contact)
            TH1_MIN = calculate_th1(TH2_MIN, right_position, contact)
            th1 = sol[0]
            th2 = sol[1]

            if (th1 <= TH1_MAX and th1 >= TH1_MIN and th2 >= TH2_MIN and th2 <= TH2_MAX):
                return True
            else:
                print("range_limit_exceeding for sliding")
                return False
        else:
            return False

    elif action=="rotate_clockwise":
        if(np.mod(contact, 2) == 0):
            if (left_position + OBJECT_BREADTH < FINGER_END and left_position + OBJECT_BREADTH > FINGER_START and right_position - OBJECT_LENGTH < FINGER_END and right_position - OBJECT_LENGTH > FINGER_START):
                th1 = theta_conversion(left_position, right_position, contact, action)
                th2 = calculate_th2(th1, left_position, contact)
                TH2_MAX = calculate_th2(TH1_MAX, left_position, contact)
                TH1_MIN = calculate_th1(TH2_MIN, right_position, contact)

                if(th1<=TH1_MAX and th1>=TH1_MIN and th2>=TH2_MIN and th2<=TH2_MAX):
                    return True
                else:
                        print ("range_limit_exceeding for rotate clockwise")
                        return False

            else:
                return False

        else:
            if (left_position + OBJECT_LENGTH < FINGER_END and left_position + OBJECT_LENGTH > FINGER_START and right_position - OBJECT_BREADTH < FINGER_END and right_position - OBJECT_BREADTH > FINGER_START):
                th1 = theta_conversion(left_position, right_position, contact, action)
                th2 = calculate_th2(th1, left_position, contact)
                TH2_MAX = calculate_th2(TH1_MAX, left_position, contact)
                TH1_MIN = calculate_th1(TH2_MIN, right_position, contact)

                if(th1<=TH1_MAX and th1>=TH1_MIN and th2>=TH2_MIN and th2<=TH2_MAX):
                    return True
                else:
                        print ("range_limit_exceeding for rotate clockwise")
                        return False
            else:
                return False

    elif action=="rotate_anticlockwise":
        if (np.mod(contact, 2) == 0):
            if (left_position - OBJECT_LENGTH < FINGER_END and left_position - OBJECT_LENGTH > FINGER_START and right_position + OBJECT_BREADTH < FINGER_END and right_position + OBJECT_BREADTH > FINGER_START):
                th2=theta_conversion(left_position, right_position, contact, action)
                th1 = calculate_th1(th2, right_position, contact)
                TH2_MAX = calculate_th2(TH1_MAX, left_position, contact)
                TH1_MIN = calculate_th1(TH2_MIN, right_position, contact)

                if(th2>=TH2_MIN  and th2<=TH2_MAX and th1<=TH1_MAX and th1>=TH1_MIN):
                    return True
                else:
                        print ("range_limit_exceeding for rotate anticlockwise")
                        return False
            else:
                return False
        else:
            if (left_position - OBJECT_BREADTH < FINGER_END and left_position - OBJECT_BREADTH > FINGER_START and right_position + OBJECT_LENGTH < FINGER_END and right_position + OBJECT_LENGTH > FINGER_START):
                th2=theta_conversion(left_position, right_position, contact, action)
                th1 = calculate_th1(th2, right_position, contact)
                TH2_MAX = calculate_th2(TH1_MAX, left_position, contact)
                TH1_MIN = calculate_th1(TH2_MIN, right_position, contact)

                if(th2>=TH2_MIN  and th2<=TH2_MAX and th1<=TH1_MAX and th1>=TH1_MIN):
                    return True
                else:
                        print ("range_limit_exceeding for rotate anticlockwise")
                        return False
            else:
                return False

def action_left_equations(variables, contact):
    global left_position
    global right_position

    (th1, th2) = variables
    if np.mod(contact, 2) == 0:
        eqn1 = FINGER_WIDTH * sin(th1) + FINGER_WIDTH * sin(th2) + left_position * cos(th1) + OBJECT_LENGTH * sin(th2) - PALM_WIDTH - right_position * cos(th2)
        eqn2 = -FINGER_WIDTH * cos(th1) - FINGER_WIDTH * cos(th2) + left_position * sin(th1) - OBJECT_LENGTH * cos(th2) - right_position * sin(th2)
    else:
        eqn1 = FINGER_WIDTH * sin(th1) + FINGER_WIDTH * sin(th2) + left_position * cos(th1) + OBJECT_BREADTH * sin(th2) - PALM_WIDTH - right_position * cos(th2)
        eqn2 = -FINGER_WIDTH * cos(th1) - FINGER_WIDTH * cos(th2) + left_position * sin(th1) - OBJECT_BREADTH * cos(th2) - right_position * sin(th2)

def action_right_equations(variables, contact):
    global left_position
    global right_position

    (th1, th2) = variables
    if np.mod(contact, 2) == 0:
        eqn1 = FINGER_WIDTH*sin(th1) + FINGER_WIDTH*sin(th2) + left_position*cos(th1) + OBJECT_LENGTH * sin(th1) - PALM_WIDTH - right_position * cos(th2)
        eqn2 = -FINGER_WIDTH*cos(th1) + FINGER_WIDTH*cos(th2) + left_position*sin(th1) - OBJECT_LENGTH * cos(th1) - right_position * sin(th2)
    else:
        eqn1 = FINGER_WIDTH * sin(th1) + FINGER_WIDTH * sin(th2) + left_position * cos(th1) + OBJECT_BREADTH * sin(th1) - PALM_WIDTH - right_position * cos(th2)
        eqn2 = -FINGER_WIDTH * cos(th1) + FINGER_WIDTH * cos(th2) + left_position * sin(th1) - OBJECT_BREADTH * cos(th1) - right_position * sin(th2)

    return [eqn1, eqn2]

def theta_conversion(left, right, contact, action_name):
    global left_position
    global right_position

    left_position = left
    right_position = right

    if (action_name == 'r_plus' or action_name == 'r_minus'):
        for i in range(31):
            initial_guess = (i / 10.0, i / 10.0)
            solution = opt.fsolve(action_right_equations, initial_guess, args = contact, full_output=True)
            if solution[2] == 1 and solution[0][0] > 0 and solution[0][0] < 3.14 and solution[0][1] < 3.14 and solution[0][
                1] > 0:
                return solution[0]

        return (None, None)

    elif(action_name == 'l_plus' or action_name == 'l_minus'):
        for i in range(31):
            initial_guess = (i/10.0, i/10.0)
            solution = opt.fsolve(action_left_equations, initial_guess, args = contact, full_output = True)
            if solution[2] == 1 and solution[0][0] > 0 and solution[0][0] < 3.14 and solution[0][1] < 3.14 and solution[0][1] > 0:
                return solution[0]

        return (None, None)

    elif(action_name == 'rotate_clockwise'):
        if np.mod(contact, 2) == 0:
            # solution = np.pi - np.arccos((((right_position-OBJECT_LENGTH)**2 + OBJECT_LENGTH**2 - PALM_WIDTH**2 - (left_position + OBJECT_BREADTH)**2)/(2*PALM_WIDTH*(left_position + OBJECT_BREADTH))))
            solution = np.pi - np.arccos((((right_position - OBJECT_SIZE) ** 2 + OBJECT_SIZE ** 2 - PALM_WIDTH ** 2 - (left_position + OBJECT_SIZE) ** 2) / (2 * PALM_WIDTH * (left_position + OBJECT_SIZE))))
        else:
            # solution = np.pi - np.arccos((((right_position - OBJECT_BREADTH) ** 2 + OBJECT_BREADTH ** 2 - PALM_WIDTH ** 2 - (left_position + OBJECT_LENGTH) ** 2) / (2 * PALM_WIDTH * (left_position + OBJECT_LENGTH))))
            solution = np.pi - np.arccos((((right_position - OBJECT_SIZE) ** 2 + OBJECT_SIZE ** 2 - PALM_WIDTH ** 2 - (left_position + OBJECT_SIZE) ** 2) / (2 * PALM_WIDTH * (left_position + OBJECT_SIZE))))
        return (solution)

    elif (action_name == 'rotate anticlockwise'):
        if np.mod(contact, 2) == 0:
            solution = np.arccos(((left_position - OBJECT_SIZE) ** 2 + OBJECT_SIZE ** 2 - (right_position + OBJECT_SIZE) ** 2 - PALM_WIDTH ** 2) / (2 * PALM_WIDTH * (right_position + OBJECT_SIZE)))
            # solution = np.arccos(((left_position - OBJECT_LENGTH)**2 + OBJECT_LENGTH**2 - (right_position+OBJECT_BREADTH)**2 - PALM_WIDTH**2)/(2*PALM_WIDTH*(right_position + OBJECT_BREADTH)))
        else:
            solution = np.arccos(((left_position - OBJECT_SIZE) ** 2 + OBJECT_SIZE ** 2 - (right_position + OBJECT_SIZE) ** 2 - PALM_WIDTH ** 2) / (2 * PALM_WIDTH * (right_position + OBJECT_SIZE)))
            # solution = np.arccos(((left_position - OBJECT_BREADTH)**2 + OBJECT_BREADTH**2 - (right_position + OBJECT_LENGTH)**2 - PALM_WIDTH**2)/(2*PALM_WIDTH*(right_position + OBJECT_LENGTH)))
        return (solution)

def calculate_th1(th1, pose_r, contact):

class node:

    def __init__(self, res, pose_l, pose_r, pose_o, p, a):
        self.orientation = pose_o
        self.parent = p
        self.action = a
        self.g = 0
        self.h = 0
        self.f = 0
        self.theta = [0, 0]
        self.contact = 0

        if (a=="l_plus"):
            self.position_l = pose_l + res
            self.position_r = pose_r
            (self.theta)= theta_conversion(self.position_l, self.position_r, self.contact, a)
        elif(a=="l_minus"):
            self.position_l = pose_l - res
            self.position_r = pose_r
            (self.theta) = theta_conversion(self.position_l, self.position_r, self.contact, a)
        elif(a=="r_plus"):
            self.position_r = pose_r + res
            self.position_l = pose_l
            (self.theta) = theta_conversion(self.position_l, self.position_r, self.contact, a)
        elif(a=="r_minus"):
            self.position_r = pose_r - res
            self.position_l = pose_l
            (self.theta) = theta_conversion(self.position_l, self.position_r, self.contact, a)
        elif(a=="rotate_clockwise"):
            if(mod(self.contact, 2) == 0):
                self.position_r = pose_r - OBJECT_LENGTH
                self.position_l = pose_l + OBJECT_BREADTH
                self.orientation= pose_o + 90
                self.theta[0] = theta_conversion(pose_l, pose_r, self.contact, a)
                self.theta[1] = calculate_th2(self.theta[0], pose_l, self.contact)

            else:
                self.position_r = pose_r - OBJECT_BREADTH
                self.position_l = pose_l + OBJECT_LENGTH
                self.orientation = pose_o + 90
                self.theta[0] = theta_conversion(pose_l, pose_r, self.contact, a)
                self.theta[1] = calculate_th2(self.theta[0], pose_l, self.contact)
            self.contact = self.contact + 1

        elif(a=="rotate_anticlockwise"):
            if (mod(self.contact, 2) == 0):
                self.position_r = pose_r + OBJECT_BREADTH
                self.position_l = pose_l - OBJECT_LENGTH
                self.orientation=pose_o-90
                self.theta[1] = theta_conversion(pose_l, pose_r, self.contact, a)
                self.theta[0] = calculate_th1(self.theta[1], pose_r, self.contact)
            else:
                self.position_r = pose_r + OBJECT_LENGTH
                self.position_l = pose_l - OBJECT_BREADTH
                self.orientation = pose_o - 90
                self.theta[1] = theta_conversion(pose_l, pose_r, self.contact, a)
                self.theta[0] = calculate_th1(self.theta[1], pose_r, self.contact)
            self.contact = self.contact - 1

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



if __name__ == "__main__":
    global current_start, current_goal, current_w_p, current_w_s, NUMBER_OF_NODES_EXPANDED

    start_list = [(7.0, 7.0, -90)]
    goal_list = [(7.9, 7.2, 90)]

    w_s_list = [2]
    w_p_list = [0.1]

    T_N = 0

    for start, goal in zip(start_list, goal_list):
        current_start = start
        current_goal = goal
        for w_s in w_s_list:
            NUMBER_OF_NODES_EXPANDED = 0
            T_N = T_N + 1
            current_w_s = w_s
            currents_w_p = w_p_list
            start_node = node(0, start[0], start[1], start[2], None, None)
            goal_node = node(0, goal[0], goal[1], goal[2], None, None)
