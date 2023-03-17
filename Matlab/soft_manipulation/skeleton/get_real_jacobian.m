function[Q] = get_real_jacobian(cur_joint_pos)
l1 = 90; l2 = 75;
th1 = cur_joint_pos(1);
th2 = cur_joint_pos(2);
Q = [-l1*sin(th1) - l2*sin(th1+th2), -l2*sin(th1+th2); l1*cos(th1) + l2*cos(th1+th2), l2*cos(th1+th2)];
end