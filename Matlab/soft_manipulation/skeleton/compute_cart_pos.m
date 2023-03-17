function [ee_pos, j2_pos] = compute_cart_pos(cur_pos)
%% Robot Model 
l1 = 90; l2 = 75;

th1 = cur_pos(1);
th2 = cur_pos(2);

eeX = l1*cos(th1) + l2*cos(th1 + th2);
eeY = l1*sin(th1) + l2*sin(th1 + th2);

j2X = l1*cos(th1);
j2Y = l1*sin(th1);

ee_pos = [eeX, eeY];
j2_pos = [j2X, j2Y];


end