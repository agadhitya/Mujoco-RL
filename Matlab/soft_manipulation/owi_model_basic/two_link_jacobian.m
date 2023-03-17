
l1 = 0.09; l2 = 0.065;
th1 = 90; th2 = 150;

Jr = [-l1*sind(th1) - l2*sind(th1+th2), -l2*sind(th1+th2); l1*cosd(th1) + l2*cosd(th1+th2), l2*cos(th1+th2)]

Jr_inv = inv(Jr)

Jr_inv*[0; -0.1]