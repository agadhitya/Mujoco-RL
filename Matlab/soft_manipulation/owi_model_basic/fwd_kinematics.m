%% Define Link lengths
l1 = 1; l2 = 1; l3 = 1; l4 = 1;

%% Forward Kinematics
A1 = transform(90, 0, l1, 90);
A2 = transform(90, l2, 0, 0);
A3 = transform(0, l3, 0, 0);
A4 = transform(-90, l4, 0, -90);

H = A1*A2*A3*A4

%% Jacobian Computation