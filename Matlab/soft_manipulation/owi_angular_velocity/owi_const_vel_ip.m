clear;
%% Reading data from csv
base_marker1 = csvread('data\exp4\base_px.csv',2,0);
ee_marker1 = csvread('data/exp4/px3.csv',2,0);
flag1 = csvread('data/exp4/flag.csv',2,0);
input_vel1 = csvread('data/exp4/input_vel.csv',2,0);

base_marker2 = csvread('data\exp5\base_px.csv',2,0);
ee_marker2 = csvread('data/exp5/px3.csv',2,0);
flag2 = csvread('data/exp5/flag.csv',2,0);
input_vel2 = csvread('data/exp5/input_vel.csv',2,0);

base_marker3 = csvread('data\exp6\base_px.csv',2,0);
ee_marker3 = csvread('data/exp6/px3.csv',2,0);
flag3 = csvread('data/exp6/flag.csv',2,0);
input_vel3 = csvread('data/exp6/input_vel.csv',2,0);
%% Calculating joint angle
theta1 = compute_angle(base_marker1,ee_marker1);
theta2 = compute_angle(base_marker2,ee_marker2);
theta3 = compute_angle(base_marker3,ee_marker3);

%% Calculating angular velocity

% Case 1: no acceleration for dropped frames
ang_vel1a = compute_angularvel1(theta1,flag1);
% Case 3: moving average for droped frames
ang_vel3a = compute_angularvel3(theta1, flag1);

% Case 1: no acceleration for dropped frames
ang_vel1b = compute_angularvel1(theta2,flag2);
% Case 3: moving average for droped frames
ang_vel3b = compute_angularvel3(theta2, flag2);

% Case 1: no acceleration for dropped frames
ang_vel1c = compute_angularvel1(theta3,flag3);
% Case 3: moving average for droped frames
ang_vel3c = compute_angularvel3(theta3, flag3);

%% Plotting

figure('NumberTitle','off','Name','OWI performance for constant vel i/p')

subplot(5,1,1)
plot(theta1)
hold on
plot(theta2)
plot(theta3)
title('Joint Angle')
legend('I/P PWM: 0.3', 'I/P PWM: 0.35', 'I/P PWM: 0.4')

subplot(5,1,2)
plot(input_vel1(:,2))
hold on
plot(input_vel2(:,2))
plot(input_vel3(:,2))
title('Input PWM')

subplot(5,1,3)
plot(ang_vel1a)
hold on
plot(ang_vel1b)
plot(ang_vel1c)
title('Measured Angular Velocity')

subplot(5,1,4)
plot(ang_vel3a)
hold on
plot(ang_vel3b)
plot(ang_vel3c)
title('Moving average')

subplot(5,1,5)
plot(flag1(:,2))
hold on
plot(flag2(:,2))
plot(flag3(:,2))
title('New frame check')

