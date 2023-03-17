clear;
%% Reading data from csv
base_marker = csvread('data\exp3\base_px.csv',2,0);
ee_marker = csvread('data/exp3/px3.csv',2,0);
flag = csvread('data/exp3/flag.csv',2,0);
input_vel = csvread('data/exp3/input_vel.csv',2,0);

%% Calculating joint angle
theta = compute_angle(base_marker,ee_marker);

%% Calculating angular velocity
% Case 1: no acceleration for dropped frames
ang_vel1 = compute_angularvel1(theta,flag);
% Case 2: assuming prev acceleration for dropped frames
ang_vel2 = compute_angularvel2(theta,flag);
% Case 3: moving average for droped frames
ang_vel3 = compute_angularvel3(theta, flag);

%% Plot
% Joint angle vs frames
figure('NumberTitle','off','Name','OWI Open Loop Performance')

subplot(6,1,1)
plot(theta)
title('Joint Angle')

% using prev ang_vel
subplot(6,1,3)
plot(ang_vel1)
title('Measured Angular Velocity')

% input vel
subplot(6,1,2)
plot(input_vel(:,2))
title('Input PWM')

% with acc
subplot(6,1,4)
plot(ang_vel2)
title('Measured Angular Velocity w/ Acc')

% Moving average
subplot(6,1,5)
plot(ang_vel3)
title('Moving average')

% Flag
subplot(6,1,6)
plot(flag(:,2))
title('New frame check')