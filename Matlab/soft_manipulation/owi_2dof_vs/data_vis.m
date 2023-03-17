clear;
%% Enter goal position
% goal = [46, 294];
% goal = [200, 250];
goal = [250, 300];
%% Read data
exp_no = "18";

str1 = "~/owi_data/vs_data/exp"+exp_no+"/base_px.csv";
str2 = "~/owi_data/vs_data/exp"+exp_no+"/shoulder.csv";
str3 = "~/owi_data/vs_data/exp"+exp_no+"/flag.csv";
str4 = "~/owi_data/vs_data/exp"+exp_no+"/input_vel.csv";
str5 = "~/owi_data/vs_data/exp"+exp_no+"/elbow.csv";
str6 = "~/owi_data/vs_data/exp"+exp_no+"/ee.csv";

base_marker = csvread(str1,2,0);
shoulder_marker = csvread(str2,2,0);
flag = csvread(str3,2,0);
input_vel = csvread(str4,2,0);
elbow_marker = csvread(str5,2,0);
ee_marker = csvread(str6,2,0);

%% compute start posn
[start, posn] = compute_start(ee_marker, input_vel)

%% Chopping off leading 0s in the arrays
[base_marker, ee_marker, shoulder_marker, flag, input_vel, elbow_marker] = process_data(base_marker, ee_marker, shoulder_marker, flag, input_vel, elbow_marker, posn);

%% Compute error
[err_x,err_y] = compute_error(goal,ee_marker);

%% Trajectory Plot
trajectory_plot(ee_marker, start, exp_no, goal);

%% Plot
figure(3)

subplot(4,1,1)
hold on
plot(err_x)
title('Error in X')
grid on
ylabel('pixels')
legend('Trial 1', 'Trial 2', 'Trial 3')

subplot(4,1,2)
hold on
plot(err_y)
title('Error in Y')
ylabel('pixels')
grid on

subplot(4,1,3)
hold on
plot(input_vel(:,2))
title('Joint 1 velocity')
ylabel('pwm')
grid on

subplot(4,1,4)
hold on
plot(input_vel(:,3))
title('Joint 2 velocity')
ylabel('pwm')
grid on

figure(2)
hold on
plot(flag(:,2))