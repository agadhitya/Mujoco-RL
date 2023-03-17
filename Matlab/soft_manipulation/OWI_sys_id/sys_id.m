clear; clc;

%% Read data
exp_no = "10"

str1 = "~/owi_data/sys_id_data/exp"+exp_no+"/base_px.csv";
str2 = "~/owi_data/sys_id_data/exp"+exp_no+"/shoulder.csv";
str3 = "~/owi_data/sys_id_data/exp"+exp_no+"/flag.csv";
str4 = "~/owi_data/sys_id_data/exp"+exp_no+"/input_vel.csv";
str5 = "~/owi_data/sys_id_data/exp"+exp_no+"/elbow.csv";
str6 = "~/owi_data/sys_id_data/exp"+exp_no+"/ee.csv";

base_marker = csvread(str1,2,0);
shoulder_marker = csvread(str2,2,0);
flag = csvread(str3,2,0);
input_vel = csvread(str4,2,0);
elbow_marker = csvread(str5,2,0);
ee_marker = csvread(str6,2,0);

%% compute start posn
[start, posn] = compute_start(ee_marker, input_vel);

%% Chopping off leading 0s in the arrays
[base_marker, ee_marker, shoulder_marker, flag, input_vel, elbow_marker] = process_data(base_marker, ee_marker, shoulder_marker, flag, input_vel, elbow_marker, posn);

%% Trajectory Plot
goal = [ee_marker(length(ee_marker),2), ee_marker(length(ee_marker),3)];

trajectory_plot(ee_marker, start, exp_no, goal);

%% Velocity Computation
time = 0.03 * length(input_vel);

dist_px = [ee_marker(length(ee_marker), 2) - ee_marker(1,2), ee_marker(length(ee_marker), 3) - ee_marker(1,3)];

vel = dist_px/time;