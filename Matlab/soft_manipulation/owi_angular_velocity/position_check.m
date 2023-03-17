clear;
%% Reading data from csv
ee_marker1 = csvread('data/exp4/px3.csv',2,0);
ee_marker2 = csvread('data/exp5/px3.csv',2,0);
ee_marker3 = csvread('data/exp6/px3.csv',2,0);

% End effector position plot

subplot(3,1,1)
plot(ee_marker1(:,2),-ee_marker1(:,3),'.')
title('input pwm: 0.3')

subplot(3,1,2)
plot(ee_marker2(:,2),-ee_marker2(:,3),'.')
title('input pwm: 0.35')

subplot(3,1,3)
plot(ee_marker3(:,2),-ee_marker3(:,3),'.')
title('input pwm: 0.4')

