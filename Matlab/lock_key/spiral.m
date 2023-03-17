% Visualize different methods for calculating spirals
% for key-in-lock insertion strategies.

clearvars; close all; clc;

%% Method 1 - Update r & theta each iteration
ii_max=50;
x_vec_1=zeros(ii_max,1);
y_vec_1=zeros(ii_max,1);
phi=0; phi_step=0.25;
r=0; r_step=0.0005;

for ii=2:ii_max
    % Get new point
    x_vec_1(ii)=x_vec_1(ii-1)+r*cos(phi);
    y_vec_1(ii)=y_vec_1(ii-1)+r*sin(phi);
    % Update spiral parameters
    r=r+r_step; phi=phi+phi_step; 
end

%% Method 2 - Lans paper (Archimedean)
a=0; % Initial distance from origin (m)
b=0.00025; %Distance between turns (m)
nmax=100; % Number of spiral points
rot=4; %Number of rotations to complete
x_vec_2=zeros(ii_max,1);
y_vec_2=zeros(ii_max,1);
for n=2:nmax
    phi=sqrt(n./nmax).*(rot.*2.*pi);
    r=(a-b.*phi);
    x_vec_2(n)=x_vec_2(n-1)+r.*cos(phi);
    y_vec_2(n)=y_vec_2(n-1)+r.*sin(phi);
end

figure(1);
hold on; grid on;
plot(x_vec_1.*100, y_vec_1*100,'o-', ... %Convert to cm
     x_vec_2.*100, y_vec_2*100,'o-'); %Convert to cm
ax = gca; set(gca, 'XAxisLocation', 'origin', 'YAxisLocation', 'origin')
legend('Method 1','Method 2','Location','Best'); axis equal;
title("Comparing Spiral Trajectory Generation Methods");
xlabel('x (cm)'); ylabel('y (cm)');