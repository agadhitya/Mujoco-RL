clear
clc

%% Robot Model
l1 = 90; l2 = 75;
% th1 = 35; th2 = -35;
syms th1 th2;
eeXex = l1*cosd(th1) + l2*cosd(th1 + th2);
eeYex = l1*sind(th1) + l2*sind(th1 + th2);

j2Xex = l1*cosd(th1);
j2Yex = l1*sind(th1);

baseX = 0;
baseY = 0;

%% Estimation Variables
goal_coeffs = zeros(4,1);
old_coeffs = zeros(4,1);
step = 0.1;
t = 1;
t_end = 50;
th1_st = 40;
th2_st = -30;
r = zeros(2,1);
old_r = [th1_st; th2_st];
ds = zeros(t_end,4);
dr = zeros(t_end,2);
th1_start = th1_st;
th2_start = th2_st;
gamma = 1;

%% Select feature points from skeleton
[skelexX, skelexY, link1PtsexX, link1PtsexY, link2PtsexX, link2PtsexY] = skeleton(j2Xex, j2Yex, eeXex, eeYex);


%% Move Robot locally
disp('Collecting shape data for small movements')
while(t<=t_end)
    skelX = eval(subs(skelexX, [th1, th2], [th1_start, th2_start]));
    skelY = eval(subs(skelexY, [th1, th2], [th1_start, th2_start]));
    link1PtsX = eval(subs(link1PtsexX, [th1, th2], [th1_start, th2_start]));
    link1PtsY = eval(subs(link1PtsexY, [th1, th2], [th1_start, th2_start]));
    link2PtsX = eval(subs(link2PtsexX, [th1, th2], [th1_start, th2_start]));
    link2PtsY = eval(subs(link2PtsexY, [th1, th2], [th1_start, th2_start]));
    eeX = eval(subs(eeXex, [th1, th2], [th1_start, th2_start]));
    eeY = eval(subs(eeYex, [th1, th2], [th1_start, th2_start]));
    j2X = eval(subs(j2Xex, th1, th1_start));
    j2Y = eval(subs(j2Yex, th1, th1_start));

    th1_start = th1_st + sind(step);
    th2_start = th2_st + sind(step);    
    r = [th1_start; th2_start];
    step = step + 10;
    
    %% Fit curve
    [curve, coeffs] = fit_curve(skelX, skelY);
    
    %% Change in Robot Shape
    % initialization step
    if t == 1
        old_coeffs = coeffs;
    end
    
    ds(t,:) = shape_change(coeffs, old_coeffs);
    old_coeffs = coeffs;
    dr(t,:) = angle_change(r, old_r);
    old_r = r;

    
    %% Plots
    figure(1)
    
    % Robot
    plot([0, j2X, eeX], [0, j2Y, eeY], 'b');
    hold on
    
    % Skeleton Points
    plot(link1PtsX(:), link1PtsY(:), 'ro');
    plot(link2PtsX(:), link2PtsY(:), 'ro');
    
    % Curve
    plot(curve, 'g');
    hold off
    %% Collect shape and pose data
    t = t+1;
end
disp('Done collecting data')

% Reset Time
t = 1;

%% Estimate Shape Jacobian
disp('Beginning Shape Jacobian estimation process')

% Inititalize estimation variables
qhat = zeros(4,2);
Jplot = [];
while t<=t_end
    [J, qhat_dot] = compute_energy_functional(ds, dr, qhat, t, gamma)
    qhat = qhat_dot + qhat;
     t = t+1;
     Jplot = [Jplot; J];
end
figure()
plot(Jplot)