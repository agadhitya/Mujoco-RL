clear
clc

% Initialize start pose
start_joint_pos = [deg2rad(68), deg2rad(35)];

% Initialize goal pose
goal_joint_pos = [deg2rad(70), deg2rad(-60)];

% Estimation variables
window = 30; % size of sliding window
step = 0.1; % sin velocity resolution
old_r = start_joint_pos; % Initializing old joint positions with current joint positions
[old_pos, temp] = compute_cart_pos(start_joint_pos); % Initializing old_pos with current ee position
[ee_goal, j2_goal] = compute_cart_pos(goal_joint_pos); % For visualizing robot in goal state
ds =[]; % Change in ee position
dr =[]; % Change in joint positions
qhat = zeros(2,2);
gamma1 = 10e-6; % learning rate for initial estimation
gamma2 = 10e-8; % learning rate for online estimation

% Visual Servoing variables
lam = .005*eye(2); % visual servoing gain
rate =100;
r = rateControl(rate); % visual servoing sampling rate
thresh = 10e-1;
it = 1; % Iterator for frame vis

% Data recording variables
ds_ = []; % ee pos change vector
dr_ = []; % joint position change vector
Jplot = []; % model error vector for visualizing result
j_dot_ = []; % velocity vector for visualizing result
err_ = []; % error vector for visualizing result
traj_ = []; % storing robot trajectory for vis

%% Estimating Initial Jacobian
% Move Robot locally to collect estimation data
while(it<=window)
    % generate sinusoidal vel
    cur_joint_pos = start_joint_pos + [sin(step) + 2*((rand - 0.5)), sin(step) + 5*((rand - 0.5))];

    % current robot state
    [ee_pos, j2_pos] = compute_cart_pos(cur_joint_pos);
    
    % compute change in state
    ds = [ds; shape_change(ee_pos, old_pos)];
    old_pos = ee_pos;
    
    dr = [dr; angle_change(cur_joint_pos, old_r)];
    old_r = cur_joint_pos;
    
    % Increase step
    step = step + 0.01;
    
    % Data collection
    ds_ = [ds_; ds(it,:)]; % storing change in ee pos for vis
    dr_ = [dr_; dr(it,:)]; % storing change in joint positions for vis
    traj_ = [traj_; ee_pos];
    err_ = [err_; compute_err(cur_joint_pos, goal_joint_pos)]; % Store errors for vis
    
    % Plot robot state
    plot([0, j2_pos(1), ee_pos(1)], [0, j2_pos(2), ee_pos(2)], 'r'); % current robot state
    hold on
    plot(traj_(:,1),traj_(:,2), 'mo')
    %plot([0, j2_goal(1), ee_goal(1)], [0, j2_goal(2), ee_goal(2)], 'g'); % goal state
    plot(ee_goal(1),ee_goal(2),'g*', 'MarkerSize',12)
    txt = append('Step#:',int2str(it));
    text(50,5,txt)
    axis([-165 165 -165 165])
    title('Robot Trajectory')
    legend({'robot','cartesian velocity','trajectory','goal'},'Location','northeastoutside')
    pbaspect([1 1 1])
    grid on
    %pause(0.1);
    hold off
    
    % Increase iterator
    it = it +1;
end

% Estimate Jacobian for local movements
count = 1;
while count<=window
    [J, qhat_dot] = compute_energy_functional(ds, dr, qhat, count, gamma1);
    qhat = qhat + qhat_dot;
    Jplot = [Jplot; J];
    count = count+1;
end
%% Error at start of VS
err = [thresh, thresh]; % Initializing error

%% Initial Jacobian
Q = qhat

%% Control Loop
disp('Entering Control Loop')
reset(r) % Resetting rate control object
while(abs(norm(err)) >= thresh)
    
    %% Compute error
    err = compute_err(cur_joint_pos, goal_joint_pos);
    
    %% Error Saturation
%     if(norm(err) >= 3) 
%         err(1) = 2*(err(1)/(abs(err(1))));
%         err(2) = 2*(err(2)/(abs(err(2))));
%     end
    
    %% Generate Velocity
    [j_dot, r_dot] = velocity_gen(Q, lam, err);
    
    %% Update robot state
    j_update = (1/rate)*j_dot;
    cur_joint_pos = cur_joint_pos + j_update;
    
    %% Compute end effector Position
    [ee_pos, j2_pos] = compute_cart_pos(cur_joint_pos);
    
    %% Compute change in state
    ds(1,:) = shape_change(ee_pos, old_pos); % replacing oldest data instance
    ds = transpose(circshift(transpose(ds),-1,2)); % left circular shift
    old_pos = ee_pos;

    dr(1,:) = angle_change(cur_joint_pos, old_r);
    dr = transpose(circshift(transpose(dr),-1,2)); % left circular shift
    old_r = cur_joint_pos;
    
    %% Update Jacobian
    [J, qhat_dot] = compute_energy_functional(ds, dr, Q, window, gamma2);
    Q = Q + qhat_dot;
    
    
    %% Data Collection
    ds_ = [ds_; ds(window,:)]; % storing change in ee pos for vis
    dr_ = [dr_; dr(window,:)]; % storing change in joint positions for vis
    traj_ = [traj_; ee_pos];
    Jplot = [Jplot; J]; % Store model errors for vis
    j_dot_ = [j_dot_; j_dot]; % Store velocities for vis
    err_ = [err_; err]; % Store errors for vis
    
    %% Visualize robot state
    figure(1)
    plot([0, j2_pos(1), ee_pos(1)], [0, j2_pos(2), ee_pos(2)], 'r'); % current robot state
    hold on
    quiver(ee_pos(1), ee_pos(2), r_dot(1), r_dot(2),100, 'c')
    plot(traj_(1:window,1),traj_(1:window,2), 'mo')
    plot(traj_(window:end,1),traj_(window:end,2), 'b-')
    %plot([0, j2_goal(1), ee_goal(1)], [0, j2_goal(2), ee_goal(2)], 'g'); % goal state
    plot(ee_goal(1),ee_goal(2),'g*', 'MarkerSize',12)
    txt = append('Step#:',int2str(it));
    text(50,5,txt)
    axis([-165 165 -165 165])
    title('Robot Trajectory')
    legend({'robot','cartesian velocity','trajectory','goal'},'Location','northeastoutside')
    pbaspect([1 1 1])
    grid on
    %pause(0.1);
    hold off
    
    %% Increment iterator
    it = it+1;
    
    %% Rate control
%     r.TotalElapsedTime 0.02s
    waitfor(r);
end

%% Plot Results
disp('Generating results')

figure()
plot(Jplot(window:end))
title('Model Error')
xlabel('Steps')
ylabel('Estimation Error')

figure()
subplot(2,1,1)
plot(j_dot_(:,1))
title('Velocity J1')
xlabel('time')
ylabel('Velocity')

subplot(2,1,2)
plot(j_dot_(:,2))
title('Velocity J2')
xlabel('time')
ylabel('Velocity')

figure()
plot(err_(window+1:end,1), 'b')
hold on
plot(err_(window+1:end,2), 'g')
legend('ErrorX', 'ErrorY')
title('Pose Error')
xlabel('Steps')
ylabel('Estimation Error')