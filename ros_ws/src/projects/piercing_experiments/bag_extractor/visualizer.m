% Base File Paths
str = "/home/agandhi2/FF/csv/exp";
str_ = "/home/agandhi2/FF/csv/exp_";

% Pre-allocated variables
x = "";
goal_x = [0.0, 0.0, 0.0, 0.0];
goal_y = [0.0, 0.0, 0.0, 0.0];
plength= 0.0;
result = [0.0, 0.0, 0.0, 0.0;0.0, 0.0, 0.0, 0.0;0.0, 0.0, 0.0, 0.0;0.0, 0.0, 0.0, 0.0];

% Reading CSV files as matrices 
for exp = 1:4
x = num2str(exp);
plength = 0.0;
str_mod_pos = strcat(str, x);
str_mod_act = strcat(str_,x);

pos = readmatrix(str_mod_pos);
act = readmatrix(str_mod_act);
goal = readmatrix('/home/agandhi2/FF/csv/goal.csv');

% Size of Matrix
pos_size = size(pos);
act_size = size(act);

% Extracting Final Object Position
final_x = pos(pos_size(1),2);
final_y = pos(pos_size(1),3);

% Extracting Object Goal Position 
goal_x(exp) = goal(exp,1);
goal_y(exp) = goal(exp,2);

% Computing Error
err_x = goal_x(exp) - final_x;
err_y = goal_y(exp) - final_y;
error = norm(err_x, err_y);

% Computing the Number of Switching Actions
prev = 0; count = 0;
for i = 1 : act_size(1)
    if(act(i,2) == 0)
        continue;
    end
    if(act(i,2) ~= 0 && act(i,2) ~= prev)
        start_time = act(i,1);
        prev = act(i,2);
        count  = count+1;
    else
        continue
    end
end

% Computing Experiment Time
exp_time = (start_time - act(1,1))/(10^9); % Converting to Seconds

% Computing Path Length
for len = 1:(pos_size-1)
   dx = pos(len+1,2) - pos(len,2);
   dy = pos(len+1,3) - pos(len,3);
   if(dx == 0 && dy  == 0)
           continue;
       else
   plength = plength + norm(dx, dy);
   end
end
plength = plength*1000; % Converting to mm

% Concatenate Results Matrix
result(exp,:) = [exp, exp_time, error, plength];
end

% Write Results to File
writematrix(result,'/home/agandhi2/FF/result.csv');