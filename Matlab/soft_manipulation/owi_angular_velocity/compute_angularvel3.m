function [ang_vel3] = compute_angularvel3(theta, flag)
time = 1/30;
for i = 25:length(theta)
    if(flag(i,2) == 1)
        ang_vel3(i) = (theta(i) - theta(i-1))/time;
        ang_vel3(i);
    else
        sum = ang_vel3(i-3) + ang_vel3(i-2) + ang_vel3(i-1);
        ang_vel3(i) = sum/3;
        theta(i) = theta(i-1) + (ang_vel3(i) * time);
    end
end
end