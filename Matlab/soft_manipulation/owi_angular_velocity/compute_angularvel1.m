function [ang_vel] = compute_angularvel1(theta,flag)
time = 1/30;
for i = 25:length(theta)
    if(flag(i,2) == 1)
        ang_vel(i) = (theta(i) - theta(i-1))/time;
    else
        ang_vel(i) = ang_vel(i-1);
        theta(i) = theta(i-1) + (ang_vel(i) * time);
    end
end


end