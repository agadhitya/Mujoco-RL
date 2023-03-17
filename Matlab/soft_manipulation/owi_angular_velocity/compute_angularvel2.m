function [ang_vel2] = compute_angularvel2(theta, flag)
time = 1/30;
ang_vel2(1) = 0;
acc = 0;
for i = 25:length(theta)
    if(flag(i,2) == 1)
        ang_vel2(i) = (theta(i) - theta(i-1))/time;
        acc = (ang_vel2(i) - ang_vel2(i-1))/time;
    else
        ang_vel2(i) = ang_vel2(i-1) + (acc * time);
        theta(i) = theta(i-1) + (ang_vel2(i)*time);
    end
end


end