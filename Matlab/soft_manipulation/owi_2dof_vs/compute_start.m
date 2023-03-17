function[start, posn] = compute_start(ee, input_vel)
flag = 0;
posn = 0;
start = [0,0];
    for i = 1:length(input_vel)
        if(input_vel(i,2) ==0 && input_vel(i,3) ==0)
            continue;
        else
            start = [ee(i,2),ee(i,3)];
            posn = i;
            break;
        end
    end
end