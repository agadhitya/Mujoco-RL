function[err_x, err_y] = compute_error(goal, ee)
for i=1:length(ee)
    err_x(i) = ee(i,2) - goal(1);
    err_y(i) = ee(i,3) - goal(2);
end
end