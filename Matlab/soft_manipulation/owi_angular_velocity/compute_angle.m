function [theta] = compute_angle(base,ee)
theta = [0];

for i=1:size(base)
    if(ee(i,2) ~= base(i,2))
        if(ee(i,2) < base(i,2))
            theta(i) = (atan((ee(i,3) - base(i,3))/(ee(i,2) - base(i,2))))*57.2958;
        else
            theta(i) = 180 + ((atan((ee(i,3) - base(i,3))/(ee(i,2) - base(i,2))))*57.2958);
        end
    else
        theta(i) = 90;
    end
end


end