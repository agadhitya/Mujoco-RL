%function [skelX, skelY, link1PtsX, link1PtsY, link2PtsX, link2PtsY] = skeleton(j2X, j2Y, eeX, eeY)
function [skel] = skeleton(j2_cart_pos, ee_cart_pos)
j2X = j2_cart_pos(1);
j2Y = j2_cart_pos(2);

eeX = ee_cart_pos(1);
eeY = ee_cart_pos(2);

% Points on Link1
link1PtsX = linspace(0, j2X, 10);
link1PtsY = linspace(0, j2Y, 10);

% Points on Link2
link2PtsX = linspace(j2X, eeX, 8);
link2PtsY = linspace(j2Y, eeY, 8);
    
% Concatenating skeleton points
skelX = transpose([link1PtsX, link2PtsX(2:end)]);
skelY = transpose([link1PtsY, link2PtsY(2:end)]);

skel = [skelX skelY];
end