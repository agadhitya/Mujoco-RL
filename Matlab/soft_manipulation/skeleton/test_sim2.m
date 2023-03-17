clear;

l1 = 60; 
l2 = -30;
b = 5;

th1 = 30;
th2 = 30;
% syms th1, th2;

eeX = l1*cosd(th1) + l2*cosd(th1 + th2);
eeY = l1*sind(th1) + l2*sind(th1 + th2);

j2X = l1*cosd(th1);
j2Y = l1*sind(th1);

baseX = 0;
baseY = 0;
R1 = [cosd(th1), -sind(th1); sind(th1), cosd(th1)];
R2 = [cosd(th1+th2), -sind(th1+th2); sind(th1+th2), cosd(th1+th2)];

v1 = R1*[0; b/2];
v2 = R1*[0; -b/2];
v3 = R1*[l1; -b/2];
v4 = R1*[l1;b/2];

q1 = R2*[0; b/2] + [j2X; j2Y];
q2 = R2*[0; -b/2]+ [j2X; j2Y];
q3 = R2*[l2; -b/2]+ [j2X; j2Y];
q4 = R2*[l2;b/2]+ [j2X; j2Y];

pgon1 = polyshape([v1(1) v2(1) v3(1) v4(1)],[v1(2) v2(2) v3(2) v4(2)]);
pgon2 = polyshape([q1(1) q2(1) q3(1) q4(1)],[q1(2) q2(2) q3(2) q4(2)]);
%% Equation of Link1
link1PtsX = linspace(0, j2X, 3);
link1PtsY = linspace(0, j2Y, 3);

%% Equation of Link2
link2PtsX = linspace(j2X, eeX, 3);
link2PtsY = linspace(j2Y, eeY, 3);

%% Plot robot
figure(1)
plot([0, j2X, eeX], [0, j2Y, eeY], 'b');
hold on
plot(link1PtsX(:), link1PtsY(:), 'r')
plot(link2PtsX(:), link2PtsY(:), 'g')

%% Fit a 2nd Degree Polynomial
curvePtsX = [link1PtsX, link2PtsX];
curvePtsY = [link1PtsY, link2PtsY];
xpts = [0, j2X, eeX];
ypts = [0, j2Y, eeY];
p = polyfit(curvePtsX,curvePtsY,4);
x1 = linspace(0,eeX);
f1 = polyval(p,x1);
% plot(pgon1)
% plot(pgon2)
% plot(x1, f1, 'm--')

%% Boundary
polyin = polyshape({[v1(1),v2(1), q2(1), q1(1), q4(1), q3(1), v3(1), v4(1)]},{[v1(2),v2(2), q2(2), q1(2), q4(2), q3(2), v3(2), v4(2)]})
% plot(polyin)
[x,y] = boundary(polyin)
plot(x,y)
 