clear;

l1 = 90; 
l2 = 75;

th1 = 30;
th2 = -30
% syms th1, th2;

eeX = l1*cosd(th1) + l2*cosd(th1 + th2);
eeY = l1*sind(th1) + l2*sind(th1 + th2);

j2X = l1*cosd(th1);
j2Y = l1*sind(th1);

baseX = 0;
baseY = 0;

%% Equation of Link1
link1PtsX = linspace(0, j2X, 10);
link1PtsY = linspace(0, j2Y, 10);

%% Equation of Link2
link2PtsX = linspace(j2X, eeX, 8);
link2PtsY = linspace(j2Y, eeY, 8);

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
% p = polyfit(curvePtsX,curvePtsY,6);
% x1 = linspace(0,eeX);
% f1 = polyval(p,x1);
f=fit(transpose(curvePtsX),transpose(curvePtsY),'poly6')
% plot(x1, f1, 'm--')
plot(f, curvePtsX, curvePtsY)

