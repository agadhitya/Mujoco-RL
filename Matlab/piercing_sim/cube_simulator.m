clear; close all; clc;
c = cube();
% Generate piercing forces
vectors = c.generate_force();
% Run simulations
%c.food_plate_simulator(vectors);
%% Max piercing angle trend along various static friction values
u = 0.1:0.02:0.9;
max_p = 1:41;
for i = 1:41;
    max_p(i) = c.max_piercing_angle(u(i));
end

%% Max force trend along various piercing angles
i = 1;
max_f = 1:151;
for p = 15:0.5:90
    max_f(i) = c.max_force(p);
    i = i+1;
end  % for loop
p = 15:0.5:90;

%% Food fork interaction acceptable angles
acceptable_angle = 1:41; fall = true; angle = 0; i = 1;
for u1 = 0.1:0.04:1.8
while(fall==true)
    vector = [cosd(angle),0,-sind(angle)];
    fall = c.fall_checker(vector,u1);
    if(fall ~= true)
        acceptable_angle(i) = angle;
        i = i+1;
    else
        angle = angle+.1;
    end  % if
end  % while
angle = 0; fall = true;
end  % for loop
u1 = 0.1:0.04:1.8;

  %% Plot Results
figure('Name','Force and Piercing Angle Trends')
subplot(2,2,1)
plot(p,max_f)
xlabel('Piercing Angles')
ylabel('Maximum Force')
legend('u_s: 0.25, mass: 0.05kg')
subplot(2,2,2)
plot(u,max_p)
grid on
xlabel('Static Friction Co-efficient')
ylabel('Angle for No Slide Condition')
subplot(2,2,3)
plot(u1,acceptable_angle)
grid on
xlabel('Static Friction Co-efficient')
ylabel('Angle for No Fall condition')
pl = [max_p(1)*ones(1,43);max_p(5)*ones(1,43);max_p(10)*ones(1,43);max_p(15)*ones(1,43);max_p(20)*ones(1,43);max_p(25)*ones(1,43);max_p(30)*ones(1,43);max_p(35)*ones(1,43);max_p(40)*ones(1,43)];
figure('Name', 'Acceptable Piercing Angles for various cases')
hold on
grid on
plot(u1,acceptable_angle,u1,pl(1,:),u1,pl(2,:),u1,pl(3,:),u1,pl(4,:),u1,pl(5,:),u1,pl(6,:),u1,pl(7,:),u1,pl(8,:),u1,pl(9,:))
legend('No fall condition','u_s: 0.1','u_s: .18','u_s: .28','u_s: .38','u_s: .48','u_s: .58','u_s: .68','u_s: .78','u_s: .88')