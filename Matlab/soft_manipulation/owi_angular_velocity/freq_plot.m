%% Importing data
% Array1=csvread('owi_test_data\final_vel_step1.5\freq_test1a.csv',2,0);
% Array2=csvread('owi_test_data\final_vel_step1.5\freq_test1b.csv',2,0);
% Array3=csvread('owi_test_data\final_vel_step1.5\freq_test1c.csv',2,0);
% Array4=csvread('owi_test_data\final_vel_step1.5\freq_test1d.csv',2,0);
% Array5=csvread('owi_test_data\final_vel_step1.5\freq_test1e.csv',2,0);

Array1=csvread('freq_test1a.csv',2,0);
Array2=csvread('freq_test1b.csv',2,0);
Array3=csvread('freq_test1c.csv',2,0);
Array4=csvread('freq_test1d.csv',2,0);
Array5=csvread('freq_test1e.csv',2,0);

%% Storing Relevant data
input_vel = Array2(:,2);
angular_vel = Array1(:, 2);
pos_x = Array3(:,2);
pos_y = Array3(:,3);
flag = Array4(:,2);
action = Array5(:,2);
input_vel_1=[0];
input_vel_2 = [0];
input_vel_3 = [0];
angular_vel_1 = [0];
angular_vel_2 = [0];
angular_vel_3 = [0];
pos_x_1 = [0];
pos_x_2 = [0];
pos_x_3 = [0];
pos_y_1 = [0];
pos_y_2 = [0];
pos_y_3 = [0];
datapoint_1 = [0];
datapoint_2 = [0];
datapoint_3 = [0];

%% Velocity Plot
figure(1)
hold on
grid on
title('End-effetor Velocity')
xlabel('frame #')
ylabel('velocity/pwm')
for i = 1:size(action)
    if(action(i) == 1)
        datapoint_1 = [datapoint_1,i];
        input_vel_1 = [input_vel_1,input_vel(i)];
        %angular_vel_1 = [angular_vel_1,angular_vel(i)];
        if(abs(angular_vel(i)) <= 10)
            angular_vel_1 = [angular_vel_1,angular_vel(i)];
        else
            %s = size(angular_vel_2)
            if(angular_vel(i)>0)
                angular_vel_1 = [angular_vel_1,1];
                %angular_vel_2 = [angular_vel_2,angular_vel_2(s(2))];
            else
                angular_vel_1 = [angular_vel_1,-1];
            end
        end
    elseif(action(i) == 2)
        datapoint_2 = [datapoint_2,i];
        input_vel_2 = [input_vel_2,input_vel(i)];
        if(abs(angular_vel(i)) <= 10)
            angular_vel_2 = [angular_vel_2,angular_vel(i)];
        else
            %s = size(angular_vel_2)
            if(angular_vel(i)>0)
                angular_vel_2 = [angular_vel_2,1];
                %angular_vel_2 = [angular_vel_2,angular_vel_2(s(2))];
            else
                angular_vel_2 = [angular_vel_2,-1];
            end
        end
    elseif(action(i) == -1)
        datapoint_3 = [datapoint_3,i];
        input_vel_3 = [input_vel_3,input_vel(i)];
        if(abs(angular_vel(i)) <= 10)
            angular_vel_3 = [angular_vel_3,angular_vel(i)];
        else
            %s = size(angular_vel_2)
            if(angular_vel(i)>0)
                angular_vel_3 = [angular_vel_3,1];
                %angular_vel_2 = [angular_vel_2,angular_vel_2(s(2))];
            else
                angular_vel_3 = [angular_vel_3,-1];
            end
        end
    end

end
        plot(datapoint_1(2:end),input_vel_1(2:end),'r:')
        plot(datapoint_1(2:end),angular_vel_1(2:end),'c')
        plot(datapoint_2(2:end),input_vel_2(2:end),'g:')
        plot(datapoint_2(2:end),angular_vel_2(2:end),'m')
        plot(datapoint_3(2:end),input_vel_3(2:end),'b:')
        plot(datapoint_3(2:end),angular_vel_3(2:end),'k')
        %legend({'+ pwm', '- pwm', '0 pwm'})
        legend({'+ pwm','angular vel', '- pwm', 'angular vel','0 pwm', 'angular vel'});
%% Position Plot
figure(2)
hold on
title('End-effector Position (Image frame)')
xlabel('X')
ylabel('Y')
for i = 1:size(action)
    if(action(i) == 1)
        datapoint_4 = [datapoint_4,i];
        pos_x_1 = [pos_x_1,pos_x(i)];
        pos_y_1 = [pos_y_1,pos_y(i)];
    elseif(action(i) == 2)
        datapoint_5 = [datapoint_5,i];
        pos_x_2 = [pos_x_2,pos_x(i)];
        pos_y_2 = [pos_y_2,pos_y(i)];
    elseif(action(i) == -1)
        pos_x_3 = [pos_x_3,pos_x(i)];
        pos_y_3 = [pos_y_3,pos_y(i)];
    end
end
    plot(pos_x_1(2:end),pos_y_1(2:end),'co')
    plot(pos_x_2(2:end),pos_y_2(2:end),'mo')
    plot(pos_x_3(2:end),pos_y_3(2:end),'ko')
    %legend('- pwm')
    legend('+ pwm', '- pwm', '0 pwm')
    
figure(3)
s = size(pos_x_1)
for i= 1:s(2)
    plot(pos_x_1(i),pos_y_1(i),'ro')
    pause(1.0)
    plot(pos_x_1(1:i),pos_y_1(1:i),'co')
    hold on
    pause(1.0)
end

figure(4)
s = size(pos_x_2)
for i = 1:s(2)
    plot(pos_x_2(i),pos_y_2(i),'ro')
    pause(1.0)
    plot(pos_x_2(1:i),pos_y_2(1:i),'co')
    hold on
    pause(1.0)
end