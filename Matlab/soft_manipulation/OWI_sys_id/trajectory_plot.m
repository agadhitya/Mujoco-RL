function[] = trajectory_plot(ee, start, exp_no, goal)

%% Read image at start posn
str = "~/owi_data/sys_id_data/exp"+exp_no+"/frame0000.jpg";
img = imread(str);
size(img)

figure(1)
imshow(img);
axis on
hold on
plot(start(1),start(2),'g+', 'MarkerSize', 15, 'LineWidth', 3)
plot(ee(:,2),ee(:,3),'b.', 'MarkerSize', 7, 'LineWidth', 2)
% line([start(1),goal(1)],[start(2),goal(2)], 'Color', 'green', 'LineWidth', 3)
plot(goal(1),goal(2),'r+', 'MarkerSize', 15, 'LineWidth', 3)
str_loc = "~/owi_data/sys_id_data/exp"+exp_no+"/trajectory.png";
saveas(gcf, str_loc)

end