function [err] = compute_coeffs_err(cur_joint_pos, goal_joint_pos)

% Current curve coeffs
% [ee_cart_pos, j2_cart_pos] = compute_cart_pos(cur_joint_pos);
% cur_skeleton = skeleton(j2_cart_pos, ee_cart_pos);
% [curve,cur_coeffs] = fit_curve(cur_skeleton);
[curve, cur_coeffs] = compute_coeffs(cur_joint_pos);
% Goal curve coeffs
% [ee_cart_goal, j2_cart_goal] = compute_cart_pos(goal_joint_pos);
% goal_skeleton = skeleton(j2_cart_goal, ee_cart_goal);
% [goal_curve, goal_coeffs] = fit_curve(goal_skeleton);
[goal_curve, goal_coeffs] = compute_coeffs(goal_joint_pos);
err = cur_coeffs - goal_coeffs;
end