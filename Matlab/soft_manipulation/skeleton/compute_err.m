function [err] = compute_err(cur_joint_pos, goal_joint_pos)
    [cur_ee_pos, cur_j1_pos] = compute_cart_pos(cur_joint_pos);
    [goal_ee_pos, goal_j1_pos] = compute_cart_pos(goal_joint_pos);
    
    err = cur_ee_pos - goal_ee_pos;
end