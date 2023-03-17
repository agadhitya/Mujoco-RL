function[j_dot, r_dot] = velocity_gen(Q, lam, err)
%r_dot = -lam*(pinv(transpose(Q)*Q))*transpose(Q)*transpose(err); %TO DO: change to pseudo inv only

% Cartesian velocity
r_dot = -lam*transpose(err);

% Joint Velocity    
j_dot = transpose(pinv(Q)*r_dot);
end