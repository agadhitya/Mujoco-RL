This folder contains a mathematical simulation for shape servoing of a 2 link robot. The list of functions and scripts is provided below.

## List of Scripts
1. adaptive_jacobian.m: This script implements a mathematical visual-servoing scheme to test the online Jacobian estimation
2. ee_jacobian.m: This script estimates offline the Jacobian for a 2 link planar robot based of the change in end effector pose data and change in angle position data
3. main.m: This script estimates offline the shape Jacobian for a 2 link planar robot based of the change in curve parameters for its skeleton and change in angle position data
4. test_sim: This is an initial test script to check the visulaization of a 2 link planar robot and its skeleton and for fitting a curve to the skeleton
5. test_sim2.m: This is an initial test script to try to extract shape contours for a 2 link planar robot

## List of Functions
1. angle_change.m
2. compute_energy_functional.m
3. compute_pos.m
4. fit_curve.m
5. shape_change.m
6. skeleton.m

### Function Descriptions
#### angle_change.m
Returns change in angle given the cur angles and old angles.

`dr = angle_change(cur_r, old_r);`

> Inputs cur_r and old_r can be any dimension as long as they have the same dimension.

#### compute_energy_functional.m
Inplements algorithm 1 from the paper, "Fourier-Based Shape Servoing: A New Feedback Method to Actively Deform Soft Objects into Desired 2-D Image Contours".
Returns the model error (J) and change in Jacobian elements(qhat_dot).

`[J, qhat_dot] = compute_energy_functional(dS, dR, qhat, t);`

> dS and dR are the change in shape and change in angle vectors of length T X 2, where T is the total sample instances
> qhat is the approximated Jacobian that is being estimated
> t is the time/sample instance

#### compute_pos.m
Defines the robot model and computes the position of the end-effector and j2 for the 2 link planar robot given the current angle positions.

`[ee_pos, j2_pos] = compute_pos(cur_pos);`

> cur_pos is a 2X1 vector containing the angle positions for j1 and j2
> ee_pos is a 2X1 vector containing the end effector pose
> j2_pos is a 2X1 vector containing joint2 position
> base position is assumed to be [0, 0] for visualization purposes

#### fit_curve.m

#### shape_change.m
Similar to angle_change

#### skeleton.m
