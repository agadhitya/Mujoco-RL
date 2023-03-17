# Dexterous In Hand Manipulation
This project investigates in-hand manipulation strategies based on the integration of a variable friction (VF) hand and Franka Emika Panda arm. This folder currently contains low and high level planning nodes for both Franka arm and variable friction hand, setups for manipulation environments and experiments, user interfaces for parameter modification and robot operation, simulation plugins, and a moveit setup for combined Franka arm and variable friction hand.

## Packages:

### arm_controls
This package is used for high level motion planning of the Franka arm.

### gripper_controls
This package is based on the motion planning nodes in variable_friction_gripper project, which prepared to be used on hardware. Same logic and functionality is transferred to the gazebo simulation in this package.

### manipulation_env
This package is used to generate a manipulation environment.

### manipulation_exp
This package is used for a higher level planner implementation that combines services advertised by arm_controls and gripper_controls into some sequential manipulation strategies.

### manipulation_planning
This package is used to read and execute planned manipulation sequences.

### planning_ui
This package contains an easy-to-use user interface to perform experiments allowing users to adjust the initial states, execute manipulation plans, compute results, and reset simulation.

### simulation_plugins
This package contains gazebo plugins to switch between low and high friction states, and modify other friction related simulation parameters.

### simulation_ui
This package contains an easy-to-use user interface to modify simulation parameters and send motion primitive commands to the gazebo simulation.

### vf_moveit_config
This package contains moveit config and launch files for arm and gripper setup.

## Testing example:
1. Source the workspace
```
  source ~/mer_lab/ros_ws/devel/setup.bash
```
2. Launch the manipulation environment including integrated arm & gripper
```
  roslaunch manipulation_env franka_vf.launch
```
3. Sequence through following motion
```
  rosservice call /ih_manip/reset_sequence "start: 0"
```
```
  rosservice call /ih_manip/prepare_grasp "start: 0"
```
```
  rosservice call /ih_manip/object_grasp "start: 0"
```
```
  rosservice call /ih_manip/prehensile_pushing "start: 0"
```
```
  rosservice call /ih_manip/gravity_exploit "start: 0"
```
