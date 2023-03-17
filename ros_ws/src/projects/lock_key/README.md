# lock_key
Key-in-Lock Insertion using FT sensing and RGBD vision.

## Instructions for Usage
### Physical Panda Robot
1. Prepare the Panda.
    1. Turn on computer near robot. Password is merlab.
    2. Power up arm (switch is on controller box underneath arm). Wait until the lights on the base of the arm are solid yellow.
    3. Unlock the joints using [desk](https://172.16.0.2/desk/ "desk").
    4. Disengage E-Stop button (rotate the button)
2. Move Panda to "start" 
    1. ```roslaunch franka_example_controllers move_to_start.launch robot_ip:=172.16.0.2 load_gripper:=true```
3. Launch key insertion procedure.
	1. ```roslaunch lock_key actual_panda.launch```
	2. ```rosrun lock_key actual_smach.py```

### Simulation (using deprecated method)
1. ```roslaunch lock_key sim.launch```
2. ```roslaunch lock_key run.launch```

**Note** commander.py should be able to be used with Gazebo by setting gazebo=True and modifying the ee_link. Current vision system does not work in simulation.

## Package contents
#### gazebo-pkgs
A collection of tools and plugins for Gazebo. Includes plugin for rigidly attaching key to arm when it is grasped.
#### general-message-pkgs
Collection of various message packages which can be useful to a broader range of other packages. Dependancy of gazebo-pkgs.
#### lock_key
Contains core launch, src, and script files. 
#### lock_key_msgs
*Should* contain all custom action, msg, and srv definitions for lock_key. Note that actual server and client source files using the custom definitions should still be placed in lock_key/src.
#### trac-ik
Provides an alternative IK solver to the inverse Jacobian methods in KDL. Seems to work better on actual arm.

## Useful commands
- Open/Close gripper on actual Panda (Set epsilon values to 0.05).
    - ```rostopic pub /franka_gripper/grasp/goal franka_gripper/GraspActionGoal ...```
- Send commands to the arm through Rviz
	- ```roslaunch panda_moveit_config panda_control_moveit_rviz.launch robot_ip:=172.16.0.2 load_gripper:=true```

## Open Issues
- [ ] Verify trac-ik respects joint limit parameters.
- [ ] Finish adding insertion plane orientation capabilities to vision system.
- [ ] Place additional constraints on lock plane detection (to not detect vertical planes).
- [ ] Add "return key to table" to SMACH.
- [ ] Make it easier to switch between Gazebo/Actual Panda development.
- [ ] Improve color segmentation for key/lock finding.
- [ ] Perform spiral search on any plane (not just horizontal).
- [ ] Tune spiral search parameters for higher success rate.
- [ ] Look for memory leaks. Is system stable over longer periods of time?
- [ ] Move actions and srv definitions in lock_key to lock_key_msgs. Update affected src files.
- [ ] Add collision objects to scene.