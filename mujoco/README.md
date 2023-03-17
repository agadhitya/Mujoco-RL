# mujoco
Contains files for use with Mujoco Simulator.

## Setup
**Assumptions:** 
- Mujoco is at ~/.mujoco/mujoco200. 
- mjkey.txt is placed in ~/.mujoco, /.mujoco/mujoco200, and /.mujoco/mujoco200/bin.
- Add this line to the bottom of your ~/.bashrc.
	- ```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<YOUR_USERNAME>/.mujoco/mujoco200/bin```

## Notes
Here are a few options for connecting ROS, but none have worked smoothly:
- https://github.com/shadow-robot/mujoco_ros_pkgs
- https://github.com/saga0619/mujoco_ros_sim