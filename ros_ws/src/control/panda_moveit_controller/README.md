# panda_moveit_controller
A control designed for use with the panda_arm_hand.xacro file and moveit

Therefore, it launches a jointTrajectoryController for the robot body.

To launch the controllers, you can either launch main.launch, or include it in your own project-specific launch file.  Make sure the robot is properly loaded first and exposes its hardware interfaces before launching, otherwise the controller will not find the joints

Do note that these controllers are launched under the /panda namespace, to avoid collision with others
