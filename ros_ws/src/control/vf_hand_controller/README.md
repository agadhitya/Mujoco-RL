# vf_hand_controller
Controller package designed for use with the vf_hand.xacro file

It includes both position and effort controllers. During launch it spawns position controllers and loads the effort controllers as stopped.

To launch the controllers, you can either launch main.launch, or include it in your own project-specific launch file.  Make sure the robot is properly loaded first and exposes its hardware interfaces before launching, otherwise the controller will not find the joints.

Do note that these controllers are launched under the /vf_hand namespace, to avoid collision with others.
