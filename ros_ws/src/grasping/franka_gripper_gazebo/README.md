# franka_gripper_gazebo
Package for applying a force command on the panda's fingers.  Designed primarily for gazebo

## Launching
Launch the node and specify the topics on which the finger controllers listen to.  The default is "/panda/panda_finger1_controller/command", which is compatible with the panda_finger_torque_controller package.

To provide a different topic, simply send the node arguments as follows:
1. Through the command line: "rosrun franka_gripper_gazebo gripper_node {finger1} {finger2}"
2. In a launch file: "node name="gripper_node" pkg="franka_gripper_gazebo" type="gripper_node" args="{finger1} {finger2}"

## Commands
Sending commands is done through a single service message: "franka_gripper_gazebo/GripMsg.h".

The topic is "gazebo_franka_grip"
