# gripper_controls
This package is based on the motion planning nodes in variable_friction_gripper project, which prepared to be used on hardware. Same logic and functionality is transferred to the gazebo simulation in this package.

Low level controller is based on the class LowLvlController and deals with low level tasks such as receiving and publishing effort, position references, friction values for the left and right fingers.

High level controller is based on the class Hand. It advertises services of high level commands such as, Hold_object, Slide_Left_Finger_Down, Slide_Left_Finger_Up, Slide_Right_Finger_Down, Slide_Right_Finger_Up, Rotate_clockwise, Rotate_anticlockwise, by using the low level commands.

Also a sequential command node is included, that helps testing the services advertised by the high level controller. It is possible to use the torque publisher and a plotting method of your own choice to analyse the gripper behavior during tasks.

high_lvl_ctrl and low_lvl_ctrl nodes can either be run seperately, or you may use the launch file that launches both controllers.

> This package contains:
> 1. yaml config files for low level and high level controller parameters
> 2. launch file for gripper controller
> 3. Implementation of high and low level controllers
> 4. A sequential command node for testing the advertised services
> 4. A torque publisher for analyzing the gripper behavior

Following services are the main functionalities provided by this package.

#### Available Services:

Do not forget to source the workspace
```
  source ~/mer_lab/ros_ws/devel/setup.bash
```
1. Slide object down on left finger, provide left finger position reference (+ for opening, - for closing direction)
```
  rosservice call /Slide_Left_Finger_Down "data: 0.0"
```
2. Slide object up on left finger, provide right finger position reference (+ for opening, - for closing direction)
```
  rosservice call /Slide_Left_Finger_Up "data: 0.0"
```
3. Slide object down on right finger, provide right finger position reference (+ for opening, - for closing direction)
```
  rosservice call /Slide_Right_Finger_Down "data: 0.0"
```
4. Slide object up on right finger, provide left finger position reference (+ for opening, - for closing direction)
```
  rosservice call /Slide_Right_Finger_Up "data: 0.0"
```
5. Rotate object anticlockwise, provide left finger position reference (+ for opening, - for closing direction)
```
  rosservice call /Rotate_anticlockwise "data: 0.0"
```
6. Rotate object anticlockwise, provide right finger position reference (+ for opening, - for closing direction)
```
  rosservice call /Rotate_clockwise "data: 0.0"
```
7. Hold object, provide position references for both fingers (+ for opening, - for closing direction)
```
  rosservice call /Hold_object "left: 0.0 right: 0.0"
```
7. Set friction of selected finger (finger: 0 for left, 1 for right, high_friction: 0 for low_friction, 1 for high_friction)
```
  rosservice call /set_friction "finger:0 high_friction:false"
```
