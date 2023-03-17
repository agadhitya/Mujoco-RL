# Variable_friction_finger
## Steps to run the finger gripper node
### Setting things up:
### Workspace:
1. Download the repository into a catkin workspace.
2. Make the dev and build files by running this command.
> catkin_make

### Arduino:
1. Create the ros libraries for the arduino.
> rosrun rosserial_client make_libraries path_to_library 
2. This will create a ros_lib foder on the given path.
3. Now open the arduino installation folder and inside the libraries paste the ros_lib folder(Note: If one exists, delete it and paste the new one)
6. Copy the arduino code and burn it to the arduino board.
7. Check whether the arduino is connected to ACM0 port and the baud rate is set as 57600.

### Connecting the dynamixel:
(Note: This code is tested with XM430-W350 dynamixels(Protocol 2.0) motors only. For other motors, not supported)
1. Connect the dynamixel to USB0 and set the baud rate to 57600.
2. Set the Left Finger Motor ID:20 and Right Finger Motor Id:21 in the dynamixel wizard.
3. Incase if you want to change the motor id in the friction finger code, open the friction_finger_gripper/src/controller.cpp and modify as mentioned in the code.

### Launching the finger_gripper:
1. Run the following commands in different terminals:
> roscore
2. Run the launch file
> roslaunch friction_finger_gripper MotionPlanner
This launch file contains the controllers for the dynamixel motors and the arduino 


### Extracting the limits for the motor positions:
1. Keep the finger in home position(open position) and check whether the thread is attached tight with the finger. Then run the command
> rosservice call /read_pos 
2. This will give the current position of the motors.
Example: [1.0371184371184372, 0.6036630036630036]  -->[Left,Right]
3. Similarly the fingers at 90degree and run the same command. From this you can understand how the limit and the range of values which can be given.
4. To convert the motor's encoder values to the finger angles, run /read_pos for 3-4 positions of the finger and measure the angle the finger makes with the horizontal axis
5. Find the best fit values of (a, b) where Encoder_value = a*finger_angle + b
6. The procedure is the same for both the left and the right finger
7. Once the values are calculated, update these values in /src/friction_finger_gripper/config/ff_parameters.yaml

### Testing
### Available services:
(Note: The commands given below are with default values, change it according to your motors and need)
1. Hold the object: Enter the position values for left and right motor accordingly
> rosservice call /Hold_object "left: 0.0
right: 0.0" 

2. Slide Left up: Need to enter Right finger motor values, because during left finger up action, right finger is position controlled and the left is torque controlled
> rosservice call /Slide_Left_Finger_Up "data: 0.0" 

3. Slide Right up: Left motor value
> rosservice call /Slide_Right_Finger_Up "data: 0.0" 

4. Slide Left Down: Left motor value
> rosservice call /Slide_Left_Finger_Down "data: 0.0" 

5. Slide Right Down: Right motor value
> rosservice call /Slide_Right_Finger_Down "data: 0.0" 

6. Rotate clockwise: Left motor value
> rosservice call /Rotate_clockwise "data: 0.0"

7. Rotate anticlockwise: Right motor value
> rosservice call /Rotate_anticlockwise "data: 0.0"

8. Read Motor Position:
> rosservice call /read_pos

### Planner and Controller

### Motion Planner and Visual Servoing
1. To generate the plan run plan_arbitraryShape.py
2. To test it for object of a different shape change the variables X_vertices, Y_vertices
3. X_vertices, Y_vertices are the X and Y coordinates of the object
4. Similarly to change the control equations for Visual servoing change the variables X_vertices, Y_vertices in the VisualServoing_arbitraryShape.py

There are 3 methods for this 1. Motion Planner, 2. Visual Servoing, 3. Hybrid (Planner + Visual Servoing)
1. The choose the method you want, you can change the value of method in /src/friction_finger_gripper/config/ff_parameters.yaml
2. When MotionPlanner.launch is executed the Planner and the visual_servoing file is also executed


