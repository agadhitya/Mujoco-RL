# arm_controls
This package is used for high level motion planning of the Franka arm. Using the services advertised by the moveit_planner package, high level commands such as rest_pose, initialize_arm_pose, grasp_pose, move_up and move_down are implemented. To run the node, you can either launch arm_controller.launch or include the arm_ctrl node in your project specific launch file.
> This package contains:
> 1. yaml config files for parameters such as, initial, rest and grasp pose
> 2. launch file for arm_ctrl node
> 3. arm_ctrl node
> 4. service files for arm motion

Following services are the main functionalities provided by this package.

#### Available Services:

Do not forget to source the workspace
```
  source ~/mer_lab/ros_ws/devel/setup.bash
```
1. Reset arm (assign 0 to joints)
```
  rosservice call /reset_arm "execute: true"
```
2. Initialize arm pose (near grasp position)
```
  rosservice call /initialize_arm_pose "execute: true"
```
3. Get to grasp pose (object is between fingers)
```
  rosservice call /grasp_pose "execute: true"
```
4. Resting pose away from the objects
```
  rosservice call /rest_pose "execute: true"
```
5. Move up (from current position)
```
  rosservice call /move_up "val: 0.0"
```
6. Move down (from current position)
```
  rosservice call /move_down "val: 0.0"
```
7. Move left (from current position)
```
  rosservice call /move_left "val: 0.0"
```
8. Move right (from current position)
```
  rosservice call /move_right "val: 0.0"
```
9. Move forward (from current position)
```
  rosservice call /move_forward "val: 0.0"
```
10. Move back (from current position)
```
  rosservice call /move_back "val: 0.0"
```
