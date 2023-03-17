# Moveit Planner
A package used for generic motion planning using the moveit package

## Starting node
Make sure moveit is already running before starting up this node

The executable in this package is called 'main_node':

'rosrun moveit_planner main_node'

This node can optionally be given as an argument the planning_group name to properly connect moveit to different groups.

By default, the planning_group is the 'arm' group, but different names can be given:
1. Through the command line: rosrun moveit_planner main_node panda_arm
2. Through a launch file: node name="moveit_planner" pkg="moveit_planner" type="main_node" args="panda_arm"

## Services
This package exposes several services to interact with moveit, they are:
1. Pose Target
   - Topic: "move_to_pose"
   - Message: "moveit_planner/MovePose.h"
2. Joint Space Target
   - Topic: "move_to_joint_space"
   - Message: "moveit_planner/MoveJoint.h"
3. Cartesian Movement
   - Topic: "cartesian_move"
   - Message: "moveit_planner/MoveCart.h"
4. Set Velocity Scaling
   - Topic: "set_velocity_scaling"
   - Message: "moveit_planner/SetVelocity.h"
5. Move Away from Target (moves a "distance" away from "pose" in that pose's z-direction)
   - Topic: "move_away_point"
   - Message: "moveit_planner/MoveAway.h"
6. Add Collision Object
   - Topic: "add_collision_object"
   - Message: "moveit_planner/AddCollision.h"