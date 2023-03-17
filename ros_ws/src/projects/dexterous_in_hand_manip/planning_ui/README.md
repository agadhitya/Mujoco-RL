# planning_ui

This package contains an easy-to-use user interface to be used with the gazebo simulation. UI window is designed using Qt (window design is in *forms/mainWindow.ui*). Package contains the main node: *plan_ui*

## plan_ui

*plan_ui* is the node for the UI, controlling the functionalities on the UI window. It calls ros services upon interaction with the buttons and text boxes on the UI window. This node utilizes the services advertised by *manipulation_exp*, *manipulation_planning*, *gripper_controls*, and *arm_controls* packages.

## How to use?

Run *plan_ui* with:

```
  rosrun planning_ui plan_ui
```
