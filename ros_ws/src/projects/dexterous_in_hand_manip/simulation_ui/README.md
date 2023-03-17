# simulation_ui

This package contains an easy-to-use user interface to be used with the gazebo simulation. UI window is designed using Qt (window design is in *forms/mainWindow.ui*). Package contains two main nodes: *sim_ui* and *param_setter*

## sim_ui

*sim_ui* is the node for the UI, controlling the functionalities on the UI window. It calls ros services upon interaction with the buttons and sliders on the UI window.

## param_setter

*param_setter* is a node advertising services for setting simulation parameters and publishing the parameters on designated ros topics. These topics are subscribed by the simulation plugin allowing real-time modification of simulation parameters.

## How to use?

Launch *simulation_ui.launch* to run both nodes and access the UI window.
