# lock_key
Contains core launch, src, and script files. 

## Main File Descriptions
### launch
- actual_panda.launch
	- Launches all nodes requires for key insertion on physical robot.
- run.launch
	- Secondary bringup file for Gazebo simulation. Unpauses Gazebo.
- sim.launch
	- Main bringup file for Gazebo simulation.
- vision.launch
	- Launches Realsense D435 node and associated key and lock finding nodes.

### scripts
- actual_smach.py
	- Defines SMACH/state machine for key insertion procedure.
- commander.py
	- Defines high-level interface for using MoveIt Commander with the Panda arm.
- image_utils.py
	- Contains image processing utility functions using cv2.
- key_finder_node.py
	- ROS node for lock and key color segmentation and centroid finding. Publishes lock and key centroids from color segmentation. Publishes visualization image. Publishes key angle from PCA.
- lockkey_finder_srv.py
	- ROS Service Server for retrieving average of the past 30 lock and key positions. Publishes lock and key positions as topics. Publishes key yaw as a topic.
- plane_finder_client.py
	- ROS node for calling plane finder service. Publishes point clouds, lock orientation, and lock plane coefficients.
- rosbag_utils.py
	- Saves messages from color and depth image ROS topics to JPEGs. Topics and bag are input by user at the bottom of this file.

### src
**Note:** Most .cpp files are deprecated and not used for key insertion. The following .cpp files are still used:
- find_planes.cpp
	- Finds and publishes separate key and lock point clouds. Finds primary lock plane and publishes associated coefficients.
- wrench_server.cpp
	- Servces getWrench and getAveWrench services for retrieving FT data from Panda.

The following .cpp file is not used anymore, but may be useful:
- actual_controller.cpp
	- Previously used as the main file for executing key insertion. It is now deprecated due to poor Cartesian planner performance and the inability to actually set velocity scaling when using the moveCart service.
