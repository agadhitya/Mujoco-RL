This package contains 2 classes that can be repurposed in other packages.

1. Movement Class
2. Recorder Class


1. Movement Class Public Methods -

    For more information on how to call the services in the class, check out the source code in main_pierce.cpp

    Movement(ros::Nodehandl& _n) 
            This is the default constructor of the class
    
    bool prepose_callback(piercing_experiments::PrePose::Request& req, piercing_experiments::PrePose::Response& res)

    This Service can move the robot to a specified position and orientation.

    bool cartesian_path_callback(piercing_experiments::CartesianPath::Request& req, piercing_experiments::CartesianPath::Response& res)

    This service can move the robot in cartesian co-ordinates by a specified distance

2. Recorder Class - 

    The recorder class allows you to record topics of your choice to a rosbag. To use the class, an instance of the class should be created in a ROS Node. For an example on how to do this check the source code in record.cpp

    To add new topics to the Record Class follow the comments in the class

    NOTE: Make sure to comment out topics you are not publishing to. If you don't do this there will be several blank topics written to the bag file and rosbag play will not be able to play the file. If you are usinf rqt_bag the you will be able to view the data but if you are exporting data from the bag file rosbag will be unable to use the bag.
