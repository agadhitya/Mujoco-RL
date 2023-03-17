#include <ros/ros.h>
// Services
#include "lock_key/imgCapture.h"
#include "lock_key/findKey.h"
#include "moveit_planner/MovePose.h"
#include "moveit_planner/MoveCart.h"
#include "franka_gripper_gazebo/GripMsg.h"
#include "franka_pos_grasping_gazebo/GripPos.h"
#include "moveit_planner/GetPose.h"
#include "moveit_planner/MoveJoint.h"
#include "moveit_planner/SetVelocity.h"
#include "moveit_planner/AddCollision.h"
#include <moveit_msgs/CollisionObject.h>
// Actions
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <lock_key/SpiralInsertAction.h>
// Publish Marker
#include <visualization_msgs/Marker.h>
// Transform Listener
#include <tf/transform_listener.h>
#include <geometry_msgs/PointStamped.h>

typedef actionlib::SimpleActionClient<lock_key::SpiralInsertAction> SpiralClient;

void wait_for_service(int32_t timeout){
    ros::service::waitForService("move_to_pose",timeout);
    ros::service::waitForService("imgCapture", timeout);
    ros::service::waitForService("gripPosServer", timeout);
    ros::service::waitForService("get_pose", timeout);
    ros::service::waitForService("move_to_joint_space", timeout);
    ros::service::waitForService("set_velocity_scaling", timeout);
    ros::service::waitForService("add_collision_object",timeout);
    ROS_INFO("SERVICES READY");
}

int main(int argc, char **argv){

    ros::init(argc, argv, "control");
    ros::NodeHandle n;

    //Waiting for services to be available
    int32_t timeout = 1000;
    wait_for_service(timeout);

    // Transform Listener

    tf::TransformListener listener;

    //Declaring Clients
    ros::ServiceClient imgCaptureClient = n.serviceClient<lock_key::imgCapture>("imgCaptureServer");
    ros::ServiceClient findKeyClient = n.serviceClient<lock_key::findKey>("findKeyServer");
    ros::ServiceClient movePoseClient = n.serviceClient<moveit_planner::MovePose>("move_to_pose");
    ros::ServiceClient movePoseClient2 = n.serviceClient<moveit_planner::MovePose>("move_to_pose");
    ros::ServiceClient moveCartClient = n.serviceClient<moveit_planner::MoveCart>("cartesian_move");
    ros::ServiceClient gripperClient = n.serviceClient<franka_gripper_gazebo::GripMsg>("gazebo_franka_grip");
    ros::ServiceClient gripperPosClient = n.serviceClient<franka_pos_grasping_gazebo::GripPos>("gripPosServer");
    ros::ServiceClient getPoseClient = n.serviceClient<moveit_planner::GetPose>("get_pose");
    ros::ServiceClient jointSpaceClient = n.serviceClient<moveit_planner::MoveJoint>("move_to_joint_space");
    ros::ServiceClient velScalingClient = n.serviceClient<moveit_planner::SetVelocity>("set_velocity_scaling");
    ros::ServiceClient addCollisionClient = n.serviceClient<moveit_planner::AddCollision>("add_collision_object");

    // Client Objects
    lock_key::imgCapture reqImg;
    lock_key::findKey keyImg;
    moveit_planner::MovePose pose;
    moveit_planner::MovePose pose2;
    moveit_planner::MoveCart cart;
    moveit_planner::MoveCart cart2;
    moveit_planner::MoveCart cart3;
    franka_gripper_gazebo::GripMsg grip;
    franka_pos_grasping_gazebo::GripPos grasp;
    moveit_planner::GetPose curPose;
    moveit_planner::MoveJoint jpos;
    moveit_planner::SetVelocity velscale;

    //Testing get pose
    getPoseClient.call(curPose);
    geometry_msgs::Pose botpose;
    botpose = curPose.response.pose;
    ROS_INFO_STREAM(botpose);

    //Client Calls
    imgCaptureClient.call(reqImg);
    keyImg.request.pc2 = reqImg.response.pc2;
    ROS_INFO("IMAGE RECEIVED");
    findKeyClient.call(keyImg);
    ROS_INFO("Key found");

    //Transforming Key position to robot frame
    geometry_msgs::PointStamped keypose;
    listener.transformPoint("panda_link0",keyImg.response.p,keypose);
    ROS_INFO_STREAM(keypose);
    // Visualization and Debugging
    ros::Publisher cloud_pub = n.advertise<sensor_msgs::PointCloud2>("debugPC2", 1);
    ros::Publisher centroid_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 1);

    visualization_msgs::Marker marker;

    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = .005;
    marker.scale.y = .005;
    marker.scale.z = .005;
    marker.color.a = 1;
    marker.color.r = 1;
    marker.color.g = 1;
    marker.color.b = 1;
    marker.pose.orientation.w = 1;
    marker.header.frame_id="/panda_camera_optical_link";
    marker.header.stamp = ros::Time();
    marker.id = 0;
    marker.pose.position.x = keyImg.response.p.point.x;
    marker.pose.position.y = keyImg.response.p.point.y;
    marker.pose.position.z = keyImg.response.p.point.z;

    ROS_INFO_STREAM(marker.pose);

    // int count =0;
    //  while (count<50)
    //  {
    //     cloud_pub.publish(keyImg.response.pc2);
    //     centroid_pub.publish(marker);
    //     ROS_INFO("KEY Published");
    //     count += 1;
    //  }

    // TO Do : Move this stuff to Pre Grasp Service
    // Opening Gripper
    //  grasp.request.finger_pos = 0.04;
    //  gripperPosClient.call(grasp);
    
    //Manipulating position so as to create pre-grasp pose
    keypose.point.x -= 0.06;
    keypose.point.z -= 0.006;
    
    //Setting pre-grasp orientation
    pose.request.val.position = keypose.point;
    pose.request.val.orientation.x = 0.0;//0.3826834;// 0.7071068; //-0.4082483;
    pose.request.val.orientation.y = 0.7071068; //0.8164966;
    pose.request.val.orientation.z = 0.0; //-0.4082483;
    pose.request.val.orientation.w = 0.7071068;

    // Opening Gripper and moving to pre-grasp pose
    pose.request.execute = true;
    grip.request.force = 50.0;
    gripperClient.call(grip);
    movePoseClient.call(pose);

    // Here instead of calling movePoseClient as done initially
    // We are using Move to joint after recording the joint positions for the position
    // This is done to improve reliability

    jpos.request.execute = true;
    jpos.request.val.push_back(0.04043039654191989);
    jpos.request.val.push_back(0.04002915885797289);

    jpos.request.val.push_back(1.874648704741471);
    jpos.request.val.push_back(1.691488029202242);
    jpos.request.val.push_back(-1.761014720549876);
    jpos.request.val.push_back(-2.4737772829115077);
    jpos.request.val.push_back(2.764633621468974);
    jpos.request.val.push_back(2.5131015865487534);
    jpos.request.val.push_back(2.7243020563317195);

    //jointSpaceClient.call(jpos);


    // Move to grasp
    geometry_msgs::Pose p;
    p.position = keypose.point;
    p.orientation = pose.request.val.orientation;
    p.position.x += 0.06;   
    cart.request.val.push_back(p);
    cart.request.execute = true;
    moveCartClient.call(cart);
    ros::Duration(1).sleep();
    
    // Grasp
    // For the Position Controller
    ROS_INFO("Gripping");
    // grasp.request.finger_pos = 0.0;
    // gripperPosClient.call(grasp);
    // ros::Duration(3).sleep();


    // For the force controller
    grip.request.force = -0.10;
    gripperClient.call(grip);
    ROS_INFO("Gripped");
    ros::Duration(1).sleep();
    grip.request.force = -20.0;
    gripperClient.call(grip);
    // ros::Duration(0.1).sleep();
    // grip.request.force = -20.0;
    // gripperClient.call(grip);
    // ros::Duration(0.1).sleep();
    // grip.request.force = -125.0; //-100.0
    // gripperClient.call(grip);
    // ros::Duration(0.2).sleep();
    // Attach key to robot
    // First, remove key from environment (this part works)
    // moveit_planner::AddCollision collisionObj;
    // moveit_msgs::CollisionObject remove_object;
    // remove_object.id = "key";
    // remove_object.header.frame_id = "panda_link0";
    // remove_object.operation = remove_object.REMOVE;
    // collisionObj.request.collObject = remove_object;
    // addCollisionClient.call(collisionObj);

    // ROS_INFO("Attaching the object to the right wrist and removing it from the world.");
    // moveit_msgs::PlanningScene planning_scene;
    // planning_scene.world.collision_objects.clear();
    // planning_scene.world.collision_objects.push_back(remove_object);
    // planning_scene.robot_state.attached_collision_objects.push_back(attached_object);
    // planning_scene_diff_publisher.publish(planning_scene);

    // Move away
    ROS_INFO("Moving away");
    cart2.request.val.push_back(p);
    p.position.x -= 0.04;
    cart2.request.val.push_back(p);
    p.position.z += 0.05;
    cart2.request.val.push_back(p);
    cart2.request.execute = true;
    moveCartClient.call(cart2);
    ros::Duration(1).sleep();
    
    //Move to Lock
    ROS_INFO("Moving to Lock");
    
    // reducing velocity
    velscale.request.velScaling = 0.3;
    velScalingClient.call(velscale);
    
    // Get padlock goal position
    n.getParam("padlock_goal_x",pose2.request.val.position.x);
    pose2.request.val.position.x+=0.04; //Add depth/2 of box
    n.getParam("padlock_goal_y",pose2.request.val.position.y);
    pose2.request.val.position.y-=0.0125; //Subtract width/2 of box
    n.getParam("padlock_goal_z",pose2.request.val.position.z);
    pose2.request.val.position.z+=0.094; //Add height of box and extra

    //pose2.request.val.position.x = 0.65;
    //pose2.request.val.position.y = 0.0;
    //pose2.request.val.position.z = 0.444; //0.45
    pose2.request.val.orientation.w = 0;//.00653015;
    pose2.request.val.orientation.x = 0;//.915333;
    pose2.request.val.orientation.y = 1; //.402644;
    pose2.request.val.orientation.z = 0; //.000463674;
    pose2.request.execute = true;
    movePoseClient2.call(pose2);

    // normal velocity
    velscale.request.velScaling = 1.0;
    velScalingClient.call(velscale);

    ROS_INFO("Waiting for Insertion Action Server");
    SpiralClient client("spiral_insert_key", true); // true -> don't need ros::spin()
    client.waitForServer();

    lock_key::SpiralInsertGoal goal;

    ROS_INFO("Retrieving spiral parameters");
    n.getParam("spiral_Ft", goal.Ft);
    n.getParam("spiral_Fd", goal.Fd);
    n.getParam("spiral_Fi", goal.Fi);
    n.getParam("spiral_delta_max", goal.delta_max);

    ROS_INFO("Sending action goal");
    client.sendGoal(goal);

    // ros::Duration(1).sleep();
    // p.position = pose2.request.val.position;
    // p.orientation = pose2.request.val.orientation;
    // cart3.request.val.push_back(p);
    // p.position.z -= 0.1;
    // cart3.request.val.push_back(p);
    // cart3.request.execute = true;
    // moveCartClient.call(cart3);


    ros::spin();

    return 0;
}
