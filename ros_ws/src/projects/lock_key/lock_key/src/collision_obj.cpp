#include <ros/ros.h>
#include "moveit_planner/AddCollision.h"
#include <moveit_msgs/CollisionObject.h>

int main(int argc, char **argv){
    ros::init(argc, argv, "collision_obj");
    ros::NodeHandle n;

    // Wait for Service
    int32_t timeout = 1000;
    ros::service::waitForService("add_collision_object",timeout);
    
    // Creating Service Client object
    ros::ServiceClient addCollisionClient = n.serviceClient<moveit_planner::AddCollision>("add_collision_object");

    // Creating Message Object
    moveit_planner::AddCollision collisionObj;
    moveit_planner::AddCollision collisionObj2;

    // Adding the padlock as a collision object
    moveit_msgs::CollisionObject padlock;
    padlock.header.frame_id = "panda_link0";
    padlock.id = "padlock";

    shape_msgs::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[0] = 0.08;
    primitive.dimensions[1] = 0.05;
    primitive.dimensions[2] = 0.025;

    geometry_msgs::Pose padlock_pose;
    padlock_pose.orientation.x = 0.7071068;
    padlock_pose.orientation.y = 0.0;
    padlock_pose.orientation.z = 0.0;
    padlock_pose.orientation.w = 0.7071068;

    // Get padlock goal position
    n.getParam("padlock_goal_x",padlock_pose.position.x);
    n.getParam("padlock_goal_y",padlock_pose.position.y);
    n.getParam("padlock_goal_z",padlock_pose.position.z);
    // "Move" CF to center of box to comply with shape_msg/SolidPrimitive format
    padlock_pose.position.x+= primitive.dimensions[0]/2.0;
    padlock_pose.position.y+= primitive.dimensions[1]/2.0;
    padlock_pose.position.z+= primitive.dimensions[2]/2.0;

    padlock.primitives.push_back(primitive);
    padlock.primitive_poses.push_back(padlock_pose);
    padlock.operation = padlock.ADD;

    // Sending objects to client
    collisionObj.request.collObject = padlock;
    addCollisionClient.call(collisionObj);

   // Adding the key as a collision object
    moveit_msgs::CollisionObject key;
    key.header.frame_id = "panda_link0";
    key.id = "key";

    shape_msgs::SolidPrimitive primitive2;
    primitive2.type = primitive2.BOX;
    primitive2.dimensions.resize(3);
    primitive2.dimensions[0] = 0.0145;     
    primitive2.dimensions[1] = 0.0055;
    primitive2.dimensions[2] = 0.06;

    geometry_msgs::Pose key_pose;
    key_pose.orientation.x = 0.0;
    key_pose.orientation.y = -0.7071068;
    key_pose.orientation.z = 0.0;
    key_pose.orientation.w = 0.7071068;
    n.getParam("key_x",key_pose.position.x);
    n.getParam("key_y",key_pose.position.y);
    n.getParam("key_z",key_pose.position.z);
    // "Move" CF to center of box to comply with shape_msg/SolidPrimitive format
    key_pose.position.x+= primitive2.dimensions[0]/2.0;
    key_pose.position.y+= primitive2.dimensions[1]/2.0;
    key_pose.position.z+= primitive2.dimensions[2]/2.0;
    // key_pose.position.x = 0.007250;
    // key_pose.position.y = 0.002750;
    // key_pose.position.z = 0.10000;

    key.primitives.push_back(primitive);
    key.primitive_poses.push_back(key_pose);
    key.operation = key.ADD;

    // Sending objects to client
    collisionObj2.request.collObject = key;
    addCollisionClient.call(collisionObj2);

    // Adding the table as a collision object
    // moveit_msgs::CollisionObject surface;
    // surface.header.frame_id = "panda_link0";
    // surface.id = "table_surface";

    // shape_msgs::SolidPrimitive primitive;
    // primitive.type = primitive.BOX;
    // primitive.dimensions.resize(3);
    // primitive.dimensions[0] = 0.6;
    // primitive.dimensions[1] = 1;
    // primitive.dimensions[2] = 0.005;

    // geometry_msgs::Pose surface_pose;
    // surface_pose.orientation.w = 1;
    // surface_pose.position.x = 0.75;
    // surface_pose.position.y = 0.0;
    // surface_pose.position.z = 0.4;

    // surface.primitives.push_back(primitive);
    // surface.primitive_poses.push_back(surface_pose);
    // surface.operation = surface.ADD;

    // // Sending objects to client
    // collisionObj.request.collObject = surface;
    // addCollisionClient.call(collisionObj);
    
    return 0;
}

