#include "gripper_controls/Hand.h"
#include <ros/ros.h>

int main(int argc, char **argv){
  ros::init(argc, argv, "High_level_action_server");
  ros::NodeHandle n;

  // to be able to run both arm+gripper and gripper
  std::string topic_js;
  if(argc == 1) {
    ROS_INFO("No arm argument given, defaulting to gripper only");
    topic_js = "joint_states";
  }
  else if(argc == 2) {
    ROS_INFO_STREAM("Starting controller with namespace " << argv[1]);
    std::string ns;
    ns = argv[1];
    topic_js = ns + "/joint_states";
  }

  // Construct Hand object
  Hand my_hand(topic_js);
  // Advertise services
  ros::ServiceServer service_slfd = n.advertiseService("Slide_Left_Finger_Down", &Hand::slide_left_down, &my_hand);
  ros::ServiceServer service_slfu = n.advertiseService("Slide_Left_Finger_Up", &Hand::slide_left_up, &my_hand);
  ros::ServiceServer service_srfd = n.advertiseService("Slide_Right_Finger_Down", &Hand::slide_right_down, &my_hand);
  ros::ServiceServer service_srfu = n.advertiseService("Slide_Right_Finger_Up", &Hand::slide_right_up, &my_hand);
  ros::ServiceServer service_ranti = n.advertiseService("Rotate_anticlockwise", &Hand::rotate_anticlockwise, &my_hand);
  ros::ServiceServer service_cw = n.advertiseService("Rotate_clockwise", &Hand::rotate_clockwise, &my_hand);
  ros::ServiceServer service_hold = n.advertiseService("Hold_object", &Hand::hold_object, &my_hand);
  ros::spin();
  return 0;
}
