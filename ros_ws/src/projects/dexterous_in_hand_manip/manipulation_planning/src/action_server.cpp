#include "manipulation_planning/ManipulationActions.h"
#include <ros/ros.h>

int main(int argc, char **argv){
  ros::init(argc, argv, "manipulation_actions_server");
  ros::NodeHandle n;
  ManipulationActions manip_acts;
  // Advertise services for manipulation primitives
  ros::ServiceServer srv_slfd = n.advertiseService("plan_exe/slide_left_down", &ManipulationActions::slide_left_down, &manip_acts);
  ros::ServiceServer srv_slfu = n.advertiseService("plan_exe/slide_left_up", &ManipulationActions::slide_left_up, &manip_acts);
  ros::ServiceServer srv_srfd = n.advertiseService("plan_exe/slide_right_down", &ManipulationActions::slide_right_down, &manip_acts);
  ros::ServiceServer srv_srfu = n.advertiseService("plan_exe/slide_right_up", &ManipulationActions::slide_right_up, &manip_acts);
  ros::ServiceServer srv_rc = n.advertiseService("plan_exe/rotate_cw", &ManipulationActions::rotate_cw, &manip_acts);
  ros::ServiceServer srv_rac = n.advertiseService("plan_exe/rotate_acw", &ManipulationActions::rotate_acw, &manip_acts);
  ros::ServiceServer srv_mcu = n.advertiseService("plan_exe/move_contact_up", &ManipulationActions::move_contact_up, &manip_acts);
  ros::ServiceServer srv_mcd = n.advertiseService("plan_exe/move_contact_down", &ManipulationActions::move_contact_down, &manip_acts);
  ros::spin();
  return 0;
}
