#include "manipulation_exp/ManipulationSequence.h"
#include <ros/ros.h>

int main(int argc, char **argv){
  ros::init(argc, argv, "manipulation_sequence_server");
  ros::NodeHandle n;
  ManipulationSequence manip_seq;
  ros::ServiceServer srv_reset = n.advertiseService("ih_manip/reset_sequence", &ManipulationSequence::reset_sequence, &manip_seq);
  ros::ServiceServer srv_prep = n.advertiseService("ih_manip/prepare_grasp", &ManipulationSequence::prepare_grasp, &manip_seq);
  ros::ServiceServer srv_grasp = n.advertiseService("ih_manip/object_grasp", &ManipulationSequence::object_grasp, &manip_seq);
  ros::ServiceServer srv_gravexp = n.advertiseService("ih_manip/gravity_exploit", &ManipulationSequence::gravity_exploit, &manip_seq);
  ros::ServiceServer srv_prepush = n.advertiseService("ih_manip/prehensile_pushing", &ManipulationSequence::prehensile_pushing, &manip_seq);
  ros::ServiceServer srv_slide = n.advertiseService("ih_manip/sliding_test", &ManipulationSequence::sliding_test, &manip_seq);
  ros::ServiceServer srv_rotat = n.advertiseService("ih_manip/rotating_test", &ManipulationSequence::rotating_test, &manip_seq);
  ros::spin();
  return 0;
}
