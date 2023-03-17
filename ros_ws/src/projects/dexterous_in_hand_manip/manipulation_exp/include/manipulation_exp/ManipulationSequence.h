#ifndef _MANIPULATIONSEQUENCE_H_
#define _MANIPULATIONSEQUENCE_H_
#include "ros/ros.h"
#include <manipulation_exp/Sequence.h>

class ManipulationSequence
{
private:
  ros::NodeHandle n_;
  ros::Duration dur;
  bool reset_world();
  bool release_object();
  bool go_to_rest();
  bool initialize_arm();
  bool approach_object();
  bool grasp_object();
  bool move_up(float val);
  bool move_down(float val);
  bool decrease_friction();
  bool increase_friction();
  bool slide_left_down();
  bool slide_left_up();
  bool slide_right_down();
  bool slide_right_up();
  bool rotate_clockwise();
  bool rotate_anticlockwise();

public:
  ManipulationSequence();
  bool reset_sequence(manipulation_exp::Sequence::Request &req, manipulation_exp::Sequence::Response &res);
  bool prepare_grasp(manipulation_exp::Sequence::Request &req, manipulation_exp::Sequence::Response &res);
  bool object_grasp(manipulation_exp::Sequence::Request &req, manipulation_exp::Sequence::Response &res);
  bool gravity_exploit(manipulation_exp::Sequence::Request &req, manipulation_exp::Sequence::Response &res);
  bool prehensile_pushing(manipulation_exp::Sequence::Request &req, manipulation_exp::Sequence::Response &res);
  bool sliding_test(manipulation_exp::Sequence::Request &req, manipulation_exp::Sequence::Response &res);
  bool rotating_test(manipulation_exp::Sequence::Request &req, manipulation_exp::Sequence::Response &res);
};

#endif // _MANIPULATIONSEQUENCE_H_
