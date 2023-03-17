#ifndef _MANIPULATIONACTIONS_H_
#define _MANIPULATIONACTIONS_H_
#include "ros/ros.h"
#include <manipulation_planning/HandAct.h>
#include <manipulation_planning/ArmAct.h>

class ManipulationActions
{
private:
  ros::NodeHandle n_;
  ros::Duration dur;
  bool move_up(float val);
  bool move_down(float val);
  bool decrease_friction();
  bool increase_friction();
  bool slide_left_finger_down(float val);
  bool slide_left_finger_up(float val);
  bool slide_right_finger_down(float val);
  bool slide_right_finger_up(float val);
  bool rotate_clockwise(float val);
  bool rotate_anticlockwise(float val);

public:
  ManipulationActions();
  bool slide_left_down(manipulation_planning::HandAct::Request &req, manipulation_planning::HandAct::Response &res);
  bool slide_left_up(manipulation_planning::HandAct::Request &req, manipulation_planning::HandAct::Response &res);
  bool slide_right_down(manipulation_planning::HandAct::Request &req, manipulation_planning::HandAct::Response &res);
  bool slide_right_up(manipulation_planning::HandAct::Request &req, manipulation_planning::HandAct::Response &res);
  bool rotate_cw(manipulation_planning::HandAct::Request &req, manipulation_planning::HandAct::Response &res);
  bool rotate_acw(manipulation_planning::HandAct::Request &req, manipulation_planning::HandAct::Response &res);
  bool move_contact_up(manipulation_planning::ArmAct::Request &req, manipulation_planning::ArmAct::Response &res);
  bool move_contact_down(manipulation_planning::ArmAct::Request &req, manipulation_planning::ArmAct::Response &res);
};

#endif // _MANIPULATIONACTIONS_H_
