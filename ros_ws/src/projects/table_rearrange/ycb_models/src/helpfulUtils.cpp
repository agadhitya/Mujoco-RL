#include <ros/ros.h>
#include <ros/package.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>

#include <string>
#include <fstream>
#include <sstream>

geometry_msgs::Point getPosition(const std::string& param, ros::NodeHandle& nh, bool& found) {
  geometry_msgs::Point ret;
  if(!nh.getParam(param + "/x", ret.x) ||
     !nh.getParam(param + "/y", ret.y) ||
     !nh.getParam(param + "/z", ret.z)) {
    ret.x = 0; ret.y = 0; ret.z = 0; // Position to return if not found
    found = false;
    return ret;
  }

  found = true;
  return ret;
}

geometry_msgs::Quaternion getQuaternion(const std::string& param, ros::NodeHandle& nh, bool& found) {
  geometry_msgs::Quaternion ret;
  if(!nh.getParam(param + "/x", ret.x) ||
     !nh.getParam(param + "/y", ret.y) ||
     !nh.getParam(param + "/z", ret.z) ||
     !nh.getParam(param + "/w", ret.w)) {
    ret.w = 1; ret.x = 0; ret.y = 0; ret.z = 0; // Quaternion to return if not found
    found = false;
    return ret;
  }

  found = true;
  return ret;
}

std::string slurp(std::ifstream& in) {
  std::ostringstream sstr;
  sstr << in.rdbuf();
  return sstr.str();
}

std::string loadLocal(const std::string& name, bool& found) {
  std::string ycbPath = ros::package::getPath("ycb_models") + "/desc/";
  std::ifstream ifs;
  ifs.open(ycbPath + name + ".urdf");
  if(ifs.fail()) {
    found = false;
    return "";
  }

  found = true;
  std::string ret = slurp(ifs);
  ifs.close();
  return ret;
}
