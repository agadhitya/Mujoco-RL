#include <ros/ros.h>
#include <simulation_ui/SetParam.h>
#include <simulation_ui/SimParam.h>
#include <simulation_ui/SimFriction.h>

class ParamSetter{
private:
  double muTorsion;
  bool usePR;
  double patchRadius;
  double surfaceRadius;
  double lowFriction;
  double highFriction;
public:
  ParamSetter(){
    muTorsion = 0.5;
    usePR = true;
    patchRadius = 0.5;
    surfaceRadius = 0.5;
    lowFriction = 0.1;
    highFriction = 10000;
  }
  bool muTorsion_setter(simulation_ui::SetParam::Request &req, simulation_ui::SetParam::Response &res){
    muTorsion = req.param;
    res.success = 1;
    return 1;
  }

  bool usePR_setter(simulation_ui::SetParam::Request &req, simulation_ui::SetParam::Response &res){
    if (req.param)
      usePR = true;
    else
      usePR = false;
    res.success = 1;
    return 1;
  }

  bool patchRadius_setter(simulation_ui::SetParam::Request &req, simulation_ui::SetParam::Response &res){
    patchRadius = req.param;
    res.success = 1;
    return 1;
  }
  bool surfaceRadius_setter(simulation_ui::SetParam::Request &req, simulation_ui::SetParam::Response &res){
    surfaceRadius = req.param;
    res.success = 1;
    return 1;
  }
  bool lowFriction_setter(simulation_ui::SetParam::Request &req, simulation_ui::SetParam::Response &res){
    lowFriction = req.param;
    res.success = 1;
    return 1;
  }
  bool highFriction_setter(simulation_ui::SetParam::Request &req, simulation_ui::SetParam::Response &res){
    highFriction = req.param;
    res.success = 1;
    return 1;
  }
  double get_muTorsion(){
    return muTorsion;
  }
  bool get_usePR(){
    return usePR;
  }
  double get_patchRadius(){
    return patchRadius;
  }
  double get_surfaceRadius(){
    return surfaceRadius;
  }
  double get_lowFriction(){
    return lowFriction;
  }
  double get_highFriction(){
    return highFriction;
  }
};

int main(int argc, char **argv){
  ros::init(argc, argv, "param_setter_server");
  ros::NodeHandle n;
  ParamSetter param_setter;

  // Advertise services for UI
  ros::ServiceServer service_set_muTorsion = n.advertiseService("set_muTorsion", &ParamSetter::muTorsion_setter, &param_setter);
  ros::ServiceServer service_set_usePR = n.advertiseService("set_usePR", &ParamSetter::usePR_setter, &param_setter);
  ros::ServiceServer service_set_patchRadius = n.advertiseService("set_patchRadius", &ParamSetter::patchRadius_setter, &param_setter);
  ros::ServiceServer service_set_surfaceRadius = n.advertiseService("set_surfaceRadius", &ParamSetter::surfaceRadius_setter, &param_setter);
  ros::ServiceServer service_set_lowFriction = n.advertiseService("set_lowfriction", &ParamSetter::lowFriction_setter, &param_setter);
  ros::ServiceServer service_set_highFriction = n.advertiseService("set_highfriction", &ParamSetter::highFriction_setter, &param_setter);

  // Publishers (simulation parameters and friction parameters)
  ros::Publisher param_pub = n.advertise<simulation_ui::SimParam>("/vf_hand/parameter_setter", 1000);
  ros::Publisher friction_pub = n.advertise<simulation_ui::SimFriction>("/vf_hand/friction_limits", 1000);

  ros::Rate loop_rate(10);

  while(ros::ok()){

    // Get current parameters and generate parameter message
    simulation_ui::SimParam param_msg;
    param_msg.MuTorsion = param_setter.get_muTorsion();
    param_msg.usePR = param_setter.get_usePR();
    param_msg.PatchRadius = param_setter.get_patchRadius();
    param_msg.SurfaceRadius = param_setter.get_surfaceRadius();
    // Publish parameter message
    param_pub.publish(param_msg);

    // Get current friction limits and generate friction message
    simulation_ui::SimFriction friction_msg;
    friction_msg.low = param_setter.get_lowFriction();
    friction_msg.high = param_setter.get_highFriction();
    //  Publish friction message
    friction_pub.publish(friction_msg);
    
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
