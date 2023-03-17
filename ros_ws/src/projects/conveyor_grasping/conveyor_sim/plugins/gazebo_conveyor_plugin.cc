#include <functional>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>

namespace gazebo {
  class ConveyorPlugin : public ModelPlugin {
  public:
    ConveyorPlugin() : ModelPlugin() {}

    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) {
      model = _model;
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(std::bind(&ConveyorPlugin::OnUpdate, this));

      if(!_sdf->HasElement("speed"))
	gzerr << "No conveyor speed specified\n";
      if(!_sdf->HasElement("length"))
	gzerr << "No conveyor length specified\n";
      if(!_sdf->HasElement("root"))
	gzerr << "No conveyor root specified\n";

      speed = _sdf->GetElement("speed")->Get<double>();
      length = _sdf->GetElement("length")->Get<double>();
      root = _sdf->GetElement("root")->Get<double>();
    }

    void OnUpdate() {
      this->model->SetLinearVel(ignition::math::Vector3d(0, speed, 0));
      curPose = this->model->GetLink()->WorldPose();
      if(curPose.Pos().Y() - root >= length) {
	curPose.Pos().Y() = root; // Hardcoded start position, should change to initial position
	this->model->SetLinkWorldPose(curPose, "canonical");
      }
    }
  private:
    // Parameters
    double speed;
    double length;
    double root;

    // Other
    ignition::math::Pose3d curPose;
    physics::ModelPtr model;
    event::ConnectionPtr updateConnection;
  };

  GZ_REGISTER_MODEL_PLUGIN(ConveyorPlugin);
}
