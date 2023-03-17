#include "ros/ros.h"

#include "geometry_msgs/Pose.h"
#include "gazebo_msgs/ModelStates.h"
#include "gazebo_msgs/DeleteModel.h"

double zThreshold = -0.5;	// z value below which we delete (hardcoded for now)

ros::ServiceClient deleteModelClient;
gazebo_msgs::DeleteModel deleteModelMsg;

// Determines whether object must be deleted
// For now, delete any model that falls down
inline bool canDelete(const geometry_msgs::Pose& modelPose) {
  return modelPose.position.z <= zThreshold;
}

// Function called every time model positions are published by gazebo
void recvModelStates(const gazebo_msgs::ModelStates& modelPoses) {
  // Do not look at name for now, just check all
  for(int i = 0; i < modelPoses.pose.size(); ++i)
    if(canDelete(modelPoses.pose[i])) {
      ROS_INFO_STREAM("Deleting model " << modelPoses.name[i]);
      deleteModelMsg.request.model_name = modelPoses.name[i];
      if(!deleteModelClient.call(deleteModelMsg))
	ROS_ERROR_STREAM("Could not delete model " << modelPoses.name[i]);
      ROS_INFO_STREAM("Deleted");
    }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "object_despawner_node");
  ros::NodeHandle nh;

  // Initialize services
  ros::service::waitForService("/gazebo/delete_model");
  deleteModelClient = nh.serviceClient<gazebo_msgs::DeleteModel>("/gazebo/delete_model", true);

  // Initialize subscriber
  ros::Subscriber modelStateSub = nh.subscribe("/gazebo/model_states", 1, recvModelStates);

  ros::spin();
  return 0;
}
