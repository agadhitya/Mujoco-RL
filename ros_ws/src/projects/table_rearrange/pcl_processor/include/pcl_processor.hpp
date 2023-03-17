#include "ros/ros.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class PCLProcessor {
public:
  PCLProcessor(ros::NodeHandle& n) : nh{n} {
    setupServices();
  };
  ~PCLProcessor() {};

  // PCL Functions
private:
  ros::NodeHandle nh;

  void setupServices() {
    
  }
};
