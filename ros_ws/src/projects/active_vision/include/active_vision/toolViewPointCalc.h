#ifndef TOOLVIEWPOINTCALC
#define TOOLVIEWPOINTCALC

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <map>
#include <string>
#include <math.h>

static double minAngle = 20*(M_PI/180.0);
static pcl::PointXYZ defautCentre(0,0,0);

pcl::PointXYZ sphericalToCartesian(std::vector<double> &pose, pcl::PointXYZ &centre);

std::vector<double> cartesianToSpherical(pcl::PointXYZ &point, pcl::PointXYZ &centre);

bool checkValidPose(std::vector<double> pose);

bool checkIfNewPose(std::vector<std::vector<double>> &oldPoses, std::vector<double> &pose, int type);

double disBtwSpherical(std::vector<double> &poseA,std::vector<double> &poseB);

std::vector<double> calcExplorationPoseA(std::vector<double> &startPose, int dir, double step=minAngle);

std::vector<double> calcExplorationPoseB(std::vector<double> &startPose, int dir, double step=minAngle);

std::vector<double> calcExplorationPose(std::vector<double> &startPose, int dir, int mode, double step=minAngle);

#endif
