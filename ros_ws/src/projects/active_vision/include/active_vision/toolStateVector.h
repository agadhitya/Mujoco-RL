#ifndef TOOLSTATEVECTOR
#define TOOLSTATEVECTOR

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <tuple>
#include <boost/make_shared.hpp>
#include <sstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/common.h>

// Typedef for convinience
typedef pcl::PointCloud<pcl::PointXYZRGB> ptCldColor;

void passThroughFilter(ptCldColor::Ptr ptrPtCld,pcl::PointXYZRGB min, pcl::PointXYZRGB max);

class HAFStVec1{
private:
  ptCldColor::Ptr ptrPtCldObj{new ptCldColor};
  ptCldColor::Ptr ptrPtCldUnexp{new ptCldColor};
  std::vector<double> kinViewsphere = {};
  int gridDim;
  std::vector<float> stateVec = {};
  bool dataSet = false;
  bool maintainScale = false;

public:
  void setInput(ptCldColor::Ptr obj, ptCldColor::Ptr unexp, std::vector<double> &kinect);

  void setGridDim(int dim);

  void setMaintainScale(bool val);

  std::vector<float> getStateVec();

  void print();

  void calculate();

  void saveToCSV(std::fstream &saveTo, std::string PCDpath, std::string label);
};

#endif
