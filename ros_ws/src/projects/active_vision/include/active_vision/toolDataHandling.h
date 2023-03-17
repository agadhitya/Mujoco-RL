#ifndef TOOLDATAHANDLING
#define TOOLDATAHANDLING

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <time.h>
#include <tuple>
#include <boost/make_shared.hpp>
#include <sstream>
#include <stdexcept>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <ros/ros.h>
#include <ros/package.h>
#include <ros/master.h>


// Typedef for convinience
typedef pcl::PointCloud<pcl::PointXYZRGB> ptCldColor;

// Struct representing the node of a tree. Each node is a data point
struct nodeData{
  int nodeID{-1};
  int parentID{-1};
  std::vector<int> childIDs = {};
  std::string dirs = "";
  int checked{0};
  int configID{-1};
  int allowedSteps{15};
  int possibleSolChildIndex{-1};

  void reset(){
    nodeID = -1;
    parentID = -1;
    dirs = "";
    childIDs.clear();
    checked = 0;
    configID = -1;
    allowedSteps = 15;
    possibleSolChildIndex = -1;
  }

  void print(){
    std::cout << std::setw(3) << nodeID;
    std::cout << " (" << std::setw(3) << parentID << ")";
    std::cout << " [" << std::setw(3) << configID << "]";
    std::cout << " <" << checked << "," << std::setw(2) << allowedSteps << ">";
    std::cout << std::setw(7) << dirs << " : ";
    for(int i = 0; i < childIDs.size(); i++){
      std::cout << childIDs[i] << " -> ";
    }
    std::cout << "\\" << std::endl;
  }
};

// Struct which stores the data which would be saved to the PC
struct RouteData{
  int parentID{-1};
  int childID{-1};
  ptCldColor obj,unexp,env;
  std::vector<ptCldColor> detailedEnv;
  std::vector<float> timer;
  std::string objType;
  std::vector<double> objPose;
  bool goodInitialGrasp{false};
  bool success{false};
  float graspQuality{-1};
  int nSteps{0};
  float EffnSteps{-1};
  std::vector<std::vector<double>> path;
  int direction{-1};
  int type{0};
  std::string filename;
  nodeData nodeInfo;

  void reset(){
    parentID = -1; childID = -1;
    obj.clear();unexp.clear();env.clear();
    detailedEnv.clear();
    timer.clear();
    objType.clear(); objPose.clear();
    goodInitialGrasp = false; success = false;
    graspQuality = -1;
    nSteps = 0; path.clear();
    EffnSteps = -1;
    direction = -1;
    type = 0;
    filename.clear();
    nodeInfo.reset();
  }
};

// Function to print the RouteData
void printRouteData(RouteData &in);

// Function to save the RouteData
void saveData(RouteData &in, std::fstream &saveTo, std::string &dir, bool all=true);

// Function to get the current time
std::string getCurTime();

// Function to save the pointcloud to PC
void savePointCloud(ptCldColor::Ptr cloud, std::string dir, std::string prefix, int type);

// Function to read the pointcloud from PC
void readPointCloud(ptCldColor::Ptr cloud, std::string dir, std::string prefix, int type);

// Function to read a csv file
std::vector<std::vector<std::string>> readCSV(std::string filename);

// Function the convert the data to a tree format
std::vector<nodeData> convertToTree(std::vector<std::vector<std::string>> &data);

// Struct to store the info of objects available for use
struct objectInfo{
public:
	int ID;
	std::string fileName;
	std::string description;
	int nPoses;
	std::vector<std::vector<double>> poses;

	void setFromROSParam(ros::NodeHandle &nh, std::string param, int num);
	void printObjectInfo(bool all = false);
};

// Function to read all the object info from the objects.yaml file
void readObjectsList(ros::NodeHandle &nh, std::string param, std::map<int,objectInfo> &res);

#endif
