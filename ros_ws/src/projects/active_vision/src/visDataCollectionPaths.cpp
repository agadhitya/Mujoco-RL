#include <active_vision/toolVisualization.h>
#include <active_vision/toolViewPointCalc.h>
#include <active_vision/toolDataHandling.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <boost/thread.hpp>
#include <boost/make_shared.hpp>

int arrID = 0;
int sphereID = 0;

void help(){
  std::cout << "******* Data collection paths Visualizer Help *******" << std::endl;
  std::cout << "Arguments : [CSV filename]" << std::endl;
  std::cout << "CSV filename : CSV file name inside the 'misc' directory of the project (Data.csv)" << std::endl;
  std::cout << "*******" << std::endl;
}

void addArrows(std::vector<double> &homePose,std::string dirs,int &mode, ptCldVis::Ptr viewer, bool sphere, int vp){
  // Calculating the path and adding arrows
  pcl::PointXYZ table,a1,a2; table.x = 0.55; table.y = 0; table.z = 0;
  std::vector<std::vector<double>> path = {homePose};
  std::vector<double> nextPose;
  for(int i = 0; i < dirs.length(); i++){
    int dir = dirs[i] - '0';
    nextPose = calcExplorationPose(path.back(),dir,mode);
    if(checkValidPose(nextPose) == true){
      path.push_back(nextPose);
      a1 = sphericalToCartesian(path[i],table);
      a2 = sphericalToCartesian(path[i+1],table);
      ::arrID++;
      viewer->addArrow(a2,a1,0,1,0,false,"Arr_"+std::to_string(::arrID),vp);
    }
  }
  if(sphere == true){
    for(int i=1; i<=8; i++){
      nextPose = calcExplorationPose(path.back(),i,mode);
      a1 = sphericalToCartesian(nextPose,table);
      ::sphereID++;
      viewer->addSphere(a1,0.04,1,0,0,"Sphere_"+std::to_string(::sphereID),vp);
    }
  }
  a1 = sphericalToCartesian(path.back(),table);
  ::sphereID++;
  viewer->addSphere(a1,0.04,0,1,0,"Sphere_"+std::to_string(::sphereID),vp);
  nextPose = path[0]; nextPose[0]*=3;
  a1 = sphericalToCartesian(nextPose,table);
  viewer->setCameraPosition(a1.x,a1.y,a1.z,table.x,table.y,table.z,0,0,1);
}

int main(int argc, char** argv){
  if(argc != 2){
    std::cout << "ERROR. Incorrect number of arguments." << std::endl;
    help(); return(-1);
  }

  ros::NodeHandle nh;
  std::string directory;
  nh.getParam("/active_vision/data_dir", directory);
  std::string csvFile(argv[1]);
  std::vector<std::vector<std::string>> data = readCSV(directory+csvFile);

  std::vector<nodeData> tree = convertToTree(data);
  // for(int i = 0; i < tree.size(); i++){
  //   tree[i].print();
  // }

  std::vector<double> homePose={1.4,M_PI,45*M_PI/180};
  pcl::PointXYZ table; table.x = 0.55; table.y = 0; table.z = 0;

  // Setting up the point cloud visualizer
  ptCldVis::Ptr viewer(new ptCldVis ("PCL Viewer")); std::vector<int> vp;
  setupViewer(viewer, 1, vp);
  keyboardEvent keyPress(viewer,1); keyPress.help();
  keyPress.mode = 2;

  viewer->initCameraParameters();
  viewer->setCameraPosition(0,0,5,table.x,table.y,table.z,0,0,1);

  int nodeID = 0;
  while(keyPress.ok){
    addViewsphere(viewer,vp[0],table,homePose[0],false);
    viewer->addCube(table.x-0.25,table.x+0.25,
                    table.y-0.50,table.y+0.50,
                    table.z-0.01,table.z,0.9,0.8,0.4,"Table",vp[0]);

    ::sphereID = 0;
    ::arrID = 0;
    if(nodeID == tree.size()){
      viewer->addText("Home Pos : All",5,5,25,1,0,0,"Pos",vp[0]);
      for(int i = 0; i < tree.size(); i++){
        addArrows(homePose,tree[i].dirs,keyPress.mode, viewer, false, vp[0]);
      }
    }else{
      viewer->addText("Home Pos : "+std::to_string(nodeID),5,5,25,1,0,0,"Pos",vp[0]);
      addArrows(homePose,tree[nodeID].dirs,keyPress.mode, viewer, true, vp[0]);
    }

    keyPress.called = false;
    while(!viewer->wasStopped() && keyPress.called==false){
      viewer->spinOnce(100);
      boost::this_thread::sleep (boost::posix_time::microseconds(100000));
    }
    viewer->resetStoppedFlag();
    viewer->removeAllShapes();

    nodeID += keyPress.counter;
    if(nodeID < 0) nodeID = 0;
    if(nodeID > tree.size()) nodeID = tree.size();
  }
}
