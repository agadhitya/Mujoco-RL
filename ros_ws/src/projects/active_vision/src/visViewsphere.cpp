#include <active_vision/toolVisualization.h>
#include <active_vision/toolViewPointCalc.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <boost/make_shared.hpp>
#include <boost/thread.hpp>

void help(){
  std::cout << "******* Viewsphere Visualizer Help *******" << std::endl;
  std::cout << "Use numpad/normal keys to go to a direction N(1),NE(2),E(3),SE(4),S(5),SW(6),W(7),NW(8)" << std::endl;
  std::cout << "Use numpad/normal key 0 to reset." << std::endl;
  std::cout << "Use space to switch between modes." << std::endl;
  std::cout << "*******" << std::endl;
}

int main(int argc, char** argv){
  help();
  std::vector<double> homePose={1.0,M_PI,45*M_PI/180};
  std::vector<double> nextPose;
  std::vector<double> temp;
  std::vector<std::vector<double>> path;
  path.push_back(homePose);
  std::vector<int> arrIDs = {0};
  pcl::PointXYZ table,a1,a2;
  table.x = 0.55; table.y = 0; table.z = 0;

  // Setting up the point cloud visualizer
  ptCldVis::Ptr viewer(new ptCldVis ("PCL Viewer")); std::vector<int> vp;
  setupViewer(viewer, 1, vp);
  keyboardEvent keyPress(viewer,2); keyPress.help();
  keyPress.mode = 2;

  addViewsphere(viewer,vp[0],table,homePose[0],false);

  viewer->addCube(table.x-0.25,table.x+0.25,
                  table.y-0.50,table.y+0.50,
                  table.z-0.01,table.z,0.9,0.8,0.4,"Table",vp[0]);

  viewer->addText("8(NW)\n7(W)\n6(SW)",5,70,20,1,0,0,"Key1",vp[0]);
  viewer->addText("1(N)\n0(Reset)\n5(S)",70,70,20,1,0,0,"Key2",vp[0]);
  viewer->addText("2(NE)\n3(E)\n4(SE)",160,70,20,1,0,0,"Key3",vp[0]);

  viewer->initCameraParameters();
  viewer->setCameraPosition(0,0,5,table.x,table.y,table.z,0,0,1);

  while(keyPress.ok){
    // Adding the Kinect position
    if(keyPress.dir >=1 && keyPress.dir <= 8){
      nextPose = calcExplorationPose(path.back(),keyPress.dir,keyPress.mode);
      if(checkValidPose(nextPose) == true && checkIfNewPose(path,nextPose,keyPress.mode) == true){
        path.push_back(nextPose);
      }
    }
    else if(keyPress.dir == 0){
      path.clear();
      path.push_back(homePose);
    }

    for(int i=1; i<=8; i++){
      nextPose = calcExplorationPose(path.back(),i,keyPress.mode);
      a1 = sphericalToCartesian(nextPose,table);
      viewer->addSphere(a1,0.04,1,0,0,"Sphere"+std::to_string(i),vp[0]);
    }
    a1 = sphericalToCartesian(path.back(),table);
    viewer->addSphere(a1,0.04,0,1,0,"Sphere",vp[0]);

    for(int i = 0; i < path.size()-1; i++){
      a1 = sphericalToCartesian(path[i],table);
      a2 = sphericalToCartesian(path[i+1],table);
      viewer->addArrow(a2,a1,0,1,0,false,"Arr"+std::to_string(arrIDs.back()),vp[0]);
      arrIDs.push_back(arrIDs.back()+1);
    }

    viewer->addText("Polar : "+std::to_string(round(path.back()[1]*180/M_PI)),5,5,25,1,0,0,"Polar",vp[0]);
    viewer->addText("Azhimuthal : "+std::to_string(round(path.back()[2]*180/M_PI)),5,35,25,1,0,0,"Azhi",vp[0]);
    viewer->addText("Mode " +std::to_string(keyPress.mode)+" (Space to switch)",5,140,20,1,0,0,"Mode",vp[0]);

    temp = path.back(); temp[0]*=3;
    a1 = sphericalToCartesian(temp,table);
    viewer->setCameraPosition(a1.x,a1.y,a1.z,table.x,table.y,table.z,0,0,1);

    keyPress.called = false;
    while(!viewer->wasStopped() && keyPress.called==false){
      viewer->spinOnce(100);
      boost::this_thread::sleep (boost::posix_time::microseconds(100000));
    }
    viewer->resetStoppedFlag();
    viewer->removeShape("Mode",vp[0]);
    viewer->removeShape("Sphere",vp[0]);
    for(int i = 1; i <= 8; i++) {
      viewer->removeShape("Sphere"+std::to_string(i),vp[0]);
    }

    // for(int i = 0; i < arrIDs.size(); i++){
    //   viewer->removeShape("Arr"+std::to_string(arrIDs[i]),vp[0]);
    // }
    // arrIDs = {0};

    viewer->removeShape("Polar",vp[0]);
    viewer->removeShape("Azhi",vp[0]);
  }
}
