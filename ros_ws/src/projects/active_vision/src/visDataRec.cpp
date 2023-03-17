#include <active_vision/environment.h>
#include <active_vision/toolVisualization.h>

void passThroughFilterZ(ptCldColor::Ptr ptrPtCld,float minZ, float maxZ){

  ptCldColor::ConstPtr cPtrPtCld{ptrPtCld};
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud(cPtrPtCld);
  pass.setFilterFieldName("z"); pass.setFilterLimits(minZ,maxZ); pass.filter(*ptrPtCld);
}

void help(){
  std::cout << "******* Recorded Data Visualizer Help *******" << std::endl;
  std::cout << "Arguments : [Directory] [CSV filename] [Visual]" << std::endl;
  std::cout << "Directory : Directory where csv and pcd files are there (./DataRecAV/)" << std::endl;
  std::cout << "CSV filename : CSV file name (Data.csv)" << std::endl;
  std::cout << "Visual : 1 (result.pcd only), 2 (obj,unexp,result.pcd)" << std::endl;
  std::cout << "*******" << std::endl;
}

int main(int argc, char** argv){
  if(argc != 4){
    std::cout << "ERROR. Incorrect number of arguments." << std::endl;
    help(); return(-1);
  }
  int visual;
  visual = std::atoi(argv[3]);
  if(visual != 1 && visual != 2){
    std::cout << "ERROR. Incorrect visual." << std::endl;
    help(); return(-1);
  }

  std::string directory(argv[1]);
  std::string csvFile(argv[2]);
  std::vector<std::vector<std::string>> data;
  data = readCSV(directory+csvFile);

  int typeColID = 11;
  int pathColID = 12;
  int dirColID = 13;
  int nStepColID = 14;
  int stepColID = 15;
  int nSteps,type;

  std::vector<double> pose={0,0,0};
  pcl::PointXYZ sphereCentre;
  pcl::PointXYZ table,a1,a2;
  table.x = 0.45; table.y = 0; table.z = 0;

  ptCldColor::Ptr ptrPtCldTemp{new ptCldColor};

  // Setting up the point cloud visualizer
  ptCldVis::Ptr viewer(new ptCldVis ("PCL Viewer")); std::vector<int> vp;
  if(visual == 1)      setupViewer(viewer, 1, vp);
  else if(visual == 2) setupViewer(viewer, 3, vp);
  viewer->setCameraPosition(-2,0,7,table.x,table.y,table.z,0,0,1);
  keyboardEvent keyPress(viewer,1); keyPress.help();
  viewer->removeCoordinateSystem();

  int i = 0;
  while(keyPress.ok){
    if (pathColID <= data[i].size()){
      if(visual==2){
        readPointCloud(ptrPtCldTemp,directory,data[i][pathColID-1],1);
        addRGB(viewer,ptrPtCldTemp,"Obj",vp[0]);
        pcl::PointXYZRGB minPtObj, maxPtObj;
        pcl::getMinMax3D(*ptrPtCldTemp, minPtObj, maxPtObj);
        readPointCloud(ptrPtCldTemp,directory,data[i][pathColID-1],2);
        passThroughFilterZ(ptrPtCldTemp,minPtObj.z,maxPtObj.z+0.05);
        addRGB(viewer,ptrPtCldTemp,"Unexp",vp[1]);
      }
      readPointCloud(ptrPtCldTemp,directory,data[i][pathColID-1],3);
      addRGB(viewer,ptrPtCldTemp,"Env",vp.back());

      type = std::atoi(data[i][typeColID-1].c_str());
      nSteps = std::atoi(data[i][nStepColID-1].c_str());

      viewer->addText("Data No : "+std::to_string(i+1),5,5,25,1,0,0,"Data",vp.back());
      viewer->addText("Direction : "+dirLookup[std::atoi(data[i][dirColID-1].c_str())],5,35,25,1,0,0,"Dir",vp.back());
      viewer->addText("nSteps : "+std::to_string(nSteps),5,65,25,1,0,0,"nSteps",vp.back());
      viewer->addText("Data for step "+std::to_string(type),5,95,25,1,0,0,"type",vp.back());

      for (int j = stepColID; j < stepColID+(nSteps+type-1)*3; j+=3){
        pose[0] = std::atof(data[i][j-1].c_str());
        pose[1] = std::atof(data[i][j-1+1].c_str());
        pose[2] = std::atof(data[i][j-1+2].c_str());
        a1.x = table.x+pose[0]*sin(pose[2])*cos(pose[1]);
        a1.y = table.y+pose[0]*sin(pose[2])*sin(pose[1]);
        a1.z = table.z+pose[0]*cos(pose[2]);

        pose[0] = std::atof(data[i][j-1+3].c_str());
        pose[1] = std::atof(data[i][j-1+4].c_str());
        pose[2] = std::atof(data[i][j-1+5].c_str());
        a2.x = table.x+pose[0]*sin(pose[2])*cos(pose[1]);
        a2.y = table.y+pose[0]*sin(pose[2])*sin(pose[1]);
        a2.z = table.z+pose[0]*cos(pose[2]);

        if(j-1 < stepColID+(type-1)*3-1){
          viewer->addArrow(a2,a1,1,0,0,false,std::to_string(j),vp.back());
        }else{
          viewer->addArrow(a2,a1,0,1,0,false,std::to_string(j),vp.back());
        }
      }

      pose[0] = std::atof(data[i][stepColID+(type-1)*3-1].c_str());
      pose[1] = std::atof(data[i][stepColID+(type-1)*3-1+1].c_str());
      pose[2] = std::atof(data[i][stepColID+(type-1)*3-1+2].c_str());
      a1.x = table.x+pose[0]*sin(pose[2])*cos(pose[1]);
      a1.y = table.y+pose[0]*sin(pose[2])*sin(pose[1]);
      a1.z = table.z+pose[0]*cos(pose[2]);

      viewer->addSphere(a1,0.04,1,0,0,"Cam",vp.back());
      addViewsphere(viewer,vp.back(),table,pose[0],true);

      keyPress.called = false;
      while(!viewer->wasStopped() && keyPress.called==false){
        viewer->spinOnce(100);
        boost::this_thread::sleep (boost::posix_time::microseconds(100000));
      }

      viewer->resetStoppedFlag();
      viewer->removeAllPointClouds();
      viewer->removeAllShapes();
    }else{
      std::cout << "No data found for i = " << i+1 << std::endl;
      keyPress.ok = false;
    }
    if(keyPress.skipTo == true){
      keyPress.skipTo = false;
      std::cout << "Enter the Data Number you want to go : "; std::cin >> i;
    }
    else{
      i += keyPress.counter;
    }
    if(i < 0) i = 0;
    if(i > data.size()-1) i = data.size()-1;
  }
}
