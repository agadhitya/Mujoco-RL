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
  std::cout << "Arguments : [Directory] [CSV filename]" << std::endl;
  std::cout << "Directory : Directory where csv and pcd files are there (./DataRecAV/)" << std::endl;
  std::cout << "CSV filename : CSV file name (Data.csv)" << std::endl;
  std::cout << "*******" << std::endl;
}

int main(int argc, char** argv){
  if(argc != 3){
    std::cout << "ERROR. Incorrect number of arguments." << std::endl;
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
  table.x = 0.45; table.y = 0; table.z = 0.125;

  ptCldColor::Ptr ptrPtCldTemp{new ptCldColor};

  // Setting up the point cloud visualizer
  ptCldVis::Ptr viewer(new ptCldVis ("PCL Viewer")); std::vector<int> vp;
  setupViewer(viewer, 8, vp);
  keyboardEvent keyPress(viewer,1); keyPress.help();
  viewer->removeCoordinateSystem();

  int i = 0;
  while(keyPress.ok){
    if (pathColID <= data[i].size()){
      nSteps = std::atoi(data[i][nStepColID-1].c_str());
      type = std::atoi(data[i][typeColID-1].c_str());

      viewer->addText("Data No : "+std::to_string(i+1),5,5,25,1,0,0,"Data",vp.back());
      viewer->addText("nSteps : "+std::to_string(nSteps),5,65,25,1,0,0,"nSteps",vp.back());

      for(int j = 0; j <= nSteps; j++){
        if(pcl::io::loadPCDFile(directory+data[i][pathColID-1]+"_detailed_"+std::to_string(j)+".pcd",*ptrPtCldTemp) != -1){
          addRGB(viewer,ptrPtCldTemp,"Data"+std::to_string(j),vp[j]);

          pose[0] = std::atof(data[i][3*j+stepColID-1].c_str());
          pose[1] = std::atof(data[i][3*j+stepColID-1+1].c_str());
          pose[2] = std::atof(data[i][3*j+stepColID-1+2].c_str());
          viewer->addText(std::to_string(j),5,5,25,1,0,0,"Data"+std::to_string(j),vp[j]);
          pose[0] *= 2; setCamView(viewer,pose,table,j+1);
        }
      }

      pose[0] = std::atof(data[i][stepColID-1].c_str());
      pose[1] = std::atof(data[i][stepColID-1+1].c_str());
      pose[2] = std::atof(data[i][stepColID-1+2].c_str());
      readPointCloud(ptrPtCldTemp,directory,data[i][pathColID-1],3);
      addRGB(viewer,ptrPtCldTemp,"Env",vp[6]);
      viewer->addText("Grasp",5,5,25,1,0,0,"Grasp",vp[6]);
      pose[0] *= 2; setCamView(viewer,pose,table,6); setCamView(viewer,pose,table,7);

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
          for(int k = 0; k <= nSteps; k++){
            viewer->addArrow(a2,a1,1,0,0,false,std::to_string(k)+","+std::to_string(j),vp[k]);
          }
          viewer->addArrow(a2,a1,1,0,0,false,std::to_string(6)+","+std::to_string(j),vp[6]);
        }else{
          for(int k = 0; k <= nSteps; k++){
            viewer->addArrow(a2,a1,0,1,0,false,std::to_string(k)+","+std::to_string(j),vp[k]);
          }
          viewer->addArrow(a2,a1,0,1,0,false,std::to_string(6)+","+std::to_string(j),vp[6]);
        }
      }

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
