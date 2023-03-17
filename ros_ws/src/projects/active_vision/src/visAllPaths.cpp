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

  int yawColID = 5;
  int typeColID = 11;
  int pathColID = 12;
  int dirColID = 13;
  int nStepColID = 14;
  int stepColID = 15;
  int nSteps,type;
  double yaw = 0;

  std::vector<double> pose={0,0,0};
  pcl::PointXYZ sphereCentre;
  pcl::PointXYZ table,a1,a2;
  table.x = 0.55; table.y = 0; table.z = 0;

  ptCldColor::Ptr ptrPtCldTemp{new ptCldColor};

  // Setting up the point cloud visualizer
  ptCldVis::Ptr viewer(new ptCldVis ("PCL Viewer")); std::vector<int> vp;
  setupViewer(viewer, 1, vp);
  viewer->setCameraPosition(-2,0,7,table.x,table.y,table.z,0,0,1);
  viewer->addCube(table.x-0.25,table.x+0.25,
                table.y-0.50,table.y+0.50,
                table.z-0.01,table.z,0.9,0.8,0.4,"Table",vp[0]);
  keyboardEvent keyPress(viewer,1); keyPress.help();


  int i = 0;
  //Controls the decimal to round values to- 100=2 places, 10=1, etc..
  int precision = 10;
  for (int i = 0; i < data.size(); ++i)
  {
    type = std::atoi(data[i][typeColID-1].c_str());
    nSteps = std::atoi(data[i][nStepColID-1].c_str());
    yaw = std::atof(data[i][yawColID-2].c_str());

    for (int j = stepColID; j < stepColID+(nSteps+type-1)*3; j+=3){
      pose[0] = std::atof(data[i][j-1].c_str());
      pose[1] = std::atof(data[i][j-1+1].c_str());
      pose[2] = std::atof(data[i][j-1+2].c_str());
      a1.x = table.x+pose[0]*sin(pose[2])*cos(pose[1]+yaw);
      a1.y = table.y+pose[0]*sin(pose[2])*sin(pose[1]+yaw);
      a1.z = table.z+pose[0]*cos(pose[2]);
      if(stepColID != j){
        a1.x = floor(a1.x*precision+.5)/precision;
        a1.y = floor(a1.y*precision+.5)/precision;
        a1.z = floor(a1.z*precision+.5)/precision;
      }

      pose[0] = std::atof(data[i][j-1+3].c_str());
      pose[1] = std::atof(data[i][j-1+4].c_str());
      pose[2] = std::atof(data[i][j-1+5].c_str());
      a2.x = table.x+pose[0]*sin(pose[2])*cos(pose[1]+yaw);
      a2.y = table.y+pose[0]*sin(pose[2])*sin(pose[1]+yaw);
      a2.z = table.z+pose[0]*cos(pose[2]);
      a2.x = floor(a2.x*precision+.5)/precision;
      a2.y = floor(a2.y*precision+.5)/precision;
      a2.z = floor(a2.z*precision+.5)/precision;
      std::cout << "**********" << std::endl;
      std::cout << a1 << std::endl;
      std::cout << a2 << std::endl;

      if(j-1 < stepColID+(type-1)*3-1){
        viewer->addArrow(a2,a1,1,0,0,false,std::to_string(i)+"_"+std::to_string(j),vp.back());
      }else{
        viewer->addArrow(a2,a1,1-.2*((j-stepColID)/3),1,1-.2*((j-stepColID)/3),false,std::to_string(i)+"_"+std::to_string(j),vp.back());
      }
    }

    pose[0] = std::atof(data[i][stepColID+(type-1)*3-1].c_str());
    pose[1] = std::atof(data[i][stepColID+(type-1)*3-1+1].c_str());
    pose[2] = std::atof(data[i][stepColID+(type-1)*3-1+2].c_str());
    a1.x = table.x+pose[0]*sin(pose[2])*cos(pose[1]+yaw);
    a1.y = table.y+pose[0]*sin(pose[2])*sin(pose[1]+yaw);
    a1.z = table.z+pose[0]*cos(pose[2]);

    viewer->addSphere(a1,0.04,1,0,0,"Cam_"+std::to_string(i),vp.back());

  }
  addViewsphere(viewer,vp.back(),table,pose[0],true);
  keyPress.called = false;
  while(!viewer->wasStopped() && keyPress.called==false){
    viewer->spinOnce(100);
    boost::this_thread::sleep (boost::posix_time::microseconds(100000));
  }

  viewer->resetStoppedFlag();
  viewer->removeAllPointClouds();
  viewer->removeAllShapes();
}
