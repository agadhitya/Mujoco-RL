#include <active_vision/toolStateVector.h>

void passThroughFilter(ptCldColor::Ptr ptrPtCld,pcl::PointXYZRGB min, pcl::PointXYZRGB max){

  ptCldColor::ConstPtr cPtrPtCld{ptrPtCld};
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud(cPtrPtCld);
  pass.setFilterFieldName("x"); pass.setFilterLimits(min.x,max.x); pass.filter(*ptrPtCld);
  pass.setFilterFieldName("y"); pass.setFilterLimits(min.y,max.y); pass.filter(*ptrPtCld);
  pass.setFilterFieldName("z"); pass.setFilterLimits(min.z,max.z); pass.filter(*ptrPtCld);
}

void HAFStVec1::setInput(ptCldColor::Ptr obj, ptCldColor::Ptr unexp, std::vector<double> &kinect){
  ptrPtCldObj = obj;
  ptrPtCldUnexp = unexp;
  kinViewsphere = kinect;
  dataSet = true;
}

void HAFStVec1::setGridDim(int dim){
  gridDim = dim;
}

void HAFStVec1::setMaintainScale(bool val){
  maintainScale = val;
}

std::vector<float> HAFStVec1::getStateVec(){
  return(stateVec);
}

void HAFStVec1::print(){
  if(dataSet != true) return;

  static std::string red = "\x1b[31m";
  static std::string green = "\x1b[32m";
  static std::string off = "\x1b[0m";

  std::cout << "\t Object\t\t\t\tUnexplored" << std::endl;
  for(int i = 0; i < gridDim; i++){
    // Object State Vector
    for(int j = 0; j < gridDim; j++){
      if (stateVec[i*gridDim+j] != 0) std::cout << red;
      std::cout << std::fixed << std::setfill('0') << std::setw(5) << std::setprecision(2) << stateVec[i*gridDim+j] << " ";
      if (stateVec[i*gridDim+j] != 0) std::cout << off;
    }
    std::cout << "\t";
    // unexplored State Vector
    for(int j = 0; j < gridDim; j++) {
      if (stateVec[gridDim*gridDim+i*gridDim+j] != 0) std::cout << red;
      std::cout << std::fixed << std::setfill('0') << std::setw(5) << std::setprecision(2) << stateVec[gridDim*gridDim+i*gridDim+j] << " ";
      if (stateVec[gridDim*gridDim+i*gridDim+j] != 0) std::cout << off;
    }
    std::cout << std::endl;
  }
  // Kinect Position
  std::cout << "Polar Angle : " << stateVec[2*gridDim*gridDim]*180/M_PI
            << ", Azhimuthal Angle : " << stateVec[2*gridDim*gridDim+1]*180/M_PI << std::endl << std::endl;
}

void HAFStVec1::calculate(){
  if(dataSet != true) return;

  Eigen::Vector3f pt;
  float gridSize[2]={};
  int gridLoc[2]={};

  pcl::PointXYZRGB minPtObj, maxPtObj;
  pcl::getMinMax3D(*ptrPtCldObj, minPtObj, maxPtObj);
  // printf("1---%f,%f,%f,%f\n",minPtObj.x,maxPtObj.x,minPtObj.y,maxPtObj.y);
  if(maintainScale == true){
    float midPointX,midPointY;
    midPointX = (maxPtObj.x+minPtObj.x)/2;
    midPointY = (maxPtObj.y+minPtObj.y)/2;
    if((maxPtObj.x-minPtObj.x) > (maxPtObj.y-minPtObj.y)){
      minPtObj.y = midPointY - (maxPtObj.x-minPtObj.x)/2;
      maxPtObj.y = midPointY + (maxPtObj.x-minPtObj.x)/2;
    }else{
      minPtObj.x = midPointX - (maxPtObj.y-minPtObj.y)/2;
      maxPtObj.x = midPointX + (maxPtObj.y-minPtObj.y)/2;
    }
  }
  // printf("2---%f,%f,%f,%f\n",minPtObj.x,maxPtObj.x,minPtObj.y,maxPtObj.y);

  float stateObj[gridDim][gridDim] = {};
  gridSize[0] = (maxPtObj.x-minPtObj.x)/gridDim;
  gridSize[1] = (maxPtObj.y-minPtObj.y)/gridDim;

  for(int i = 0; i < ptrPtCldObj->size(); i++){
    pt = ptrPtCldObj->points[i].getVector3fMap();
    pt[0] = pt[0]-minPtObj.x; pt[1] = pt[1]-minPtObj.y; pt[2] = pt[2]-minPtObj.z;
    if (pt[0] < 0 || pt[1] < 0 || pt[2] < 0) printf("ERROR HAFStVec1.calculate : Object point outside\n");

    gridLoc[0] = (int)(pt[0]/gridSize[0]);
    gridLoc[0] = std::min(gridLoc[0],gridDim-1);        // Ensuring that the grid loc for the extreme point is inside the grid
    gridLoc[0] = (gridDim-1)-gridLoc[0];                // Reversing so that the array and grid indices match
    gridLoc[1] = (int)(pt[1]/gridSize[1]);
    gridLoc[1] = std::min(gridLoc[1],gridDim-1);        // Ensuring that the grid loc for the extreme point is inside the grid

    stateObj[gridLoc[0]][gridLoc[1]] = std::max(stateObj[gridLoc[0]][gridLoc[1]],pt[2]);
  }

  float scale = 3;
  pcl::PointXYZRGB minPtUnexp, maxPtUnexp;
  minPtUnexp.x = minPtObj.x - (scale-1)*(maxPtObj.x-minPtObj.x)/2;
  minPtUnexp.y = minPtObj.y - (scale-1)*(maxPtObj.y-minPtObj.y)/2;
  minPtUnexp.z = minPtObj.z;
  maxPtUnexp.x = maxPtObj.x + (scale-1)*(maxPtObj.x-minPtObj.x)/2;
  maxPtUnexp.y = maxPtObj.y + (scale-1)*(maxPtObj.y-minPtObj.y)/2;
  maxPtUnexp.z = maxPtObj.z + 0.05;
  passThroughFilter(ptrPtCldUnexp,minPtUnexp,maxPtUnexp);

  float stateUnexp[gridDim][gridDim] = {};
  gridSize[0] = (maxPtUnexp.x-minPtUnexp.x)/gridDim;
  gridSize[1] = (maxPtUnexp.y-minPtUnexp.y)/gridDim;
  for(int i = 0; i < ptrPtCldUnexp->size(); i++){
    pt = ptrPtCldUnexp->points[i].getVector3fMap();
    pt[0] = pt[0]-minPtUnexp.x; pt[1] = pt[1]-minPtUnexp.y; pt[2] = pt[2]-minPtUnexp.z;
    if (pt[0] < 0 || pt[1] < 0 || pt[2] < 0) printf("ERROR HAFStVec1.calculate : Unexp point outside\n");

    gridLoc[0] = (int)(pt[0]/gridSize[0]);
    gridLoc[0] = std::min(gridLoc[0],gridDim-1);        // Ensuring that the grid loc for the extreme point is inside the grid
    gridLoc[0] = (gridDim-1)-gridLoc[0];                // Reversing so that the array and grid indices match
    gridLoc[1] = (int)(pt[1]/gridSize[1]);
    gridLoc[1] = std::min(gridLoc[1],gridDim-1);        // Ensuring that the grid loc for the extreme point is inside the grid
    stateUnexp[gridLoc[0]][gridLoc[1]] = std::max(stateUnexp[gridLoc[0]][gridLoc[1]],pt[2]);
  }

  stateVec.clear();
  // Storing both in the state vector
  for(int i = 0; i < gridDim; i++){
    for(int j = 0; j < gridDim; j++){
      stateVec.push_back(stateObj[i][j]*100);
    }
  }
  for(int i = 0; i < gridDim; i++){
    for(int j = 0; j < gridDim; j++){
      stateVec.push_back(stateUnexp[i][j]*100);
    }
  }
  // Adding kinect position
  stateVec.push_back(kinViewsphere[1]);
  stateVec.push_back(kinViewsphere[2]);
}

void HAFStVec1::saveToCSV(std::fstream &saveTo, std::string PCDpath, std::string label){
  saveTo << PCDpath << ",,";
  for(int i = 0; i < stateVec.size(); i++){
    saveTo << stateVec[i] << ",";
  }
  saveTo << "," << label << "\n" ;
}
