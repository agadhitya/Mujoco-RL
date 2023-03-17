#include <active_vision/environment.h>
#include <active_vision/toolViewPointCalc.h>
#include <active_vision/toolVisualization.h>
#include <active_vision/heuristicPolicySRV.h>
// #include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <set>

float voxelGridSize;
pcl::PointXYZ table;
ros::ServiceClient IKClient;
std::string simulationMode;
Eigen::MatrixXf projectionMat;

bool heuristicDiagonalPref = true;
bool compareDirections(std::vector<int> A, std::vector<int> B){
  if(::heuristicDiagonalPref){
    bool isADiagonal, isBDiagonal;
    if((A[1]/45)%2 == 1) isADiagonal = true;
    else                 isADiagonal = false;

    if((B[1]/45)%2 == 1) isBDiagonal = true;
    else                 isBDiagonal = false;

    if((isADiagonal && isBDiagonal) || (!isADiagonal && !isBDiagonal)){
      return (A[2] > B[2]);
    }

    if(isADiagonal && !isBDiagonal){
      if(abs(A[2]-B[2]) >= 20) return (A[2] > B[2]);
      else                     return true;
    }

    if(!isADiagonal && isBDiagonal){
      if(abs(A[2]-B[2]) >= 20) return (A[2] > B[2]);
      else                     return false;
    }

  }else{
    return (A[2] > B[2]);
  }

}

Eigen::Affine3f tfKinect(std::vector<double> &pose){
  std::vector<double> cartesian = {0,0,0,0,0,0};
  cartesian[0] = ::table.x+pose[0]*sin(pose[2])*cos(pose[1]);
  cartesian[1] = ::table.y+pose[0]*sin(pose[2])*sin(pose[1]);
  cartesian[2] = ::table.z+pose[0]*cos(pose[2]);
  cartesian[3] = 0;
  cartesian[4] = M_PI/2-pose[2];
  cartesian[5] = M_PI+pose[1];

  Eigen::Affine3f tfKinOptGaz,tfGazWorld;
  tfKinOptGaz = pcl::getTransformation(0,0,0,-M_PI/2,0,-M_PI/2);
  tfGazWorld = pcl::getTransformation(cartesian[0],cartesian[1],cartesian[2],\
                                      cartesian[3],cartesian[4],cartesian[5]);
  return(tfGazWorld * tfKinOptGaz);
}

geometry_msgs::Pose viewsphereToFranka(std::vector<double> &pose){

  Eigen::Matrix3f rotMat;
  rotMat = Eigen::AngleAxisf(M_PI+pose[1], Eigen::Vector3f::UnitZ()) * Eigen:: AngleAxisf(M_PI/2-pose[2], Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX());

  // Incorporating the kinect translation offset
  Eigen::Matrix4f tfMat; tfMat.setIdentity();
  tfMat.block<3,3>(0,0) = rotMat;
  tfMat(0,3) = ::table.x+pose[0]*sin(pose[2])*cos(pose[1]);
  tfMat(1,3) = ::table.y+pose[0]*sin(pose[2])*sin(pose[1]);
  tfMat(2,3) = ::table.z+pose[0]*cos(pose[2]);

  tfMat *= pcl::getTransformation(0,0,0,0,-M_PI/2,M_PI).matrix();
  Eigen::Matrix4f kinectOffset; kinectOffset.setIdentity();
  kinectOffset(0,3) = -0.037796115635;
  kinectOffset(1,3) = +0.0298131982299;
  kinectOffset(2,3) = -0.059671236405;
  if(tfMat(0,0) < 0) tfMat *= pcl::getTransformation(0,0,0,0,0,M_PI).matrix();
  tfMat *= kinectOffset;

  if(::simulationMode == "FRANKA"){
    tfMat *= pcl::getTransformation(0,0,0,0,0,M_PI/4).matrix();
  }

  Eigen::Quaternionf quat(tfMat.block<3,3>(0,0));

  geometry_msgs::Pose p;
  p.position.x = tfMat(0,3);  p.position.y = tfMat(1,3);  p.position.z = tfMat(2,3);
  p.orientation.x = quat.x(); p.orientation.y = quat.y(); p.orientation.z = quat.z(); p.orientation.w = quat.w();

  return p;
}

bool checkKinectOK(std::vector<double> &pose){
  bool check1 = checkValidPose(pose);
  bool check2 = true;
  if(::simulationMode != "SIMULATION"){
    geometry_msgs::Pose p = viewsphereToFranka(pose);
    check2 = checkFrankReach(::IKClient,p) || checkFrankReach(::IKClient,p);
  }
  return (check1 && check2);
}

// For object point cloud
bool objCheckIfOccluded(pcl::PointXYZRGB &home, pcl::PointXYZRGB &check, pcl::Normal &normal, float radius){

  // Plane
  Eigen::Vector3f n = {normal.normal_x,normal.normal_y,normal.normal_z};
  Eigen::Vector3f p0 = {check.x,check.y,check.z};
  // Line
  Eigen::Vector3f l = {-home.x,-home.y,-home.z};
  Eigen::Vector3f l0 = {home.x,home.y,home.z};

  float d = 999;
  if(l.dot(n) != 0)
    d = ((p0 - l0).dot(n)) / (l.dot(n));

  pcl::PointXYZRGB intersectionPt;
  intersectionPt.getVector3fMap() = l0 + l*d;
  float distance = pcl::euclideanDistance(check,intersectionPt);

  if(d <= 0 || distance > radius) return false;
  else                            return true;
}

// For unexplored point cloud
bool unexpCheckIfOccluded(pcl::PointXYZRGB &home, pcl::PointXYZRGB &check, float radius){

  float disEuc = pcl::euclideanDistance(home,check);
  float disPerpVP = 0;
  float disParaVP = 0;
  float angle = 0;
  if(disEuc > 0){
    Eigen::Vector3f homeToVP,homeToCheck;
    homeToVP = {-home.x,-home.y,-home.z}; homeToVP.normalize();
    homeToCheck = {check.x-home.x,check.y-home.y,check.z-home.z}; homeToCheck.normalize();
    angle = atan2(homeToVP.cross(homeToCheck).norm(), homeToVP.dot(homeToCheck));
    disPerpVP = disEuc*sin(angle);
    disParaVP = disEuc*cos(angle);
  }

  if(disParaVP <= radius*sqrt(2) && disPerpVP <= radius && angle < 75.0/180.0*M_PI) return true;
  else return false;

}

void extractUsefulUnexpPts(ptCldColor::Ptr obj, ptCldColor::Ptr unexp, ptCldColor::Ptr res){
  res->clear();

  // Calculating the normal vectors
  static ptCldNormal::Ptr normal{new ptCldNormal};
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;  // Normal Estimation
  pcl::search::Search<pcl::PointXYZRGB>::Ptr KdTree{new pcl::search::KdTree<pcl::PointXYZRGB>};
  ptCldColor::ConstPtr cObj{obj};
  ne.setInputCloud(cObj);
  ne.setSearchMethod(KdTree);
  ne.setKSearch(10);
  ne.compute(*normal);

  // pcl::PointXYZRGB minPtObj, maxPtObj;
  // pcl::getMinMax3D(*obj, minPtObj, maxPtObj);

  pcl::PointXYZRGB min,max;
  min.x = -0.080; max.x = 0.080;
  min.y = -0.015; max.y = 0.015;
  min.z = -0.015; max.z = 0.015;

  Eigen::Affine3f tf;

  pcl::CropBox<pcl::PointXYZRGB> cpBox;
  cpBox.setInputCloud(unexp);
  cpBox.setMin(min.getVector4fMap());
  cpBox.setMax(max.getVector4fMap());

  std::vector<int> tempIndices;
  std::set<int> usefulIndices;

  for(int i = 0; i < normal->width; i++){
    tf = calcTfFromNormal(normal->points[i],obj->points[i]);

    cpBox.setRotation(getEuler(tf));
    cpBox.setTranslation(getTranslation(tf));
    cpBox.filter(tempIndices);

    for(int j = 0; j < tempIndices.size(); j++){
      usefulIndices.insert(tempIndices[j]);
    }
  }

  // Extracting the indices
  for(auto idx : usefulIndices){
    if(unexp->points[idx].z >= ::table.z+0.01){
      res->points.push_back(unexp->points[idx]);
      res->points.back().r = 0;
      res->points.back().g = 200;
      res->points.back().b = 0;
    }
  }
  res->width = res->points.size();
  res->height = 1;
}

std::set<int> findVisibleUsefulUnexp(ptCldColor::Ptr obj, ptCldColor::Ptr unexp, ptCldColor::Ptr usefulUnexp, std::vector<double> &pose){

  Eigen::Affine3f tf = tfKinect(pose);
  static ptCldColor::Ptr tempObj{new ptCldColor};            pcl::transformPointCloud(*obj, *tempObj, homoMatTranspose(tf));
  static ptCldColor::Ptr tempUsefulUnexp{new ptCldColor};    pcl::transformPointCloud(*usefulUnexp, *tempUsefulUnexp, homoMatTranspose(tf));
  static ptCldColor::Ptr tempUsefulUnexpObj{new ptCldColor};

  pcl::transformPointCloud(*unexp, *tempUsefulUnexpObj, homoMatTranspose(tf));
  *tempUsefulUnexpObj += *tempObj;

  // Calculating the transoformed normal vectors
  static ptCldNormal::Ptr tempNormal{new ptCldNormal};
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;  // Normal Estimation
  pcl::search::Search<pcl::PointXYZRGB>::Ptr KdTree{new pcl::search::KdTree<pcl::PointXYZRGB>};
  ptCldColor::ConstPtr ctempObj{tempObj};
  ne.setInputCloud(ctempObj);
  ne.setSearchMethod(KdTree);
  ne.setKSearch(10);
  ne.compute(*tempNormal);

  pcl::PointXYZRGB cam; cam.x = 0; cam.y = 0; cam.z = 0;

  std::vector<int> tempIndices;
  std::set<int> objClearIndices; objClearIndices.clear();
  std::set<int> finalIndices; finalIndices.clear();

  int scale = 2;
  int w = 75*scale;
  int maskRadius;
  cv::Mat patchMask, surface; float minPatchArea;

  Eigen::Vector3f proj;
  std::vector<cv::Point> projPts;

  // std::vector<int> indicesOK,indicesNotOK;
  // static ptCldVis::Ptr viewer(new ptCldVis ("PCL Viewer"));
  // std::vector<int> vp;
  // setupViewer(viewer, 1, vp);
  // viewer->removeCoordinateSystem();
  // keyboardEvent keyPress(viewer,1);

  Eigen::Affine3f tf1;
  pcl::CropBox<pcl::PointXYZRGB> cpBoxObj1;
  pcl::CropBox<pcl::PointXYZRGB> cpBoxObj2;

  pcl::PointXYZRGB cpBoxMin,cpBoxMax;
  cpBoxMin.x = -0.01;                 cpBoxMax.x = ::voxelGridSize*2.5;
  cpBoxMin.y = -::voxelGridSize*2.5;  cpBoxMax.y = ::voxelGridSize*2.5;
  cpBoxMin.z = -::voxelGridSize*2.5;  cpBoxMax.z = ::voxelGridSize*2.5;

  cpBoxObj1.setInputCloud(tempObj);

  cpBoxObj2.setInputCloud(tempUsefulUnexpObj);
  cpBoxObj2.setMin(cpBoxMin.getVector4fMap());
  cpBoxObj2.setMax(cpBoxMax.getVector4fMap());

  int stepSize = std::max(1,int(round(double(tempUsefulUnexp->size())/1000.0)));

  for(int i = 0; i < tempUsefulUnexp->points.size(); i+=stepSize){
    // indicesOK.clear(); indicesNotOK.clear();
    projPts.clear(); projPts.push_back(cv::Point(w/2,w/2));

    Eigen::Vector3f projCentre = projectionMat*(tempUsefulUnexp->points[i].getVector4fMap()); projCentre = projCentre/projCentre[2];
    // Converting the voxelGridSize to pixels (Scaled up by 2) as it would change based on depth
    maskRadius = round((projectionMat(0,0)*::voxelGridSize*1.1/tempUsefulUnexp->points[i].z) * scale);

    pcl::Normal normal;
    normal.normal_x = -1*tempUsefulUnexp->points[i].x;
    normal.normal_y = -1*tempUsefulUnexp->points[i].y;
    normal.normal_z = -1*tempUsefulUnexp->points[i].z;
    tf1 = calcTfFromNormal(normal,tempUsefulUnexp->points[i]);

    cpBoxMax.x = sqrt(pow(tempUsefulUnexp->points[i].x,2)+pow(tempUsefulUnexp->points[i].y,2)+pow(tempUsefulUnexp->points[i].z,2));
    cpBoxObj1.setMin(cpBoxMin.getVector4fMap()); cpBoxObj1.setMax(cpBoxMax.getVector4fMap());
    cpBoxObj1.setRotation(getEuler(tf1));   cpBoxObj1.setTranslation(getTranslation(tf1));
    cpBoxObj1.filter(tempIndices);

    for(auto j : tempIndices){
      bool tempResult = objCheckIfOccluded(tempUsefulUnexp->points[i],tempObj->points[j],tempNormal->points[j],::voxelGridSize*2.5);
      if(tempResult){
        Eigen::Vector3f projCurrent = projectionMat*(tempObj->points[j].getVector4fMap()); projCurrent = projCurrent/projCurrent[2];
        proj[1] = round((-projCentre[0]+projCurrent[0])*scale+w/2);
        proj[0] = round((-projCentre[1]+projCurrent[1])*scale+w/2);
        if(proj[0] >= 0 && proj[0] < w && proj[1] >= 0 && proj[1] < w){
          projPts.push_back(cv::Point(proj[1],proj[0]));
        }
      }
      // if(tempResult) indicesOK.push_back(j);
      // else indicesNotOK.push_back(j);
    }

    surface = cv::Mat::zeros(w-1,w-1,CV_8UC1);
    minPatchArea = 0.33*M_PI*pow(maskRadius-0.5,2)+1;
    patchMask = cv::Mat::zeros(w-1,w-1,CV_8UC1);
    cv::circle(patchMask, cv::Point(w/2,w/2), maskRadius, 255, cv::FILLED, cv::LINE_8);

    if(projPts.size() > 3){
      std::vector<std::vector<cv::Point>> hullPts(1);
      cv::convexHull(projPts, hullPts[0]);
      cv::drawContours(surface, hullPts, -1, 255,-1);
    }

    cv::Mat surfacePatch;
    surface.copyTo(surfacePatch,patchMask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(surfacePatch, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    float area = 0;
    cv::Point2f vtx[4];
    if(contours.size() == 1){
      cv::Moments mu = cv::moments(contours[0]);
      area = mu.m00;
    }

    // if(keyPress.ok){
    //   addRGB(viewer,tempObj,"tempObj",vp[0]);
    //   addRGB(viewer,tempUsefulUnexp,"tempUsefulUnexp",vp[0]);
    //   viewer->addSphere(tempUsefulUnexp->points[i],0.001,0,1,0,"P1",vp[0]);
    //   for(int x=0; x < indicesOK.size(); x++){
    //     viewer->addSphere(tempObj->points[indicesOK[x]],0.001,1,0,0,"P"+std::to_string(x+2),vp[0]);
    //     viewer->addLine(tempObj->points[indicesOK[x]],tempUsefulUnexp->points[i],1,0,0,"L"+std::to_string(x+2),vp[0]);
    //   }
    //   // for(int x=0; x < indicesNotOK.size(); x++){
    //   //   viewer->addLine(tempObj->points[indicesNotOK[x]],tempUsefulUnexp->points[i],0,0,1,"LL"+std::to_string(x+2),vp[0]);
    //   // }
    //   viewer->addLine(cam,tempUsefulUnexp->points[i],0,1,0,"L1",vp[0]);
    //   std::cout << maskRadius << "," << area << "," << M_PI*pow(maskRadius-0.5,2)+1 << std::endl;
    //   cv::circle(surfacePatch, cv::Point(w/2,w/2), 2, 127, cv::FILLED, cv::LINE_8);
    //   cv::Mat res;
    //   cv::resize(surfacePatch, res, cv::Size(), 4, 4);
    //   cv::imshow("Projection",res);
    //   cv::waitKey(10);
    //   while(!viewer->wasStopped() && keyPress.called==false){
    //     viewer->spinOnce(100);
    //     boost::this_thread::sleep (boost::posix_time::microseconds(100000));
    //   }
    //   viewer->resetStoppedFlag();
    //   viewer->removeAllPointClouds();
    //   viewer->removeAllShapes();
    // }

    if(area < minPatchArea) objClearIndices.insert(i);
  }

  // *****************STAGE 1 END*****************************

  // static ptCldVis::Ptr viewer(new ptCldVis ("PCL Viewer"));
  // std::vector<int> vp;
  // setupViewer(viewer, 1, vp);
  // viewer->removeCoordinateSystem();
  // keyboardEvent keyPress(viewer,1);
  // static ptCldColor::Ptr tempUsefulObjCleared{new ptCldColor}; tempUsefulObjCleared->clear();
  // for(auto i : objClearIndices){
  //   tempUsefulObjCleared->points.push_back(tempUsefulUnexp->points[i]);
  // }
  // tempUsefulObjCleared->width = tempUsefulObjCleared->points.size();
  // tempUsefulObjCleared->height = 1;

  // Check of the object cleared useful unxplored occlude each other
  for(auto i : objClearIndices){
    // indicesOK.clear(); indicesNotOK.clear();
    projPts.clear(); projPts.push_back(cv::Point(w/2,w/2));

    Eigen::Vector3f projCentre = projectionMat*(tempUsefulUnexp->points[i].getVector4fMap()); projCentre = projCentre/projCentre[2];
    // Converting the voxelGridSize to pixels (Scaled up by 2) as it would change based on depth
    maskRadius = round((projectionMat(0,0)*::voxelGridSize*1.1/tempUsefulUnexp->points[i].z) * scale);

    pcl::Normal normal;
    normal.normal_x = -1*tempUsefulUnexp->points[i].x;
    normal.normal_y = -1*tempUsefulUnexp->points[i].y;
    normal.normal_z = -1*tempUsefulUnexp->points[i].z;
    tf1 = calcTfFromNormal(normal,tempUsefulUnexp->points[i]);

    cpBoxObj2.setRotation(getEuler(tf1));   cpBoxObj2.setTranslation(getTranslation(tf1));
    cpBoxObj2.filter(tempIndices);

    for(auto j : tempIndices){
      bool tempResult = unexpCheckIfOccluded(tempUsefulUnexp->points[i],tempUsefulUnexpObj->points[j],::voxelGridSize*2);
      if(tempResult){
        Eigen::Vector3f projCurrent = projectionMat*(tempUsefulUnexpObj->points[j].getVector4fMap()); projCurrent = projCurrent/projCurrent[2];
        proj[1] = round((-projCentre[0]+projCurrent[0])*scale+w/2);
        proj[0] = round((-projCentre[1]+projCurrent[1])*scale+w/2);
        if(proj[0] >= 0 && proj[0] < w && proj[1] >= 0 && proj[1] < w){
          projPts.push_back(cv::Point(proj[1],proj[0]));
        }
      }
      // if(tempResult) indicesOK.push_back(j);
      // else indicesNotOK.push_back(j);
    }

    surface = cv::Mat::zeros(w-1,w-1,CV_8UC1);
    minPatchArea = 0.5*M_PI*pow(maskRadius-0.5,2)+1;
    patchMask = cv::Mat::zeros(w-1,w-1,CV_8UC1);
    cv::circle(patchMask, cv::Point(w/2,w/2), maskRadius, 255, cv::FILLED, cv::LINE_8);

    if(projPts.size() > 3){
      std::vector<std::vector<cv::Point>> hullPts(1);
      cv::convexHull(projPts, hullPts[0]);
      cv::drawContours(surface, hullPts, -1, 255,-1);
    }

    cv::Mat surfacePatch;
    surface.copyTo(surfacePatch,patchMask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(surfacePatch, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    float area = 0;
    cv::Point2f vtx[4];
    if(contours.size() == 1){
      cv::Moments mu = cv::moments(contours[0]);
      area = mu.m00;
    }

    // if(keyPress.ok){
    //   addRGB(viewer,tempObj,"tempObj",vp[0]);
    //   addRGB(viewer,tempUsefulObjCleared,"tempUsefulObjCleared",vp[0]);
    //   viewer->addSphere(tempUsefulUnexp->points[i],0.001,0,1,0,"P1",vp[0]);
    //   for(int x=0; x < indicesOK.size(); x++){
    //     viewer->addSphere(tempUsefulUnexp->points[indicesOK[x]],0.001,1,0,0,"P"+std::to_string(x+2),vp[0]);
    //     viewer->addLine(tempUsefulUnexp->points[indicesOK[x]],tempUsefulUnexp->points[i],1,0,0,"L"+std::to_string(x+2),vp[0]);
    //   }
    //   // for(int x=0; x < indicesNotOK.size(); x++){
    //   //   viewer->addLine(tempUsefulUnexp->points[indicesNotOK[x]],tempUsefulUnexp->points[i],0,0,1,"LL"+std::to_string(x+2),vp[0]);
    //   // }
    //   viewer->addLine(cam,tempUsefulUnexp->points[i],0,1,0,"L1",vp[0]);
    //   std::cout << maskRadius << "," << area << "," << M_PI*pow(maskRadius-0.5,2)+1 << std::endl;
    //   cv::circle(surfacePatch, cv::Point(w/2,w/2), 2, 127, cv::FILLED, cv::LINE_8);
    //     cv::Mat res;
    //     cv::resize(surfacePatch, res, cv::Size(), 4, 4);
    //     cv::imshow("Projection",res);
    //     cv::waitKey(10);
    //   while(!viewer->wasStopped() && keyPress.called==false){
    //     viewer->spinOnce(100);
    //     boost::this_thread::sleep (boost::posix_time::microseconds(100000));
    //   }
    //   viewer->resetStoppedFlag();
    //   viewer->removeAllPointClouds();
    //   viewer->removeAllShapes();
    // }

    if(area < minPatchArea) finalIndices.insert(i);
  }

  return finalIndices;
}

// int findDirection(ptCldColor::Ptr obj, ptCldColor::Ptr unexp, std::vector<double> &pose, int mode, int minVis, ptCldVis::Ptr viewer){
int findDirection(ptCldColor::Ptr obj, ptCldColor::Ptr unexp, std::vector<double> &pose, int mode, int minVis){

  static ptCldColor::Ptr usefulUnexp{new ptCldColor};
  extractUsefulUnexpPts(obj,unexp,usefulUnexp);
  int threshold = 75;
  std::set<int> visibleIndices;
  std::vector<std::set<int>> indices;
  std::vector<int> res = {0,0,0,0,0,0,0,0};
  std::vector<int> step = {0,0,0,0,0,0,0,0};
  std::vector<int> ok = {0,0,0,0,0,0,0,0};

  for(int i = 1 ; i <= 8; i++){
    std::vector<double> goToPose = calcExplorationPose(pose,i,mode,20*(M_PI/180.0));
    ok[i-1] = checkKinectOK(goToPose);
  }
  int maxRes = 0;

  int stepSize = std::max(1,int(round(double(usefulUnexp->size())/1000.0)));

  for(int i = 1 ; i <= 8; i++){
    visibleIndices.clear();
    if(ok[i-1] == 1){
      std::vector<double> newPose = calcExplorationPose(pose,i,mode,20*(M_PI/180.0)); step[i-1] = 20;
      visibleIndices = findVisibleUsefulUnexp(obj,unexp,usefulUnexp,newPose);
    }
    indices.push_back(visibleIndices);
  }

  for(int i = 0 ; i < 8; i++) res[i] = indices[i].size();
  maxRes = *std::max_element(res.begin(), res.end());

  if(maxRes*stepSize < minVis){
    for(int i = 1 ; i <= 8; i++){
      visibleIndices.clear();
      if(ok[i-1] == 1){
        std::vector<double> newPose = calcExplorationPose(pose,i,mode,40*(M_PI/180.0)); step[i-1] = 40;
        visibleIndices = findVisibleUsefulUnexp(obj,unexp,usefulUnexp,newPose);
      }
      indices[i-1].insert(visibleIndices.begin(),visibleIndices.end());
    }
    for(int i = 0 ; i < 8; i++) res[i] = indices[i].size();
    maxRes = *std::max_element(res.begin(), res.end());
  }

  for(int i = 0 ; i < 8; i++){
    if(maxRes != 0) res[i] = round(res[i]*100.0/maxRes);
    else            res[i] = 100;
  }

  std::vector<std::vector<int>> resDetailed;
  for(int i = 0 ; i < 8; i++) resDetailed.push_back({i+1,45*i,res[i]});
  std::sort(resDetailed.begin(),resDetailed.end(),compareDirections);
  // for(int i = 0 ; i < 8; i++) std::cout << resDetailed[i][0] << ":" << resDetailed[i][2] << ",";
  // std::cout << std::endl;

  // /***/
  // std::vector<int> vp;
  // setupViewer(viewer, 9, vp);
  // setCamView(viewer,{pose[0]/2,pose[1],pose[2]},::table,vp[0]);
  // addRGB(viewer,obj,"obj0",vp[0]);
  // addRGB(viewer,unexp,"unexp",vp[0]);
  // addRGB(viewer,usefulUnexp,"usefulUnexp",vp[0]);
  // viewer->removeCoordinateSystem();

  // ptCldColor::Ptr result{new ptCldColor};
  // for(int i = 1 ; i <= 8; i++){
  //   if(ok[i-1] == 0) continue;
  //   result->clear();
  //   for(auto idx : indices[i-1]){
  //     result->points.push_back(usefulUnexp->points[idx]);
  //     result->points.back().r = 255;
  //     result->points.back().g = 255;
  //     result->points.back().b = 255;
  //   }
  //   std::vector<double> newPose = calcExplorationPose(pose,i,mode,step[i-1]*(M_PI/180.0));
  //   setCamView(viewer,{newPose[0],newPose[1],newPose[2]},::table,vp[i]);
  //   addRGB(viewer,obj,"obj"+std::to_string(i),vp[i]);
  //   addRGB(viewer,result,"result"+std::to_string(i),vp[i]);
  //   viewer->addText(std::to_string(i)+","+std::to_string(step[i-1])+","+std::to_string(res[i-1]),2,2,20,1,0,0,"data"+std::to_string(i),vp[i]);
  // }
  // while(!viewer->wasStopped()){
  //   viewer->spinOnce(100);
  //   boost::this_thread::sleep (boost::posix_time::microseconds(100000));
  // }
  // viewer->resetStoppedFlag();
  // viewer->removeAllPointClouds();
  // viewer->removeAllShapes();
  /***/

  return resDetailed[0][0];
  // return std::distance(res.begin(), std::max_element(res.begin(), res.end())) + 1;
}

bool heuristicPolicy(active_vision::heuristicPolicySRV::Request  &req,
                     active_vision::heuristicPolicySRV::Response &res){

  static ptCldColor::Ptr obj{new ptCldColor};
  static ptCldColor::Ptr unexp{new ptCldColor};
  static std::vector<std::vector<double>> path;
  static std::vector<double> temp;
  static int mode;
  static std::vector<int> nUnexp;

  pcl::fromROSMsg(req.object, *obj);
  pcl::fromROSMsg(req.unexplored, *unexp);
  temp = req.path.data; path.clear();
  for(int i = 0; i < temp.size(); i=i+3){
    path.push_back({temp[i],temp[i+1],temp[i+2]});
  }
  mode = req.mode;

  res.direction = findDirection(obj,unexp,path.back(),mode,req.minPtsVis)-1;

  ROS_INFO_STREAM("Heuristic Policy Service Called. Direction -> "<< dirLookup[res.direction+1]);

  return true;
}

int main(int argc, char** argv){
  ros::init(argc, argv, "HeuristicPolicyServer");

  ros::NodeHandle nh;
  nh.getParam("/active_vision/environment/voxelGridSize", ::voxelGridSize);
  nh.getParam("/active_vision/policyTester/heuristicDiagonalPref", ::heuristicDiagonalPref);
  nh.getParam("/active_vision/simulationMode", ::simulationMode);
  std::vector<double> tableCentre;
  nh.getParam("/active_vision/environment/tableCentre", tableCentre);
  ::IKClient =  nh.serviceClient<moveit_planner::Inv>("inverse_kinematics_collision_check");
  ::table.x = tableCentre[0];
  ::table.y = tableCentre[1];
  ::table.z = tableCentre[2];

  ::projectionMat.resize(3,4);
  if(::simulationMode == "FRANKA"){
    ::projectionMat << 383.28009033203125, 0.0, 323.4447021484375, 0.0,
                       0.0, 383.28009033203125, 237.4062042236328, 0.0,
                       0.0, 0.0, 1.0, 0.0;
  }else{
    ::projectionMat << 554.254691191187, 0.0, 320.5, 0.0,
                       0.0, 554.254691191187, 240.5, 0.0,
                       0.0, 0.0, 1.0, 0.0;
  }

  ros::ServiceServer service = nh.advertiseService("/active_vision/heuristic_policy", heuristicPolicy);
  ROS_INFO("3D Heuristic policy service ready.");
  ros::spin();
  return 0;
}

// int main (int argc, char** argv){
//   // Initialize ROS
//   ros::init(argc, argv, "Test");
//   ros::NodeHandle nh;
//   nh.getParam("/active_vision/environment/voxelGridSize", ::voxelGridSize);
//   nh.getParam("/active_vision/policyTester/heuristicDiagonalPref", ::heuristicDiagonalPref);
//   nh.getParam("/active_vision/simulationMode", ::simulationMode);
//   std::vector<double> tableCentre;
//   nh.getParam("/active_vision/environment/tableCentre", tableCentre);

//   ::IKClient =  nh.serviceClient<moveit_planner::Inv>("inverse_kinematics_collision_check");
//   ::table.x = tableCentre[0];
//   ::table.y = tableCentre[1];
//   ::table.z = tableCentre[2];
//   environment activeVision(&nh);
//   activeVision.loadGripper();
//   ::projectionMat.resize(3,4);
//   if(::simulationMode == "FRANKA"){
//     ::projectionMat << 383.28009033203125, 0.0, 323.4447021484375, 0.0,
//                        0.0, 383.28009033203125, 237.4062042236328, 0.0,
//                        0.0, 0.0, 1.0, 0.0;
//   }else{
//     ::projectionMat << 554.254691191187, 0.0, 320.5, 0.0,
//                        0.0, 554.254691191187, 240.5, 0.0,
//                        0.0, 0.0, 1.0, 0.0;
//   }

//   // Delay to ensure all publishers and subscribers are connected
//   boost::this_thread::sleep(boost::posix_time::milliseconds(500));

//   if(argc != 4){
//     std::cout << "ERROR. Incorrect number of arguments." << std::endl;
//     return(-1);
//   }

//   int objID = std::atoi(argv[1]);
//   int pose = std::atoi(argv[2]);
//   int yaw = std::atoi(argv[3]);
//   int dir;
//   std::vector<double> newPose = {activeVision.viewsphereRad,-M_PI,M_PI/7};
//   std::vector<double> tempPose;
//   activeVision.spawnObject(objID,pose,yaw*M_PI/180);

//   singlePass(activeVision,newPose,true,true,2);
//   int minPtsVis;

//   ptCldVis::Ptr viewer(new ptCldVis ("PCL Viewer"));

//   int nSteps = 0;
//   while(activeVision.selectedGrasp == -1 && nSteps <= 5){
//     minPtsVis = 0.20*(activeVision.ptrPtCldObject->points.size());
//     dir = findDirection(activeVision.ptrPtCldObject,activeVision.ptrPtCldUnexp,newPose,2,minPtsVis,viewer);
//     // dir = findDirection(activeVision.ptrPtCldObject,activeVision.ptrPtCldUnexp,newPose,2,minPtsVis);
//     std::cout << "Direction selected : " << dir << std::endl;

//     tempPose = calcExplorationPose(newPose,dir,2);
//     if(activeVision.moveKinectViewsphere(tempPose,true)){
//       singlePass(activeVision,tempPose,false,true,2);
//       newPose = tempPose;
//     }
//     nSteps++;
//   }

//   if(activeVision.selectedGrasp != -1){
//     std::cout << "Centroid : " << activeVision.cenObject[0] <<  ","
//                                << activeVision.cenObject[1] <<  ","
//                                << activeVision.cenObject[2] << std::endl;
//     std::cout << "Selected Grasp ID : " << activeVision.selectedGrasp << std::endl;
//     std::cout << "Selected Grasp Quality : " << activeVision.graspsPossible[activeVision.selectedGrasp].quality << std::endl;
//     std::cout << "graspData.pose = {" << activeVision.graspsPossible[activeVision.selectedGrasp].pose[0] << "," <<
//                                         activeVision.graspsPossible[activeVision.selectedGrasp].pose[1] << "," <<
//                                         activeVision.graspsPossible[activeVision.selectedGrasp].pose[2] << "," <<
//                                         activeVision.graspsPossible[activeVision.selectedGrasp].pose[3] << "," <<
//                                         activeVision.graspsPossible[activeVision.selectedGrasp].pose[4] << "," <<
//                                         activeVision.graspsPossible[activeVision.selectedGrasp].pose[5] << "};" << std::endl;
//     std::cout << "graspData.addnlPitch = " << activeVision.graspsPossible[activeVision.selectedGrasp].addnlPitch << ";" << std::endl;
//     std::cout << "graspData.gripperWidth = " << activeVision.graspsPossible[activeVision.selectedGrasp].gripperWidth << ";" << std::endl;

//     std::vector<int> vp;
//     setupViewer(viewer, 1, vp);
//     keyboardEvent keyPress(viewer,1);
//     setCamView(viewer,{newPose[0],newPose[1],newPose[2]},::table,vp[0]);
//     activeVision.updateGripper(activeVision.selectedGrasp,0);
//     addRGB(viewer,activeVision.ptrPtCldEnv,"Env",vp[0]);
//     addRGB(viewer,activeVision.ptrPtCldUnexp,"Unexp",vp[0]);
//     addRGB(viewer,activeVision.ptrPtCldGripper,"Gripper",vp[0]);
//     // viewer->removeCoordinateSystem();
//     viewer->spinOnce(100);
//     boost::this_thread::sleep (boost::posix_time::microseconds(100000));
//     std::cout << "Press Q to exit" << std::endl;
//     while(!viewer->wasStopped()){
//       viewer->spinOnce(100);
//       boost::this_thread::sleep (boost::posix_time::microseconds(100000));
//     }

//     if(::simulationMode != "SIMULATION"){
//       activeVision.graspObject(activeVision.graspsPossible[activeVision.selectedGrasp]);
//     }
//   }else{
//     std::cout << "No grasp found" << std::endl;
//   }


//   activeVision.deleteObject(objID);
//   boost::this_thread::sleep(boost::posix_time::milliseconds(500));
// }
