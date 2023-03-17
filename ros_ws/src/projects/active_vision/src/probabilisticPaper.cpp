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
std::vector<std::vector<double>> camPoses;
std::vector<int> camPoseOK;

int testAll;
int maxSteps;
int nSaved;
bool recordPtCldEachStep;
int visual;

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

void setupCamPoses(float rad){
  ::camPoses.push_back({rad,M_PI,M_PI/6.5}); // Home
  ::camPoseOK.push_back(checkKinectOK(::camPoses.back()));
  // Circle 1
  float j = 90.0/4.0;
  for(int i = 0; i < 360; i+=40){
    ::camPoses.push_back({rad,i*M_PI/180,j*M_PI/180});
    ::camPoseOK.push_back(checkKinectOK(::camPoses.back()));
  }
  // Circle 2
  j += 90.0/4.0;
  for(int i = 60; i <= 360-60; i+=20){
    if(i > 160 && i < 200) continue;
    ::camPoses.push_back({rad,i*M_PI/180,j*M_PI/180});
    ::camPoseOK.push_back(checkKinectOK(::camPoses.back()));
  }
  // Circle 3
  j += 90.0/4.0;
  for(int i = 60; i <= 360-60; i+=20){
    if(i > 160 && i < 200) continue;
    ::camPoses.push_back({rad,i*M_PI/180,j*M_PI/180});
    ::camPoseOK.push_back(checkKinectOK(::camPoses.back()));
  }
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

  // Considering all points above the table
  for(int i = 0; i < unexp->width; i++){
    if(unexp->points[i].z >= ::table.z+0.01){
      res->points.push_back(unexp->points[i]);
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

  int stepSize = std::max(1,int(round(double(tempUsefulUnexp->size())/2000.0)));

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

int findPose(ptCldColor::Ptr obj, ptCldColor::Ptr unexp){

  static ptCldColor::Ptr usefulUnexp{new ptCldColor};
  extractUsefulUnexpPts(obj,unexp,usefulUnexp);
  std::set<int> visibleIndices;
  std::vector<int> res; res.clear();

  for(int i = 0; i < ::camPoses.size(); i++){
    bool poseOK = checkKinectOK(::camPoses[i]);
    visibleIndices.clear();
    if(poseOK) visibleIndices = findVisibleUsefulUnexp(obj,unexp,usefulUnexp,::camPoses[i]);
    res.push_back(visibleIndices.size());
    std::cout << i << "," << std::flush;
  }
  std::cout << std::endl;

  return std::distance(res.begin(), std::max_element(res.begin(), res.end())) + 1;
}

void findGrasp(environment &kinectControl, int object, int objPoseCode, int objYaw, std::string dir, std::string saveLocation){

  std::fstream fout;
 	fout.open(dir+saveLocation, std::ios::out | std::ios::app);
  kinectControl.moveObject(object,objPoseCode,objYaw*(M_PI/180.0));

  int poseID;
  std::vector<int> poseIDs = {0};

  std::vector<double> timeRecord;
  timeRecord = singlePass(kinectControl,::camPoses[0],true,true,2);

  RouteData data;
  data.path = {::camPoses[0]};
	data.nSteps = 0;
  data.type = 1;
	data.objType = kinectControl.objectDict[object].description;
	data.objPose = {kinectControl.objectDict[object].poses[objPoseCode][1],
									kinectControl.objectDict[object].poses[objPoseCode][2],
									objYaw*(M_PI/180.0)};
  data.goodInitialGrasp = (kinectControl.selectedGrasp != -1);
  data.obj = *kinectControl.ptrPtCldObject;
  data.unexp = *kinectControl.ptrPtCldUnexp;
  data.env = *kinectControl.ptrPtCldEnv;
	data.timer.push_back(timeRecord[0]);
	data.timer.push_back(timeRecord[1]);
	data.detailedEnv.push_back(*kinectControl.ptrPtCldEnv);

  std::chrono::high_resolution_clock::time_point start, end;

  while(kinectControl.selectedGrasp == -1 && data.nSteps <= ::maxSteps){
    start = std::chrono::high_resolution_clock::now();
    poseID = findPose(kinectControl.ptrPtCldObject,kinectControl.ptrPtCldUnexp) - 1; poseIDs.push_back(poseID);
    end = std::chrono::high_resolution_clock::now();
    data.timer.push_back((std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count());

    std::cout << "Pose selected : " << poseID << std::endl;
    timeRecord = singlePass(kinectControl,::camPoses[poseID],false,true,2);
    data.timer.push_back(timeRecord[0]);
    data.timer.push_back(timeRecord[1]);
    data.detailedEnv.push_back(*kinectControl.ptrPtCldEnv);
    data.path.push_back(::camPoses[poseID]);
    data.nSteps++;
  }

  if(kinectControl.selectedGrasp != -1){
    data.graspQuality = kinectControl.graspsPossible[kinectControl.selectedGrasp].quality;
		kinectControl.updateGripper(kinectControl.selectedGrasp,0);
		data.env = *kinectControl.ptrPtCldEnv + *kinectControl.ptrPtCldGripper;
    data.filename = getCurTime()+"_"+std::to_string(::nSaved);
    ::nSaved++;
    double distance = 0;
    for(int i = 0; i < poseIDs.size()-1; i++){
      distance += disBtwSpherical(::camPoses[poseIDs[i]],::camPoses[poseIDs[i+1]]);
    }
    double oneStep = 20.0/180.0*M_PI*kinectControl.viewsphereRad;
    data.EffnSteps = distance/oneStep;
    saveData(data, fout, dir, ::recordPtCldEachStep);
  }

  if(::visual == 1){
    ptCldVis::Ptr viewer(new ptCldVis ("PCL Viewer"));
    std::vector<int> vp;
    setupViewer(viewer, 1, vp);
    keyboardEvent keyPress(viewer,1);

    // Viewing the the camPoses
    pcl::PointXYZ a1,a2;
    double rad = kinectControl.viewsphereRad;
    addViewsphere(viewer,vp[0],::table,rad,false);
    for(int i = 0; i < ::camPoses.size(); i++){
      a1 = sphericalToCartesian(::camPoses[i],::table);
      if(camPoseOK[i] == 1) viewer->addSphere(a1,0.02,0,1,0,"Sphere"+std::to_string(i),vp[0]);
      else viewer->addSphere(a1,0.02,1,0,0,"Sphere"+std::to_string(i),vp[0]);
    }

    addRGB(viewer,kinectControl.ptrPtCldEnv,"Env",vp[0]);
    addRGB(viewer,kinectControl.ptrPtCldUnexp,"Unexp",vp[0]);
    addRGB(viewer,kinectControl.ptrPtCldGripper,"Gripper",vp[0]);

    for(int i = 0; i < poseIDs.size()-1; i++){
      a1 = sphericalToCartesian(::camPoses[poseIDs[i]],::table);
      a2 = sphericalToCartesian(::camPoses[poseIDs[i+1]],::table);
      viewer->addArrow(a2,a1,0,1,0,false,std::to_string(i),vp[0]);
    }

    viewer->setCameraPosition(0,0,2,::table.x,::table.y,::table.z,0,0,1);

    viewer->spinOnce(100);
    boost::this_thread::sleep (boost::posix_time::microseconds(100000));

    std::cout << "Press Q to exit" << std::endl;
    while(!viewer->wasStopped()){
      viewer->spinOnce(100);
      boost::this_thread::sleep (boost::posix_time::microseconds(100000));
    }

  }

  kinectControl.reset();
	fout.close();
}

int main(int argc, char** argv){
	chdir(getenv("HOME"));

	ros::init(argc, argv, "Policy_Tester");
 	ros::NodeHandle nh;
 	environment kinectControl(&nh); sleep(1);
  kinectControl.loadGripper();

	bool relativePath; nh.getParam("/active_vision/policyTester/relative_path", relativePath);
	std::string temp;
	nh.getParam("/active_vision/policyTester/directory", temp);
	std::string dir;
	if(relativePath) dir = ros::package::getPath("active_vision") + temp;
  else             dir = temp;

	std::string time = getCurTime();
	nh.getParam("/active_vision/policyTester/csvName", temp);
	std::string csvName;
	if(temp == "default.csv") csvName = time+"_dataRec.csv";
	else	csvName = temp;

	if(csvName.substr(csvName.size() - 4) != ".csv"){
    ROS_ERROR("Incorrect file name.");
    return(-1);
	}

	int objID;
	nh.getParam("/active_vision/policyTester/objID", objID);
	if(kinectControl.objectDict.count(objID) == 0){
		std::cout << "ERROR. Incorrect Object ID." << std::endl;
    return(-1);
	}

	nh.getParam("/active_vision/policyTester/maxSteps", ::maxSteps);
	nh.getParam("/active_vision/simulationMode", ::simulationMode);

  nh.getParam("/active_vision/environment/voxelGridSize", ::voxelGridSize);
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

  setupCamPoses(kinectControl.viewsphereRad);

	if(::simulationMode != "FRANKA"){
		printf("Do you want to test for all possible poses (0->No, 1->Yes) : ");
		std::cin >> ::testAll;
		if(::testAll < 0 && ::testAll > 1) ::testAll = 0;
	}else{
		::testAll = 0;
	}

	kinectControl.spawnObject(objID,0,0);
	::nSaved = 0;
	std::chrono::high_resolution_clock::time_point start, end;
	double elapsed;
	if(testAll == 0){
    ::visual = 1;
		int objPoseCode, objYaw;
		if(::simulationMode != "FRANKA"){
			printf("Enter the object pose code (0-%d) : ", int(kinectControl.objectDict[objID].nPoses-1));
			std::cin >> objPoseCode;
			if(objPoseCode < 0 || objPoseCode > kinectControl.objectDict[objID].nPoses-1) objPoseCode = 0;
			printf("Enter the object yaw (deg) (0-360) : ");
			std::cin >> objYaw;
		}else{
			objPoseCode = 0;
			objYaw = 0;
		}

		findGrasp(kinectControl, objID, objPoseCode, objYaw, dir, csvName);

		if(kinectControl.selectedGrasp != -1) std::cout << "Grasp found." << std::endl;
		else																  std::cout << "Grasp not found." << std::endl;
	}else{
		// Reading the yawValues csv file
    ::visual = 0;
		std::string yawAnglesCSVDir;
		nh.getParam("/active_vision/policyTester/yawAnglesCSVDir", yawAnglesCSVDir);
		yawAnglesCSVDir = ros::package::getPath("active_vision") + yawAnglesCSVDir;
		std::string yawAnglesCSV;
		nh.getParam("/active_vision/policyTester/yawAnglesCSV", yawAnglesCSV);
		std::vector<std::vector<std::string>> yawAngle = readCSV(yawAnglesCSVDir+yawAnglesCSV);

		// Converting it to a dictionary format (First column is the key)
		std::map<std::string,std::vector<int>> yawAngleDict;
		for(int row = 0; row < yawAngle.size(); row++){
			yawAngleDict.insert({yawAngle[row][0],{}});
			for(int col = 1; col < yawAngle[row].size(); col++){
				yawAngleDict[yawAngle[row][0]].push_back(std::stoi(yawAngle[row][col]));
			}
		}

		int uniformYawStepSize; nh.getParam("/active_vision/policyTester/uniformYawStepSize", uniformYawStepSize);
		int nDataPoints; nh.getParam("/active_vision/policyTester/nDataPoints", nDataPoints);
		if(::simulationMode == "FRANKASIMULATION") nDataPoints = 3;
		for(int objPoseCode = 0; objPoseCode < kinectControl.objectDict[objID].nPoses; objPoseCode+=1){
			int tempNDataPoints = nDataPoints;

			// Checking for the uniform steps
			for(int objYaw = kinectControl.objectDict[objID].poses[objPoseCode][3]; objYaw < kinectControl.objectDict[objID].poses[objPoseCode][4]; objYaw+=uniformYawStepSize){
				printf("%d - Obj #%d, Pose #%d, Yaw %d \t",nDataPoints-tempNDataPoints+1,objID,objPoseCode,objYaw);
				start = std::chrono::high_resolution_clock::now();
				findGrasp(kinectControl, objID, objPoseCode, objYaw, dir, csvName);
				end = std::chrono::high_resolution_clock::now();
				elapsed = (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count();
				std::cout << "\tTime(ms) : " << elapsed << std::endl;
				tempNDataPoints--;
			}

			std::string key = std::to_string(int(kinectControl.objectDict[objID].poses[objPoseCode][3]))+"-"+
			                  std::to_string(int(kinectControl.objectDict[objID].poses[objPoseCode][4]));

			if(yawAngleDict.count(key) == 0) continue;

			// Checking for the random steps
			for(int ctr = 0; ctr < yawAngleDict[key].size() && tempNDataPoints > 0; ctr++){
				printf("%d - Obj #%d, Pose #%d, Yaw %d \t",nDataPoints-tempNDataPoints+1,objID,objPoseCode,yawAngleDict[key][ctr]);
				start = std::chrono::high_resolution_clock::now();
				findGrasp(kinectControl, objID, objPoseCode, yawAngleDict[key][ctr], dir, csvName);
				end = std::chrono::high_resolution_clock::now();
				elapsed = (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count();
				std::cout << "\tTime(ms) : " << elapsed << std::endl;
				tempNDataPoints--;
			}
		}
	}

	kinectControl.deleteObject(objID);

	printf("Data saved to : %s\n",csvName.c_str());
 }
