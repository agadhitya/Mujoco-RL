#include <active_vision/environment.h>
#include <active_vision/toolVisualization.h>

// Function to check if a ROS NODE/TOPIC/SERVICE exists
bool ROSCheck(std::string type, std::string name){
  bool status = true;
  if(type == "NODE"){
    // Get the list of available nodes and search in them
    std::vector<std::string> nodeList;
    ros::master::getNodes(nodeList);
    status = std::find(nodeList.begin(), nodeList.end(), name) != nodeList.end();
  }else if(type == "TOPIC"){
    // Get the list of available topics and search in them
    ros::master::V_TopicInfo topicList;
    ros::master::getTopics(topicList);
    status = false;
    for(int i = 0; i < topicList.size(); i++){
      if(topicList[i].name == name){status = true; break;}
    }
  }else if(type == "SERVICE"){
    // Check if the service is running
    status = ros::service::exists(name, false);
  }
  if(!status) ROS_INFO_STREAM("Waiting for " << name << "...");
  return status;
}

// ***STRUCT : graspPoint***
graspPoint::graspPoint(){
  quality = 0;              // Quality of the grasp
  gripperWidth = 0.05;      // Gripper width for the grasp
  pose = {0,0,0,0,0,0};     // Base pose of the gripper from where additional pitch is used to change its orientation
  addnlPitch = 0;           // Pitch relative to the Base Pose
  distance = 999;           // Pitch relative to the Base Pose
}

// Function to compare grasp point for sorting
bool compareGrasp(graspPoint A, graspPoint B){
  // Approach 2
  if(abs(A.lineDistance - B.lineDistance) <= 0.01){
    if(abs(A.pose[2] - B.pose[2]) <= 0.06){
      if(abs(A.distance - B.distance) <= 0.01){
        return(A.quality > B.quality);
      }else{
        return(A.distance < B.distance);
      }
    }
    else{
      return(A.pose[2] > B.pose[2]);
    }
  }else{
   return(A.lineDistance < B.lineDistance);
  }
  // Approach 1
  // if(abs(A.distance - B.distance) <= 0.005){
  //   return(A.quality > B.quality);
  // }else{
  //   return(A.distance < B.distance);
  // }
}
// ***END***

// Funstion to transpose a homogenous matrix
Eigen::Affine3f homoMatTranspose(const Eigen::Affine3f& tf){
  Eigen::Affine3f tfTranspose;
  tfTranspose.setIdentity();
  tfTranspose.matrix().block<3,3>(0,0) = tf.rotation().transpose();
  tfTranspose.matrix().block<3,1>(0,3) = -1*tf.rotation().transpose()*tf.translation();
  return(tfTranspose);
}

// Get Rotation Part of a Affine3f
Eigen::Vector3f getEuler(const Eigen::Affine3f& tf){
  return Eigen::Vector3f(atan2f(tf(2,1), tf(2,2)),
                         asinf(-tf(2,0)),
                         atan2f(tf(1,0), tf(0,0)));
}

// Get Translational Part of a Affine3f
Eigen::Vector3f getTranslation(const Eigen::Affine3f& tf){
  return Eigen::Vector3f(tf(0,3), tf(1,3), tf(2,3));
}

Eigen::Affine3f calcTfFromNormal(pcl::Normal normal, pcl::PointXYZRGB point){

  Eigen::Matrix4f tfMat; tfMat.setIdentity();
  Eigen::Affine3f tf;
  Eigen::Matrix3f rot;
  Eigen::Vector3f trans;
  calcTfFromNormal(normal, point, rot, trans);

  tfMat.block<3,3>(0,0) = rot;
  tfMat.block<3,1>(0,3) = trans;
  tf.matrix() = tfMat;

  return tf;
}

void calcTfFromNormal(pcl::Normal normal, pcl::PointXYZRGB point, Eigen::Matrix3f &rot, Eigen::Vector3f &trans){
  Eigen::Matrix4f tfMat; tfMat.setIdentity();
  Eigen::Affine3f tf;

  Eigen::Vector3f xAxis,yAxis,zAxis,xyPlane(0,0,1);

  xAxis = {normal.normal_x,normal.normal_y,normal.normal_z}; xAxis.normalize();
  yAxis = xAxis.cross(xyPlane);
  if(yAxis.norm() == 0) yAxis << normal.normal_x,0,0;
  yAxis.normalize();
  zAxis = xAxis.cross(yAxis);

  rot << xAxis[0], yAxis[0], zAxis[0],
         xAxis[1], yAxis[1], zAxis[1],
         xAxis[2], yAxis[2], zAxis[2];
  trans = point.getVector3fMap();
}

// Estimating if the contact patch is sufficient
bool isContactPatchOk(ptCldColor::Ptr obj, ptCldNormal::Ptr normal, long int ptIdx, float voxelGridSize){
  int w = 100;
  cv::Mat patchMask = cv::Mat::zeros(w+1,w+1,CV_8UC1);
  int patchWidth = 26;
  float patchArea = patchWidth*patchWidth;
  circle(patchMask, cv::Point(w/2,w/2), int(patchWidth/2.0*sqrt(2)), 255, cv::FILLED, cv::LINE_8);

  pcl::PointXYZRGB minBound,maxBound;
  minBound.x = -0.005; maxBound.x = 0.005;
  minBound.y = -0.030; maxBound.y = 0.030;
  minBound.z = -0.030; maxBound.z = 0.030;

  pcl::CropBox<pcl::PointXYZRGB> cpBox;
  cpBox.setInputCloud(obj);
  cpBox.setMin(minBound.getVector4fMap());
  cpBox.setMax(maxBound.getVector4fMap());

  Eigen::Affine3f tf = calcTfFromNormal(normal->points[ptIdx],obj->points[ptIdx]);
  cpBox.setRotation(getEuler(tf));
  cpBox.setTranslation(getTranslation(tf));

  static ptCldColor::Ptr objFiltered{new ptCldColor};
  cpBox.filter(*objFiltered);
  Eigen::Affine3f tfTranspose = homoMatTranspose(tf);
  pcl::transformPointCloud(*objFiltered, *objFiltered, tfTranspose);

  cv::Mat surface = cv::Mat::zeros(w+1,w+1,CV_8UC1);
  Eigen::Vector3f proj;
  std::vector<cv::Point> projPts;
  for(int i = 0; i<objFiltered->points.size(); i++){
    proj[0] = int(objFiltered->points[i].y*1000+w/2);
    proj[1] = int(objFiltered->points[i].z*1000+w/2);
    if(proj[0] >= 0 && proj[0] < w+1 && proj[1] >= 0 && proj[1] < w+1){
      // uchar &intensity = surface.at<uchar>(proj[1],proj[0]);
      // intensity = 255;
      projPts.push_back(cv::Point(proj[1],proj[0]));
    }else{
      ROS_WARN("Error in isContactPatchOk projection.");
    }
  }

  std::vector<std::vector<cv::Point>> hullPts(1);
  cv::convexHull(projPts, hullPts[0]);
  cv::drawContours(surface, hullPts, -1, 255,-1);

  cv::Mat surfacePatch; surface.copyTo(surfacePatch, patchMask);
  // Finding the minimum inscribed circle
  double max_val; cv::Point max_loc; cv::Mat1f dt;
  cv::distanceTransform(surfacePatch, dt, cv::DIST_L2, 5, cv::DIST_LABEL_PIXEL);
  cv::minMaxLoc(dt, nullptr, &max_val, nullptr, &max_loc);

  if(max_val >= patchWidth/1.9) return true;
  else                          return false;

  // std::vector<std::vector<cv::Point>> contours;
  // cv::findContours(surfacePatch, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  // if(contours.size() == 1){
  //   cv::RotatedRect bBox = cv::minAreaRect(contours[0]);
  //   float dimRatio = std::max(bBox.size.width,bBox.size.height) / std::min(bBox.size.width,bBox.size.height);
  //   cv::Moments mu = cv::moments(contours[0]);
  //
  //   double max_val;
  //   cv::Point max_loc;
  //   cv::Mat1f dt;
  //
  //   cv::circle(surfacePatch, max_loc, max_val, 127, cv::FILLED, cv::LINE_8);
  //
  //   float cx,cy,area;
  //   cx = mu.m10 / (mu.m00 + 1e-5);
  //   cy = mu.m01 / (mu.m00 + 1e-5);
  //   area = mu.m00;
  //
  //   // if(dimRatio <= 1.75 && area >= patchArea*0.8 && sqrt(pow(cx - w/2,2) + pow(cy - w/2,2)) <= 4){
  //   //   circle(surfacePatch, cv::Point(10,10), int(5), 255, cv::FILLED, cv::LINE_8);
  //   //   cv::imshow("Projection",surfacePatch);
  //   //   cv::waitKey(2);
  //   //   std::cout << 1 << "," << max_val << "," << sqrt(pow(cx - w/2,2) + pow(cy - w/2,2)) << "," << sqrt(pow(max_loc.x - w/2,2) + pow(max_loc.y - w/2,2)) << std::endl;
  //   //   return true;
  //   // }
  // }

}

/*! \brief Convert RGB to HSV color space
  Converts a given set of RGB values `r', `g', `b' into HSV
  coordinates. The input RGB values are in the range [0, 1], and the
  output HSV values are in the ranges h = [0, 360], and s, v = [0,
  1], respectively.
*/
void RGBtoHSV(float fR, float fG, float fB, float& fH, float& fS, float& fV){
  float fCMax = std::max(std::max(fR, fG), fB);
  float fCMin = std::min(std::min(fR, fG), fB);
  float fDelta = fCMax - fCMin;

  if(fDelta > 0){
    if(fCMax == fR)       fH = 60 * (std::fmod(((fG - fB) / fDelta), 6));
    else if(fCMax == fG)  fH = 60 * (((fB - fR) / fDelta) + 2);
    else if(fCMax == fB)  fH = 60 * (((fR - fG) / fDelta) + 4);

    if(fCMax > 0) fS = fDelta / fCMax;
    else          fS = 0;

    fV = fCMax;
  }else{
    fH = 0;
    fS = 0;
    fV = fCMax;
  }

  if(fH < 0) fH = 360 + fH;
}

// Check if franka can reach the point
bool checkFrankReach(ros::ServiceClient &IKClient, geometry_msgs::Pose &p){
  moveit_planner::Inv poseIKMsg;
  poseIKMsg.request.pose = p;
  return IKClient.call(poseIKMsg);
}

// Function to fill holes in a ordered point cloud
void holeFilling(ptCldColor::ConstPtr input, ptCldColor::Ptr output){
  ptCldColor::Ptr tempOutput{new ptCldColor};
  *output = *input;
  *tempOutput = *output;
  for(int p = 0; p < 5; p++){
    // Hole Filling
    for(int i = 0; i < output->width ; i++){
      for(int j = 0; j < output->height ; j++){
        if(i < output->width / 6 * 1)  continue;
        if(i > output->width / 6 * 5)  continue;
        if(j < output->height / 6 * 1) continue;
        if(j > output->height / 6 * 5) continue;
        int index = j*(output->width)+i;
        if(output->points[index].z == 0){
          double x,y,z,wtSum;
          x = 0;y = 0;z = 0; wtSum = 0;
          int ctr = 0;

          int var;
          int k = 20;
          // Left
          var = i - 1;
          while(i - var < k){
            if(output->points[j*(output->width)+var].z != 0){
              double wt = 1 - (0.8*(i-var))/(1.414*k);
              wtSum+=wt;
              z += wt*output->points[j*(output->width)+var].z;
              ctr++;
              break;
            }
            var--;
          }
          // Right
          var = i + 1;
          while(var - i < k){
            if(output->points[j*(output->width)+var].z != 0){
              double wt = 1 - (0.8*(var-i))/(1.414*k);
              wtSum+=wt;
              z += wt*output->points[j*(output->width)+var].z;
              ctr++;
              break;
            }
            var++;
          }
          // Top
          var = j - 1;
          while(j - var < k){
            if(output->points[var*(output->width)+i].z != 0){
              double wt = 1 - (0.8*(j-var))/(1.414*k);
              wtSum+=wt;
              z += wt*output->points[var*(output->width)+i].z;
              ctr++;
              break;
            }
            var--;
          }
          // Bottom
          var = j + 1;
          while(var - j < k){
            if(output->points[var*(output->width)+i].z != 0){
              double wt = 1 - (0.8*(var-j))/(1.414*k);
              wtSum+=wt;
              z += wt*output->points[var*(output->width)+i].z;
              ctr++;
              break;
            }
            var++;
          }

          if(ctr > 2 && wtSum > 1){
            float calcX = x / wtSum;
            float calcY = y / wtSum;
            float calcZ = z / wtSum;
            tempOutput->points[index].z = calcZ;
            tempOutput->points[index].x = (i - 323.4447021484375)*calcZ/383.28009033203125;
            tempOutput->points[index].y = (j - 237.4062042236328)*calcZ/383.28009033203125;
            tempOutput->points[index].r = 0;
            tempOutput->points[index].g = 0;
            tempOutput->points[index].b = 255;
            // ctrrr ++;
          }
        }
      }
    }
    *output = *tempOutput;
  }
}

// ******************** ENVIRONMENT CLASS FUNCTIONS START ********************
// Environment class constructor
environment::environment(ros::NodeHandle *nh){

  nh->getParam("/active_vision/simulationMode", simulationMode);

  // Checking if the required topics and services are running
  bool allOK = false;
  while(!allOK){
    boost::this_thread::sleep(boost::posix_time::seconds(1));
    allOK  = true;
    if(simulationMode == "SIMULATION"){
      allOK *= ROSCheck("NODE","/gazebo"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","/gazebo/set_model_state"); if(!allOK) continue;  // This a gazebo subscribed topic / gazebo service
      allOK *= ROSCheck("TOPIC","/camera/depth/points"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","/gazebo/get_model_state"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","/gazebo/spawn_sdf_model"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","/gazebo/delete_model"); if(!allOK) continue;
    }else if(simulationMode == "FRANKASIMULATION"){
      allOK *= ROSCheck("NODE","/gazebo"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","/gazebo/set_model_state"); if(!allOK) continue;  // This a gazebo subscribed topic / gazebo service
      allOK *= ROSCheck("TOPIC","/camera/depth/points"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","/gazebo/get_model_state"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","/gazebo/spawn_sdf_model"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","/gazebo/delete_model"); if(!allOK) continue;

      allOK *= ROSCheck("SERVICE","get_pose"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","move_to_pose"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","cartesian_move"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","inverse_kinematics_collision_check"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","set_velocity_scaling"); if(!allOK) continue;
      // allOK *= ROSCheck("SERVICE","move_to_joint_space"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","add_collision_object"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","gripPosServer"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","set_constraints"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","clear_constraints"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","move_to_named_state"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","one_joint_with_time"); if(!allOK) continue;
    }else if(simulationMode == "FRANKA"){
      allOK *= ROSCheck("TOPIC","/camera/depth/color/points"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","get_pose"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","move_to_pose"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","cartesian_move"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","inverse_kinematics_collision_check"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","add_collision_object"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","set_constraints"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","clear_constraints"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","move_to_named_state"); if(!allOK) continue;
      allOK *= ROSCheck("SERVICE","one_joint_with_time"); if(!allOK) continue;
    }
  }

  if(simulationMode == "SIMULATION"){
    pubObjPose = nh->advertise<gazebo_msgs::ModelState> ("/gazebo/set_model_state", 1);
    subKinectPtCld = nh->subscribe ("/camera/depth/points", 2, &environment::cbPtCld, this);
    gazeboSpawnModel = nh->serviceClient< gazebo_msgs::SpawnModel> ("/gazebo/spawn_sdf_model");
    gazeboCheckModel = nh->serviceClient< gazebo_msgs::GetModelState> ("/gazebo/get_model_state");
    gazeboDeleteModel = nh->serviceClient< gazebo_msgs::DeleteModel> ("/gazebo/delete_model");
  }else if(simulationMode == "FRANKASIMULATION"){
    pubObjPose = nh->advertise<gazebo_msgs::ModelState> ("/gazebo/set_model_state", 1);
    subKinectPtCld = nh->subscribe ("/camera/depth/points", 2, &environment::cbPtCld, this);
    gazeboSpawnModel = nh->serviceClient< gazebo_msgs::SpawnModel> ("/gazebo/spawn_sdf_model");
    gazeboCheckModel = nh->serviceClient< gazebo_msgs::GetModelState> ("/gazebo/get_model_state");
    gazeboDeleteModel = nh->serviceClient< gazebo_msgs::DeleteModel> ("/gazebo/delete_model");

    getPoseClient = nh->serviceClient<moveit_planner::GetPose>("get_pose");
    poseClient = nh->serviceClient<moveit_planner::MovePose>("move_to_pose");
    cartMoveClient = nh->serviceClient<moveit_planner::MoveCart>("cartesian_move");
    IKClient =  nh->serviceClient<moveit_planner::Inv>("inverse_kinematics_collision_check");
    velScalingClient = nh->serviceClient<moveit_planner::SetVelocity>("set_velocity_scaling");
    // jointSpaceClient = nh->serviceClient<moveit_planner::MoveJoint>("move_to_joint_space");
    collisionClient = nh->serviceClient<moveit_planner::AddCollision>("add_collision_object");
    gripperPosClient = nh->serviceClient<franka_pos_grasping_gazebo::GripPos>("gripPosServer");
    setConstClient = nh->serviceClient<moveit_planner::SetConstraints>("set_constraints");
    clearConstClient = nh->serviceClient<std_srvs::Empty>("clear_constraints");
    namedStateClient = nh->serviceClient<moveit_planner::MoveNamedState>("move_to_named_state");
    oneJointWithTimeClient = nh->serviceClient<moveit_planner::SetJointWithTime>("one_joint_with_time");

    listener.waitForTransform("panda_link0","camera_optical_link",ros::Time(0),ros::Duration(1.0));
    listener.waitForTransform("camera_optical_link","panda_link0",ros::Time(0),ros::Duration(1.0));
  }else if(simulationMode == "FRANKA"){
    subKinectPtCld = nh->subscribe ("/camera/depth/color/points", 1, &environment::cbPtCld, this);
    getPoseClient = nh->serviceClient<moveit_planner::GetPose>("get_pose");
    poseClient = nh->serviceClient<moveit_planner::MovePose>("move_to_pose");
    cartMoveClient = nh->serviceClient<moveit_planner::MoveCart>("cartesian_move");
    IKClient =  nh->serviceClient<moveit_planner::Inv>("inverse_kinematics_collision_check");
    velScalingClient = nh->serviceClient<moveit_planner::SetVelocity>("set_velocity_scaling");
    collisionClient = nh->serviceClient<moveit_planner::AddCollision>("add_collision_object");
    setConstClient = nh->serviceClient<moveit_planner::SetConstraints>("set_constraints");
    clearConstClient = nh->serviceClient<std_srvs::Empty>("clear_constraints");
    namedStateClient = nh->serviceClient<moveit_planner::MoveNamedState>("move_to_named_state");
    oneJointWithTimeClient = nh->serviceClient<moveit_planner::SetJointWithTime>("one_joint_with_time");

    listener.waitForTransform("panda_link0","camera_depth_optical_frame",ros::Time(0),ros::Duration(1.0));
    listener.waitForTransform("camera_depth_optical_frame","panda_link0",ros::Time(0),ros::Duration(1.0));
  }

  // NOT USED (JUST FOR REFERENCE)
  /*subKinectRGB = nh->subscribe ("/camera/color/image_raw", 1, &environment::cbImgRgb, this);
  subKinectDepth = nh->subscribe ("/camera/depth/image_raw", 1, &environment::cbImgDepth, this);*/

  readFlag[3] = {};           // Flag used to read data from kinect only when needed
  nh->getParam("/active_vision/environment/fingerZOffset", fingerZOffset); // Z axis offset between gripper hand and finger

  // Transform : Kinect Optical Frame to Kinect Gazebo frame
  tfKinOptGaz = pcl::getTransformation(0,0,0,-M_PI/2,0,-M_PI/2);

  // Camera projection matrix
  projectionMat.resize(3,4);
  if(simulationMode == "FRANKA"){
    projectionMat << 383.28009033203125, 0.0, 323.4447021484375, 0.0,
                     0.0, 383.28009033203125, 237.4062042236328, 0.0,
                     0.0, 0.0, 1.0, 0.0;
  }else{
    projectionMat << 554.254691191187, 0.0, 320.5, 0.0,
                     0.0, 554.254691191187, 240.5, 0.0,
                     0.0, 0.0, 1.0, 0.0;
  }


  path = ros::package::getPath("active_vision");  // Path to the active_vision package folder

  readObjectsList(*nh,"/active_vision/objectsInfo/",objectDict);

  nh->getParam("/active_vision/environment/voxelGridSizeUnexp", voxelGridSizeUnexp); // Voxel Grid size for unexplored point cloud
  nh->getParam("/active_vision/environment/voxelGridSize", voxelGridSize); // Voxel Grid size for environment

  nh->getParam("/active_vision/environment/viewsphereRad", viewsphereRad);
  nh->getParam("/active_vision/environment/tableCentre", tableCentre); // Co-ordinates of table centre
  if(simulationMode == "FRANKA") tableCentre[2] = 0.125;

  minUnexp = {0,0,0};
  maxUnexp = {0,0,0};
  nh->getParam("/active_vision/environment/scale", scale); // Scale value for unexplored point cloud generation

  nh->getParam("/active_vision/environment/maxGripperWidth", maxGripperWidth); // Gripper max width
  nh->getParam("/active_vision/environment/minGraspQuality", minGraspQuality); // Min grasp quality threshold
  selectedGrasp = -1; // Index of the selected grasp

  nh->getParam("/active_vision/environment/addNoise", addNoise);
  nh->getParam("/active_vision/environment/depthNoise", depthNoise);

  nh->getParam("/active_vision/environment/graspCurvatureConstraint", graspCurvatureConstraint);
  nh->getParam("/active_vision/environment/graspSurPatchConstraint", graspSurPatchConstraint);
  ros::Rate r(60);

  if(simulationMode != "SIMULATION"){
    moveit_planner::SetVelocity velscale;
    velscale.request.velScaling = 0.5;
    velScalingClient.call(velscale);
    moveFrankaHome();
    editMoveItCollisions("TABLE","ADD");
    maxGripperWidth += 0.01;
  }
}

// Function to reset the environment
void environment::reset(){
  ptrPtCldEnv->clear();
  ptrPtCldUnexp->clear();
  configurations.clear();
  if(simulationMode != "SIMULATION") moveFrankaHome();
}

// Store the configuration
int environment::saveConfiguration(std::string name){
  stateConfig configTemp;
  configTemp.env = *ptrPtCldEnv;
  configTemp.unexp = *ptrPtCldUnexp;
  configTemp.kinectPose = lastKinectPoseViewsphere;
  configTemp.description = name;
  configTemp.unexpMin = minUnexp;
  configTemp.unexpMax = maxUnexp;
  configurations.push_back(configTemp);
  //std::cout << "State saved : " << name << std::endl;
  return configurations.size()-1;
}

// Rollback to a configuration
void environment::rollbackConfiguration(int index){
  *ptrPtCldEnv = configurations[index].env;
  *ptrPtCldUnexp = configurations[index].unexp;
  lastKinectPoseViewsphere = configurations[index].kinectPose;
  minUnexp = configurations[index].unexpMin;
  maxUnexp = configurations[index].unexpMax;
  //std::cout << "Rolled back to state : " << configurations[index].description << std::endl;
}

// 1A: Callback function to point cloud subscriber
void environment::cbPtCld(const ptCldColor::ConstPtr& msg){
  if(readFlag[0]==1){
    *ptrPtCldLast = *msg;
    if(addNoise == true){
      // Looping through all the points
      for(int i = 0; i < ptrPtCldLast->points.size(); i++){
        if(!isnan(ptrPtCldLast->points[i].z)){
          // if z value is not nan then add noise
          float stdDev = (ptrPtCldLast->points[i].z)*depthNoise/100;
          std::normal_distribution<float> normDistb{0,stdDev};
          float noise = normDistb(generator);
          // Truncating to 1 sigma limit
          noise = std::min(noise,stdDev); noise = std::max(-stdDev,noise);
          ptrPtCldLast->points[i].z += noise;
        }
      }
    }
    readFlag[0] = 0;
    if(simulationMode == "FRANKA") holeFilling(cPtrPtCldLast,ptrPtCldLastFill);
    else *ptrPtCldLastFill = *cPtrPtCldLast;
  }
}

// Function to set noise variables
void environment::setPtCldNoise(float num){

  depthNoise = abs(num);
  if(depthNoise != 0) addNoise = true;
  else addNoise = false;
}

// NOT USED (JUST FOR REFERENCE)
/*// 1B: Callback function to RGB image subscriber
void cbImgRgb (const sensor_msgs::ImageConstPtr& msg){
  if (readFlag[1]==1) {
    ptrRgbLast = cv_bridge::toCvShare(msg);
    readFlag[1] = 0;
  }
}

// 1C: Callback function to RGB image subscriber
void cbImgDepth (const sensor_msgs::ImageConstPtr& msg){
  if (readFlag[2]==1) {
    ptrDepthLast = cv_bridge::toCvShare(msg);
    readFlag[2] = 0;
  }
}*/

// 2A: Spawning objects in gazebo on the table centre for a given pose option and yaw
void environment::spawnObject(int objectID, int choice, float yaw){

  if(simulationMode == "FRANKA") return;

  if(tableCentre[2] > 0) spawnBoxOnTable();

  gazebo_msgs::SpawnModel spawnObj;
  gazebo_msgs::GetModelState checkObj;
  geometry_msgs::Pose pose;

  pose.position.x = 0;
  pose.position.y = 0;
  pose.position.z = objectDict[objectID].poses[0][0];
  pose.orientation.x = 0;
  pose.orientation.y = 0;
  pose.orientation.z = 0;
  pose.orientation.w = 1;

  spawnObj.request.model_name = objectDict[objectID].description;

  if(objectDict[objectID].fileName.substr(0,3) == "YCB"){
    std::ifstream ifs(path+"/models/ycbAV/sdf/"+objectDict[objectID].fileName+".sdf");
    std::string sdfFile( (std::istreambuf_iterator<char>(ifs)),
                         (std::istreambuf_iterator<char>()));
    spawnObj.request.model_xml = sdfFile;
  }else{
    std::ifstream ifs(path+"/models/"+objectDict[objectID].fileName+"/model.sdf");
    std::string sdfFile( (std::istreambuf_iterator<char>(ifs)),
                         (std::istreambuf_iterator<char>()));
    spawnObj.request.model_xml = sdfFile;
  }

  spawnObj.request.reference_frame = "world";
  spawnObj.request.initial_pose = pose;

  checkObj.request.model_name = objectDict[objectID].description;

  gazeboSpawnModel.call(spawnObj);
  gazeboCheckModel.call(checkObj);
  while(not checkObj.response.success){
    boost::this_thread::sleep(boost::posix_time::milliseconds(100));
    gazeboCheckModel.call(checkObj);
  }
  moveObject(objectID,choice,yaw);
}
void environment::spawnBoxOnTable(){
  gazebo_msgs::SpawnModel spawnObj;
  gazebo_msgs::GetModelState checkObj;
  geometry_msgs::Pose pose;

  pose.position.x = tableCentre[0];
  pose.position.y = tableCentre[1];
  pose.position.z = tableCentre[2];
  pose.orientation.x = 0;
  pose.orientation.y = 0;
  pose.orientation.z = 0;
  pose.orientation.w = 1;

  spawnObj.request.model_name = "boxOnTable";

  std::ifstream ifs(path+"/models/boxOnTable/model.sdf");
  std::string sdfFile( (std::istreambuf_iterator<char>(ifs)),
                       (std::istreambuf_iterator<char>()));
  spawnObj.request.model_xml = sdfFile;

  spawnObj.request.reference_frame = "world";
  spawnObj.request.initial_pose = pose;

  checkObj.request.model_name = "boxOnTable";

  gazeboSpawnModel.call(spawnObj);
  gazeboCheckModel.call(checkObj);
  while(not checkObj.response.success){
    boost::this_thread::sleep(boost::posix_time::milliseconds(100));
    gazeboCheckModel.call(checkObj);
  }
}

// 2B: Function to move the object. Same args as spawnObject
void environment::moveObject(int objectID, int choice, float yaw){

  if(simulationMode == "FRANKA") return;

  gazebo_msgs::GetModelState checkObj;

  if(choice >= objectDict[objectID].nPoses){
    choice = 0;
    printf("WARNING moveObject: Pose choice invalid. Setting choice to 0.\n");
  }

  //Create Matrix3x3 from Euler Angles
  tf::Matrix3x3 m_rot;
  m_rot.setEulerYPR(yaw, objectDict[objectID].poses[choice][2], objectDict[objectID].poses[choice][1]);

  // Convert into quaternion
  tf::Quaternion quat;
  m_rot.getRotation(quat);

  // Converting it to the required gazebo format
  gazebo_msgs::ModelState ModelState;
  ModelState.model_name = objectDict[objectID].description;
  ModelState.reference_frame = "world";
  ModelState.pose.position.x = tableCentre[0];
  ModelState.pose.position.y = tableCentre[1];
  ModelState.pose.position.z = tableCentre[2]+objectDict[objectID].poses[choice][0];
  ModelState.pose.orientation.x = quat.x();
  ModelState.pose.orientation.y = quat.y();
  ModelState.pose.orientation.z = quat.z();
  ModelState.pose.orientation.w = quat.w();

  // Publishing it to gazebo
  pubObjPose.publish(ModelState);
  checkObj.request.model_name = objectDict[objectID].description;
  gazeboCheckModel.call(checkObj);
  ros::spinOnce();
  while((not checkObj.response.success) or (abs(checkObj.response.pose.position.z - (tableCentre[2]+objectDict[objectID].poses[choice][0]) > 0.2))){
    printf("Object not positioned yet, pose z = %1.4f, not %1.4f, diff of %1.4f\n", checkObj.response.pose.position.z, tableCentre[2]+objectDict[objectID].poses[choice][0], abs(checkObj.response.pose.position.z - (tableCentre[2]+objectDict[objectID].poses[choice][0])));
    boost::this_thread::sleep(boost::posix_time::milliseconds(100));
    gazeboCheckModel.call(checkObj);
    ros::spinOnce();
  }
}

// 3: Deleting objects in gazebo
void environment::deleteObject(int objectID){

  if(simulationMode == "FRANKA") return;

  gazebo_msgs::DeleteModel deleteObj;
  deleteObj.request.model_name = objectDict[objectID].description;

  gazeboDeleteModel.call(deleteObj);

  if(tableCentre[2] > 0){
    deleteObj.request.model_name = "boxOnTable";
    gazeboDeleteModel.call(deleteObj);
  }
}

// 4: Load Gripper Hand and Finger file
void environment::loadGripper(){
  std::string pathToHand = path+"/models/gripperAV/hand1.ply";
  std::string pathToFinger = path+"/models/gripperAV/finger1.ply";
  std::string pathToCamera = path+"/models/gripperAV/realsense.ply";
  // Gripper Hand
  if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(pathToHand, *ptrPtCldGrpHnd) == -1){
    PCL_ERROR ("Couldn't read file hand.ply \n");
  }
  // Gripper Camera
  if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(pathToCamera, *ptrPtCldGrpCam) == -1){
    PCL_ERROR ("Couldn't read file realsense.ply \n");
  }
  // Gripper Left Finger
  if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(pathToFinger, *ptrPtCldGrpLfgr) == -1){
    PCL_ERROR ("Couldn't read file finger.ply \n");
  }
  // Gripper Right Finger (Mirror of the left one)
  pcl::transformPointCloud(*ptrPtCldGrpLfgr, *ptrPtCldGrpRfgr, pcl::getTransformation(0,0,0,0,0,M_PI));

  // Find the min max 3D coordinates for the three segments
  pcl::getMinMax3D(*ptrPtCldGrpHnd,minPtGrp[0],maxPtGrp[0]);
  pcl::getMinMax3D(*ptrPtCldGrpLfgr,minPtGrp[1],maxPtGrp[1]);
  pcl::getMinMax3D(*ptrPtCldGrpRfgr,minPtGrp[2],maxPtGrp[2]);
  pcl::getMinMax3D(*ptrPtCldGrpCam,minPtGrp[3],maxPtGrp[3]);
  std::cout << "Ignore the PLY reader error on 'face' and 'rgb'." << std::endl;
}

// 5: Update gripper
// 0 -> Visualization
// 1 -> Axis Collision Check
// 2 -> Gripper Collision Check
void environment::updateGripper(int index ,int choice){

  // Gripper orientation which is used in cropbox during collision check
  tfGripper = pcl::getTransformation(graspsPossible[index].pose[0],graspsPossible[index].pose[1],
                                     graspsPossible[index].pose[2],graspsPossible[index].pose[3],
                                     graspsPossible[index].pose[4],graspsPossible[index].pose[5])*
              pcl::getTransformation(0,0,0,0,graspsPossible[index].addnlPitch,0)*
              pcl::getTransformation(0,0,-0.0447-fingerZOffset,0,0,0);

  if (choice == 0) {
    // Adding the gripper hand
    *ptrPtCldGripper=*ptrPtCldGrpHnd;

    // Translating the left finger and adding
    pcl::transformPointCloud(*ptrPtCldGrpLfgr, *ptrPtCldTemp,
                            pcl::getTransformation(0,graspsPossible[index].gripperWidth/2,fingerZOffset,0,0,0));
    *ptrPtCldGripper += *ptrPtCldTemp;

    // Translating the right finger and adding
    pcl::transformPointCloud(*ptrPtCldGrpRfgr, *ptrPtCldTemp,
                            pcl::getTransformation(0,-graspsPossible[index].gripperWidth/2,fingerZOffset,0,0,0));
    *ptrPtCldGripper += *ptrPtCldTemp;

    // adding the camera
    *ptrPtCldGripper += *ptrPtCldGrpCam;

    pcl::transformPointCloud(*ptrPtCldGripper, *ptrPtCldGripper, tfGripper);
    ptrPtCldTemp->clear();

  } else if (choice == 1) {
    // Left Finger Basic Check
    minPtCol[3].x = -0.0125; maxPtCol[3].x = 0.0125;
    minPtCol[3].y =  0.0;    maxPtCol[3].y = 0.0250;
    minPtCol[3].z =  0.0322; maxPtCol[3].z = 0.0572;
    // Applying transformation for the gripper width
    minPtCol[3] = pcl::transformPoint(minPtCol[3],pcl::getTransformation(0,graspsPossible[index].gripperWidth/2,fingerZOffset,0,0,0));
    maxPtCol[3] = pcl::transformPoint(maxPtCol[3],pcl::getTransformation(0,graspsPossible[index].gripperWidth/2,fingerZOffset,0,0,0));

    // Right Finger Basic Check
    minPtCol[4].x = -0.0125; maxPtCol[4].x = 0.0125;
    minPtCol[4].y = -0.0250; maxPtCol[4].y = 0.0;
    minPtCol[4].z =  0.0322; maxPtCol[4].z = 0.0572;
    // Applying transformation for the gripper width
    minPtCol[4] = pcl::transformPoint(minPtCol[4],pcl::getTransformation(0,-graspsPossible[index].gripperWidth/2,fingerZOffset,0,0,0));
    maxPtCol[4] = pcl::transformPoint(maxPtCol[4],pcl::getTransformation(0,-graspsPossible[index].gripperWidth/2,fingerZOffset,0,0,0));

  } else if (choice == 2) {
    // Hand
    minPtCol[0] = minPtGrp[0];
    maxPtCol[0] = maxPtGrp[0];

    // Camera
    minPtCol[5] = minPtGrp[3];
    maxPtCol[5] = maxPtGrp[3]; maxPtCol[5].x += 0.005; maxPtCol[5].z += 0.005;

    // Left Finger
    minPtCol[1] = pcl::transformPoint(minPtGrp[1],pcl::getTransformation(0,graspsPossible[index].gripperWidth/2,fingerZOffset,0,0,0));
    maxPtCol[1] = pcl::transformPoint(maxPtGrp[1],pcl::getTransformation(0,graspsPossible[index].gripperWidth/2,fingerZOffset,0,0,0));

    // Right Finger
    minPtCol[2] = pcl::transformPoint(minPtGrp[2],pcl::getTransformation(0,-graspsPossible[index].gripperWidth/2,fingerZOffset,0,0,0));
    maxPtCol[2] = pcl::transformPoint(maxPtGrp[2],pcl::getTransformation(0,-graspsPossible[index].gripperWidth/2,fingerZOffset,0,0,0));
  }
}

// 6A: Function to move the kinect. Args: Array of X,Y,Z,Roll,Pitch,Yaw
bool environment::moveKinectCartesian(std::vector<double> pose, bool execute){
  if(simulationMode == "SIMULATION"){
    //Create Matrix3x3 from Euler Angles
    tf::Matrix3x3 rotMat;
    rotMat.setEulerYPR(pose[5], pose[4], pose[3]);

    // Convert into quaternion
    tf::Quaternion quat;
    rotMat.getRotation(quat);

    // Converting it to the required gazebo format
    gazebo_msgs::ModelState ModelState;
    ModelState.model_name = "Kinect";           // This should be the name of kinect in gazebo
    ModelState.reference_frame = "world";
    ModelState.pose.position.x = pose[0];
    ModelState.pose.position.y = pose[1];
    ModelState.pose.position.z = pose[2];
    ModelState.pose.orientation.x = quat.x();
    ModelState.pose.orientation.y = quat.y();
    ModelState.pose.orientation.z = quat.z();
    ModelState.pose.orientation.w = quat.w();

    // Publishing it to gazebo
    pubObjPose.publish(ModelState);
    ros::spinOnce();
    boost::this_thread::sleep(boost::posix_time::milliseconds(250));

    // Storing the kinect pose
    lastKinectPoseCartesian = pose;
  }
  else if(simulationMode == "FRANKASIMULATION" || simulationMode == "FRANKA"){
    // Create Matrix3x3 from Euler Angles
    // Additional rotation of PI about Z and -PI/2 about Y so that kinect frame orientation aligns with the gripper
    Eigen::Matrix3f rotMat;
    rotMat = Eigen::AngleAxisf(pose[5], Eigen::Vector3f::UnitZ()) * Eigen:: AngleAxisf(pose[4], Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(pose[3], Eigen::Vector3f::UnitX());

    // Incorporating the kinect translation offset
    Eigen::Matrix4f tfMat; tfMat.setIdentity();
    tfMat.block<3,3>(0,0) = rotMat;
    tfMat(0,3) = pose[0];
    tfMat(1,3) = pose[1];
    tfMat(2,3) = pose[2];

    geometry_msgs::Pose p;
    bool res = moveFranka(tfMat,"JOINT",true,execute,p);
    if(!res) return false;

    if(execute){
      // Storing the kinect pose
      if(simulationMode != "FRANKA")
        boost::this_thread::sleep(boost::posix_time::milliseconds(250));
      else
        boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
      lastKinectPoseCartesian = pose;
    }
  }
  return true;
}

// 6B: Funtion to move the Kinect in a viewsphere which has the table cente as its centre
// R (Radius)
// Phi (Azhimuthal angle) -> 0 to 2*PI
// Theta (Polar Angle)) -> 0 to PI/2
bool environment::moveKinectViewsphere(std::vector<double> pose, bool execute){

  if(simulationMode == "SIMULATION"){
    //Create Matrix3x3 from Euler Angles
    tf::Matrix3x3 rotMat;
    //rotMat.setEulerYPR(M_PI+pose[1], M_PI/2-pose[2], 0);
    rotMat.setEulerYPR(M_PI+pose[1], M_PI/2-pose[2], 0);

    // Convert into quaternion
    tf::Quaternion quat;
    rotMat.getRotation(quat);

    // Converting it to the required gazebo format
    gazebo_msgs::ModelState ModelState;
    ModelState.model_name = "Kinect";           // This should be the name of kinect in gazebo
    ModelState.reference_frame = "world";
    ModelState.pose.position.x = tableCentre[0]+pose[0]*sin(pose[2])*cos(pose[1]);
    ModelState.pose.position.y = tableCentre[1]+pose[0]*sin(pose[2])*sin(pose[1]);
    ModelState.pose.position.z = tableCentre[2]+pose[0]*cos(pose[2]);
    ModelState.pose.orientation.x = quat.x();
    ModelState.pose.orientation.y = quat.y();
    ModelState.pose.orientation.z = quat.z();
    ModelState.pose.orientation.w = quat.w();

    // Publishing it to gazebo
    pubObjPose.publish(ModelState);
    ros::spinOnce();
    boost::this_thread::sleep(boost::posix_time::milliseconds(250));

    // Storing the kinect pose
    lastKinectPoseViewsphere = pose;
    lastKinectPoseCartesian = {ModelState.pose.position.x,
                               ModelState.pose.position.y,
                               ModelState.pose.position.z,
                               0,M_PI/2-pose[2],M_PI+pose[1]};
  }
  else if(simulationMode == "FRANKASIMULATION" || simulationMode == "FRANKA"){

    // Create Matrix3x3 from Euler Angles
    // Additional rotation of PI about Z and -PI/2 about Y so that kinect frame orientation aligns with the gripper
    Eigen::Matrix3f rotMat;
    rotMat = Eigen::AngleAxisf(M_PI+pose[1], Eigen::Vector3f::UnitZ()) * Eigen:: AngleAxisf(M_PI/2-pose[2], Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX());

    // Incorporating the kinect translation offset
    Eigen::Matrix4f tfMat; tfMat.setIdentity();
    tfMat.block<3,3>(0,0) = rotMat;
    tfMat(0,3) = tableCentre[0]+pose[0]*sin(pose[2])*cos(pose[1]);
    tfMat(1,3) = tableCentre[1]+pose[0]*sin(pose[2])*sin(pose[1]);
    tfMat(2,3) = tableCentre[2]+pose[0]*cos(pose[2]);

    geometry_msgs::Pose p;
    bool res = moveFranka(tfMat,"JOINT",true,execute,p);
    if(!res) return false;

    if(execute){
      if(simulationMode != "FRANKA")
        boost::this_thread::sleep(boost::posix_time::milliseconds(250));
      else
        boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
      // Storing the kinect pose
      lastKinectPoseViewsphere = pose;
      lastKinectPoseCartesian = {tfMat(0,3),
                                 tfMat(1,3),
                                 tfMat(2,3),
                                 0,M_PI/2-pose[2],M_PI+pose[1]};
    }

  }
  return true;

}

bool environment::moveFranka(Eigen::Matrix4f tfMat, std::string mode ,bool isKinect ,bool execute, geometry_msgs::Pose &p){

  tfMat *= pcl::getTransformation(0,0,0,0,-M_PI/2,M_PI).matrix();
  if(isKinect){
    Eigen::Matrix4f kinectOffset; kinectOffset.setIdentity();
    kinectOffset(0,3) = -0.037796115635;
    kinectOffset(1,3) = +0.0298131982299;
    kinectOffset(2,3) = -0.059671236405;
    if(tfMat(0,0) < 0) tfMat *= pcl::getTransformation(0,0,0,0,0,M_PI).matrix();
    tfMat *= kinectOffset;
  }

  if(simulationMode == "FRANKA"){
    tfMat *= pcl::getTransformation(0,0,0,0,0,M_PI/4).matrix();
  }

  Eigen::Quaternionf quat(tfMat.block<3,3>(0,0));

  p.position.x = tfMat(0,3);  p.position.y = tfMat(1,3);  p.position.z = tfMat(2,3);
  p.orientation.x = quat.x(); p.orientation.y = quat.y(); p.orientation.z = quat.z(); p.orientation.w = quat.w();

  // if(execute){
  //   std::cout << p.position.x << "," <<
  //                p.position.y << "," <<
  //                p.position.z << "," <<
  //                p.orientation.x << "," <<
  //                p.orientation.y << "," <<
  //                p.orientation.z << "," <<
  //                p.orientation.w << std::endl;
  // }

  if(!checkFrankReach(IKClient,p)) return false;

  // Moveit move command.
  if(execute){
    bool res = true;
    if(isKinect){
      editMoveItCollisions("OBJECT","ADD");
      addVisibilityConstraint();
    }
    if(mode == "JOINT"){
      moveit_planner::MovePose movePoseMsg;
      movePoseMsg.request.val = p;
      movePoseMsg.request.execute = true;
      res = poseClient.call(movePoseMsg);
    }else if(mode == "CARTESIAN"){
      moveit_planner::MoveCart moveCartMsg;
      moveCartMsg.request.val.push_back(p);
      moveCartMsg.request.time = 0;
      moveCartMsg.request.execute = true;
      res = cartMoveClient.call(moveCartMsg);
    }

    if(isKinect){
      moveGripper(0.08);
      editMoveItCollisions("OBJECT","REMOVE");
      clearAllConstraints();
    }

    ros::Duration(1).sleep();

    float frankaOffset[3] = {0.0,0.0,0.0};
    float frankaX = tfMat(0,3)-frankaOffset[0];
    float frankaY = tfMat(1,3)-frankaOffset[1];
    float frankaZ = tfMat(2,3)-frankaOffset[2];

    // std::cout << frankaX << "," <<
    //              frankaY << "," <<
    //              frankaZ << "," <<
    //              quat.x() << "," <<
    //              quat.y() << "," <<
    //              quat.z() << "," <<
    //              quat.w() << std::endl;

    if(simulationMode == "FRANKA") return res;

    moveit_planner::GetPose curPose;
    getPoseClient.call(curPose);

    // std::cout << curPose.response.pose.position.x << "," <<
    //              curPose.response.pose.position.y << "," <<
    //              curPose.response.pose.position.z << "," <<
    //              curPose.response.pose.orientation.x << "," <<
    //              curPose.response.pose.orientation.y << "," <<
    //              curPose.response.pose.orientation.z << "," <<
    //              curPose.response.pose.orientation.w << std::endl;

    float positionError = sqrt(pow(curPose.response.pose.position.x-frankaX,2) +
                               pow(curPose.response.pose.position.y-frankaY,2) +
                               pow(curPose.response.pose.position.z-frankaZ,2));
    float orientationError = curPose.response.pose.orientation.x * p.orientation.x +
                             curPose.response.pose.orientation.y * p.orientation.y +
                             curPose.response.pose.orientation.z * p.orientation.z +
                             curPose.response.pose.orientation.w * p.orientation.w;

    if(positionError > 5e-03 || 1-abs(orientationError) > 1e-03) return false;
  }

  return true;
}

void environment::moveGripper(double Grasp_Width, bool grasp){
  if(simulationMode == "FRANKA"){
    if(grasp){
      actionlib::SimpleActionClient<franka_gripper::GraspAction> ac("franka_gripper/grasp", true);
      ac.waitForServer();

      franka_gripper::GraspGoal goal;
      goal.width = Grasp_Width;   // Distance between fingers [m]
      goal.speed = 0.1;           // Closing speed. [m/s]
      goal.force = 40;            // Grasping (continuous) force [N]
      goal.epsilon.inner = 0.05;  // Maximum tolerated deviation when the actual grasped width is
                                  // smaller than the commanded grasp width.
      goal.epsilon.outer = 0.05;  // Maximum tolerated deviation when the actual grasped width is
                                  // larger than the commanded grasp width.
      ac.sendGoal(goal);          // Sending the Grasp command to gripper

      bool finished_before_timeout = ac.waitForResult(ros::Duration(5));

      // if (finished_before_timeout){
      // ROS_INFO("Gripper action finished.");
      // }
      // else {
      // ROS_INFO("Gripper action did not finish before the time out.");
      // }
    }else{
      actionlib::SimpleActionClient<franka_gripper::MoveAction> ac("franka_gripper/move", true);
      ac.waitForServer();

      franka_gripper::MoveGoal goal;
      goal.width = Grasp_Width;   // Distance between fingers [m]
      goal.speed = 0.1;           // Closing speed. [m/s]
      ac.sendGoal(goal);
      bool finished_before_timeout = ac.waitForResult(ros::Duration(5));
    }
  }else if(simulationMode == "FRANKASIMULATION"){
    franka_pos_grasping_gazebo::GripPos grasp;
    grasp.request.finger_pos = Grasp_Width/2.0;
    gripperPosClient.call(grasp);
    ros::Duration(1).sleep();
  }

}

void environment::moveFrankaHome(bool gripper){
  std::cout << "Moving to home pose..." << std::endl;
  static moveit_planner::MoveNamedState namedState; namedState.request.name = "ready";
  namedStateClient.call(namedState);
  if(gripper) moveGripper(0.08);
}

void environment::addVisibilityConstraint(){
  moveit_msgs::VisibilityConstraint visConstraint;
  visConstraint.target_radius = 0.125;

  geometry_msgs::Pose pose;
  pose.orientation.x = 0;
  pose.orientation.y = 0;
  pose.orientation.z = 0;
  pose.orientation.w = 1;

  geometry_msgs::PoseStamped tgtPose;
  tgtPose.header.frame_id = "panda_link0";
  pose.position.x = tableCentre[0];
  pose.position.y = tableCentre[1];
  pose.position.z = tableCentre[2];
  tgtPose.pose = pose;
  visConstraint.target_pose = tgtPose;

  visConstraint.cone_sides = 4;

  geometry_msgs::PoseStamped sensorPose;
  sensorPose.header.frame_id = "panda_hand";
  pose.position.x =  0.037796115635;
  pose.position.y = -0.0298131982299;
  pose.position.z =  0.059671236405;
  sensorPose.pose = pose;
  visConstraint.sensor_pose = sensorPose;

  visConstraint.max_view_angle = 85*M_PI/180;

  visConstraint.sensor_view_direction = visConstraint.SENSOR_Z;

  visConstraint.weight = 1.0;

  moveit_planner::SetConstraints constraintsMsg;
  constraintsMsg.request.constraints.visibility_constraints = {visConstraint};

  setConstClient.call(constraintsMsg);
}

void environment::addOrientationConstraint(Eigen::Affine3f tf){
  tf = tf*pcl::getTransformation(0,0,0,0,-M_PI/2,M_PI);
  Eigen::Quaternionf quat(tf.matrix().block<3,3>(0,0));

  moveit_planner::SetConstraints constraintsMsg;
  moveit_msgs::OrientationConstraint orientConstraint;
  orientConstraint.header.frame_id = "/world";
  orientConstraint.link_name = "panda_hand";
  orientConstraint.orientation.x = quat.x();
  orientConstraint.orientation.y = quat.y();
  orientConstraint.orientation.z = quat.z();
  orientConstraint.orientation.w = quat.w();
  orientConstraint.absolute_x_axis_tolerance = 0.01;
  orientConstraint.absolute_y_axis_tolerance = 0.01;
  orientConstraint.absolute_z_axis_tolerance = 0.01;
  orientConstraint.weight = 1.0;

  constraintsMsg.request.constraints.orientation_constraints = {orientConstraint};
  setConstClient.call(constraintsMsg);
}

void environment::clearAllConstraints(){
  std_srvs::Empty emptyMsg;
  clearConstClient.call(emptyMsg);
}

// 7: Function to read the kinect data.
void environment::readKinect(){
  readFlag[0] = 1; // readFlag[1] = 1; readFlag[2] = 1;
  while (readFlag[0]==1) {
    ros::spinOnce();
    r.sleep();
  }
}

// 8: Function to Fuse last data with existing data
void environment::fuseLastData(){
  ptrPtCldTemp->clear();

  ICP_Tf = Eigen::Matrix4f::Identity();

  if(simulationMode == "SIMULATION"){
    // Transform : Kinect Gazebo Frame to Gazebo World frame
    tfGazWorld = pcl::getTransformation(lastKinectPoseCartesian[0],lastKinectPoseCartesian[1],lastKinectPoseCartesian[2],\
                                        lastKinectPoseCartesian[3],lastKinectPoseCartesian[4],lastKinectPoseCartesian[5]);
    // Apply transformation
    Eigen::Affine3f tf = tfGazWorld * tfKinOptGaz;
    pcl::transformPointCloud(*ptrPtCldLast, *ptrPtCldTemp, tf);
  }else if(simulationMode == "FRANKASIMULATION"){
    tf::StampedTransform transform;
    listener.lookupTransform("panda_link0", "camera_optical_link", ros::Time(0), transform);
    pcl_ros::transformPointCloud(*ptrPtCldLast, *ptrPtCldTemp, transform);
  }else{
    tf::StampedTransform transform;
    listener.lookupTransform("panda_link0", "camera_depth_optical_frame", ros::Time(0), transform);
    pcl_ros::transformPointCloud(*ptrPtCldLast, *ptrPtCldTemp, transform);
    ICPRegistration();
  }

  // Fuse the two pointclouds (except for the first time) and downsample again
  if(ptrPtCldEnv->width == 0) *ptrPtCldEnv = *ptrPtCldTemp;
  else                        *ptrPtCldEnv += *ptrPtCldTemp;

  // Downsample using voxel grid
  voxelGrid.setInputCloud(cPtrPtCldEnv);
  voxelGrid.setLeafSize(voxelGridSize, voxelGridSize, voxelGridSize);
  voxelGrid.filter(*ptrPtCldEnv);

  // Using pass through filter to focus on the required region
  pass.setInputCloud(cPtrPtCldEnv);
  pass.setFilterFieldName("x"); pass.setFilterLimits( 0.20,0.70); pass.filter(*ptrPtCldEnv);
  pass.setFilterFieldName("y"); pass.setFilterLimits(-0.25,0.25); pass.filter(*ptrPtCldEnv);
  pass.setFilterFieldName("z"); pass.setFilterLimits(-0.02,0.60); pass.filter(*ptrPtCldEnv);
  ptrPtCldTemp->clear();
}

void environment::ICPRegistration(){

  if(ptrPtCldEnv->width == 0) return;

  ptCldColor::Ptr ptrPtCldSrc{new ptCldColor}; ptCldColor::ConstPtr cPtrPtCldSrc{ptrPtCldSrc};
  *ptrPtCldSrc = *ptrPtCldTemp;
  voxelGrid.setInputCloud(cPtrPtCldSrc);
  voxelGrid.setLeafSize(voxelGridSize, voxelGridSize, voxelGridSize);
  voxelGrid.filter(*ptrPtCldSrc);

  ptCldColor::Ptr ptrPtCldTgt{new ptCldColor}; ptCldColor::ConstPtr cPtrPtCldTgt{ptrPtCldTgt};
  *ptrPtCldTgt = *ptrPtCldEnv;

  ptCldColor::Ptr ptrPtCldSrcTemp{new ptCldColor}; ptCldColor::ConstPtr cPtrPtCldSrcTemp{ptrPtCldSrcTemp};
  ptCldColor::Ptr ptrPtCldTgtTemp{new ptCldColor}; ptCldColor::ConstPtr cPtrPtCldTgtTemp{ptrPtCldTgtTemp};

  // Aligning with table
  *ptrPtCldSrcTemp = *ptrPtCldSrc;
  *ptrPtCldTgtTemp = *ptrPtCldTgt;

  // Extracting table
  pass.setInputCloud(cPtrPtCldSrcTemp);
  pass.setFilterFieldName("z"); pass.setFilterLimits(tableCentre[2]-0.01,tableCentre[2]+0.01);  pass.filter(*ptrPtCldSrcTemp);

  pass.setInputCloud(cPtrPtCldTgtTemp);
  pass.setFilterFieldName("z"); pass.setFilterLimits(tableCentre[2]-0.01,tableCentre[2]+0.01);  pass.filter(*ptrPtCldTgtTemp);

  // ICP to align the table
  pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp1;
  icp1.setInputSource(ptrPtCldSrcTemp);
  icp1.setInputTarget(ptrPtCldTgtTemp);
  icp1.setMaxCorrespondenceDistance(0.02);

  Eigen::Matrix4f Ti1 = Eigen::Matrix4f::Identity();
  icp1.setMaximumIterations(10);
  icp1.align(*ptrPtCldSrcTemp);
  Ti1 = icp1.getFinalTransformation();
  pcl::transformPointCloud(*ptrPtCldSrc, *ptrPtCldSrc, Ti1);
  pcl::transformPointCloud(*ptrPtCldTemp, *ptrPtCldTemp, Ti1);

  // Align the object
  *ptrPtCldSrcTemp = *ptrPtCldSrc;
  *ptrPtCldTgtTemp = *ptrPtCldTgt;

  pass.setInputCloud(cPtrPtCldSrcTemp);
  pass.setFilterFieldName("x"); pass.setFilterLimits(minPtObj.x-0.04,maxPtObj.x+0.04);  pass.filter(*ptrPtCldSrcTemp);
  pass.setFilterFieldName("y"); pass.setFilterLimits(minPtObj.y-0.04,maxPtObj.y+0.04);  pass.filter(*ptrPtCldSrcTemp);
  pass.setFilterFieldName("z"); pass.setFilterLimits(tableCentre[2]+0.01,1);   pass.filter(*ptrPtCldSrcTemp);

  pass.setInputCloud(cPtrPtCldTgtTemp);
  pass.setFilterFieldName("x"); pass.setFilterLimits(minPtObj.x-0.05,maxPtObj.x+0.05);  pass.filter(*ptrPtCldTgtTemp);
  pass.setFilterFieldName("y"); pass.setFilterLimits(minPtObj.y-0.05,maxPtObj.y+0.05);  pass.filter(*ptrPtCldTgtTemp);
  pass.setFilterFieldName("z"); pass.setFilterLimits(tableCentre[2]+0.01,1);   pass.filter(*ptrPtCldTgtTemp);

  pcl::registration::WarpPointXY<pcl::PointXYZRGB, pcl::PointXYZRGB>::Ptr warp_fcn
    (new pcl::registration::WarpPointXY<pcl::PointXYZRGB, pcl::PointXYZRGB>);

  // Create a TransformationEstimationLM object, and set the warp to it
  pcl::registration::TransformationEstimationLM<pcl::PointXYZRGB, pcl::PointXYZRGB>::Ptr te
    (new pcl::registration::TransformationEstimationLM<pcl::PointXYZRGB, pcl::PointXYZRGB>);

  te->setWarpFunction(warp_fcn);

  // ICP to align the object
  pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp2;
  icp2.setTransformationEstimation(te);
  icp2.setInputSource(ptrPtCldSrcTemp);
  icp2.setInputTarget(ptrPtCldTgtTemp);
  icp2.setMaxCorrespondenceDistance(0.015);

  Eigen::Matrix4f Ti2 = Eigen::Matrix4f::Identity(), prev;
  icp2.setMaximumIterations(10);
  icp2.align(*ptrPtCldSrcTemp);
  Ti2 = icp2.getFinalTransformation();

  pcl::transformPointCloud(*ptrPtCldTemp, *ptrPtCldTemp, Ti2);

  ICP_Tf = Ti1 * Ti2;
}

// 9: Extracting the major plane (Table) and object
void environment::dataExtract(){
  dataExtractPlaneSeg();
  if(simulationMode != "FRANKA") dataColorCorrection();

  // Generating the normals for the object point cloud. Used from grasp synthesis
  pcl::search::Search<pcl::PointXYZRGB>::Ptr KdTree{new pcl::search::KdTree<pcl::PointXYZRGB>};
  ne.setInputCloud(cPtrPtCldObject);
  ne.setSearchMethod(KdTree);
  ne.setKSearch(10);
  ne.compute(*ptrObjNormal);

  float avgCurvature;
  if(graspCurvatureConstraint){
    for(int i = 0; i < cPtrPtCldObject->points.size(); i++) avgCurvature += ptrObjNormal->points[i].curvature;
    avgCurvature /= cPtrPtCldObject->points.size();
  }

  useForGrasp.resize(cPtrPtCldObject->points.size());
  for(int i = 0; i < cPtrPtCldObject->points.size(); i++){
    useForGrasp[i] = true;

    // Curvature check
    if(graspCurvatureConstraint && ptrObjNormal->points[i].curvature > 2*avgCurvature){
      useForGrasp[i] = false;
    }

    // Surface Patch constraint check
    if(graspSurPatchConstraint && !(isContactPatchOk(ptrPtCldObject,ptrObjNormal,i,voxelGridSize))){
      useForGrasp[i] = false;
    }
  }

}

void environment::dataExtractPlaneSeg(){
  // Find the major plane and get its coefficients and indices
  seg.setInputCloud(cPtrPtCldEnv);
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(1000);
  seg.setDistanceThreshold(0.005+viewsphereRad*depthNoise/100);
  Eigen::Vector3f axis = Eigen::Vector3f(0.0,0.0,1.0); //y axis
  seg.setAxis(axis);
  seg.setEpsAngle(10.0f*(M_PI/180.0f) );
  seg.segment(*tableIndices,*tableCoeff);

  if (tableIndices->indices.size () == 0){
    std::cerr << "No table found in the environment" << std::endl;
    return;
  }

  // Seperating the table and storing its point
  extract.setInputCloud(cPtrPtCldEnv);
  extract.setIndices(tableIndices);
  extract.setNegative(false); extract.filter(*ptrPtCldTable);
  extract.setNegative(true);  extract.filter(*ptrPtCldObject);

  pcl::compute3DCentroid(*ptrPtCldTable, cenTable);
  pcl::getMinMax3D(*ptrPtCldTable, minTable, maxTable);
  //std::cout << tableCoeff->values[0] << " " << tableCoeff->values[1] << " " << tableCoeff->values[2] << " " << tableCoeff->values[3] << std::endl;
  //std::cout << minTable.z << " " << cenTable[2] << " " << maxTable.z << std::endl;

  // Using convex hull to get the table boundary which would be like a rectangle
  cvHull.setInputCloud(cPtrPtCldTable);
  cvHull.setDimension(2);
  cvHull.reconstruct(*ptrPtCldHull);

  // Double checking the hull dimensions
  if(cvHull.getDimension() != 2){
    std::cerr << "Convex hull dimension != 2" << std::endl;
    return;
  }

  // Using polygonal prism and hull the extract object above the table
  prism.setInputCloud(cPtrPtCldObject);
  prism.setInputPlanarHull(cPtrPtCldHull);
  prism.setViewPoint(tableCentre[0],tableCentre[1],tableCentre[2]+1);        // Ensuring normals point above the table
  prism.setHeightLimits(0.01,1.5f);                                 // Z height (min, max) in m
  prism.segment(*objectIndices);

  // Using extract to get the point cloud
  extract.setInputCloud(cPtrPtCldObject);
  extract.setNegative(false);
  extract.setIndices(objectIndices);
  extract.filter(*ptrPtCldObject);

  pass.setInputCloud(cPtrPtCldObject);
  pass.setFilterFieldName("x"); pass.setFilterLimits(tableCentre[0]-0.15,tableCentre[0]+0.15); pass.filter(*ptrPtCldObject);
  pass.setFilterFieldName("y"); pass.setFilterLimits(tableCentre[1]-0.15,tableCentre[1]+0.15); pass.filter(*ptrPtCldObject);
  pass.setFilterFieldName("z"); pass.setFilterLimits(tableCentre[2]+0.01,tableCentre[2]+0.30); pass.filter(*ptrPtCldObject);

  // Getting the min and max co-ordinates of the object
  pcl::compute3DCentroid(*ptrPtCldObject, cenObject);
  pcl::getMinMax3D(*ptrPtCldObject, minPtObj, maxPtObj);
}

void environment::dataColorCorrection(){
  // Green Color for table (Hue range 81 to 140)
  // Removing all non-green points from table
  // for(int i = 0; i < ptrPtCldTable->points.size(); i++){
  //   float r,g,b,h,s,v;
  //   r = static_cast<float>(ptrPtCldTable->points[i].r);
  //   g = static_cast<float>(ptrPtCldTable->points[i].g);
  //   b = static_cast<float>(ptrPtCldTable->points[i].b);
  //   RGBtoHSV(r/255.0,g/255.0,b/255.0,h,s,v);
  //   if(!(h >= 81 && h <=140)){
  //     ptrPtCldTable->points.erase(ptrPtCldTable->points.begin()+i);
  //     i--;
  //   }
  // }
  // ptrPtCldTable->width = ptrPtCldTable->points.size();
  // ptrPtCldTable->height = 1;
  // pcl::compute3DCentroid(*ptrPtCldTable, cenTable);
  // pcl::getMinMax3D(*ptrPtCldTable, minTable, maxTable);

  // Removing all green points from object and manipulator points
  for(int i = 0; i < ptrPtCldObject->points.size(); i++){
    float r,g,b,h,s,v;
    r = static_cast<float>(ptrPtCldObject->points[i].r);
    g = static_cast<float>(ptrPtCldObject->points[i].g);
    b = static_cast<float>(ptrPtCldObject->points[i].b);
    RGBtoHSV(r/255.0,g/255.0,b/255.0,h,s,v);
    if(h >= 81 && h <=140 || ptrPtCldObject->points[i].x <= 0.1){
      ptrPtCldObject->points.erase(ptrPtCldObject->points.begin()+i);
      i--;
    }
  }
  ptrPtCldObject->width = ptrPtCldObject->points.size();
  ptrPtCldObject->height = 1;

  // Getting the min and max co-ordinates of the object
  pcl::compute3DCentroid(*ptrPtCldObject, cenObject);
  pcl::getMinMax3D(*ptrPtCldObject, minPtObj, maxPtObj);
}

// 10: Generating unexplored point cloud
void environment::genUnexploredPtCld(){
  if(ptrPtCldUnexp->width != 0){
    std::cerr << "Unexplored point cloud already created. Not creating new one." << std::endl;
    return;
  }

  // Setting the min and max limits based on the object dimension and scale.
  // Min of 0.40m on each side
  // Note: Z scale is only used on +z axis
  minUnexp[0] = (minPtObj.x-std::max((scale-1)*(maxPtObj.x-minPtObj.x)/2,0.30f));
  minUnexp[1] = (minPtObj.y-std::max((scale-1)*(maxPtObj.y-minPtObj.y)/2,0.30f));
  minUnexp[2] = tableCentre[2]-0.01;
  maxUnexp[0] = (maxPtObj.x+std::max((scale-1)*(maxPtObj.x-minPtObj.x)/2,0.30f));
  maxUnexp[1] = (maxPtObj.y+std::max((scale-1)*(maxPtObj.y-minPtObj.y)/2,0.30f));
  maxUnexp[2] = (maxPtObj.z+std::max((scale-1)*(maxPtObj.z-minPtObj.z)/2,0.25f));

  pcl::PointXYZRGB ptTemp;
  for(float x = minUnexp[0]; x < maxUnexp[0]; x+=voxelGridSizeUnexp){
    for(float y = minUnexp[1]; y < maxUnexp[1]; y+=voxelGridSizeUnexp){
      for(float z = minUnexp[2]; z < maxUnexp[2]; z+=voxelGridSizeUnexp){
        ptTemp.x = x; ptTemp.y = y; ptTemp.z = z;
        ptrPtCldUnexp->points.push_back(ptTemp);
      }
    }
  }
  ptrPtCldUnexp->width = ptrPtCldUnexp->points.size();
  ptrPtCldUnexp->height = 1;
}

// 11: Updating the unexplored point cloud
void environment::updateUnexploredPtCld(){

  ptCldColor::Ptr lastFilled{new ptCldColor};
  if(simulationMode == "SIMULATION"){
    // Transforming the point cloud to Kinect frame from world frame
    Eigen::Affine3f tf = tfGazWorld*tfKinOptGaz;
    Eigen::Affine3f tfTranspose = homoMatTranspose(tf);
    pcl::transformPointCloud(*ptrPtCldUnexp, *ptrPtCldTemp, tfTranspose);
  }else if(simulationMode == "FRANKASIMULATION"){
    tf::StampedTransform transform;
    listener.lookupTransform("camera_optical_link", "panda_link0", ros::Time(0), transform);
    pcl_ros::transformPointCloud(*ptrPtCldUnexp, *ptrPtCldTemp, transform);
  }else{
    tf::StampedTransform transform;
    listener.lookupTransform("camera_depth_optical_frame", "panda_link0", ros::Time(0), transform);
    pcl::transformPointCloud(*ptrPtCldUnexp, *ptrPtCldTemp, ICP_Tf.inverse());
    pcl_ros::transformPointCloud(*ptrPtCldTemp, *ptrPtCldTemp, transform);
  }

  Eigen::Vector4f ptTemp;
  Eigen::Vector3f proj;
  static pcl::PointIndices::Ptr occludedIndices(new pcl::PointIndices());
  occludedIndices->indices.clear();
  int projIndex;

  // Looping through all the points and finding occluded ones.
  // Using the camera projection matrix to project 3D point to camera plane
  for(int i = 0; i < ptrPtCldTemp->width; i++){
    ptTemp = ptrPtCldTemp->points[i].getVector4fMap();
    proj = projectionMat*ptTemp;
    proj = proj/proj[2];
    proj[0] = round(proj[0]);
    proj[1] = round(proj[1]);

    // Leave Points below the table as it is
    if(ptrPtCldUnexp->points[i].z < cenTable[2]){
      occludedIndices->indices.push_back(i);
    }else{
      if(proj[0] >=0 && proj[0] <= 640-1 && proj[1] >=0 && proj[1] <= 480-1){
        projIndex = proj[1]*(ptrPtCldLastFill->width)+proj[0];
        // If the z value of unexplored pt is greater than the corresponding
        // projected point in Kinect Raw data then that point is occluded.
        if(ptrPtCldLastFill->points[projIndex].z > 0.10 && ptrPtCldLastFill->points[projIndex].z <= ptTemp[2]*0.99){
          occludedIndices->indices.push_back(i);
        }
      }
    }
  }

  // Only keeping the occluded points
  extract.setInputCloud(cPtrPtCldUnexp);
  extract.setIndices(occludedIndices);
  extract.setNegative(false);
  extract.filter(*ptrPtCldUnexp);

  if(minUnexp[0] > minPtObj.x || minUnexp[1] > minPtObj.y ||
     maxUnexp[0] < maxPtObj.x || maxUnexp[1] < maxPtObj.y || maxUnexp[2] < maxPtObj.z){
    ROS_WARN("Unexplored point cloud initially generated smaller than the object.");
  }

  ptrPtCldTemp->clear();

  // Recalculating centroid based on unexplored point cloud
  *ptrPtCldTemp = *ptrPtCldUnexp + *ptrPtCldObject;
  pass.setInputCloud(cPtrPtCldTemp);
  pass.setFilterFieldName("x"); pass.setFilterLimits(minPtObj.x,maxPtObj.x); pass.filter(*ptrPtCldTemp);
  pass.setFilterFieldName("y"); pass.setFilterLimits(minPtObj.y,maxPtObj.y); pass.filter(*ptrPtCldTemp);
  pass.setFilterFieldName("z"); pass.setFilterLimits(minPtObj.z,maxPtObj.z); pass.filter(*ptrPtCldTemp);
  pcl::compute3DCentroid(*ptrPtCldTemp, cenObject);

  ptrPtCldTemp->clear();
}

// 12: Finding normals and pairs of grasp points from object point cloud
void environment::graspsynthesis(){

  graspsPossible.clear();   // Clear the vector
  selectedGrasp = -1;

  graspPoint graspTemp;
  Eigen::Vector3f vectA, vectB;
  double A,B;

  pcl::PointXYZRGB centroidObj,centroidGrasp;
  Eigen::Vector4f zAxis(0,0,1,0);
  centroidObj.x = cenObject[0]; centroidObj.y = cenObject[1]; centroidObj.z = cenObject[2];
  int stepSize = std::max(1,int(round(double(ptrPtCldObject->size())/1000.0)));
  // Using brute force search
  // Checking for each pair of points (alternate points skipped to speedup the process)
  for(int i = 0; i < ptrPtCldObject->size()-1; i+=stepSize){
    if(useForGrasp[i] == false) continue;
    for(int j = i+1; j < ptrPtCldObject->size(); j+=stepSize){
      if(useForGrasp[j] == false) continue;

      graspTemp.p1 = ptrPtCldObject->points[i];
      graspTemp.p2 = ptrPtCldObject->points[j];

      // Ignoring point closer to table
      if(graspTemp.p1.z <= tableCentre[2]+0.01 || graspTemp.p2.z <= tableCentre[2]+0.01) continue;

      // Vector connecting the two grasp points and its distance
      vectA = graspTemp.p1.getVector3fMap() - graspTemp.p2.getVector3fMap();
      vectB = graspTemp.p2.getVector3fMap() - graspTemp.p1.getVector3fMap();
      graspTemp.gripperWidth = vectA.norm() + voxelGridSize; // Giving a tolerance based on voxel grid size

      // If grasp width is greater than the limit then skip the rest
      if(graspTemp.gripperWidth > maxGripperWidth) continue;

      // Using normals to find the angle
      A = std::min(pcl::getAngle3D(vectA,ptrObjNormal->points[i].getNormalVector3fMap()),
                   pcl::getAngle3D(vectB,ptrObjNormal->points[i].getNormalVector3fMap()))*180/M_PI;
      B = std::min(pcl::getAngle3D(vectA,ptrObjNormal->points[j].getNormalVector3fMap()),
                   pcl::getAngle3D(vectB,ptrObjNormal->points[j].getNormalVector3fMap()))*180/M_PI;

      graspTemp.quality = 180 - ( A + B );
      // If grasp quality is less than the min requirement then skip the rest
      if(graspTemp.quality < minGraspQuality) continue;

      centroidGrasp.x = (graspTemp.p1.x + graspTemp.p2.x) / 2;
      centroidGrasp.y = (graspTemp.p1.y + graspTemp.p2.y) / 2;
      centroidGrasp.z = (graspTemp.p1.z + graspTemp.p2.z) / 2;
      graspTemp.distance = pcl::euclideanDistance(centroidGrasp,centroidObj);
      graspTemp.lineDistance = pcl::sqrPointToLineDistance(centroidGrasp.getVector4fMap(),centroidObj.getVector4fMap(),zAxis);
      graspTemp.aboveCOG = centroidGrasp.z >= centroidObj.z;
      centroidGrasp.z = centroidObj.z; graspTemp.lineDistance = pcl::euclideanDistance(centroidGrasp,centroidObj);

      // Push this into the vector
      graspsPossible.push_back(graspTemp);
      findGripperPose(graspsPossible.size()-1);
    }
  }

  std::sort(graspsPossible.begin(),graspsPossible.end(),compareGrasp);
  // For thin objects grasp pair would not be feasible, so each point is considered as a grasp pair
  // Adding these grasps in the end
  Eigen::Vector3f xyPlaneA(0,0,1);
  Eigen::Vector3f xyPlaneB(0,0,-1);
  for(int i = 0; i < ptrPtCldObject->size(); i+=stepSize){
    if(useForGrasp[i] == false) continue;

    A = std::min(pcl::getAngle3D(xyPlaneA,ptrObjNormal->points[i].getNormalVector3fMap()),
                 pcl::getAngle3D(xyPlaneB,ptrObjNormal->points[i].getNormalVector3fMap()))*180/M_PI;

    // If the point is too close to table and its normal vector is along z axis this skip it
    if (A > 45 && ptrPtCldObject->points[i].z < tableCentre[2]+0.01){
      continue;
    }

    graspTemp.p1 = ptrPtCldObject->points[i];
    // Translating it along the +ve normal vector
    graspTemp.p1.x += (voxelGridSize)/2*ptrObjNormal->points[i].normal_x;
    graspTemp.p1.y += (voxelGridSize)/2*ptrObjNormal->points[i].normal_y;
    graspTemp.p1.z += (voxelGridSize)/2*ptrObjNormal->points[i].normal_z;

    graspTemp.p2 = ptrPtCldObject->points[i];
    // Translating it along the -ve normal vector
    graspTemp.p2.x -= (voxelGridSize)/2*ptrObjNormal->points[i].normal_x;
    graspTemp.p2.y -= (voxelGridSize)/2*ptrObjNormal->points[i].normal_y;
    graspTemp.p2.z -= (voxelGridSize)/2*ptrObjNormal->points[i].normal_z;

    graspTemp.gripperWidth = voxelGridSize;
    graspTemp.quality = 180;

    centroidGrasp.x = (graspTemp.p1.x + graspTemp.p2.x) / 2;
    centroidGrasp.y = (graspTemp.p1.y + graspTemp.p2.y) / 2;
    centroidGrasp.z = (graspTemp.p1.z + graspTemp.p2.z) / 2;
    graspTemp.distance = pcl::euclideanDistance(centroidGrasp,centroidObj);
    graspTemp.lineDistance = pcl::sqrPointToLineDistance(centroidGrasp.getVector4fMap(),centroidObj.getVector4fMap(),zAxis);
    graspTemp.aboveCOG = centroidGrasp.z >= centroidObj.z;
    centroidGrasp.z = centroidObj.z; graspTemp.lineDistance = pcl::euclideanDistance(centroidGrasp,centroidObj);

    graspsPossible.push_back(graspTemp);
    findGripperPose(graspsPossible.size()-1);
  }

}

// 13: Given a grasp point pair find the gripper orientation
void environment::findGripperPose(int index){

  Eigen::Vector3f xAxis,yAxis,zAxis;
  Eigen::Vector3f xyPlane(0,0,1);

  // Calculating the x,y,z axis which is at the midpoint of the fingers
  yAxis = graspsPossible[index].p1.getVector3fMap() - graspsPossible[index].p2.getVector3fMap(); yAxis.normalize();
  zAxis = yAxis.cross(xyPlane); zAxis.normalize();
  xAxis = yAxis.cross(zAxis);

  // Finding RPY based on the axis directions
  tf::Matrix3x3 rotMat, pitch180, yaw180;
  pitch180.setRPY(0,M_PI,0);
  yaw180.setRPY(0,0,M_PI);
  double Roll,Pitch,Yaw;
  rotMat.setValue(xAxis[0],yAxis[0],zAxis[0],
                  xAxis[1],yAxis[1],zAxis[1],
                  xAxis[2],yAxis[2],zAxis[2]);


  if(rotMat.getColumn(2).getX() < 0) rotMat *= pitch180;
  if(rotMat.getColumn(0).getZ() < 0) rotMat *= yaw180;

  // std::cout << rotMat.getColumn(0).getX()  << "," << rotMat.getColumn(0).getY() << "," << rotMat.getColumn(0).getZ() << std::endl;
  // std::cout << rotMat.getColumn(1).getX()  << "," << rotMat.getColumn(1).getY() << "," << rotMat.getColumn(1).getZ() << std::endl;
  // std::cout << rotMat.getColumn(2).getX()  << "," << rotMat.getColumn(2).getY() << "," << rotMat.getColumn(2).getZ() << std::endl;

  rotMat.getRPY(Roll,Pitch,Yaw);

  // Setting the coordinates as the midpoint between the two fingers i.e. midpoint of the two pointclouds
  std::vector<float> pose = {0,0,0,0,0,0};
  pose[0] = (graspsPossible[index].p1.x + graspsPossible[index].p2.x)/2;
  pose[1] = (graspsPossible[index].p1.y + graspsPossible[index].p2.y)/2;
  pose[2] = (graspsPossible[index].p1.z + graspsPossible[index].p2.z)/2;
  pose[3] = Roll; pose[4] = Pitch; pose[5] = Yaw;

  graspsPossible[index].pose = pose;
}

// 14: Collision check for gripper and unexplored point cloud
void environment::collisionCheck(){
  ptrPtCldCollided->clear();    // Reset the collision cloud
  *ptrPtCldCollCheck = *ptrPtCldUnexp + *ptrPtCldObject;
  // *ptrPtCldCollCheck += *ptrPtCldTable;
  cpBox.setInputCloud(ptrPtCldCollCheck);

  bool stop = false;
  int nOrientations;
  std::vector<int> orientations;
  // nOrientations = 8; orientations = {0,45,90,135,180,225,270,315};
  // nOrientations = 8; orientations = {90,45,135,0,180,315,225,270};
  // nOrientations = 9; orientations = {90,68,112,45,135,22,157,0,180};
  // nOrientations = 9; orientations = {-45,-68,-22,-90,0,-112,-135,-157,-180};
  //ELEPHANT: rational?
  nOrientations = 7; orientations = {-90,-68,-122,-45,-135,-22,-157,0,180};
  // ptCldVis::Ptr viewer(new ptCldVis ("PCL Viewer")); std::vector<int> vp;
  // setupViewer(viewer, 1, vp);
  // viewer->setCameraPosition(-2,0,7,0.45,0,0,0,0,1);
  // keyboardEvent keyPress(viewer,1);
  // Loop through all the possible grasps available
  for(int i = 0; (i < graspsPossible.size()) && (stop == false); i++){
    // std::cout << graspsPossible[i].aboveCOG << "," << graspsPossible[i].lineDistance << std::endl;
    findGripperPose(i);

    // Do axis collision check
    updateGripper(i,1);
    for (int j = 3; j < 5; j++) {
      cpBox.setMin(minPtCol[j].getVector4fMap());
      cpBox.setMax(maxPtCol[j].getVector4fMap());
      cpBox.setRotation(getEuler(tfGripper));
      cpBox.setTranslation(getTranslation(tfGripper));
      cpBox.filter(*ptrPtCldCollided);
      // If collision detected then exit this loop and check next grasp pair
      if(ptrPtCldCollided->size() > 0) break;
    }
    // if(keyPress.ok){
    //   updateGripper(i,0);
    //   for(int ii = 0; ii < ptrPtCldCollided->size(); ii++){
    //     ptrPtCldCollided->points[ii].r = 0;
    //     ptrPtCldCollided->points[ii].g = 255;
    //     ptrPtCldCollided->points[ii].b = 0;
    //   }
    //   addRGB(viewer,ptrPtCldUnexp,"Unexp",vp[0]);
    //   addRGB(viewer,ptrPtCldObject,"Env",vp[0]);
    //   addRGB(viewer,ptrPtCldGripper,"Gripper",vp[0]);
    //   addRGB(viewer,ptrPtCldCollided,"Coll",vp[0]);
    //
    //   keyPress.called = false;
    //   while(!viewer->wasStopped() && keyPress.called==false){
    //     viewer->spinOnce(100);
    //     boost::this_thread::sleep (boost::posix_time::microseconds(100000));
    //   }
    //   viewer->resetStoppedFlag();
    //   viewer->removeAllPointClouds();
    // }
    // std::cout << ptrPtCldCollided->size() << std::endl;
    // Move to next grasp if collision found
    if(ptrPtCldCollided->size() > 0) continue;

    // Do gripper collision check for each orientation
    for(int j = 0; (j < nOrientations) && (stop == false); j++){
      // graspsPossible[i].addnlPitch = j*(2*M_PI)/nOrientations;
      graspsPossible[i].addnlPitch = orientations[j]*M_PI/180;
      if(simulationMode != "SIMULATION" && (!checkPreGrasp(graspsPossible[i]))) continue;
      updateGripper(i,2);
      // if(keyPress.ok){
      //   updateGripper(i,0);
      //   addRGB(viewer,ptrPtCldEnv,"Env",vp[0]);
      //   addRGB(viewer,ptrPtCldGripper,"Gripper",vp[0]);
      //
      //   keyPress.called = false;
      //   while(!viewer->wasStopped() && keyPress.called==false){
      //     viewer->spinOnce(100);
      //     boost::this_thread::sleep (boost::posix_time::microseconds(100000));
      //   }
      //   viewer->resetStoppedFlag();
      //   viewer->removeAllPointClouds();
      // }
      for(int k = 5; k >= 0; k--){
        if(k == 4 || k == 3) continue;
        // ptrPtCldCollided->clear();    // Reset the collision cloud
        cpBox.setMin(minPtCol[k].getVector4fMap());
        cpBox.setMax(maxPtCol[k].getVector4fMap());
        cpBox.setRotation(getEuler(tfGripper));
        cpBox.setTranslation(getTranslation(tfGripper));
        cpBox.filter(*ptrPtCldCollided);

        // If collision detected then exit this loop and check next orientation
        if(k == 2 || k == 1){
          if(ptrPtCldCollided->size() > 5) break;
        }else{
          if(ptrPtCldCollided->size() > 0) break;
        }
      }
      // If this doesn't have collision, this grasp is OK. So exit the loop. No more orientation or grasp check required
      if(ptrPtCldCollided->size() == 0){
        selectedGrasp = i;
        stop = true;
        // std::cout << "fff..." << checkPreGrasp(graspsPossible[i]) << "***" << std::endl;
        // std::cout << selectedGrasp << std::endl;
      }
    }
  }
}

// 15: Grasp and Collision check combined. After finding each grasp collision check is done
int environment::graspAndCollisionCheck(){

  graspsPossible.clear();   // Clear the vector
  selectedGrasp = -1;

  graspPoint graspTemp;
  Eigen::Vector3f vectA, vectB;
  double A,B;

  pcl::PointXYZRGB centroidObj,centroidGrasp;
  Eigen::Vector4f zAxis(0,0,1,0);
  centroidObj.x = cenObject[0]; centroidObj.y = cenObject[1]; centroidObj.z = cenObject[2];

  int nGrasp = 0;
  // std::cout << "IN1" << std::endl;
  for (int i = 0; i < ptrPtCldObject->size()-1; i++){
    if(useForGrasp[i] == false) continue;
    for (int j = i+1; j < ptrPtCldObject->size(); j++){
      if(useForGrasp[j] == false) continue;

      graspTemp.p1 = ptrPtCldObject->points[i];
      graspTemp.p2 = ptrPtCldObject->points[j];

      // Ignoring point closer to table
      if(graspTemp.p1.z <= tableCentre[2]+0.01 || graspTemp.p2.z <= tableCentre[2]+0.01) continue;

      // Vector connecting the two grasp points and its distance
      vectA = graspTemp.p1.getVector3fMap() - graspTemp.p2.getVector3fMap();
      vectB = graspTemp.p2.getVector3fMap() - graspTemp.p1.getVector3fMap();
      graspTemp.gripperWidth = vectA.norm() + voxelGridSize; // Giving a tolerance based on voxel grid size

      // If grasp width is greater than the limit then skip the rest
      if(graspTemp.gripperWidth > maxGripperWidth) continue;

      // Using normals to find the angle
      A = std::min(pcl::getAngle3D(vectA,ptrObjNormal->points[i].getNormalVector3fMap()),
                   pcl::getAngle3D(vectB,ptrObjNormal->points[i].getNormalVector3fMap()))*180/M_PI;
      B = std::min(pcl::getAngle3D(vectA,ptrObjNormal->points[j].getNormalVector3fMap()),
                   pcl::getAngle3D(vectB,ptrObjNormal->points[j].getNormalVector3fMap()))*180/M_PI;

      graspTemp.quality = 180 - ( A + B );

      // If grasp quality is less than the min requirement then skip the rest
      if (graspTemp.quality < minGraspQuality) continue;

      centroidGrasp.x = (graspTemp.p1.x + graspTemp.p2.x) / 2;
      centroidGrasp.y = (graspTemp.p1.y + graspTemp.p2.y) / 2;
      centroidGrasp.z = (graspTemp.p1.z + graspTemp.p2.z) / 2;
      graspTemp.distance = pcl::euclideanDistance(centroidGrasp,centroidObj);
      graspTemp.lineDistance = pcl::sqrPointToLineDistance(centroidGrasp.getVector4fMap(),centroidObj.getVector4fMap(),zAxis);
      graspTemp.aboveCOG = centroidGrasp.z >= centroidObj.z;
      centroidGrasp.z = centroidObj.z; graspTemp.lineDistance = pcl::euclideanDistance(centroidGrasp,centroidObj);

      nGrasp++;
      // Push this into the vector
      if (graspsPossible.size() == 0){
        graspsPossible.push_back(graspTemp);
      }else{
        graspsPossible[0] = graspTemp;
      }
      collisionCheck();
      if(selectedGrasp == 0) {
        return(nGrasp);
      }
    }
  }
  // std::sort(graspsPossible.begin(),graspsPossible.end(),compareGrasp);

  // For thin objects grasp pair would not be feasible, so each point is considered as a grasp pair
  // Adding these grasps in the end
  Eigen::Vector3f xyPlaneA(0,0,1);
  Eigen::Vector3f xyPlaneB(0,0,-1);
  for(int i = 0; i < ptrPtCldObject->size(); i++){
    if(useForGrasp[i] == false) continue;

    A = std::min(pcl::getAngle3D(xyPlaneA,ptrObjNormal->points[i].getNormalVector3fMap()),
                 pcl::getAngle3D(xyPlaneB,ptrObjNormal->points[i].getNormalVector3fMap()))*180/M_PI;

    // If the point is too close to table and its normal vector is along z axis this skip it
    if (A > 45 && ptrPtCldObject->points[i].z < tableCentre[2]+0.01){
      continue;
    }

    graspTemp.p1 = ptrPtCldObject->points[i];
    // Translating it along the +ve normal vector
    graspTemp.p1.x += (voxelGridSize)/2*ptrObjNormal->points[i].normal_x;
    graspTemp.p1.y += (voxelGridSize)/2*ptrObjNormal->points[i].normal_y;
    graspTemp.p1.z += (voxelGridSize)/2*ptrObjNormal->points[i].normal_z;

    graspTemp.p2 = ptrPtCldObject->points[i];
    // Translating it along the -ve normal vector
    graspTemp.p2.x -= (voxelGridSize)/2*ptrObjNormal->points[i].normal_x;
    graspTemp.p2.y -= (voxelGridSize)/2*ptrObjNormal->points[i].normal_y;
    graspTemp.p2.z -= (voxelGridSize)/2*ptrObjNormal->points[i].normal_z;

    graspTemp.gripperWidth = voxelGridSize;
    graspTemp.quality = 180;

    centroidGrasp.x = (graspTemp.p1.x + graspTemp.p2.x) / 2;
    centroidGrasp.y = (graspTemp.p1.y + graspTemp.p2.y) / 2;
    centroidGrasp.z = (graspTemp.p1.z + graspTemp.p2.z) / 2;
    graspTemp.distance = pcl::euclideanDistance(centroidGrasp,centroidObj);
    graspTemp.lineDistance = pcl::sqrPointToLineDistance(centroidGrasp.getVector4fMap(),centroidObj.getVector4fMap(),zAxis);
    graspTemp.aboveCOG = centroidGrasp.z >= centroidObj.z;
    centroidGrasp.z = centroidObj.z; graspTemp.lineDistance = pcl::euclideanDistance(centroidGrasp,centroidObj);

    nGrasp++;
    if (graspsPossible.size() == 0){
      graspsPossible.push_back(graspTemp);
    }else{
      graspsPossible[0] = graspTemp;
    }
    collisionCheck();
    if(selectedGrasp == 0){
      return(nGrasp);
    }
  }
  return(nGrasp);
}

// 16: Modify moveit collision elements
void environment::editMoveItCollisions(std::string object, std::string mode){

  if(object == "OBJECT" && ptrPtCldObject->points.size() == 0) return;

  moveit_planner::AddCollision collisionObjMsg;

  collisionObjMsg.request.collObject.header.frame_id = "/world";
  collisionObjMsg.request.collObject.id = object;

  if(mode == "ADD"){
    shape_msgs::SolidPrimitive primitive;

    geometry_msgs::Pose pose;
    pose.orientation.x = 0;
    pose.orientation.y = 0;
    pose.orientation.z = 0;
    pose.orientation.w = 1;

    if(object == "TABLE"){
      primitive.type = primitive.BOX;
      primitive.dimensions.resize(3);
      primitive.dimensions[0] = 0.7;
      primitive.dimensions[1] = 0.7;
      primitive.dimensions[2] = 0.01;

      pose.position.x = tableCentre[0];
      pose.position.y = tableCentre[1];
      pose.position.z = tableCentre[2];
      if(simulationMode == "FRANKA") pose.position.z -= 0.03;
    }else if(object == "OBJECT"){
      // Object size limited to 25x25x25 cm

      // primitive.type = primitive.BOX;
      // primitive.dimensions.resize(3);
      // primitive.dimensions[0] = (maxPtObj.x - minPtObj.x)*1.1;
      // primitive.dimensions[1] = (maxPtObj.y - minPtObj.y)*1.1;
      // primitive.dimensions[2] = (maxPtObj.z)*1.1;

      primitive.type = primitive.CYLINDER;
      primitive.dimensions.resize(2);
      primitive.dimensions[0] = std::min((maxPtObj.z-tableCentre[2])*1.1,0.25);
      primitive.dimensions[1] = std::min(double(std::max(maxPtObj.x - minPtObj.x,maxPtObj.y - minPtObj.y)),0.25)/2.0;
      if(simulationMode == "FRANKA"){
        primitive.dimensions[0] -= 0.05;
        primitive.dimensions[1] -= 0.03;
      }
      if(primitive.dimensions[0] < 0) primitive.dimensions[0] = 0.01;
      if(primitive.dimensions[1] < 0) primitive.dimensions[1] = 0.01;

      pose.position.x = cenObject[0];
      pose.position.y = cenObject[1];
      pose.position.z = tableCentre[2]+primitive.dimensions[0]/2;
    }
    collisionObjMsg.request.collObject.primitives.push_back(primitive);
    collisionObjMsg.request.collObject.primitive_poses.push_back(pose);
    collisionObjMsg.request.collObject.operation = collisionObjMsg.request.collObject.ADD;
  }else if(mode == "REMOVE"){
    collisionObjMsg.request.collObject.operation = collisionObjMsg.request.collObject.REMOVE;
  }

  collisionClient.call(collisionObjMsg);
}

// 17: Check is pregrasp is viable
bool environment::checkPreGrasp(graspPoint &graspData){
  Eigen::Affine3f tfGrasp = pcl::getTransformation(graspData.pose[0],graspData.pose[1],
                                                   graspData.pose[2],graspData.pose[3],
                                                   graspData.pose[4],graspData.pose[5])*
                            pcl::getTransformation(0,0,0,0,graspData.addnlPitch,0);

  // if(tfGrasp(0,0) < 0) tfGrasp = tfGrasp*pcl::getTransformation(0,0,0,0,0,M_PI);
  // Checking approach vector
  // Eigen::Vector3f approachVec(tfGrasp(0,2),tfGrasp(1,2),tfGrasp(2,2));
  // Eigen::Vector3f xAxis(1,0,0), yAxis(0,1,0);
  // float angleX = abs(pcl::getAngle3D(approachVec,xAxis,true));
  // float angleY = abs(pcl::getAngle3D(approachVec,yAxis,true));
  //
  // if(angleY > 60 && angleY < 120 && (angleX < 60 || (180-angleX) < 30)) return false;

  tfGrasp = tfGrasp * pcl::getTransformation(0.0,0,-0.0447-fingerZOffset,0,0,0)*
                      pcl::getTransformation(0,0,0,0,-M_PI/2,-M_PI);

  Eigen::Affine3f tfPreGrasp;

  float clearance = 0;
  float rad = std::min(double(std::max(maxPtObj.x - minPtObj.x,maxPtObj.y - minPtObj.y)),0.25)/2.0;
  do{
    tfPreGrasp = tfGrasp*pcl::getTransformation(-clearance+0.0447+fingerZOffset,0,0,0,0,0);
    clearance += 0.01;
  }while((sqrt(pow(tfPreGrasp(0,3)-cenObject[0],2) + pow(tfPreGrasp(1,3)-cenObject[1],2)) <= rad &&
          tfPreGrasp(2,3)>0 && tfPreGrasp(2,3)<0.25));

  tfPreGrasp = tfGrasp*pcl::getTransformation(-clearance,0,0,0,0,0);

  geometry_msgs::Pose p;
  bool res = moveFranka(tfGrasp.matrix(),"JOINT",false,false,p) *
             moveFranka(tfPreGrasp.matrix(),"JOINT",false,false,p);
  return res;
}

// 17: Object grasping pipeline
void environment::graspObject(graspPoint &graspData){

  if(simulationMode == "SIMULATION") return;

  int temp;
  std::cout << "Do you want to run the grasp test? (1/0)"; std::cin >> temp;
  if(temp == 0) return;

  clearAllConstraints();
  editMoveItCollisions("OBJECT","ADD");

  if(simulationMode == "FRANKA"){
    graspData.pose[0] += 0.010;
    graspData.pose[1] -= 0.010;
  }

  Eigen::Affine3f tfGrasp = pcl::getTransformation(graspData.pose[0],graspData.pose[1],
                                                   graspData.pose[2],graspData.pose[3],
                                                   graspData.pose[4],graspData.pose[5])*
                            pcl::getTransformation(0,0,0,0,graspData.addnlPitch,0);

  // if(tfGrasp(0,0) < 0) tfGrasp = tfGrasp*pcl::getTransformation(0,0,0,0,0,M_PI);

  tfGrasp = tfGrasp * pcl::getTransformation(0.0,0,-0.0447-fingerZOffset,0,0,0)*
                      pcl::getTransformation(0,0,0,0,-M_PI/2,-M_PI);


  // std::cout << tfGrasp(0,0) << "," << tfGrasp(0,1) << "," << tfGrasp(0,2) << "," << tfGrasp(0,3) << "\n"
  //           << tfGrasp(1,0) << "," << tfGrasp(1,1) << "," << tfGrasp(1,2) << "," << tfGrasp(1,3) << "\n"
  //           << tfGrasp(2,0) << "," << tfGrasp(2,1) << "," << tfGrasp(2,2) << "," << tfGrasp(2,3) << std::endl;

  Eigen::Affine3f tfGraspMoveUp = tfGrasp; tfGraspMoveUp(2,3)+=0.1;
  Eigen::Affine3f tfPreGrasp;
  geometry_msgs::Pose p;

  float clearance = 0;
  float rad = std::min(double(std::max(maxPtObj.x - minPtObj.x,maxPtObj.y - minPtObj.y)),0.25)/2.0;
  do{
    tfPreGrasp = tfGrasp*pcl::getTransformation(-clearance+0.0447+fingerZOffset,0,0,0,0,0);
    clearance += 0.01;
  }while((sqrt(pow(tfPreGrasp(0,3)-cenObject[0],2) + pow(tfPreGrasp(1,3)-cenObject[1],2)) <= rad &&
          tfPreGrasp(2,3)>0 && tfPreGrasp(2,3)<0.25));

  tfPreGrasp = tfGrasp*pcl::getTransformation(-clearance*1.1,0,0,0,0,0);

  bool res = true;

  // std::cout << moveFranka(tfGrasp.matrix(),"JOINT",false,false,p) << std::endl;
  // std::cout << moveFranka(tfPreGrasp.matrix(),"JOINT",false,false,p) << std::endl;

  moveFrankaHome();

  std::cout << "Moving to Pre Grasp..." << std::endl;
  res = moveFranka(tfPreGrasp.matrix(),"JOINT",false,true,p);
  if(!res){
    tfPreGrasp = tfGrasp*pcl::getTransformation(-clearance-0.07,0,0,0,0,0);
    res = moveFranka(tfPreGrasp.matrix(),"JOINT",false,true,p);
  }
  if(!res){
    tfPreGrasp = tfGrasp*pcl::getTransformation(-clearance-0.15,0,0,0,0,0);
    res = moveFranka(tfPreGrasp.matrix(),"JOINT",false,true,p);
  }
  editMoveItCollisions("OBJECT","REMOVE");
  moveGripper(graspData.gripperWidth+0.04);

  // Move to Grasp and grasp
  if(res){
    std::cout << "Enter any key to start grasping..."; std::cin >> temp;
    res = moveFranka(tfGrasp.matrix(),"CARTESIAN",false,true,p);
    if(simulationMode == "FRANKA") moveGripper(graspData.gripperWidth-voxelGridSize-0.02,true);
    else                           moveGripper(graspData.gripperWidth-1.5*voxelGridSize,true);
  }else std::cout << "Skipping Grasp" << std::endl;

  // Lift test
  if(res){
    std::cout << "Do you want to run the Lift test? (1/0) : "; std::cin >> temp;
    if(temp == 1){
      std::cout << "Lift test running" << std::endl;
      res = moveFranka(tfGraspMoveUp.matrix(),"CARTESIAN",false,true,p);
      std::cout << "Lift test ended" << std::endl;
    }
  }else std::cout << "Skipping Lift Test" << std::endl;

  // Rotation test
  if(res){
    std::cout << "Do you want to run the Rotation test? (1/0) : "; std::cin >> temp;
    if(temp == 1){
      moveFrankaHome(false);
      std::cout << "Rotation test running" << std::endl;
      moveit_planner::SetJointWithTime setJointWithTimeMsg;
      setJointWithTimeMsg.request.joint_name = "panda_joint7";
      for(int i = 0; i < 2; i++){
        setJointWithTimeMsg.request.joint_angle = 0.000; oneJointWithTimeClient.call(setJointWithTimeMsg);
        setJointWithTimeMsg.request.joint_angle = 1.570; oneJointWithTimeClient.call(setJointWithTimeMsg);
      }
      setJointWithTimeMsg.request.joint_angle = 0.785; oneJointWithTimeClient.call(setJointWithTimeMsg);
      std::cout << "Rotation test ended" << std::endl;
    }
  }else std::cout << "Skipping Rotation Test" << std::endl;

  // Shaking test
  if(res){
    std::cout << "Do you want to run the Shaking test? (1/0) : "; std::cin >> temp;
    if(temp == 1){
      moveFrankaHome(false);
      std::cout << "Shaking test running" << std::endl;
      moveit_planner::SetJointWithTime setJointWithTimeMsg;
      setJointWithTimeMsg.request.joint_name = "panda_joint5";
      for(int i = 0; i < 4; i++){
        setJointWithTimeMsg.request.joint_angle = -0.35; oneJointWithTimeClient.call(setJointWithTimeMsg);
        setJointWithTimeMsg.request.joint_angle =  0.35; oneJointWithTimeClient.call(setJointWithTimeMsg);
      }
      setJointWithTimeMsg.request.joint_angle =  0.00; oneJointWithTimeClient.call(setJointWithTimeMsg);
      std::cout << "Shaking test ended" << std::endl;
    }
  }else std::cout << "Skipping Shaking test" << std::endl;

  res = moveFranka(tfGraspMoveUp.matrix(),"JOINT",false,true,p);
  // Go to same location and release
  if(res){
    std::cout << "Enter any key to place the object back..."; std::cin >> temp;
    res = moveFranka(tfGrasp.matrix(),"CARTESIAN",false,true,p);
    moveGripper(graspData.gripperWidth+0.04);
  }else std::cout << "Skipping placing object back" << std::endl;

  // Move to Post-grasp
  if(res){
    std::cout << "Moving to Retreat...";
    res = moveFranka(tfPreGrasp.matrix(),"CARTESIAN",false,true,p);
    std::cout << res << std::endl;
    editMoveItCollisions("OBJECT","ADD");
  }else std::cout << "Skipping Post Grasp" << std::endl;

  moveFrankaHome();
  editMoveItCollisions("OBJECT","REMOVE");

}

// ******************** ENVIRONMENT CLASS FUNCTIONS END ********************

// Function to do a single pass
std::vector<double> singlePass(environment &av, std::vector<double> kinectPose, bool firstTime, bool findGrasp, int mode){
  std::vector<double> timer;
  std::chrono::high_resolution_clock::time_point start,end;
  start = std::chrono::high_resolution_clock::now();

  if(av.simulationMode == "FRANKA") printf("Running single pass...\nMoving kinect...\n");
  av.moveKinectViewsphere(kinectPose);
  if(av.simulationMode == "FRANKA") printf("Reading data...\n");
  av.readKinect();
  if(av.simulationMode == "FRANKA") printf("Fusing data...\n");
  av.fuseLastData();
  if(av.simulationMode == "FRANKA") printf("Extracting table & object...\n");
  av.dataExtract();
  if(av.simulationMode == "FRANKA") printf("Updating unexplored regions...\n");
  if (firstTime == true) av.genUnexploredPtCld();
  av.updateUnexploredPtCld();
  if(av.simulationMode == "FRANKA") printf("Grasp synthesis...\n");

  end = std::chrono::high_resolution_clock::now();
  timer.push_back((std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count());

  start = std::chrono::high_resolution_clock::now();

  if (findGrasp == true){
    if(mode == 1){
      av.graspAndCollisionCheck();
    }
    else{
      av.graspsynthesis();
      av.collisionCheck();
    }
  }
  if(av.simulationMode == "FRANKA") printf("Ending single pass...\n");

  end = std::chrono::high_resolution_clock::now();
  timer.push_back((std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count());
  return timer;
}
