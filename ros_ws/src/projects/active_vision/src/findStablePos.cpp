#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <chrono>
#include <boost/make_shared.hpp>

#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_datatypes.h>

#include <pcl_ros/point_cloud.h>

//Gazebo specific includes
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/GetModelState.h>
#include <gazebo_msgs/SpawnModel.h>
#include <gazebo_msgs/DeleteModel.h>

std::string getCurTime(){
	time_t now = time(0);
	tm *ltm = localtime(&now);
	char temp[50];
	sprintf(temp, "%04d_%02d_%02d_%02d%02d%02d", 1900 + ltm->tm_year,1 + ltm->tm_mon,ltm->tm_mday,ltm->tm_hour,ltm->tm_min,ltm->tm_sec);
	std::string name = temp;
	return(name);
}

bool comparePose(std::vector<float> &A , std::vector<float> &B){
	if (A[1] != B[1]) return(A[1]>B[1]);
	else return(A[2]>B[2]);
}

int calcError(std::vector<float> &A, int thresh){
  int res = 0;
	for (int i = 1; i < 6; i++) {
		res += int(abs(A[i]) > thresh);
	}
  return(res);
}

// Class to store data of environment and its processing
class stablePose{
private:
  ros::Rate r{10};                      // ROS sleep rate
  ros::Publisher pubObjPose;            // Publisher : Object pose
  ros::ServiceClient getObjPose;        // Service : Object pose
  ros::ServiceClient gazeboSpawnModel;  // Service : Spawn Model
  ros::ServiceClient gazeboDeleteModel; // Service : Delete Model

  std::string path;                // Path the active vision package

public:

  std::vector<std::vector<std::string>> objectDict; // List of objects which can be spawned
  std::vector<double> tableCentre;                  // Co-ordinates of table centre
  int objectID;

  stablePose(ros::NodeHandle *nh){
    pubObjPose = nh->advertise<gazebo_msgs::ModelState> ("/gazebo/set_model_state", 1);
    gazeboSpawnModel = nh->serviceClient< gazebo_msgs::SpawnModel> ("/gazebo/spawn_sdf_model");
    gazeboDeleteModel = nh->serviceClient< gazebo_msgs::DeleteModel> ("/gazebo/delete_model");
    getObjPose = nh->serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");

    path = ros::package::getPath("active_vision");  // Path to the active_vision package folder

    tableCentre = {0,0,0};       // Co-ordinates of table centre

		// Dictionary of objects to be spawned
	  objectDict = {{"drillAV","Cordless Drill"},
	                {"squarePrismAV","Square Prism"},
	                {"rectPrismAV","Rectangular Prism"},
	                {"bowlAV","Bowl"},
	                {"bowl2AV","Bowl2"},
	                {"cupAV","Cup"},
	                {"cinderBlockAV","Cinder Block"},
	                {"handleAV","Door Handle"}};
  }

  // Reading object pose
  std::vector<float> readObjPose(int type){
    gazebo_msgs::GetModelState model;
    model.request.model_name = objectDict[objectID][1];
    model.request.relative_entity_name = "world";
    getObjPose.call(model);

    std::vector<float> temp{0,0,0,0,0,0};

		if (type == 1) {
	    temp[0] = round(model.response.pose.position.x*1000);
	    temp[1] = round(model.response.pose.position.y*1000);
	    temp[2] = round(model.response.pose.position.z*1000);
	    tf::Quaternion quat(model.response.pose.orientation.x,model.response.pose.orientation.y,
	                        model.response.pose.orientation.z,model.response.pose.orientation.w);

	    double Roll,Pitch,Yaw;
	    tf::Matrix3x3(quat).getRPY(Roll, Pitch, Yaw);

	    temp[3] = round(Roll*180/M_PI);
	    temp[4] = round(Pitch*180/M_PI);
	    temp[5] = round(Yaw*180/M_PI);
		} else if (type == 2) {
			temp[0] = round(model.response.twist.linear.x*1000);
	    temp[1] = round(model.response.twist.linear.y*1000);
	    temp[2] = round(model.response.twist.linear.z*1000);
			temp[3] = round(model.response.twist.angular.x*180/M_PI);
	    temp[4] = round(model.response.twist.angular.y*180/M_PI);
	    temp[5] = round(model.response.twist.angular.z*180/M_PI);
		}
    return(temp);
  }

  // Spawning objects in gazebo
  void spawnObject(){
    gazebo_msgs::SpawnModel spawnObj;
    geometry_msgs::Pose pose;

    //Create Matrix3x3 from Euler Angles
    tf::Matrix3x3 m_rot;
    m_rot.setEulerYPR(0, 0, 0);

    // Convert into quaternion
    tf::Quaternion quat;
    m_rot.getRotation(quat);

    pose.position.x = tableCentre[0];
    pose.position.y = tableCentre[1];
    pose.position.z = tableCentre[2];
    pose.orientation.x = quat.x();
    pose.orientation.y = quat.y();
    pose.orientation.z = quat.z();
    pose.orientation.w = quat.w();

    spawnObj.request.model_name = objectDict[objectID][1];

    std::ifstream ifs(path+"/models/"+objectDict[objectID][0]+"/model.sdf");
    std::string sdfFile( (std::istreambuf_iterator<char>(ifs)),
                         (std::istreambuf_iterator<char>()));
    spawnObj.request.model_xml = sdfFile;

    spawnObj.request.reference_frame = "world";
    spawnObj.request.initial_pose = pose;

    gazeboSpawnModel.call(spawnObj);

    boost::this_thread::sleep(boost::posix_time::milliseconds(1500));
  }

  // Deleting objects in gazebo
  void deleteObject(){
    gazebo_msgs::DeleteModel deleteObj;
    deleteObj.request.model_name = objectDict[objectID][1];

    gazeboDeleteModel.call(deleteObj);
  }

  // Function to move the object. Args: Array of X,Y,Z,Roll,Pitch,Yaw
  void moveObject(std::vector<float> pose){
    //Create Matrix3x3 from Euler Angles
    tf::Matrix3x3 rotMat;
    rotMat.setEulerYPR(pose[5]*M_PI/180, pose[4]*M_PI/180, pose[3]*M_PI/180);

    // Convert into quaternion
    tf::Quaternion quat;
    rotMat.getRotation(quat);

    // Converting it to the required gazebo format
    gazebo_msgs::ModelState ModelState;
    ModelState.model_name = objectDict[objectID][1];
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
    boost::this_thread::sleep(boost::posix_time::milliseconds(500));
  }
};

int main (int argc, char** argv){
  // Initialize ROS
  ros::init (argc, argv,"findStablePoses");
  ros::NodeHandle nh;

  int objID;

  stablePose env(&nh);
  // Delay to ensure all publishers and subscribers are connected
  boost::this_thread::sleep(boost::posix_time::milliseconds(500));

  std::cout << "Objects available :" << std::endl;
  std::cout << "1: Drill" << std::endl;
  std::cout << "2: Square Prism" << std::endl;
  std::cout << "3: Rectangular Prism" << std::endl;
  std::cout << "4: Bowl" << std::endl;
  std::cout << "5: Big Bowl" << std::endl;
  std::cout << "6: Cup" << std::endl;
  std::cout << "Enter your choice : "; std::cin >> objID;

  env.objectID = objID-1;
  env.spawnObject();

  std::vector<float> pose = {0,0,0,0,0,0};
	std::vector<float> res;
	std::vector<std::vector<float>> finalPoses;

  pose[0] = env.tableCentre[0];
  pose[1] = env.tableCentre[1];
  pose[2] = env.tableCentre[2];

  int errSum;
  std::cout << "Start Time : " << getCurTime() << std::endl;
  for (int roll = 0; roll < 360; roll+=15){
    printf("Testing Roll : %d\n",roll);
    for (int pitch = 0; pitch < 360; pitch+=15){
      for (int yaw = 0; yaw < 15; yaw+=15){
        pose[3] = roll;
        pose[4] = pitch;
        pose[5] = yaw;
        env.moveObject(pose);

        errSum = 6;
        do{
					boost::this_thread::sleep(boost::posix_time::milliseconds(100));
          res = env.readObjPose(2);
          errSum = calcError(res,0);
        }while(errSum!=0);

				res = env.readObjPose(1);
				finalPoses.push_back({round(res[2]),round(res[3]),round(res[4])});
				// printf("%d,%d\n",roll,pitch);
				// printf("%d,%d, ,%f,%f,%f\n",roll,pitch,round(res[2]),round(res[3]),round(res[4]));
      }
    }
  }
  std::cout << "End Time : " << getCurTime() << std::endl;
  env.deleteObject();

	// Remove similar entries
	sort(finalPoses.begin(), finalPoses.end(), comparePose);

	// for (int j = 0; j < finalPoses.size(); j++) {
	// 	printf("%f,%f,%f\n",finalPoses[j][0],finalPoses[j][1],finalPoses[j][2]);
	// }

	int i = 0;
	while (i < finalPoses.size()-1){
		if(finalPoses[i][1] == finalPoses[i+1][1] && finalPoses[i][2] == finalPoses[i+1][2]){
			finalPoses.erase(finalPoses.begin()+i+1);
		}
		else i++;
	}

	printf("*******Printing the possible poses (Z(mm),Roll(deg),Pitch(Deg))*******\n");

	for (int j = 0; j < finalPoses.size(); j++) {
		printf("%f,%f,%f\n",finalPoses[j][0],finalPoses[j][1],finalPoses[j][2]);
	}
}
