#include <active_vision/environment.h>
#include <active_vision/toolDataHandling.h>
#include <active_vision/toolViewPointCalc.h>
#include <active_vision/toolVisualization.h>
#include <active_vision/toolStateVector.h>
#include <active_vision/heuristicPolicySRV.h>
#include <active_vision/trainedPolicySRV.h>

int mode;
int runMode;
int testAll;
int maxSteps;
int nSaved;
int HAFstVecGridSize;
std::string simulationMode;
bool recordPtCldEachStep;

void updateRouteData(environment &env, RouteData &data, bool save ,std::string name){
	data.success = (env.selectedGrasp != -1);
	if(save == true){
		data.obj = *env.ptrPtCldObject;
		data.unexp = *env.ptrPtCldUnexp;
		data.childID = env.saveConfiguration(name);
	}
	if(data.success == true){
		if(save == true) data.goodInitialGrasp = true;
		data.graspQuality = env.graspsPossible[env.selectedGrasp].quality;
		env.updateGripper(env.selectedGrasp,0);
		data.env = *env.ptrPtCldEnv + *env.ptrPtCldGripper;
	}
	else{
		if(save == true) data.goodInitialGrasp = false;
		data.env = *env.ptrPtCldEnv;
	}
}

std::vector<int> nearbyDirections(int dir){
	std::vector<int> dirs = {1,2,3,4,5,6,7,8};
	dir--;
	dirs[0] = (8+dir)%8+1;
	dirs[1] = (8+dir+1)%8+1; dirs[2] = (8+dir-1)%8+1;
	dirs[3] = (8+dir+2)%8+1; dirs[4] = (8+dir-2)%8+1;
	dirs[5] = (8+dir+3)%8+1; dirs[6] = (8+dir-3)%8+1;
	dirs[7] = (8+dir+4)%8+1;
	return dirs;
}
// Function the find the grasp
void findGrasp(environment &kinectControl, int object, int objPoseCode, int objYaw, std::string dir, std::string saveLocation, ros::ServiceClient &policy){

	std::fstream fout;
 	fout.open(dir+saveLocation, std::ios::out | std::ios::app);

 	std::vector<double> startPose = {kinectControl.viewsphereRad,M_PI,M_PI/7};
	kinectControl.moveKinectViewsphere(startPose);
	std::vector<double> nextPose;
	RouteData home, current;
	std::vector<double> timeRecord;

	static ptCldVis::Ptr viewer(new ptCldVis("PCL Viewer"));
	static std::vector<int> vp;
	setupViewer(viewer, 2, vp);
	viewer->removeCoordinateSystem();
	static keyboardEvent keyPress(viewer,2);

	kinectControl.moveObject(object,objPoseCode,objYaw*(M_PI/180.0));

	// Home Pose
	home.reset();
	home.path = {startPose};
	home.type = 1;
	home.direction = 0;
	home.nSteps = 0;
	home.objType = kinectControl.objectDict[object].description;
	home.objPose = {kinectControl.objectDict[object].poses[objPoseCode][1],
									kinectControl.objectDict[object].poses[objPoseCode][2],
									objYaw*(M_PI/180.0)};

	timeRecord = singlePass(kinectControl, startPose, true, true, 2);
	updateRouteData(kinectControl,home,true,"Home");
	home.timer.push_back(timeRecord[0]);
	home.timer.push_back(timeRecord[1]);
	home.detailedEnv.push_back(*kinectControl.ptrPtCldEnv);
	current = home;

	static ptCldColor::Ptr tempPtCld{new ptCldColor};
	static ptCldColor::Ptr tempPtCldObj{new ptCldColor};
	static ptCldColor::Ptr tempPtCldUnexp{new ptCldColor};
	pcl::PointXYZ table,a1,a2;
	table.x = kinectControl.tableCentre[0];
	table.y = kinectControl.tableCentre[1];
	table.z = kinectControl.tableCentre[2];
	std::vector<int> nUnexp;
	// printf("Starting while loop.\n");

	while(keyPress.ok){
		if(!(::runMode != 0 && ::testAll == 1)){
			viewer->resetStoppedFlag();
			viewer->removeAllPointClouds();
			viewer->removeAllShapes();

			addRGB(viewer,kinectControl.ptrPtCldObject,"Obj",vp[0]);
			addRGB(viewer,kinectControl.ptrPtCldUnexp,"Unexp",vp[0]);
			*tempPtCld = current.env; addRGB(viewer,tempPtCld,"Env",vp[1]);

			addViewsphere(viewer,vp.back(),table,current.path[0][0],false);
			a1 = sphericalToCartesian(current.path[0],table);
			viewer->addSphere(a1,0.04,1,0,0,"Start",vp[1]);
			viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,"Start");

			for(int i = 0; i < current.path.size()-1; i++){
				a1 = sphericalToCartesian(current.path[i],table);
				a2 = sphericalToCartesian(current.path[i+1],table);
				viewer->addArrow(a2,a1,0,1,0,false,std::to_string(i),vp[1]);
			}

			setCamView(viewer,current.path.back(),table);
		}

		int maxIndex = 0;
		if(current.success == true){
			current.filename = getCurTime()+"_"+std::to_string(::nSaved);
			saveData(current, fout, dir, ::recordPtCldEachStep);
			// printRouteData(current);
			::nSaved++;
			if(::testAll != 1) viewer->addText("GRASP FOUND. Saving and exiting.",4,5,25,1,0,0,"Dir1",vp[0]);
		}else{
			std::chrono::high_resolution_clock::time_point start,end;
			start = std::chrono::high_resolution_clock::now();
      		if(::runMode == 1){
       			static active_vision::heuristicPolicySRV srv;
				pcl::toROSMsg(*kinectControl.ptrPtCldObject,srv.request.object);
				pcl::toROSMsg(*kinectControl.ptrPtCldUnexp,srv.request.unexplored);
				srv.request.path.data.clear();
				for(int i = 0; i < current.path.size(); i++){
					srv.request.path.data.push_back(current.path[i][0]);
					srv.request.path.data.push_back(current.path[i][1]);
					srv.request.path.data.push_back(current.path[i][2]);
				}
				srv.request.mode = ::mode;
				srv.request.minPtsVis = 0.15*(kinectControl.ptrPtCldObject->points.size()); // Only used for 3D Heuristic
				policy.call(srv);
				maxIndex = srv.response.direction;
				if(::testAll != 1) viewer->addText("Best direction calculated : " + std::to_string(maxIndex+1) + "(" + dirLookup[maxIndex+1] + ").\nPress any key to continue.",5,30,25,1,0,0,"Dir1",vp[0]);
      		}else if(::runMode == 2){
				static HAFStVec1 stVec;
				stVec.setGridDim(::HAFstVecGridSize);
		    	stVec.setMaintainScale(true);
				*tempPtCldObj = *kinectControl.ptrPtCldObject;
				*tempPtCldUnexp = *kinectControl.ptrPtCldUnexp;
				stVec.setInput(tempPtCldObj,tempPtCldUnexp,current.path.back());
				stVec.calculate();
				static active_vision::trainedPolicySRV srv;
				srv.request.stateVec.data = stVec.getStateVec();
				policy.call(srv);
				maxIndex = srv.response.direction-1;
				if(::testAll != 1) viewer->addText("Best direction calculated : " + std::to_string(maxIndex+1) + "(" + dirLookup[maxIndex+1] + ").\nPress any key to continue.",5,30,25,1,0,0,"Dir1",vp[0]);
			}else{
        		viewer->addText("Manual Mode.\nEnter the direction.",5,30,25,1,0,0,"Dir1",vp[0]);
      		}
			end = std::chrono::high_resolution_clock::now();
			current.timer.push_back((std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count());
		}

		if(!(::runMode != 0 && ::testAll == 1)){
			keyPress.called = false;
			while(!viewer->wasStopped() && keyPress.called==false){
				viewer->spinOnce(100);
				boost::this_thread::sleep (boost::posix_time::microseconds(100000));
			}
		}
		if(::runMode != 0) keyPress.dir = maxIndex+1;

		if(current.success == true){
			keyPress.dir = 0;
			keyPress.ok = false;
		}else{
			// Limiting the steps
			if(current.nSteps > ::maxSteps){
				current.filename = getCurTime()+"_"+std::to_string(::nSaved);
				saveData(current, fout, dir, ::recordPtCldEachStep);
				// printRouteData(current);
				::nSaved++;
				keyPress.dir = 0;
				keyPress.ok = false;
			}
		}

		// Reset to home if direction is 0
		if(keyPress.dir == 0){
			kinectControl.rollbackConfiguration(home.childID);
			current = home;
			current.nSteps = 0;
		}

		if(keyPress.dir >=1 && keyPress.dir <= 8){
			std::vector<int> dirPref = nearbyDirections(keyPress.dir);
			int prfID = 0;
      		nextPose = calcExplorationPose(current.path.back(),dirPref[prfID],::mode);
			while(checkValidPose(nextPose) == false || checkIfNewPose(current.path,nextPose,::mode) == false){
				prfID++;
				nextPose = calcExplorationPose(current.path.back(),dirPref[prfID],::mode);
			}
			if(prfID!=0){
				printf("Direction modified from %d -> %d.\n", dirPref[0],dirPref[prfID]);
			}
			std::cout << dirLookup[dirPref[prfID]] << ",";
			timeRecord = singlePass(kinectControl, nextPose, false, true, 2);
			updateRouteData(kinectControl,current,false,"dummy");
			current.timer.push_back(timeRecord[0]);
			current.timer.push_back(timeRecord[1]);
			current.detailedEnv.push_back(*kinectControl.ptrPtCldEnv);
			current.path.push_back(nextPose);
			current.nSteps++;
			if(current.nSteps == 1) home.direction = keyPress.dir;
    	}
	}

	keyPress.ok = true;
	kinectControl.reset();
	fout.close();
}

void help(){
	std::cout << "******* Policy Tester Node Help *******" << std::endl;
  	std::cout << "Arguments : [RunMode]" << std::endl;
	std::cout << "RunMode : 0->Manual, 1->Heuristic, 2->Trained Policy" << std::endl;
	std::cout << "*******" << std::endl;
}

int main(int argc, char** argv){
	chdir(getenv("HOME"));

	ros::init(argc, argv, "Policy_Tester");
 	ros::NodeHandle nh;
  ros::ServiceClient policy;
 	environment kinectControl(&nh); sleep(1);
  kinectControl.loadGripper();

	bool relativePath; nh.getParam("/active_vision/policyTester/relative_path", relativePath);
	std::string temp;
	nh.getParam("/active_vision/policyTester/directory", temp);
	std::string dir;
	if(relativePath == true) dir = ros::package::getPath("active_vision") + temp;
	else										 dir = temp;

	std::string time = getCurTime();
	nh.getParam("/active_vision/policyTester/csvName", temp);
	std::string csvName;
	if(temp == "default.csv") csvName = time+"_dataRec.csv";
	else	csvName = temp;

	if(csvName.substr(csvName.size() - 4) != ".csv"){
    ROS_ERROR("Incorrect file name.");
    help(); return(-1);
	}

	int objID;
	nh.getParam("/active_vision/policyTester/objID", objID);
	if(kinectControl.objectDict.count(objID) == 0){
		std::cout << "ERROR. Incorrect Object ID." << std::endl;
    return(-1);
	}

	nh.getParam("/active_vision/kinectMode", ::mode);
	if(::mode < 1 && ::mode > 2) ::mode = 2;

	nh.getParam("/active_vision/policyTester/maxSteps", ::maxSteps);
	nh.getParam("/active_vision/policyTester/HAFstVecGridSize", ::HAFstVecGridSize);
	nh.getParam("/active_vision/policyTester/recordPtCldEachStep", ::recordPtCldEachStep);

	::runMode = std::atoi(argv[1]);
	if(::runMode < 0 && ::runMode > 2) ::runMode = 0;

  if(::runMode == 0){
  	ROS_INFO("Manual Mode Selected.");
  }
  else if(::runMode == 1){
		while(!ROSCheck("SERVICE","/active_vision/heuristic_policy")){boost::this_thread::sleep(boost::posix_time::seconds(1));}
    policy = nh.serviceClient<active_vision::heuristicPolicySRV>("/active_vision/heuristic_policy");
    ROS_INFO("Heuristic Policy Selected.");
  }else if(::runMode == 2){
		while(!ROSCheck("SERVICE","/active_vision/trained_policy")){boost::this_thread::sleep(boost::posix_time::seconds(1));}
		policy = nh.serviceClient<active_vision::trainedPolicySRV>("/active_vision/trained_policy");
    ROS_INFO("Trained Policy Selected.");
	}

	nh.getParam("/active_vision/simulationMode", ::simulationMode);

	if(::simulationMode == "SIMULATION" && ::runMode != 0){
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

		findGrasp(kinectControl, objID, objPoseCode, objYaw, dir, csvName, policy);

		if(kinectControl.selectedGrasp != -1) std::cout << "Grasp found." << std::endl;
		else																  std::cout << "Grasp not found." << std::endl;

		if(::simulationMode == "FRANKASIMULATION" && kinectControl.selectedGrasp != -1){
			std::cout << "Simulating the grasp found." << std::endl;
			kinectControl.graspObject(kinectControl.graspsPossible[kinectControl.selectedGrasp]);
		}
	}else{
		// Reading the yawValues csv file
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
				findGrasp(kinectControl, objID, objPoseCode, objYaw, dir, csvName, policy);
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
				findGrasp(kinectControl, objID, objPoseCode, yawAngleDict[key][ctr], dir, csvName, policy);
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
