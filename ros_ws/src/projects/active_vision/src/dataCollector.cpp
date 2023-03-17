#include <active_vision/environment.h>
#include <active_vision/toolViewPointCalc.h>
#include <active_vision/toolVisualization.h>

int mode;
int nRdmSearches;
int maxRndSteps;
int maxOptStep = 2.0;

std::vector<int> randomDirection(){
	std::vector<int> dirs = {1,2,3,4,5,6,7,8};
	std::vector<int> ret;
	int randIndex;
	while(dirs.size() > 0){
		randIndex = rand() % (dirs.size());
		ret.push_back(dirs[randIndex]);
		dirs.erase(dirs.begin()+randIndex);
	}
	return ret;
}

// Explore one step in all deirection from the home/parent position
// Configurations are saved
bool exploreOneStepAllDir(environment &env, RouteData &home, std::vector<RouteData> &opOneStep){

	std::vector<double> curPose;
	RouteData temp;                // Used to record the iteration details

	// Sanity check
	if(home.childID == -1 || home.success == true){
	printf("ERROR exploreOneStepAllDir: Invalid child ID / Grasp already found.\n");
	return(false);
	}

	temp.parentID = home.childID;
  	temp.success = false;
	// Going to each direction for a step and saving their configurations
  	std::vector<int> dirs = randomDirection();
	for (int i = 1; i <= 8 && temp.success == false; i++){
		// Rollback to the parent/home configuration
		env.rollbackConfiguration(temp.parentID);
		curPose = calcExplorationPose(env.lastKinectPoseViewsphere,dirs[i-1],::mode);
		std::cout << "." << std::flush;
		// If kinect pose if above the table, then proceed in that direction
		if (checkValidPose(curPose) == true && checkIfNewPose(home.path,curPose,::mode) == true){
			// Saving the direction and path taken
			temp.direction = dirs[i-1];
			temp.path = home.path;
			temp.path.push_back(curPose);

			// Doing a pass and saving the child
			singlePass(env, curPose, false, true);
			temp.success = (env.selectedGrasp != -1);
			temp.childID = env.saveConfiguration(env.configurations[temp.parentID].description+"_C"+std::to_string(dirs[i-1]));

			// Exit the loop and updating variables used for visualization
			if(temp.success == true){
				temp.graspQuality = env.graspsPossible[env.selectedGrasp].quality;
				env.updateGripper(env.selectedGrasp,0);
				temp.env = *env.ptrPtCldEnv + *env.ptrPtCldGripper;
			}else{
        temp.graspQuality = -1;
        temp.env = *env.ptrPtCldEnv;
      }
			temp.env += *env.ptrPtCldUnexp;

			opOneStep.push_back(temp);
		}
	}
	return(temp.success);
}

// Explore a max of "nSteps" random steps from the home/parent position
// Configurations not saved
bool exploreRdmStep(environment &env, RouteData &home, int nSteps, RouteData &opData){

	std::vector<double> curPose;

  // Sanity check
  if(home.childID == -1 || home.success == true){
    printf("ERROR exploreRdmStep: Invalid child ID / Grasp already found.\n");
    return(false);
  }

	opData.parentID = home.childID;
  opData.direction = home.direction;
  opData.path = home.path;

  // Rollback to the parent/home configuration
  env.rollbackConfiguration(opData.parentID);

  opData.success = false;
  // Going taking a max of nSteps from the parent position
	for (int i = 1; i <= nSteps && opData.success == false; i++){
    // Do until the pose is valid
		std::vector<int> dirs = randomDirection();
		bool foundNextPose = false;
    for(int j = 0; j < 7; j++) {
    	curPose = calcExplorationPose(env.lastKinectPoseViewsphere,dirs[j],::mode);
			if(checkValidPose(curPose) == true && checkIfNewPose(opData.path,curPose,::mode) == true){
				foundNextPose = true;
				break;
			}
    }

		if(foundNextPose == true){
			// Saving path taken
			opData.path.push_back(curPose);

			// Doing a pass
			singlePass(env, curPose, false, true);
			opData.success = (env.selectedGrasp != -1);

			// Updating variables used for visualization
			if(opData.success == true){
				opData.graspQuality = env.graspsPossible[env.selectedGrasp].quality;
				env.updateGripper(env.selectedGrasp,0);
				opData.env = *env.ptrPtCldEnv + *env.ptrPtCldGripper;
				opData.env += *env.ptrPtCldUnexp;
			}
		}else{
			break;
		}
	}
  return(opData.success);
}

void updateTemp(environment &env, RouteData &data, std::string &name){
	data.success = (env.selectedGrasp != -1);
	data.obj = *env.ptrPtCldObject;
	data.unexp = *env.ptrPtCldUnexp;
	data.childID = env.saveConfiguration(name);
	if(data.success == true){
		data.goodInitialGrasp = true;
		data.graspQuality = env.graspsPossible[env.selectedGrasp].quality;
		env.updateGripper(env.selectedGrasp,0);
		data.env = *env.ptrPtCldEnv + *env.ptrPtCldGripper;
	}
	else{
		data.goodInitialGrasp = false;
		data.env = *env.ptrPtCldEnv;
	}
	data.env += *env.ptrPtCldUnexp;
}

// Generating the home poses from which we need to search
void genHomePoses(environment &env, std::vector<double> &stPose, std::string &csvName, std::vector<RouteData> &opHomePoses){
	std::vector<std::vector<std::string>> data = readCSV(csvName);
  	std::vector<nodeData> tree = convertToTree(data);
	std::vector<double> nextPose;

	std::string name;
	int dir = 0;
  	RouteData temp; temp.reset();

	if(tree[0].parentID != -1) return;

	for(int i = 0; i < tree.size(); i++){
		std::cout << "." << std::flush;
		if(tree[i].checked != 0) continue;
		temp.type = tree[i].dirs.length()+1;
		temp.path = {stPose};
		if(i == 0){
			singlePass(env, stPose, true, true); tree[0].checked = 1;
		}else{
			env.rollbackConfiguration(tree[tree[i].parentID].configID);
			dir = tree[i].dirs.back() - '0';
			nextPose = calcExplorationPose(env.lastKinectPoseViewsphere,dir,::mode);
			singlePass(env, nextPose, false, true); tree[i].checked = 1;
		}
		// Updating the path
		for(int j = 0; j < tree[i].dirs.length(); j++){
			dir = tree[i].dirs[j] - '0';
			temp.path.push_back(calcExplorationPose(temp.path.back(),dir,::mode));
		}
		name = "H"+std::to_string(tree[i].nodeID);
		updateTemp(env,temp,name);
		// No need to add the children as home poses if current pose is successful
		if(temp.success == true){
			printf("found grasp at %d\n",i);
			tree[i].checked = 2;
			env.configurations.pop_back();
			for(int j = i; j < tree.size(); j++){
				if(tree[j].checked != 0){
					for(int k = 0; k < tree[j].childIDs.size(); k++){
						tree[tree[j].childIDs[k]].checked = 2;
					}
				}
			}
		}else{
			tree[i].configID = temp.childID;
			temp.nodeInfo = tree[i];
			opHomePoses.push_back(temp);
		}
  }
	// for(int i = 0; i < tree.size(); i++){
  //   tree[i].print();
  // }
}

// Function to save optimal paths for the given position.
int findOptimalPaths(environment &kinectControl, int object, std::string homePosesTree, std::string dir, std::string saveLocation, int currentPoseCode, int currentYaw){
	std::fstream fout;
 	fout.open(dir+saveLocation, std::ios::out | std::ios::app);

  	kinectControl.spawnObject(object,0,0);

 	std::vector<double> startPose = {kinectControl.viewsphereRad, 180*(M_PI/180.0), 45*(M_PI/180.0)};

  	bool graspFound = false;
	RouteData dataFinalTemp, cData;
	std::vector<RouteData> dataOneStep,dataTemp;

  	int nHomeConfigs;

	int nSaved=0;
	int nZeroHomePoses = 0;
	printf("BFS  *** Pose #%d/%d with Yaw %d ***\n",currentPoseCode+1,int(kinectControl.objectDict[object].nPoses),currentYaw);
	//Set up the start pose
	kinectControl.moveObject(object,currentPoseCode,currentYaw*(M_PI/180.0));
	RouteData start; 
	start.reset();
	singlePass(kinectControl, startPose, true, true); 
	std::string name = "H_"+std::to_string(currentPoseCode)+"_"+std::to_string(currentYaw);
	start.childID = kinectControl.saveConfiguration(name);
	start.path = {startPose};
	updateTemp(kinectControl, start, name);
	// Check the start pose:
	int iteration = 0;
	printf("BFS      ***Testing step %d\n", iteration);
	graspFound = start.success;
	// If no grasp was found, begin the BFS.
	if(not graspFound){
		iteration++;
		printf("BFS      ***Testing step %d\n", iteration);
    	graspFound = exploreOneStepAllDir(kinectControl,start,dataOneStep);
    	iteration++;
    } else {
    	dataOneStep.push_back(start);
    }
    while((not graspFound) and (iteration < pow(8,::maxOptStep))){
    	printf("BFS      ***Testing step %d\n", iteration);
    	cData = dataOneStep[0];
    	dataTemp.clear();
    	dataOneStep.erase(dataOneStep.begin());
    	graspFound = exploreOneStepAllDir(kinectControl,cData,dataTemp);
    	for (int i = 0; i < dataTemp.size(); ++i){
    		dataOneStep.push_back(dataTemp[i]);
    	}
    	iteration++;
    }

	printf("BFS    %d\n",dataOneStep.size());
	for (int i = 0; i < dataOneStep.size(); ++i)
	{
		printf("Item %d, passed=%d\n", i, dataOneStep[i].success);
	}
	// Iterate over the last 8 poses to find the successful one
	for (int i = 1; i < 9; ++i) {
		dataFinalTemp.reset();
		dataFinalTemp = dataOneStep[dataOneStep.size()-i];
		dataFinalTemp.nodeInfo = start.nodeInfo;
      	dataFinalTemp.objType = kinectControl.objectDict[object].description;
      	dataFinalTemp.objPose = {kinectControl.objectDict[object].poses[currentPoseCode][1],
                           			  kinectControl.objectDict[object].poses[currentPoseCode][2],
                           			  currentYaw*(M_PI/180.0)};
      	dataFinalTemp.type = start.type;
      	// Don't count the start position for path length
      	dataFinalTemp.nSteps = dataFinalTemp.path.size()-1;
      	dataFinalTemp.obj = start.obj;
      	dataFinalTemp.unexp = start.unexp;
		if(dataFinalTemp.success){
			printf("Trying direction %d.... success\n", i);
			dataFinalTemp.filename = getCurTime()+"_"+std::to_string(nSaved);
			saveData(dataFinalTemp, fout, dir, false);
			break;
		} else {
			printf("Trying direction %d.... failed\n", i);
		}
	}
    kinectControl.deleteObject(object);
    kinectControl.reset();
	fout.close();
	return(nSaved);
}

// Function the generate the data for training
int generateData(environment &kinectControl, int object, std::string homePosesTree, std::string dir, std::string saveLocation, int nDataPoints){
	std::fstream fout;
 	fout.open(dir+saveLocation, std::ios::out | std::ios::app);

  	kinectControl.spawnObject(object,0,0);

 	std::vector<double> startPose = {kinectControl.viewsphereRad, 180*(M_PI/180.0), 45*(M_PI/180.0)};

  	bool graspFound = false;
	RouteData dataFinalTemp;
	std::vector<RouteData> dataOneStep,dataHomePoses,dataFinal;

  	int nHomeConfigs;

	int nSaved=0;
	int currentPoseCode;
	int currentYaw;
	int nZeroHomePoses = 0;

	while(nSaved < nDataPoints && nZeroHomePoses <= 5){

		currentPoseCode = rand()%(kinectControl.objectDict[object].nPoses);
		currentYaw = rand()%(int(kinectControl.objectDict[object].poses[currentPoseCode][4]) -int(kinectControl.objectDict[object].poses[currentPoseCode][3]) + 1) + int(kinectControl.objectDict[object].poses[currentPoseCode][3]);

    	printf("  *** Pose #%d/%d with Yaw %d ***\n",currentPoseCode+1,int(kinectControl.objectDict[object].nPoses),currentYaw);
		kinectControl.moveObject(object,currentPoseCode,currentYaw*(M_PI/180.0));

    	// Get the homes poses to collect data for based on the step count
    	dataHomePoses.clear();
		printf("    * Generating Home Configs"); std::cout << std::flush;
	    genHomePoses(kinectControl,startPose,homePosesTree,dataHomePoses);
		dataFinal.resize(dataHomePoses.size());
	    nHomeConfigs = kinectControl.configurations.size();
		if(nHomeConfigs == 0) nZeroHomePoses++;

		std::cout << " " << dataHomePoses.size() << " configs." << std::endl;

			// for (int j = 0; j < kinectControl.configurations.size(); j++){
			//   printf("%d,%s\n",j,kinectControl.configurations[j].description.c_str());
			// }
			// std::cout << std::endl;
			// for(int poses = dataHomePoses.size()-1; poses >= 0 ; poses--){
			// 	dataHomePoses[poses].nodeInfo.print();
			// }

	    for(int poses = dataHomePoses.size()-1; poses >= 0 ; poses--){
	      	dataFinal[poses].reset();

	      	printf("      Home #%d/%d : ",poses+1,int(dataHomePoses.size())); std::cout << std::flush;
	      	graspFound = dataHomePoses[poses].success;
	      	// If home pose has a successful grasp
	      	if(graspFound == true) std::cout << "How ???" << std::endl;

	      	// If we do not find a grasp at home, then check for one step in each direction
	      	printf("8 Dir Search"); std::cout << std::flush;
	      	dataOneStep.clear();
	      	graspFound = exploreOneStepAllDir(kinectControl,dataHomePoses[poses],dataOneStep);
	      	if(graspFound == true){
				dataFinal[poses] = dataOneStep.back();
			} else {
				// If we dont find a grasp after one step, then check using random searches
				int nSteps = ::maxRndSteps;
				if(dataHomePoses[poses].nodeInfo.possibleSolChildIndex != -1){
					nSteps = dataHomePoses[poses].nodeInfo.allowedSteps-1;
				}

				for (int rdmSearchCtr = 1; rdmSearchCtr <= ::nRdmSearches && nSteps > 0; rdmSearchCtr++){
			        printf(" Random Search #%d.",rdmSearchCtr); std::cout << std::flush;

			        for(int i = 0; i < dataOneStep.size() && nSteps > 0; i++){
						dataFinalTemp.reset();
						std::cout << nSteps << std::flush;
			          	bool graspFoundTemp = exploreRdmStep(kinectControl,dataOneStep[i],nSteps,dataFinalTemp);
			          	if(graspFoundTemp == true){
			            	graspFound = true;
			            	// If its the first time entering this, then store this in expRes
			            	if(dataFinal[poses].path.size() == 0){
			              		dataFinal[poses] = dataFinalTemp;
			            	}
			            	// If not, check if current result is better than the previous result and update expRes
			            	else if((dataFinalTemp.path.size() < dataFinal[poses].path.size()) ||
			                (dataFinalTemp.path.size() == dataFinal[poses].path.size() &&
			                dataFinalTemp.graspQuality > dataFinal[poses].graspQuality)){
			              		dataFinal[poses] = dataFinalTemp;
			            	}
			            	// Update nSteps to speed up the process
			            	nSteps = std::min(nSteps,int(dataFinal[poses].path.size()-dataOneStep[i].path.size()-1));
			          	}
			        }
		      	}
			}

	      	// If no grasp found, select the child which has a grasp
	      	if(graspFound == false){
				int index = dataHomePoses[poses].nodeInfo.possibleSolChildIndex;
				if(index != -1){
					graspFound = true;
					dataFinal[poses].path = dataFinal[index].path;
					dataFinal[poses].success = dataFinal[index].success;
					dataFinal[poses].graspQuality = dataFinal[index].graspQuality;
					dataFinal[poses].env = dataFinal[index].env;
					dataFinal[poses].direction = dataFinal[index].nodeInfo.dirs[dataHomePoses[poses].type-1]-'0';
				}
	      	}

				// If still no grasp found, good luck
	      	if(graspFound == false){
				printf(" Not OK.\n");
				dataFinal[poses] = dataHomePoses[poses]; // Storing dummy data
			}

			dataFinal[poses].nodeInfo = dataHomePoses[poses].nodeInfo;
	      	dataFinal[poses].objType = kinectControl.objectDict[object].description;
	      	dataFinal[poses].objPose = {kinectControl.objectDict[object].poses[currentPoseCode][1],
	                           			  kinectControl.objectDict[object].poses[currentPoseCode][2],
	                           			  currentYaw*(M_PI/180.0)};
	      	dataFinal[poses].type = dataHomePoses[poses].type;
	      	dataFinal[poses].nSteps = dataFinal[poses].path.size()-dataFinal[poses].type;
	      	dataFinal[poses].obj = dataHomePoses[poses].obj;
	      	dataFinal[poses].unexp = dataHomePoses[poses].unexp;

	      	if(graspFound == true) printf(" OK. %d Steps, Dir : %d.\n",dataFinal[poses].nSteps,dataFinal[poses].direction);

			// Setting max steps for the parent based on the child results
			for(int index = 0; index < dataHomePoses.size(); index++){
				if(dataHomePoses[poses].nodeInfo.parentID == dataHomePoses[index].nodeInfo.nodeID){
					int steps = dataFinal[poses].path.size() - dataHomePoses[index].path.size();
					if(dataHomePoses[index].nodeInfo.allowedSteps > steps){
						dataHomePoses[index].nodeInfo.allowedSteps = steps-1;
						dataHomePoses[index].nodeInfo.possibleSolChildIndex = poses;
					}
					break;
				}
			}

			// Deleting the new configs saved
	      	kinectControl.configurations.resize(nHomeConfigs);
	    }

		// Saving the results to csv
		for(int poses = 0; poses < dataHomePoses.size() ; poses++){
			dataFinal[poses].filename = getCurTime()+"_"+std::to_string(nSaved);
			if(dataFinal[poses].nSteps > 0 && dataFinal[poses].nSteps <= ::maxRndSteps+1){
				nSaved++;
				saveData(dataFinal[poses], fout, dir);
			}
		}

		// std::cout << std::endl;
		// for(int poses = dataHomePoses.size()-1; poses >= 0 ; poses--){
		// 	dataHomePoses[poses].nodeInfo.print();
		// }
    	kinectControl.reset();
	}
  	kinectControl.deleteObject(object);
	fout.close();
	return(nSaved);
}

void help(){
  std::cout << "******* Data Collector Node Help *******" << std::endl;
  std::cout << "Arguments : [mode]" << std::endl;
  std::cout << "mode: 0 (default)=data collection, 1=BFS" << std::endl;
  std::cout << "*******" << std::endl;
}

int main (int argc, char** argv){
	ros::init(argc, argv, "Data_Collector_Node");
	srand(time(NULL)); // Randoming the seed used for rand generation

 	ros::NodeHandle nh;
	std::chrono::high_resolution_clock::time_point start,end;

 	environment kinectControl(&nh);
	sleep(1);

	kinectControl.setPtCldNoise(0.0);
  	kinectControl.loadGripper();

	bool relativePath; nh.getParam("/active_vision/dataCollector/relative_path", relativePath);

	std::string temp;
	nh.getParam("/active_vision/dataCollector/directory", temp);
	std::string dir;
	if(relativePath == true){
		dir = ros::package::getPath("active_vision") + temp;
	}else{
		dir = temp;
	}

	std::string time = getCurTime();
	nh.getParam("/active_vision/dataCollector/csvName", temp);
	std::string csvName;
	if(temp == "default.csv") csvName = time+"_dataRec.csv";
	else	csvName = temp;

	if(csvName.substr(csvName.size() - 4) != ".csv"){
		std::cout << "ERROR. Incorrect file name." << std::endl;
    return(-1);
	}

	int objID;
	nh.getParam("/active_vision/dataCollector/objID", objID);
	if(kinectControl.objectDict.count(objID) == 0){
		std::cout << "ERROR. Incorrect Object ID." << std::endl;
    return(-1);
	}

	int nDataPoints;
	nh.getParam("/active_vision/dataCollector/nData", nDataPoints);

	nh.getParam("/active_vision/dataCollector/homePosesCSV", temp);
	std::string homePosesTreeCSV = ros::package::getPath("active_vision") + "/misc/" + temp;

	nh.getParam("/active_vision/kinectMode", ::mode);
	if(::mode < 1 && ::mode > 2) ::mode = 2;

	nh.getParam("/active_vision/dataCollector/nRdmSearch", ::nRdmSearches);
	nh.getParam("/active_vision/dataCollector/maxRdmSteps", ::maxRndSteps);

	std::cout << "Start Time : " << getCurTime() << std::endl;
	start = std::chrono::high_resolution_clock::now();
	int ctrData;

	if(argc <= 1 or std::atoi(argv[1]) == 0){
		printf("***** Object #%d *****\n",objID);
		ctrData = generateData(kinectControl, objID, homePosesTreeCSV, dir, csvName, nDataPoints);
	} else if(std::atoi(argv[1]) == 1){
		// Reading the yawValues csv file
		int objPoseCode, objYaw;
		std::string yawAnglesCSVDir;
		nh.getParam("/active_vision/policyTester/yawAnglesCSVDir", yawAnglesCSVDir);
		yawAnglesCSVDir = ros::package::getPath("active_vision") + yawAnglesCSVDir;
		std::string yawAnglesCSV;
		nh.getParam("/active_vision/policyTester/yawAnglesCSV", yawAnglesCSV);
		std::vector<std::vector<std::string>> yawAngle = readCSV(yawAnglesCSVDir+yawAnglesCSV);
		nh.getParam("/active_vision/dataCollector/objID", objID);
		if(kinectControl.objectDict.count(objID) == 0){
			std::cout << "ERROR. Incorrect Object ID." << std::endl;
	    return(-1);
		}
		printf("BFS***** Object #%d *****\n",objID);

		// Converting it to a dictionary format (First column is the key)
		std::map<std::string,std::vector<int>> yawAngleDict;
		for(int row = 0; row < yawAngle.size(); row++){
			yawAngleDict.insert({yawAngle[row][0],{}});
			for(int col = 1; col < yawAngle[row].size(); col++){
				yawAngleDict[yawAngle[row][0]].push_back(std::stoi(yawAngle[row][col]));
			}
		}

		int uniformYawStepSize; nh.getParam("/active_vision/policyTester/uniformYawStepSize", uniformYawStepSize);
		for(int objPoseCode = 0; objPoseCode < kinectControl.objectDict[objID].nPoses; objPoseCode+=1){
			int tempNDataPoints = nDataPoints;

			// Checking for the uniform steps
			for(int objYaw = kinectControl.objectDict[objID].poses[objPoseCode][3]; objYaw < kinectControl.objectDict[objID].poses[objPoseCode][4]; objYaw+=uniformYawStepSize){
				findOptimalPaths(kinectControl, objID, homePosesTreeCSV, dir, csvName, objPoseCode, objYaw);
				tempNDataPoints--;
				ctrData++;
			}

			std::string key = std::to_string(int(kinectControl.objectDict[objID].poses[objPoseCode][3]))+"-"+
			                  std::to_string(int(kinectControl.objectDict[objID].poses[objPoseCode][4]));

			if(yawAngleDict.count(key) == 0) continue;

			// Checking for the random steps
			for(int ctr = 0; ctr < yawAngleDict[key].size() && tempNDataPoints > 0; ctr++){
				findOptimalPaths(kinectControl, objID, homePosesTreeCSV, dir, csvName, objPoseCode, yawAngleDict[key][ctr]);
				tempNDataPoints--;
				ctrData++;
			}
		}
	} else {
		printf("Error: unknown mode %d\n", mode);
	}

	end = std::chrono::high_resolution_clock::now();
	std::cout << "End Time : " << getCurTime() << std::endl;
	double elapsed = (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count()/1000;
	printf("Time Taken (hrs) : %1.2f\n",elapsed/60/60);
	printf("Data Collected : %d\n",ctrData);
	printf("Time per data collected (sec) : %1.2f\n",elapsed/ctrData);
	printf("Data saved to : %s\n",csvName.c_str());
 }
