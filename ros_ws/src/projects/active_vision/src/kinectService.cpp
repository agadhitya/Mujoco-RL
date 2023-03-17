#include <active_vision/environment.h>
#include <active_vision/toolDataHandling.h>
#include <active_vision/toolViewPointCalc.h>
#include <active_vision/toolVisualization.h>
#include <active_vision/toolStateVector.h>
#include <active_vision/heuristicPolicySRV.h>
#include <active_vision/trainedPolicySRV.h>
#include <active_vision/controlSRV.h>
#include <active_vision/restartObjSRV.h>

int mode;
int runMode;
int testAll;
int maxSteps;
int nSaved;
int HAFstVecGridSize;
environment *kC;
RouteData *home, *cur;
std::vector<double> startPose;
int cObj;

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

void resetEnv(){
	kC->spawnObject(cObj,0,0);
	startPose = {kC->viewsphereRad, 180*(M_PI/180.0), 45*(M_PI/180.0)};
	kC->reset();
	// Home Pose
	home = new RouteData;
	cur = new RouteData;
	home->reset();
	home->path = {startPose};
	home->type = 1;
	home->direction = 0;
	home->nSteps = 0;
	*cur = *home;
}

std::vector<float> readyStateVec(){
	static HAFStVec1 stVec;
	static ptCldColor::Ptr tempPtCldObj{new ptCldColor};
	static ptCldColor::Ptr tempPtCldUnexp{new ptCldColor};
	stVec.setGridDim(::HAFstVecGridSize);
	stVec.setMaintainScale(true);
	*tempPtCldObj = *kC->ptrPtCldObject;
	*tempPtCldUnexp = *kC->ptrPtCldUnexp;
	stVec.setInput(tempPtCldObj,tempPtCldUnexp,cur->path.back());
	stVec.calculate();
	return stVec.getStateVec();
}

bool checkDone(){
	return cur->success;
}

void print(std::vector<double> const &a){
	for(int i=0; i<a.size(); i++){
		std::cout << a.at(i) << " ";
	}
	std::cout << std::endl;
}

std::vector<double> move(int direction){
	std::vector<int> dirPref = nearbyDirections(direction);
	int prfID = 0;
	std::vector<double> nextPose = calcExplorationPose(cur->path.back(),dirPref[prfID],::mode);
	printf("Calculated next pose.\n");
	print(nextPose);
	while(checkValidPose(nextPose) == false || checkIfNewPose(cur->path,nextPose,::mode) == false){
		prfID++;
		nextPose = calcExplorationPose(cur->path.back(),dirPref[prfID],::mode);
		print(nextPose);
	}
	printf("Checked next pose.\n");
	if(prfID!=0){
		printf("Direction modified from %d -> %d.\n", dirPref[0],dirPref[prfID]);
	}
	std::cout << dirLookup[dirPref[prfID]] << ",";
	singlePass(*kC, nextPose, false, true, 2);
	printf("Finished single pass.\n");
	updateRouteData(*kC,*cur,false,"dummy");
	cur->path.push_back(nextPose);
	cur->nSteps++;
	if(cur->nSteps == 1){
		printf("Updating home direction.\n");
		home->direction = direction;
	}
	printf("Done with move.\n");
	return nextPose;
}


bool startObj(active_vision::restartObjSRV::Request  &req,
           active_vision::restartObjSRV::Response &res){
	printf("Starting reset...\n");
	kC->deleteObject(cObj);
	cObj = req.object;
	resetEnv();
	printf("Reset object.\n");
	kC->moveObject(cObj,req.objPoseCode,req.objYaw*(M_PI/180.0));
	home->objType = kC->objectDict[cObj].description;
	home->objPose = {kC->objectDict[cObj].poses[req.objPoseCode][1],
									kC->objectDict[cObj].poses[req.objPoseCode][2],
									req.objYaw*(M_PI/180.0)};
	printf("Reset home.\n");
	singlePass(*kC, startPose, true, true, 2);
	updateRouteData(*kC,*home,true,"Home");
	res.stateVec.data = readyStateVec();
	printf("Done with reset.\n");
	return true;
}

bool takeStep(active_vision::controlSRV::Request  &req,
           active_vision::controlSRV::Response &res){
	printf("Starting step...\n");
	move(req.direction);
	res.stateVec.data = readyStateVec();
	res.done = checkDone();
	res.steps = cur->nSteps;
	printf("Done with step.\n");
	return true;
}

/*// Function the find the grasp
void findGrasp(environment &kinectControl, int object, int objPoseCode, int objYaw, std::string dir, std::string saveLocation, ros::ServiceClient &policy){

	std::fstream fout;
 	fout.open(dir+saveLocation, std::ios::out | std::ios::app);
 	printf("Starting everything.\n");

 	resetEnv();
	std::vector<double> nextPose;

	static ptCldVis::Ptr viewer(new ptCldVis("PCL Viewer"));
	static std::vector<int> vp;
	setupViewer(viewer, 2, vp);
	static keyboardEvent keyPress(viewer,2);
	printf("Moving object.\n");
	kC->moveObject(object,objPoseCode,objYaw*(M_PI/180.0));
	printf("Home setup.\n");
	home->objType = kC->objectDict[object].description;
	printf("Home setup 2.\n");
	home->objPose = {kC->objectDict[object].poses[objPoseCode][1],
									kC->objectDict[object].poses[objPoseCode][2],
									objYaw*(M_PI/180.0)};
	

	
	printf("Starting first pass.\n");
	singlePass(kinectControl, startPose, true, true, 2);
	updateRouteData(kinectControl,*home,true,"Home");

	static ptCldColor::Ptr tempPtCld{new ptCldColor};
	static ptCldColor::Ptr tempPtCldObj{new ptCldColor};
	static ptCldColor::Ptr tempPtCldUnexp{new ptCldColor};
	pcl::PointXYZ table,a1,a2;
	table.x = 1.5; table.y = 0; table.z = 1;
	std::vector<int> nUnexp;
	printf("Starting while loop.\n");
	viewer->addText("Manual Mode.\nEnter the direction.",5,30,25,1,0,0,"Dir1",vp[0]);

	while(keyPress.ok){

		if(!(::runMode != 0 && ::testAll == 1)){
			viewer->resetStoppedFlag();
			viewer->removeAllPointClouds();
			viewer->removeAllShapes();

			addRGB(viewer,kinectControl.ptrPtCldObject,"Obj",vp[0]);
			addRGB(viewer,kinectControl.ptrPtCldUnexp,"Unexp",vp[0]);
			*tempPtCld = cur->env; addRGB(viewer,tempPtCld,"Env",vp[1]);

			addViewsphere(viewer,vp.back(),table,cur->path[0][0],false);
			a1 = sphericalToCartesian(cur->path[0],table);
			viewer->addSphere(a1,0.04,1,0,0,"Start",vp[1]);
			viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,"Start");

			for(int i = 0; i < cur->path.size()-1; i++){
				a1 = sphericalToCartesian(cur->path[i],table);
				a2 = sphericalToCartesian(cur->path[i+1],table);
				viewer->addArrow(a2,a1,0,1,0,false,std::to_string(i),vp[1]);
			}

			setCamView(viewer,cur->path.back(),table);
		}

		int maxIndex = 0;

		if(!(::runMode != 0 && ::testAll == 1)){
			keyPress.called = false;
			while(!viewer->wasStopped() && keyPress.called==false){
				viewer->spinOnce(100);
				boost::this_thread::sleep (boost::posix_time::microseconds(100000));
			}
		}

		if(checkDone()){
			keyPress.dir = 0;
			keyPress.ok = false;
		}else{
			// Limiting the steps
			if(cur->nSteps > ::maxSteps){
				cur->filename = getCurTime()+"_"+std::to_string(::nSaved);
				saveData(*cur, fout, dir);
				::nSaved++;
				keyPress.dir = 0;
				keyPress.ok = false;
			}
		}

		// Reset to home if direction is 0
		if(keyPress.dir == 0){
			kinectControl.rollbackConfiguration(home->childID);
			*cur = *home;
			cur->nSteps = 0;
		}

		if(keyPress.dir >=1 && keyPress.dir <= 8){
			std::vector<int> dirPref = nearbyDirections(keyPress.dir);
			int prfID = 0;
      		nextPose = calcExplorationPose(cur->path.back(),dirPref[prfID],::mode);
			while(checkValidPose(nextPose) == false || checkIfNewPose(cur->path,nextPose,::mode) == false){
				prfID++;
				nextPose = calcExplorationPose(cur->path.back(),dirPref[prfID],::mode);
			}
			if(prfID!=0){
				printf("Direction modified from %d -> %d.\n", dirPref[0],dirPref[prfID]);
			}
			std::cout << dirLookup[dirPref[prfID]] << ",";
			singlePass(kinectControl, nextPose, false, true, 2);
			updateRouteData(kinectControl,*cur,false,"dummy");
			cur->path.push_back(nextPose);
			cur->nSteps++;
			if(cur->nSteps == 1) home->direction = keyPress.dir;
    	}
	}

	keyPress.ok = true;
	resetEnv();
	fout.close();
}*/

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
 	environment kinectControl(&nh); 
 	kC = &kinectControl;
 	sleep(1);
  	kC->loadGripper();

	bool relativePath; nh.getParam("/active_vision/policyTester/relative_path", relativePath);
	std::string temp;
	nh.getParam("/active_vision/policyTester/directory", temp);
	std::string dir;
	if(relativePath == true){
		dir = ros::package::getPath("active_vision") + temp;
	}else{
		dir = temp;
	}

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
	cObj = objID;

	nh.getParam("/active_vision/kinectMode", ::mode);
	if(::mode < 1 && ::mode > 2) ::mode = 2;

	nh.getParam("/active_vision/policyTester/maxSteps", ::maxSteps);
	nh.getParam("/active_vision/policyTester/HAFstVecGridSize", ::HAFstVecGridSize);
	ros::ServiceServer service = nh.advertiseService("/active_vision/restartEnv", startObj);
	ros::ServiceServer service2 = nh.advertiseService("/active_vision/moveKinect", takeStep);
  	ROS_INFO("Kinect control service ready.");
  	ros::spin();
  	return(0);
 }
