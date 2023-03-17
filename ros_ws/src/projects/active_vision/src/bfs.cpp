#include <active_vision/environment.h>
#include <active_vision/toolViewPointCalc.h>

bool saveOnlyUsefulData = true;
int mode = 1;
int moveCost = -1;
int graspOKreward = 5;
int nSearches = 0;

struct dataPoint{
	std::vector<int> dir;
	std::vector<std::vector<double>> path;
	int parentID;
	int nodeID;
	int parentConfigID;
	float reward;
	bool success;
	int ctr[2];

	void print(){
		std::cout << std::setw(3) << nodeID <<
		             std::setw(3) << parentID <<
								 std::setw(3) << success <<
								 std::setw(5) << ctr[0] << "/" << ctr[1] <<
								 // std::setw(10) << std::setprecision(3) << reward <<
								 "\t0";
		for(int i = 0; i < dir.size(); i++) {
			std::cout << "-" << dir[i] ;
		}
		std::cout << std::endl;
	}

	void printStatus(){
		std::cout << "Testing " << std::setw(3) << nodeID << "  0";
		for(int i = 0; i < dir.size(); i++) {
			std::cout << "-" << dir[i] ;
		}
		std::cout << std::endl;
	}

	void saveToCSV(std::fstream &saveTo){
		saveTo << parentID << ","
		       << nodeID << ","
					 << success << ","
					 << ctr[0] << ","
					 << ctr[1] << ","
					 << reward << ",";

		saveTo << "0";
	  for(int i = 0; i < dir.size(); i++){
			saveTo << "-" << dir[i] ;
		}
		saveTo << "\n";
	}
};

void getChildNodes(environment &env, dataPoint &parent, std::vector<dataPoint> &nodes){
	static int nodeCtr = 0;
	dataPoint temp;
	std::vector<double> nxtPose;
	for(int i = 1; i <= 8; i++) {
		temp = parent;
		if(parent.success != true){
			temp.parentConfigID = env.saveConfiguration(std::to_string(parent.nodeID));
		}else{
			temp.parentConfigID = -1;
		}
		nxtPose = calcExplorationPose(parent.path.back(),i,::mode);
		if(checkValidPose(nxtPose) == true && checkIfNewPose(parent.path,nxtPose,::mode) == true){
			temp.dir.push_back(i);
			temp.path.push_back(nxtPose);
			temp.parentID = parent.nodeID;
			nodeCtr++;
			temp.nodeID = nodeCtr;
			nodes.push_back(temp);
		}
	}
}

void checkGrasp(environment &env, dataPoint &data){
	if(data.success != true){
		if(data.parentID != -1 && data.parentConfigID < 0){
			printf("ERROR checkGrasp: Wrong config ID for reset\n");
		}
		else{
			if(data.parentID == -1){
				singlePass(env, data.path.back(), true, true);
			}else{
				env.rollbackConfiguration(data.parentConfigID);
				singlePass(env, data.path.back(), false, true);
			}
			::nSearches++;
			data.success = (env.selectedGrasp != -1);
		}
	}
	data.reward = int(data.path.size()-1)*(::moveCost) + data.success*(::graspOKreward);
}

std::vector<dataPoint> bfsSearch(environment &env, std::vector<double> home, int depth){
	std::vector<dataPoint> nodesActive, nodesClosed;
	dataPoint root, current;
	root.dir = {};
	root.path = {home};
	root.parentID = -1;
	root.parentConfigID = -1;
	root.nodeID = 0;
	root.success = false;
	root.ctr[0] = 0; root.ctr[1] = 0;

	nodesActive.push_back(root);
	int curDepth = 0;
	while(nodesActive.size() > 0){
		current = nodesActive[0]; nodesActive.erase(nodesActive.begin());
		current.printStatus();
		checkGrasp(env,current);
		if(current.dir.size() < depth){
			getChildNodes(env,current,nodesActive);
		}
		current.ctr[0] = current.success;
		current.ctr[1] = 1;

		nodesClosed.push_back(current);
	}
	return(nodesClosed);
}

void help(){
  std::cout << "******* Supervisor Node Help *******" << std::endl;
  std::cout << "Arguments : [Object] [Depth] [Mode]" << std::endl;
	std::cout << "Object : Object ID -> 0(Drill),1(Sq Prism),2(Rect Prism)" << std::endl;
	std::cout << "Depth : 1-3" << std::endl;
	std::cout << "Mode : 1->Normal, 2->New" << std::endl;
  std::cout << "*******" << std::endl;
}

int main (int argc, char** argv){
	ros::init(argc, argv, "BFS_Node");

	if(argc != 4){
    std::cout << "ERROR. Incorrect number of arguments." << std::endl;
    help(); return(-1);
  }

 	ros::NodeHandle nh;
	std::chrono::high_resolution_clock::time_point start,end;

 	environment kinectControl(&nh);
	sleep(1);
  kinectControl.loadGripper();

	int objID = std::atoi(argv[1]);
	if(objID < 0 && objID > 2) objID = 0;
	int depth = std::atoi(argv[2]);
	if(depth < 1 && depth > 3) depth = 1;
	::mode = std::atoi(argv[3]);
	if(::mode < 1 && ::mode > 2) ::mode = 1;

	int objPoseCode, objYaw;
	printf("Enter the object pose code (0-%d) : ", int(kinectControl.objectPosesDict[objID].size()-1));
	std::cin >> objPoseCode;
	if(objPoseCode < 0 || objPoseCode > kinectControl.objectPosesDict[objID].size()-1) objPoseCode = 0;
	printf("Enter the object yaw (deg) (0-360) : ");
	std::cin >> objYaw;

	std::string dir;
	nh.getParam("/active_vision/data_dir", dir);
	std::string csvName;
	std::stringstream ss;
	ss << "bfs_" <<
	      "O" << argv[1] <<
				"P" << std::to_string(objPoseCode) <<
				"Y" << std::to_string(objYaw) <<
				"D" << argv[2] <<
				"M" << argv[3] <<
				".csv";
	csvName = ss.str();

	std::fstream fout;
 	fout.open(dir+csvName, std::ios::out);

	fout<<kinectControl.objectDict[objID][1]<<","
	    <<objPoseCode<<","
			<<objYaw<<","
			<<argv[2]<<","
			<<argv[3]<<"\n";

	kinectControl.spawnObject(objID,objPoseCode,objYaw*(M_PI/180.0));

	std::vector<double> homePos = {1.0, 180*(M_PI/180.0), 45*(M_PI/180.0)};
	std::cout << "Start Time : " << getCurTime() << std::endl;
	start = std::chrono::high_resolution_clock::now();

	printf("***** Object #%d, Depth #%d *****\n",objID,depth);

	std::vector<dataPoint> result;
	result = bfsSearch(kinectControl, homePos, depth);
	for(int i = result.size()-1; i > 0; i--){
		if(result[i].parentID == -1) printf("ERROR\n");
		result[result[i].parentID].ctr[0] += result[i].ctr[0];
		result[result[i].parentID].ctr[1] += result[i].ctr[1];
	}

	for(int i = 0; i < result.size(); i++){
		// result[i].print();
		result[i].saveToCSV(fout);
	}

	printf("Data saved to : %s\n",csvName.c_str());
	fout.close();

	std::cout << "Number of paths : " << result.size() << std::endl;
	std::cout << "Number of searches : " << ::nSearches << std::endl;

	end = std::chrono::high_resolution_clock::now();
	std::cout << "End Time : " << getCurTime() << std::endl;

	kinectControl.deleteObject(objID);

	double elapsed = (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count()/1000;
	printf("Time Taken (sec) : %1.2f\n",elapsed);
	printf("Time Taken / search (sec) : %1.2f\n",elapsed/::nSearches);
 }
