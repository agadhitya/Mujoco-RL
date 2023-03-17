#!/bin/bash

# ARGUMENTS for policyEvaluator.sh
# [Path] [csv1] [csv2] ...
# [Path] : Path (relative to active_vision pkg) to folder with state vector (Eg : /misc/State_Vector/)
# [csv1] : Optional - csv file to be used, If not mentioned all files are used
# [csv2] : Optional - csv file to be used,

# Function to check if a screen is still running
checkScreen () {
  local result=false
	shopt -s nullglob
	local screens=(/var/run/screen/S-*/*)
	shopt -u nullglob
	for s in ${screens[*]}; do
			if [[ "$s" == *"$1"* ]]; then
			  result=true
			fi
	done
  echo "$result"
}

# Script to test the provided list of policies on the provided data
cur=$(pwd)
pkgPath=$(rospack find active_vision)
src=$pkgPath"/dataCollected/testData/"

csvDataRec='dataRec.csv'
csvParams='parameters.csv'
csvStorageSummary='storageSummary.csv'

simulationMode="SIMULATION" # SIMULATION / FRANKASIMULATION / FRANKA
objectID=(12 13 14 15 16 18 19 20 21)
nDataPoints=(10 10 10 10 10 10 10 10 10)

# "RANDOM" "PCA_LR" "PCA_LDA" "PCA_LDA_LR" "HEURISTIC")
# List all the policies to be tested with the prefix "Policy"
# Policyxxx = ("Policy" "Unique description" "Param 1 name" "Param 1 value" ...)

# Policy1A=("HEURISTIC" "Heuristic")
# Policy1B=("BRICK" "Brick")
# Policy1C=("RANDOM" "Random")
Policy1D=("3DHEURISTIC" "3D_Heuristic")
# # Policy2A=("PCA_LR" "PCA_LR_95"
# #           "/active_vision/policyTester/PCAcomponents" 0.95
# #           "/active_vision/policyTester/HAFstVecGridSize" 7)
# Policy2B=("PCA_LR" "PCA_LR_85"
#            "/active_vision/policyTester/PCAcomponents" 0.85
#            "/active_vision/policyTester/HAFstVecGridSize" 5)
# # Policy2C=("PCA_LR" "PCA_LR_75"
# #           "/active_vision/policyTester/PCAcomponents" 0.75
# #           "/active_vision/policyTester/HAFstVecGridSize" 7)
# # Policy3A=("PCA_LDA" "PCA_LDA_95"
# #           "/active_vision/policyTester/PCAcomponents" 0.95
# #           "/active_vision/policyTester/HAFstVecGridSize" 7)
# Policy3B=("PCA_LDA" "PCA_LDA_85"
#             "/active_vision/policyTester/PCAcomponents" 0.85
#             "/active_vision/policyTester/HAFstVecGridSize" 5)
# # Policy3C=("PCA_LDA" "PCA_LDA_75"
# #           "/active_vision/policyTester/PCAcomponents" 0.75
# #           "/active_vision/policyTester/HAFstVecGridSize" 7)
# Policy5=("QLEARN" "Q_Learning"
#             "/active_vision/policyTester/HAFstVecGridSize" 5)
# Policy6=("PROBABILISTIC" "Prob_Paper")

files=()

stVecSrc=$pkgPath"$1"
# If csv file name is specified then use the ones specified else use all
argc=$#
argv=("$@")

if [[ "$argc" -eq "0" ]]; then
  echo "ERROR : No arguments"
  exit 1
fi

if [[ "$argc" -gt "1" ]]; then
  for (( idx=1; idx < argc; idx++)); do
    echo "Using state vector : ${argv[idx]}."
  	files+=("${argv[idx]}")
  done
else
  for file in $(ls $stVecSrc | grep .csv); do
  	echo "Using state vector : $file."
  	files+=("$file")
  done
fi

now="$(date +'%Y/%m/%d %I:%M:%S')"
printf "Started at yyyy/mm/dd hh:mm:ss format %s\n" "$now"

# Creating a screen to run ROS & gazebo
printf "Starting gazebo ...\n"
gnome-terminal -- bash -c 'screen -d -R -S session-environment' & sleep 5
# Starting the gazebo and loading the parameters
screen -S session-environment -X stuff $'roslaunch active_vision workspace.launch visual:="OFF" simulationMode:="'$simulationMode'"\n'
sleep 10

# Looping over the objects and testing each policy for each.
nDataPtsCtr=0
for objID in ${objectID[@]}; do
  rosparam set /active_vision/policyTester/objID $objID
  rosparam set /active_vision/policyTester/nDataPoints ${nDataPoints[nDataPtsCtr]}
	for vars in ${!Policy*}; do
    # Setting the policy and its parameters
    declare -n policy=$vars
		rosparam set /active_vision/policyTester/policy ${policy[0]}
    for (( idx=2; idx<${#policy[@]}; idx+=2)); do
      rosparam set ${policy[idx]} ${policy[idx+1]}
    done

    nHeuristic=0 # To ensure heuristic is run only once
    n3DHeuristic=0 # To ensure heuristic is run only once
    nProbabilistic=0 # To ensure probabilistic is run only once

    for stVec in ${files[@]}; do
      # Setting the csv state vector to be used
      rosparam set /active_vision/policyTester/csvStVecDir $1
      rosparam set /active_vision/policyTester/csvStVec $stVec

      #Save the results to policy:stVec:dataRec.csv (triming the extra .csv)
  		rosparam set /active_vision/policyTester/csvName "${policy[1]}:${stVec: 0: -4}:dataRec.csv"

      printf "Evaluating ${policy[1]} with ${stVec} on object ${objID}...\n"

      # Saving the parameters
  		rosrun active_vision saveParams.sh ${src}${csvParams}

  		# Open a terminal for the service and the tester
  		gnome-terminal -- bash -c 'screen -d -R -S session-policyService'
  		gnome-terminal -- bash -c 'screen -d -R -S session-policyTester'
  		sleep 5

  		# Start the service and the tester
      # Heurisic doesnot depend on state vector
      if [[ "$policy" == "HEURISTIC" ]]; then
        if [[ "$nHeuristic" -eq "0" ]]; then
          screen -S session-policyService -X stuff $'rosrun active_vision heuristicPolicyService 0\n'
          screen -S session-policyTester -X stuff $'rosrun active_vision policyTester 1\n1\n exit\n'
          # screen -S session-policyTester -X stuff $'sleep 7\n' #Debug line
          nHeuristic=1
        fi
      elif [[ "$policy" == "3DHEURISTIC" ]]; then
        if [[ "$n3DHeuristic" -eq "0" ]]; then
          screen -S session-policyService -X stuff $'rosrun active_vision 3DheuristicPolicyService\n'
          screen -S session-policyTester -X stuff $'rosrun active_vision policyTester 1\n1\n exit\n'
          # screen -S session-policyTester -X stuff $'sleep 7\n' #Debug line
          n3DHeuristic=1
        fi
      elif [[ "$policy" == "PROBABILISTIC" ]]; then
        if [[ "$nProbabilistic" -eq "0" ]]; then
          screen -S session-policyTester -X stuff $'rosrun active_vision probabilisticPaper\n1\n exit\n'
          # screen -S session-policyTester -X stuff $'sleep 7\n' #Debug line
          nProbabilistic=1
        fi
      else
        screen -S session-policyService -X stuff $'rosrun active_vision trainedPolicyService.py\n'
  			screen -S session-policyTester -X stuff $'rosrun active_vision policyTester 2\n1\n exit\n'
        # screen -S session-policyTester -X stuff $'sleep 7\n' #Debug line
  		fi
  		sleep 5
  		screen -S session-policyTester -X stuff $'sleep 1\nexit\n'

  		# Waiting till testing is over
  		screenOK="$(checkScreen session-policyTester)"
  		while [[ "$screenOK" == "true" ]]; do
  			printf ".";	sleep 10
  			screenOK="$(checkScreen session-policyTester)"
  		done
  		printf "\n"
  		screen -S session-policyService -X stuff "^C"
  		screen -S session-policyService -X stuff $'sleep 1\nexit\n'
  		sleep 2

    done
	done
  nDataPtsCtr=$[$nDataPtsCtr+1]
done

# Closing gazebo
printf "Closing gazebo ...\n"
screen -S session-environment -X stuff "^C"
screen -S session-environment -X stuff $'sleep 1\nexit\n'

printf "***********\n"

# Creating the summary
printf "Generating Summary...\n"
rosrun active_vision summarizerResults.py $src

dst=$pkgPath"/dataCollected/storage/"

# Create a new directory
dataNo="1"
ok=false
while [ $ok = false ]; do
	dstToCheck="$dst""Test_$dataNo""/"
	if [ ! -d $dstToCheck ]; then
		mkdir $dstToCheck
		ok=true
		printf $(basename $dstToCheck)" folder created.\n"
	fi
	dataNo=$[$dataNo+1]
done

# Copying the parameters to summary folder
csvToCheck="${dst}""${csvStorageSummary}"
lineNo="1"
while IFS= read line; do
	if [ $lineNo == "1" ]; then
		if [ ! -f $csvToCheck ]; then
			echo -n "Folder Name, Description,,,,""$line" >> $csvToCheck
		fi
	else
		echo -n $(basename $dstToCheck)", Collection,,,,""$line" >> $csvToCheck
	fi
	echo "" >> $csvToCheck
	lineNo=$[$lineNo+1]
done <"${src}${csvParams}"

printf "Folder and parameter details added to "${csvStorageSummary}".\n"

# Copying the files to the created folder
cd $src
shopt -s extglob
mv !(ReadMe.txt) $dstToCheck
shopt -u extglob
cd $cur
printf "Files Moved.\n"

printf "***********\n"

now="$(date +'%Y/%m/%d %I:%M:%S')"
printf "Ended at yyyy/mm/dd hh:mm:ss format %s\n" "$now"
