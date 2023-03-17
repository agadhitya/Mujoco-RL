#!/bin/bash

# Script to collect data, save parameters, generate its summary & move to storage folder.

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

cur=$(pwd)
pkgPath=$(rospack find active_vision)
src=$pkgPath"/dataCollected/trainingData/"

csvDataRec='dataRec.csv'
csvParams='parameters.csv'
csvStorageSummary='storageSummary.csv'
objectID=(19 20)
nData=(100 100)

now="$(date +'%Y/%m/%d %I:%M:%S')"
printf "Started at yyyy/mm/dd hh:mm:ss format %s\n" "$now"

# Creating a screen to run ROS & gazebo
printf "Starting gazebo ...\n"
gnome-terminal -- bash -c 'screen -d -R -S session-environment' & sleep 5
# Starting the gazebo and loading the parameters
screen -S session-environment -X stuff $'roslaunch active_vision workspace.launch visual:="ON"\n'
sleep 10

# Collecting data
for ((i=0;i<${#objectID[@]};++i)); do
		# Setting the rosparams
		rosparam set /active_vision/dataCollector/objID ${objectID[i]}
		rosparam set /active_vision/dataCollector/nData ${nData[i]}
		rosparam set /active_vision/dataCollector/csvName ${csvDataRec}

		# Saving the parameters
		rosrun active_vision saveParams.sh ${src}${csvParams}

		# Starting a screen and running dataCollector.cpp
		printf "Collecting "${nData[i]}" data points for Object ID : "${objectID[i]}" ..."
		gnome-terminal -- bash -c 'screen -d -R -S session-dataCollection' & sleep 5
		# Dummy line for debug
		# screen -S session-dataCollection -X stuff $'sleep 7\nexit\n' 

		# Standard data collection
		#screen -S session-dataCollection -X stuff $'rosrun active_vision dataCollector\nexit\n'

		# BFS collection
		screen -S session-dataCollection -X stuff $'rosrun active_vision dataCollector 1\nexit\n'

		# Waiting till datacollection is over
		screenOK="$(checkScreen session-dataCollection)"
		while [[ "$screenOK" == "true" ]]; do
			printf ".";	sleep 60
			screenOK="$(checkScreen session-dataCollection)"
		done
		printf "\n"

done

# Closing gazebo
printf "Closing gazebo ..."
screen -S session-environment -X stuff "^C"
screen -S session-environment -X stuff $'sleep 1\nexit\n'

# Waiting till gazebo is closed
screenOK="$(checkScreen session-environment)"
while [[ "$screenOK" == "true" ]]; do
	printf ".";	sleep 5
	screenOK="$(checkScreen session-environment)"
done
printf "\n"

printf "Gazebo closed.\n"; sleep 2

printf "***********\n"

# Get the list of csv files in source directory
csvList=()
csvList+=($src$csvDataRec)
# for file in $(find $src -name "*.csv"); do
# 	if [ "$(basename $file)" != "$csvParams" ]; then
#  		csvList+=($file)
# 	fi
# done

# Generate Summary and State Vector
for csv in ${csvList[@]}; do
	printf "Using CSV : "$(basename $csv)"\n"
	printf "Generating Summary...\n"
	rosrun active_vision summarizerDataCollected.py $csv GRAPH_SAVE
	# printf "Generating State Vector...\n"
	# rosrun active_vision genStateVec $(dirname $csv)/ $(basename $csv) 1 5
done

printf "***********\n"

dst=$pkgPath"/dataCollected/storage/"

# Create a new directory
dataNo="1"
ok=false
while [ $ok = false ]; do
	dstToCheck="$dst""Data_$dataNo""/"
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
