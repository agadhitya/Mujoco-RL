#!/bin/bash
#Takes the final .csv file as the argument
dst=$1
prefix="/active_vision/"
parameters=`rosparam list`
filename=`basename $dst`

toBeSaved=("dataCollector/objID"
           "dataCollector/csvName"
					 "dataCollector/nData"
					 "dataCollector/homePosesCSV"
					 "dataCollector/maxRdmSteps"
					 "dataCollector/nRdmSearch"
					 "policyTester/policy"
					 "policyTester/objID"
					 "policyTester/csvName"
					 "policyTester/csvStVec"
					 "policyTester/maxSteps"
					 "policyTester/PCAcomponents"
					 "environment/voxelGridSize"
					 "environment/voxelGridSizeUnexp"
					 "environment/scale"
					 "environment/maxGripperWidth"
					 "environment/minGraspQuality"
					 "environment/addNoise"
					 "environment/depthNoise")

checkParam () {
	local result=false
	for p in ${toBeSaved[*]}; do
		if [[ "$p" == *"$1"* ]]; then
			result=true
			break
		fi
	done
 	echo "$result"
}

# for ((i=0;i<${#toBeSaved[@]};++i)); do
# 	printf "Collecting "${toBeSaved[i]}"\n"
# done

# If file doesnot exist setup the header
if [[ ! -f "$dst" ]]; then
	for param in $parameters; do
		if [ ${param:0:15} == $prefix ]; then
			curParam=${param:15}
			if [[ "$(checkParam "${curParam}")" == "true" ]]; then
				echo -n "$curParam," >> $dst
			fi
		fi
	done
	echo "" >> $dst
	printf ${filename}" file setup with headers.\n"
fi

# Store the parameters to the csv
for param in $parameters; do
	if [ ${param:0:15} == $prefix ]; then
		curParam=${param:15}
		if [[ "$(checkParam "${curParam}")" == "true" ]]; then
			curValue=`rosparam get $param`
			echo -n "$curValue," >> $dst
		fi
	fi
done
printf "Parameters written to "${filename}".\n"
echo "" >> $dst
