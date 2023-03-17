#!/bin/sh

export folder_path=$HOME"/FF/"
export vid_path=$folder_path"videos/";
#export bag_path=$folder_path"square/hybrid/exp"
#export bag_path=$folder_path"square/planner/exp"
#export bag_path=$folder_path"square/visual/exp"
export bag_path=$folder_path"square/try/exp"

for i in 1 2 3 4
do
# Run the launch file
cd ~/mer_lab &&
roslaunch bag_extractor extract.launch ARG_NAME:=$bag_path$i/test.bag &&

#Create a folder to save files
cd ~/FF/ &&
mkdir exp_frames_$i &&

# Move images from the .ros folder to a folder
mv ~/.ros/frame*.jpg $HOME/FF/exp_frames_$i/ &&

# Run the stitching process
cd ~/FF/exp_frames_$i &&
ffmpeg -framerate 10 -i frame%04d.jpg -c:v libx264 -profile:v high -crf 30 -pix_fmt yuv420p output_$i.mp4 &&

mv output_$i.mp4 $vid_path
# export topics to csv
rostopic echo -b $bag_path$i/test.bag -p /object_position >~/FF/csv/exp$i.csv &&
rostopic echo -b $bag_path$i/test.bag -p /Action >~/FF/csv/exp_$i.csv &&
# Delete extracted frames
cd ~ &&
rm -r ~/FF/exp_frames_$i &&

# Run MATLAB scripts
echo "EXP $i DONE!"
done