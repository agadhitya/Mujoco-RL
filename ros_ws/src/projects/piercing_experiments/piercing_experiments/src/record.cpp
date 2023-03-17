#include <ros/ros.h>
#include "recorder.cpp"

int main(int argc, char** argv){
    ros::init(argc, argv,"record");
    ros::NodeHandle n;
    
    Recorder r(n);
    return(0);
}