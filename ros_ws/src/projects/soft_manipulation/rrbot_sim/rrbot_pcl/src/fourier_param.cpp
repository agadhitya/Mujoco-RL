#include "ros/ros.h"
#include "rrbot_pcl/fourierParam.h"

bool fourierCoeff(rrbot_pcl::fourierParam::Request &req, rrbot_pcl::fourierParam::Response &res){
    int N = 10;
    int P = (4*N) + 2;
    // Declare S vector of length P
    // Declare G(p) and loop to create G matrix of size 2L X P
    // Declare c vector o lenght 2L > P
    // Compute S = (G^TG)^-1 (G^Tc)
return true;
}

int main(int argc, char** argv){
    ros::init(argc, argv, "fourier_param");
    ros::NodeHandle n;
    
    ros::ServiceServer fourierCoeffServ = n.advertiseService("fourier_coeff", fourierCoeff); 
    return 0;
}