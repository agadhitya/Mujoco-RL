#include <ros/ros.h>
#include <bitset>

#include "owi_robot/generatecmd.h"

#define BASE_L      "00000010"
#define BASE_R      "00000001"

#define ELBOW_UP    "00010000"
#define ELBOW_DN    "00100000"

#define BYTE_SIZE   8
#define SPLITTER    "00001111"

bool cmdGenServer_callback(owi_robot::generatecmd::Request &req, owi_robot::generatecmd::Response &res){
    int m = req.motor_num;
    std::bitset<BYTE_SIZE> bset1;
    if(m == 1){
        //define bitset for wrist
    }
    else if(m == 2){
        std::bitset<BYTE_SIZE> temp(std::string(ELBOW_DN));
        bset1 = bset1 | temp ;
    }
    else if(m == 3){
        //define bitset for shoulder
    }
    else if(m == 4){
        if(req.direction == true){
            std::bitset<BYTE_SIZE> temp(std::string(BASE_L));
            bset1 = bset1 | temp;
        }
        else{
            std::bitset<BYTE_SIZE> temp(std::string(BASE_R));
            bset1 = bset1 | temp;
        }
    }
    
    std::bitset<BYTE_SIZE> splitter(std::string(SPLITTER));
    // std::cout << splitter << std::endl;

    auto right_half = std::bitset<BYTE_SIZE/2> ((bset1 & splitter).to_ulong());
    auto left_half = std::bitset<BYTE_SIZE/2> (((bset1 >> 4) & splitter).to_ulong());

    // std::cout << right_half << std::endl;
    // std::cout << left_half << std::endl;

    int b = right_half.to_ulong();
    int a = left_half.to_ulong();

    // std::cout << a << std::endl;
    // std::cout << b << std::endl;

    std::string byte1 = std::to_string(((10*a)+b));
    std::cout <<byte1 <<std::endl;

    res.str1 = byte1;
    res.str2 = "00";
    res.str3 = "00";








    return true;
}

int main(int argc, char **argv){

    ros::init(argc, argv, "cmdGenerate");
    ros::NodeHandle n;

    ros::ServiceServer cmdGen = n.advertiseService("cmdGenServer", cmdGenServer_callback);
    ros::spin();


    return 0;
}