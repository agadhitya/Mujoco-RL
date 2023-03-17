#include <ros/ros.h>
#include <math.h>

#include "owi_robot/pwm.h"
#include "owi_robot/writecmd.h"

// ros::NodeHandle* nh;

// bool pwmServer_callback(owi_robot::pwm::Request &req, owi_robot::pwm::Response &res){
//     float duty_cycle = req.duty1;

//     ros::ServiceClient write = nh->serviceClient<owi_robot::writecmd>("cmdOwiServer");
//     owi_robot::writecmd write1;
//     owi_robot::writecmd write2;

//     write1.request.str1 = "00";
//     write1.request.str2 = "00";
//     write1.request.str3 = "01";

//     write2.request.str1 = "00";
//     write2.request.str2 = "00";
//     write2.request.str3 = "00";


//     // for(int i = 0; i<=180;i++){
//     //     duty_cycle = sin(i*3.14159/180);
//         ros::service::call("cmdOwiServer", write1);
//         ROS_INFO("HII");
//     //     ros::Duration d(duty_cycle);
//     //     ros::service::call("cmdOwiServer",write2);
//     //     ros::Duration d2(1 - duty_cycle);


//     // }



//     return true;
// }
// int main(int argc, char **argv){
//     ros::init(argc, argv,"pwm");
//     // ros::NodeHandle n;
//     nh = new ros::NodeHandle;

//     ros::ServiceServer pwmSrv = nh->advertiseService("pwmServer", pwmServer_callback);
//     ros::spin();

//     return 0;
// }
int main(int argc, char **argv){
    ros::init(argc, argv,"pwm");
    ros::NodeHandle n;

    ros::ServiceClient write = n.serviceClient<owi_robot::writecmd>("cmdOwiServer");
    owi_robot::writecmd write1;
    owi_robot::writecmd write2;

    write1.request.str1 = "00";
    write1.request.str2 = "00";
    write1.request.str3 = "01";

    write2.request.str1 = "00";
    write2.request.str2 = "00";
    write2.request.str3 = "00";

    // while(true){
    //     ros::service::call("cmdOwiServer", write1);
    //     ros::Duration(0.5).sleep();
    //     ros::service::call("cmdOwiServer", write2);
    //     ros::Duration(0.5).sleep();
    // }
    float duty_cycle;
     for(int i = 0; i<=180;i++){
        duty_cycle = sin(i*3.14159/180);
        ros::service::call("cmdOwiServer", write1);
        // ROS_INFO("HII");
        ros::Duration(duty_cycle).sleep();
        ros::service::call("cmdOwiServer",write2);
        ros::Duration(1 - duty_cycle).sleep();

}
}