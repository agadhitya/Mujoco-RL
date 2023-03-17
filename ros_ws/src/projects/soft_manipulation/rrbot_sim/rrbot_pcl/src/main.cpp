#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/Image.h"
#include "pcl/point_types.h"
#include "rrbot_pcl/cloudMsg.h"
#include "rrbot_pcl/processCloud.h"
#include "rrbot_pcl/concaveHull.h"
#include "rrbot_pcl/fourierParam.h"
#include "rrbot_pcl/projectCloud.h"
#include "rrbot_pcl/skelMsg.h"

int main(int argc, char **argv){
    ros::init(argc, argv, "get_cloud");
    ros::NodeHandle n;
    
    ros::service::waitForService("project_cloud", 1000);
    ros::service::waitForService("concave_hull", 1000);
    ros::service::waitForService("process_cloud", 1000);
    ros::service::waitForService("skeletonization", 1000);
    ros::service::waitForService("get_cloud", 1000);

    ros::Publisher processedCloud__pub = n.advertise<sensor_msgs::PointCloud2>("debug_processed_cloud",1);
    ros::Publisher hull_pub = n.advertise<sensor_msgs::PointCloud2>("debug_hull_cloud",1);
    // ros::Publisher img_pub = n.advertise<sensor_msgs::Image>("debug_projection", 1);
    
    ros::ServiceClient cloudClient = n.serviceClient<rrbot_pcl::cloudMsg>("get_cloud");
    ros::ServiceClient processClient = n.serviceClient<rrbot_pcl::processCloud>("process_cloud");
    ros::ServiceClient hullClient = n.serviceClient<rrbot_pcl::concaveHull>("concave_hull");
    // ros::ServiceClient fourierClient = n.serviceClient<rrbot_pcl::fourierParam>("fourier_param");
    // ros::ServiceClient projectionClient = n.serviceClient<rrbot_pcl::projectCloud>("project_cloud");
    ros::ServiceClient skelClient = n.serviceClient<rrbot_pcl::skelMsg>("skeletonization");

    
    ros::Rate r(5);
    while(ros::ok()){
        rrbot_pcl::cloudMsg cloud;
        rrbot_pcl::processCloud processedcloud;
        rrbot_pcl::concaveHull hullcloud;
        // rrbot_pcl::fourierParam fouriercloud;
        // rrbot_pcl::projectCloud image;
        rrbot_pcl::skelMsg skel;

        cloudClient.call(cloud);
    
        processedcloud.request.pc2 = cloud.response.pc2;
        if(processClient.call(processedcloud))
            processedCloud__pub.publish(processedcloud.response.pc2);

        hullcloud.request.pc2 = processedcloud.response.pc2;
        if(hullClient.call(hullcloud))
            hull_pub.publish(hullcloud.response.pc2);

        skel.request.pc2 = processedcloud.response.pc2;
        skel.request.view = true;
        skelClient.call(skel);


        // fouriercloud.request.pc2 = hullcloud.response.pc2;
        // fourierClient.call(fouriercloud);
        
        ros::spinOnce();
        r.sleep();
    }






    ros::spin();
    return 0;
}