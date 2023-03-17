//This class records images and sensor readings in a rosbag
//If you're adding a new topic, create a new subscriber and callback
//Remember to create a new subscriber, msg and topic name object and include the msg type
//Add topic names to the the yaml and the yaml to your launch file as a rosparam
//Write the topic to the bag, to change record rate change the sleep rate
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <geometry_msgs/WrenchStamped.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/Float32.h>
#include <memory>
#include <time.h>

class Recorder{
    private:
        ros::NodeHandle n;
        rosbag::Bag* bag;

        // Message Objects
        geometry_msgs::WrenchStamped msg;
        sensor_msgs::Image img;
        geometry_msgs::Point pnt;
        std_msgs::Float32 f;
        
        // Topic Name Objects
        std::string ft_sensor; std::string camera; std::string obj_pos;
        std::string encoder_val;
        
        //Subscriber objects
        ros::Subscriber sub1;
        ros::Subscriber sub2;
        ros::Subscriber sub3;
        ros::Subscriber sub4;

        //Callbacks
        void msg_callback(const geometry_msgs::WrenchStamped message){
            msg = message;
        }
        void img_callback(const sensor_msgs::Image image){
            img = image;
        }
        void obj_pos_callback(const geometry_msgs::Point point_){
            pnt = point_;
        }
        void encoder_val_callback(const std_msgs::Float32 val){
            f = val;
        }
    public:
        Recorder(ros::NodeHandle& _n): n(_n){
            bag = new rosbag::Bag();
            
            // Change location according to your pref
            bag->open("/home/agandhi2/test.bag", rosbag::bagmode::Write);    
            
            //ROS_INFO("Opening bag");

            //Add topics to your launch file
            n.getParam("topics/topic1", ft_sensor);
            n.getParam("topics/topic2", camera);
            n.getParam("topics/topic3", obj_pos);
            n.getParam("topics/topic4", encoder_val);


            //Subscribers
            sub1 = n.subscribe(ft_sensor, 1, &Recorder::msg_callback, this);
            sub2 = n.subscribe(camera, 1, &Recorder::img_callback, this);  
            sub3 = n.subscribe(obj_pos, 1, &Recorder::obj_pos_callback, this);
            sub4 = n.subscribe(encoder_val, 1, &Recorder::encoder_val_callback, this);

            ros::Rate r{30};
            while(ros::ok()){
                ros::Time t = ros::Time::now();
                
                //Writing Topic to Bag
                bag->write(ft_sensor, t, msg);
                bag->write(camera, t, img);
                bag->write(obj_pos, t, pnt);
                bag->write(encoder_val, t, f);
                ros::spinOnce();
                r.sleep();
            }
        }

        ~Recorder(){
            bag->close();
            delete bag;
        }
};